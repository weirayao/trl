from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import torch
import evaluate
import numpy as np
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False, metadata={"help": "If you want to resume training where it left off."}
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=2)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[int] = field(default=0.001)
    model_name: Optional[str] = field(
        default="gpt2",
        metadata={"help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "The number of training epochs for the reward model."}
    )
    train_subset: Optional[int] = field(
        default=100000, metadata={"help": "The size of the subset of the training data to use"}
    )
    eval_subset: Optional[int] = field(
        default=50000, metadata={"help": "The size of the subset of the eval data to use"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the human stack-exchange-paired dataset for tuning the reward model.
# TODO: it would be nice to have these as one object, it must be possible?
train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/reward", split="train")
if script_args.train_subset > 0:
    train_dataset = train_dataset.select(range(script_args.train_subset))
eval_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/evaluation", split="train")
if script_args.eval_subset > 0:
    eval_dataset = eval_dataset.select(range(script_args.eval_subset))
# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
model_name_split = script_args.model_name.split("/")[-1]
output_name = f"{model_name_split}_stack-exchange-paired_rmts__{script_args.train_subset}_{script_args.learning_rate}"

training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="epoch",
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
)

# Load the value-head model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name,
    num_labels=1,
    torch_dtype=torch.float16 if script_args.bf16 else None,
)

# Need to do this for gpt2, because it doesn't have an official pad token.
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

num_proc = 24  # Can adjust to be higher if you have more processors.
original_columns = train_dataset.column_names


# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess_function(examples):
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
        tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_j, truncation=True)
        tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_k, truncation=True)

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

    return new_examples


train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=original_columns,
)
eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=original_columns,
)


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append({"input_ids": feature["input_ids_j"], "attention_mask": feature["attention_mask_j"]})
            features_k.append({"input_ids": feature["input_ids_k"], "attention_mask": feature["attention_mask_k"]})
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


# Define the metric that we'll use for validation.
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)


class RewardTrainer(Trainer):
    # Define how to compute the reward loss.
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


# Train the model, woohoo.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer),
)

trainer.train(script_args.resume_from_checkpoint)

# Push to the hub so you can share it with people :D
model.push_to_hub(output_name + "_hub")
tokenizer.push_to_hub(output_name + "_hub")
