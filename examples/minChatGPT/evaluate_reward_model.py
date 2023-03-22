# loads and evals a reward model using the pipeline used during rl finetuning.
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from accelerate import Accelerator

from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, HfArgumentParser, Adafactor

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler


tqdm.pandas()

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPT2 model on the IMDB dataset using PPO
# (proximal policy optimization).
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################

# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="lvwerra/gpt2-xl-stackexchange", metadata={"help": "the model name"})
    reward_model_name: Optional[str] = field(
        default="edbeeching/gpt2-xl-stackexchange_stack-exchange-paired_rmts_240000_bup",
        metadata={"help": "the model name"},
    )
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
reward_model_name = script_args.reward_model_name
dataset_name = "lvwerra/stack-exchange-paired"
config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
)

train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/rl", split="train")
train_dataset = train_dataset.select(range(100000))
# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    config, dataset_name="lvwerra/stack-exchange-paired", input_min_text_length=2, input_max_text_length=8
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, data_dir="data/rl", split="train")
    original_columns = ds.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "question": [],
            "text_j": [],
            "text_k": [],
            "response_j": [],
            "response_k": [],
        }
        for question, response_j, response_k in zip(
            examples["question"], examples["response_j"], examples["response_k"]
        ):
            question = "Question: " + question + "\n\nAnswer: "
            text_j = "Question: " + question + "\n\nAnswer: " + response_j
            text_k = "Question: " + question + "\n\nAnswer: " + response_k

            new_examples["question"].append(question)
            new_examples["text_j"].append(text_j)
            new_examples["text_k"].append(text_k)
            new_examples["response_j"].append(response_j)
            new_examples["response_k"].append(response_k)

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["text_j"]) < 512 and len(x["text_k"]) < 512, batched=False)

    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(config)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config.batch_size,
    collate_fn=collator,
    shuffle=True,
    drop_last=True,
)


sentiment_pipe = pipeline(
    "sentiment-analysis", model=reward_model_name, device_map="auto", model_kwargs={"load_in_8bit": True}
)

rewards_js = []
rewards_ks = []

for epoch, batch in tqdm(enumerate(dataloader)):
    if epoch >= 100:
        break
    texts_j = batch["text_j"]
    texts_k = batch["text_k"]

    pipe_outputs_j = sentiment_pipe(texts_j, **sent_kwargs)
    pipe_outputs_k = sentiment_pipe(texts_k, **sent_kwargs)
    rewards_j = [output[0]["score"] for output in pipe_outputs_j]
    rewards_k = [output[0]["score"] for output in pipe_outputs_k]
    rewards_js.extend(rewards_j)
    rewards_ks.extend(rewards_k)


# create a histogram of rewards

import numpy as np
import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.title(f"preffered response reward dist mean = {sum(rewards_js)/len(rewards_js)}")
plt.hist(rewards_js, bins="auto")
plt.subplot(1, 2, 2)
plt.title(f"other response reward dist mean = {sum(rewards_ks)/len(rewards_ks)}")
plt.hist(rewards_ks, bins="auto")
plt.show()
print(rewards_ks)
