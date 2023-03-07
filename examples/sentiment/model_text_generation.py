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
from typing import Optional
from dataclasses import dataclass, field

import torch
from tqdm import tqdm
import torch.nn as nn
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, PeftModel, PeftConfig
import peft

tqdm.pandas()

from transformers import pipeline, AutoTokenizer, HfArgumentParser, AutoModelForCausalLM
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, set_seed
from trl.core import LengthSampler

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

########################################################################
# NOTE for to train with a 8-bit model a more recent version of
# transformers is required, full dependecies for this example:
# pip install  bitsandbytes datasets accelerate loralib
# pip install  git+https://github.com/huggingface/transformers.git@main
# pip install git+https://github.com/huggingface/peft.git
########################################################################

# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable
    model_name: Optional[str] = field(default="edbeeching/gpt-neo-125M-imdb", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    forward_batch_size=1,
    batch_size=64,
)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": config.forward_batch_size}


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
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
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(config)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
# ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name, load_in_8bit=True, device_map="auto")
# model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name, load_in_8bit=True, device_map="auto")


def disable_peft_merge(peft_model):
    for key, module in peft_model.named_modules():
        if "lora" in key:
            print(key, module)
        # parent, target, target_name = model.base_model._get_submodules(key)
        # if isinstance(target, peft.tuners.lora.Linear):
        #     bias = target.bias is not None
        #     new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
        #     model.base_model._replace_module(parent, target_name, new_module, target)

    pass


# converting the model
merge_adapter = False
if merge_adapter:
    model_name = "edbeeching/gpt-neo-125M-imdb_adapter-imdb-peft"

    peft_config = PeftConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

    from_pretrained = True
    if from_pretrained:
        model = PeftModel.from_pretrained(model, model_name)
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    zero_weights = False
    if zero_weights:
        for key, module in model.named_modules():
            if isinstance(module, peft.tuners.lora.Linear8bitLt):
                print(key)
                module.lora_A.weight.data *= 0
                module.lora_B.weight.data *= 0
                pass

    model.eval()

    # key_list = [key for key, _ in model.base_model.model.named_modules() if "lora" not in key]
    # for key in key_list:
    #     parent, target, target_name = model.base_model._get_submodules(key)
    #     if isinstance(target, peft.tuners.lora.Linear):
    #         bias = target.bias is not None
    #         new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
    #         model.base_model._replace_module(parent, target_name, new_module, target)

# model.base_model.model.push_to_hub(f"{model_name}-adapter-removed")
# print(model)
else:
    #    model = AutoModelForCausalLM.from_pretrained(config.model_name).to(
    #        "cuda"
    #    )  # , load_in_8bit=True, device_map="auto")
    #    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    peft_model_id = "edbeeching/gpt-neo-125M-imdb-lora2"
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_id)


# PEFT
# target_modules = None
# if "gpt-neox" in script_args.model_name:
#     target_modules = ["query_key_value", "xxx"] # workaround to use 8bit training on this model

# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=target_modules,  #handled automatically by peft
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

# pretrained_model = prepare_model_for_int8_training(pretrained_model, output_embedding_layer_name="embed_out")
# hacky workaround due to issues with "EleutherAI/gpt-neox-20b"
# if "gpt-neox" in script_args.model_name:
#     for name, param in pretrained_model.named_parameters():
#         # freeze base model's layers
#         param.requires_grad = False

#         if getattr(pretrained_model, "is_loaded_in_8bit", False):
#             # cast layer norm in fp32 for stability for 8bit models
#             if param.ndim == 1 and "layer_norm" in name:
#                 param.data = param.data.to(torch.float16)
# pretrained_model = get_peft_model(pretrained_model, lora_config)


####

# model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)

# model.gradient_checkpointing_disable = model.pretrained_model.gradient_checkpointing_disable
# model.gradient_checkpointing_enable = model.pretrained_model.gradient_checkpointing_enable

# # ref_pretrained_model = AutoModelForCausalLM.from_pretrained(config.model_name, load_in_8bit=True, device_map="auto")
# # ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ref_pretrained_model)

# print(model)
# print_trainable_parameters(model)

# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
tokenizer.pad_token = tokenizer.eos_token

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config.batch_size,
    collate_fn=collator,
    shuffle=True,
    drop_last=True,
)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": -1,
}
output_min_length = 4
output_max_length = 16
output_length_sampler = LengthSampler(output_min_length, output_max_length)

model.eval()
for epoch, batch in tqdm(enumerate(dataloader)):
    query_tensors = batch["input_ids"]

    model.config.use_cache = True
    #### Get response from Causal LM
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = model.generate(input_ids=query.unsqueeze(0).to("cuda"), **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    for r in zip(batch["query"], batch["response"]):
        print(r)

    break
