# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig

from trl import SFTTrainer


tqdm.pandas()

########################################################################
# This is a fully working simple example to use trl's SFTTrainer.
#
# This example fine-tunes any causal language model (GPT-2, GPT-Neo, etc.)
# by using the SFTTrainer from trl, we will leverage PEFT library to finetune
# adapters on the model.
#
########################################################################

@dataclass
class ScriptArguments:
    """
    Define the arguments used in this script.
    """

    model_name: Optional[str] = field(default="decapoda-research/llama-7b-hf", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="ybelkada/oasst1-tiny-subset", metadata={"help": "the dataset name"})
    use_8_bit: Optional[bool] = field(default=True, metadata={"help": "use 8 bit precision"})
    use_multi_gpu: Optional[bool] = field(default=False, metadata={"help": "use multi GPU"})
    use_adapters: Optional[bool] = field(default=True, metadata={"help": "use adapters"})
    batch_size: Optional[int] = field(default=1, metadata={"help": "input batch size"})
    max_seq_length: Optional[int] = field(default=512, metadata={"help": "max sequence length"})

def get_current_device():
    return Accelerator().process_index

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

dataset = load_dataset(script_args.dataset_name, split="train[:1%]")

# We load the model
if script_args.use_multi_gpu:
    device_map = "auto"
else:
    device_map = {"":get_current_device()}
model = AutoModelForCausalLM.from_pretrained(script_args.model_name, load_in_8bit=script_args.use_8_bit, device_map=device_map if script_args.use_8_bit else None)

if script_args.use_adapters:
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None
    if script_args.use_8_bit:
        raise ValueError(
            "You need to use adapters to use 8 bit precision"
        )

if "llama" in script_args.model_name:
    tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
else:
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

with tempfile.TemporaryDirectory() as tmp_dir:
    training_arguments = TrainingArguments(
        per_device_train_batch_size=script_args.batch_size,
        max_steps=10,
        gradient_accumulation_steps=4,
        output_dir=tmp_dir,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="messages",
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
    )

    trainer.train()

    assert "adapter_model.bin" in os.listdir(tmp_dir)