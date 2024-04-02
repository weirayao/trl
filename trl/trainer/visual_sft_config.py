from dataclasses import dataclass

from transformers import TrainingArguments


@dataclass
class VisualSFTConfig(TrainingArguments):
    pass
