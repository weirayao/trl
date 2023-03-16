from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers import AutoTokenizer, pipeline

model = AutoModelForSequenceClassification.from_pretrained(
    "edbeeching/gpt2-xl-stackexchange_stack-exchange-paired_rmts_240000_bup"
)
sentiment_pipe = pipeline(
    "sentiment-analysis", model="edbeeching/gpt2-xl-stackexchange_stack-exchange-paired_rmts_240000_bup", device="cuda"
)
