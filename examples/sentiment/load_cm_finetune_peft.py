from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from peft import PeftModelForCausalLM, PeftModel
from transformers import pipeline, AutoTokenizer, HfArgumentParser, AutoModelForCausalLM

# lora_config = LoraConfig(
#     r=32,
#     lora_alpha=32,
#     target_modules=["query_key_value", "xxx"],  #handled automatically by peft
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
pretrained_model = AutoModelForCausalLM.from_pretrained("edbeeching/gpt-neo-125M-imdb", load_in_8bit=True, device_map="auto")
model = PeftModel.from_pretrained(
    pretrained_model,
    "edbeeching/gpt-neo-125M-imdb-ppo-sentiment",
    load_in_8bit=True,
    device_map="auto")

print(model)
