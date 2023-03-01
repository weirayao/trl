import torch
import peft
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

model = AutoModelForCausalLM.from_pretrained("edbeeching/gpt-neox-20b-imdb-peft-adapter-removed", load_in_8bit=True, device_map="auto")


# converting the model
# pretrained_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", torch_dtype=torch.float16)
# model = PeftModel.from_pretrained(
#     pretrained_model,
#     "edbeeching/gpt-neox-20b-imdb-imdb-peft")

# model.eval()

# key_list = [key for key, _ in model.base_model.model.named_modules() if "lora" not in key]
# for key in key_list:
#     parent, target, target_name = model.base_model._get_submodules(key)
#     if isinstance(target, peft.tuners.lora.Linear):
#         bias = target.bias is not None
#         new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
#         model.base_model._replace_module(parent, target_name, new_module, target)

# print(model)
