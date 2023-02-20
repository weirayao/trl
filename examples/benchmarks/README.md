# Benchmarks of gpt2 model size 

## Models to be benchmarked
- gpt2
- gpt2-medium
- gpt2-large
- gpt2-xl

## Step 1: Fine tuning on imdb for analysis of performance
python examples/benchmarks/cm_finetune.py --model_name=edbeeching/gpt2-imdb --log_with=wandb
- [gpt2-imdb](https://huggingface.co/edbeeching/gpt2-imdb) 
- [gpt2-medium-imdb](https://huggingface.co/edbeeching/gpt2-medium-imdb)
- [gpt2-large-imdb](https://huggingface.co/edbeeching/gpt2-large-imdb)  --per_device_train_batch_size=4 --gradient_accumulation_steps=2
- [gpt2-xl-imdb](https://huggingface.co/edbeeching/gpt2-xl-imdb)  --per_device_train_batch_size=2 --gradient_accumulation_steps=4
  
- [EleutherAI/gpt-neo-125M] () python examples/benchmarks/cm_finetune.py --model_name_or_path=EleutherAI/gpt-neo-125M --dataset_name=imdb --do_train --output_dir gpt-neo-125M-imdb --num_train_epochs=1 --push_to_hub
- [EleutherAI/gpt-neo-1.3B] () python examples/benchmarks/cm_finetune.py --model_name_or_path=EleutherAI/gpt-neo-1.3B --dataset_name=imdb --do_train --output_dir gpt-neo-1.3B-imdb --num_train_epochs=1 --push_to_hub --per_device_train_batch_size=4 --gradient_accumulation_steps=2


## Step 2: RL fine tuning with a modified version of the gpt2-sentiment.py script

 python examples/benchmarks/gpt2_finetune_imdb.py --model_name_or_path=gpt2-large --dataset_name=imdb --do_train --output_dir gpt2-large-imdb --num_train_epochs=1 --push_to_hub --per_device_train_batch_size=4 --gradient_accumulation_steps=2


