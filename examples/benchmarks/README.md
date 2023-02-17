# Benchmarks of gpt2 model size 

## Models to be benchmarked
- gpt2
- gpt2-medium
- gpt2-large
- gpt2-xl

## Step 1: Fine tuning on imdb for analysis of performance
python examples/sentiment/scripts/cm_finetune.py --model_name=edbeeching/gpt2-imdb --log_with=wandb
- [gpt2-imdb](https://huggingface.co/edbeeching/gpt2-imdb) 
- [gpt2-medium-imdb](https://huggingface.co/edbeeching/gpt2-medium-imdb)
- [gpt2-large-imdb](https://huggingface.co/edbeeching/gpt2-large-imdb)  --batch_size=4 --gradient_accumulation_steps=2
- [gpt2-xl-imdb](https://huggingface.co/edbeeching/gpt2-xl-imdb)  --batch_size=2 --gradient_accumulation_steps=4
- 
## Step 2: RL fine tuning with a modified version of the gpt2-sentiment.py script