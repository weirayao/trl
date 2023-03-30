#!/bin/bash
#SBATCH --job-name=rl_tox_debug0 # Name of job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96 # Scale this linearly with num GPUs (i.e 1 node = 96 CPUs)
#SBATCH --mem-per-cpu=11G # Important to enable "mix" use of GPUs across cluster users
#SBATCH --gres=gpu:a100:8 # Adjust number of GPUs here
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/h4/logs/nathan/%x-%j.out
#SBATCH --err=/fsx/h4/logs/nathan/%x-%j.err

set -x -e

source ~/.bashrc
conda activate llama-nol

# Adjust this as necessary. Note that the `output_dir` arg in config_xxx.yaml is defined relative to where you run the SLURM script from
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file examples/toxicity/scripts/accelerate_config_multi_gpu.yml examples/toxicity/scripts/gpt-j-6b-toxicity.py --log_with wandb
