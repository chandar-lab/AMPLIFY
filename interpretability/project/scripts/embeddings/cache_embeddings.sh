#!/bin/bash

#SBATCH --partition=long                         
#SBATCH --cpus-per-task=4                             
#SBATCH --gres=gpu:1                           
#SBATCH --mem=500G                                       
#SBATCH --time=24:00:00     
#SBATCH --job-name=embeddings                       
#SBATCH -o /network/scratch/s/shawn.whitfield/logs/slurm-%j.out  # Write the log on scratch

# uv run python3 store_embeddings.py
# uv run concat_embeddings.py
uv run get_embeddings.py