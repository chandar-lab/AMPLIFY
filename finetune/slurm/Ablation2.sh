#!/bin/bash

cd ~/SLM

# Define an array of sample_modes
sample_modes=("loss_large" "loss_small")

# Iterate over each sample_mode in the array
for sample_mode in "${sample_modes[@]}"; do
    accelerate launch \
      --multi_gpu \
      --num_processes=8 \
      main.py \
      experiments=AMPLIFY_350M \
      experiments.sample_mode="$sample_mode"

    accelerate launch \
      --multi_gpu \
      --num_processes=8 \
      main.py \
      experiments=esm2_t33_650M_UR50D \
      experiments.sample_mode="$sample_mode"
done

# full data
accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  main.py \
  experiments=AMPLIFY_350M \
  experiments.ratio=1.0

accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  main.py \
  experiments=esm2_t33_650M_UR50D \
  experiments.ratio=1.0