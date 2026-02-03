#!/bin/bash

cd ~/SLM
# Define an array of loss_weights
loss_weights=("[1, 0.0, 0.5]" "[1, 0.5, 0.0]" "[1, 0.0, 0.0]")

# Iterate over each loss_weight in the array
for loss_weight in "${loss_weights[@]}"; do
    accelerate launch \
      --multi_gpu \
      --num_processes=8 \
      main.py \
      experiments=AMPLIFY_350M \
      experiments.reference_prt_model_name="chandar-lab/AMPLIFY_120M" \
      experiments.reference_model="checkpoint/AMPLIFY_120M_None_1_0.5_0.5_loss_large_1.0_foldseek_gearnet_1/best" \
      experiments.loss_weight="$loss_weight"

    accelerate launch \
      --multi_gpu \
      --num_processes=8 \
      main.py \
      experiments=esm2_t33_650M_UR50D \
      experiments.reference_prt_model_name="facebook/esm2_t30_150M_UR50D" \
      experiments.reference_model="checkpoint/esm2_t30_150M_UR50D_None_1_0.5_0.5_loss_large_1.0_foldseek_gearnet_1/best" \
      experiments.loss_weight="$loss_weight"
done
