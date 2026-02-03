#!/bin/bash

cd ~/SLM

# Define an array of struc_token_types
struc_token_types=("protoken" "aido")

# Iterate over each struc_token_type in the array
for struc_token_type in "${struc_token_types[@]}"; do
    accelerate launch \
      --multi_gpu \
      --num_processes=8 \
      main.py \
      experiments=AMPLIFY_350M \
      experiments.reference_prt_model_name="chandar-lab/AMPLIFY_120M" \
      experiments.reference_model="checkpoint/AMPLIFY_120M_None_1_0.5_0.5_loss_large_1.0_${struc_token_type}_gearnet_1/best" \
      experiments.struc_token_type="$struc_token_type"

    accelerate launch \
      --multi_gpu \
      --num_processes=8 \
      main.py \
      experiments=esm2_t33_650M_UR50D \
      experiments.reference_prt_model_name="facebook/esm2_t30_150M_UR50D" \
      experiments.reference_model="checkpoint/esm2_t30_150M_UR50D_None_1_0.5_0.5_loss_large_1.0_${struc_token_type}_gearnet_1/best" \
      experiments.struc_token_type="$struc_token_type"
done