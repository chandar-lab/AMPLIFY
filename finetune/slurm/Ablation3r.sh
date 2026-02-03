#!/bin/bash

cd ~/SLM

struc_token_types=("protoken" "aido")

for struc_token_type in "${struc_token_types[@]}"; do
    accelerate launch \
      --multi_gpu \
      --num_processes=8 \
      main.py \
      experiments=AMPLIFY_120M \
      experiments.struc_token_type="$struc_token_type"

    accelerate launch \
      --multi_gpu \
      --num_processes=8 \
      main.py \
      experiments=esm2_t30_150M_UR50D \
      experiments.struc_token_type="$struc_token_type"
done
