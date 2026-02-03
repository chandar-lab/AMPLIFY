#!/bin/bash

cd ~/SLM

struc_embed_type="af2"

accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  main.py \
  experiments=AMPLIFY_120M \
  experiments.struc_embed_type="$struc_embed_type"

accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  main.py \
  experiments=esm2_t30_150M_UR50D \
  experiments.struc_embed_type="$struc_embed_type"