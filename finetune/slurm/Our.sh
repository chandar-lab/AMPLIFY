#!/bin/bash

cd ~/SLM

for seed in {1..3}; do
    accelerate launch \
        --multi_gpu \
        --num_processes=8 \
        main.py \
        experiments=AMPLIFY_350M \
        experiments.reference_prt_model_name="chandar-lab/AMPLIFY_120M" \
        experiments.reference_model="checkpoint/AMPLIFY_120M_None_1_0.5_0.5_loss_large_1.0_foldseek_gearnet_1/best" \
        experiments.seed=$seed

    accelerate launch \
        --multi_gpu \
        --num_processes=8 \
        main.py \
        experiments=esm2_t33_650M_UR50D \
        experiments.reference_prt_model_name="facebook/esm2_t30_150M_UR50D" \
        experiments.reference_model="checkpoint/esm2_t30_150M_UR50D_None_1_0.5_0.5_loss_large_1.0_foldseek_gearnet_1/best" \
        experiments.seed=$seed
done
