#!/bin/bash

cd ~/SLM

accelerate launch --multi_gpu --num_processes=8 main.py experiments=AMPLIFY_120M
accelerate launch --multi_gpu --num_processes=8 main.py experiments=esm2_t30_150M_UR50D
