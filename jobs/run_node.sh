#!/bin/bash

# =============================================================================
# RecAMPLIFY pre-training on a single multi-GPU node (8 GPUs, no scheduler).
#
#   VARIANT=halt_only bash jobs/run_node.sh
#
# VARIANT is one of: baseline halt_only gate_scalar gate_lowrank gate_perdim.
# Uses the token budget from training_args/rec.yaml (32768 tokens/device), which
# assumes 80 GB GPUs. Adjust the block marked "VARIABLES TO ADAPT" for your machine.
# =============================================================================

set -eo pipefail

######################################
# VARIABLES TO ADAPT
######################################
PATH_TO_MACHINE_SCRATCH="${PATH_TO_MACHINE_SCRATCH:-/tmp}"
PATH_TO_LOGS="${PATH_TO_LOGS:-/path/to/logs}"
PATH_TO_ENV="${PATH_TO_ENV:-/path/to/venv}"
PATH_TO_DATA="${PATH_TO_DATA:-/path/to/datasets}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

cd "$REPO"
source $PATH_TO_ENV/bin/activate

# shellcheck source=jobs/variant.sh
source jobs/variant.sh

export PYTHONPATH=$PYTHONPATH:./src
export OMP_NUM_THREADS=24
export HF_HOME=$PATH_TO_MACHINE_SCRATCH/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

NUM_PROCESSES="${NUM_PROCESSES:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_STEPS="${MAX_STEPS:-100000}"
OUTPUT_DIR="${OUTPUT_DIR:-$PATH_TO_LOGS/$RUN_TAG}"

echo "=============================================="
echo "RecAMPLIFY pre-training (single node)"
echo "=============================================="
echo "Variant:    $VARIANT"
echo "Run tag:    $RUN_TAG"
echo "Output dir: $OUTPUT_DIR"
echo "Processes:  $NUM_PROCESSES   grad_accum: $GRAD_ACCUM   max_steps: $MAX_STEPS"
echo "Overrides:  $VARIANT_OVERRIDES $COMMON_OVERRIDES"
echo "=============================================="

accelerate launch \
    --config_file=conf/amplify/accelerate_ddp.yaml \
    --num_machines=1 \
    --num_processes=$NUM_PROCESSES \
    --num_cpu_threads_per_process=24 \
    --enable_cpu_affinity \
    --mixed_precision=bf16 \
    scripts/pretrain_rec.py \
    +data.train_files.uniref100_train=$PATH_TO_DATA/uniref100_train.csv \
    +data.train_files.oas_train=$PATH_TO_DATA/oas_train.csv \
    +data.train_files.scop_train=$PATH_TO_DATA/scop_train.csv \
    +data.validation_files.uniprot_dev=$PATH_TO_DATA/uniprot_dev.csv \
    +data.validation_files.oas_dev=$PATH_TO_DATA/oas_dev.csv \
    +data.validation_files.scop_dev=$PATH_TO_DATA/scop_dev.csv \
    training_args.output_dir=$OUTPUT_DIR \
    training_args.max_steps=$MAX_STEPS \
    training_args.gradient_accumulation_steps=$GRAD_ACCUM \
    training_args.dataloader_num_workers=8 \
    $VARIANT_OVERRIDES \
    $COMMON_OVERRIDES \
    $EXTRA_OVERRIDES \
    wandb.dir=$OUTPUT_DIR \
    wandb.name=$RUN_TAG \
    wandb.mode=${WANDB_MODE:-offline}
