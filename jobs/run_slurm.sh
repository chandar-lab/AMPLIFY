#!/bin/bash

#SBATCH --job-name=recamplify
##SBATCH --partition=<partition>          # uncomment and set for your cluster
#SBATCH --nodes=1
#SBATCH --gres=gpu:2                      # 2x GPU (adjust: 1-4)
#SBATCH --cpus-per-task=16                # CPUs across all GPUs on the node
#SBATCH --mem=150G
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --signal=TERM@60                  # SIGTERM 60s before wall-time -> checkpoint
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# =============================================================================
# RecAMPLIFY pre-training on a SLURM cluster. Submit from the repository root.
#
#   VARIANT=halt_only sbatch jobs/run_slurm.sh
#
# VARIANT is one of: baseline halt_only gate_scalar gate_lowrank gate_perdim.
# Training arguments: lr 1e-4, 1 M steps, 32 sequences/device, grad accum 8, packing on.
#
# Any knob in jobs/variant.sh can be overridden the same way, e.g.
#   VARIANT=halt_only L_CYCLES=8 BETA=0.1 sbatch jobs/run_slurm.sh
#   VARIANT=halt_only EXTRA_OVERRIDES="training_args.eval_steps=50" sbatch jobs/run_slurm.sh
#
# Speed work (see train/tokens_per_second in the logs):
#   COMPILE=true sbatch jobs/run_slurm.sh            # torch.compile, max-autotune-no-cudagraphs
#
# Resume an interrupted run:
#   RESUME_DIR=/path/to/outputs/rec_..._j12345 VARIANT=halt_only sbatch jobs/run_slurm.sh
#
# Site-specific paths come from the environment:
#   ENV_ACTIVATE  optional script that activates the Python environment; when unset, the job
#                 uses the environment it was submitted from (sbatch exports it by default)
#   DATA_DIR      holds train/uniref100_2011.csv.tar.gz and validation/uniprot.fasta.tar.gz
#   OUTPUT_ROOT   where run directories and the persistent HF Arrow cache are written
#
# stdout/stderr are redirected below, once RUN_TAG is known, so the log filename
# matches the W&B run name.
# =============================================================================

set -eo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO"

# shellcheck source=jobs/variant.sh
source jobs/variant.sh

mkdir -p logs
exec 1>"logs/${RUN_TAG}_j${SLURM_JOB_ID}.out" 2>"logs/${RUN_TAG}_j${SLURM_JOB_ID}.err"

echo "=============================================="
echo "RecAMPLIFY pre-training (SLURM)"
echo "=============================================="
echo "Job ID:   $SLURM_JOB_ID   Node: $SLURMD_NODENAME"
echo "Start:    $(date)"
echo "Variant:  $VARIANT"
echo "Run tag:  $RUN_TAG"
echo "----------------------------------------------"

# =============================================================================
# ENVIRONMENT  (torch 2.10+cu128, flash-attn 2.8.3)
# =============================================================================
if [ -n "$ENV_ACTIVATE" ]; then source "$ENV_ACTIVATE"; fi

export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}
export PYTHONPATH=$PYTHONPATH:./src
# Packed batches change buffer size every step; expandable segments avoid the
# fragmentation that a fixed split size causes here.
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# If NCCL hangs on intra-node peer-to-peer transfers, export NCCL_P2P_DISABLE=1 before sbatch.
export NCCL_DEBUG=WARN

export MASTER_PORT=$(expr 10000 + $(echo -n "$SLURM_JOBID" | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# =============================================================================
# DATA
# =============================================================================
DATA_DIR="${DATA_DIR:-/path/to/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/outputs}"
NODE_TMP="${SLURM_TMPDIR:-${TMPDIR:-/tmp}/recamplify_$SLURM_JOB_ID}"
TRAIN_SRC="$DATA_DIR/train/uniref100_2011.csv.tar.gz"
VALID_SRC="$DATA_DIR/validation/uniprot.fasta.tar.gz"

for f in "$TRAIN_SRC" "$VALID_SRC"; do
    [ -f "$f" ] || { echo "ERROR: dataset not found at $f"; exit 1; }
done

# Train is read in place from shared storage; validation is small, so extract it
# to node-local storage.
TRAIN_FILE="$TRAIN_SRC"
mkdir -p "$NODE_TMP/data/validation"
tar -xzf "$VALID_SRC" -C "$NODE_TMP/data/validation"
VALID_FILE=$(find "$NODE_TMP/data/validation" -name "*.fasta" | head -n 1)
[ -n "$VALID_FILE" ] || { echo "ERROR: no .fasta found after extracting $VALID_SRC"; exit 1; }

echo "Train: $TRAIN_FILE"
echo "Valid: $VALID_FILE"

# HF caches: hub metadata is node-local, but the Arrow dataset cache is persisted to
# OUTPUT_ROOT so UniRef100 is not re-parsed on every job.
export HF_HOME="$NODE_TMP/hf_cache"
export HF_DATASETS_CACHE="${HF_ARROW_CACHE:-$OUTPUT_ROOT/hf_datasets_cache}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# =============================================================================
# OUTPUT
# =============================================================================
if [ -n "$RESUME_DIR" ]; then
    OUTPUT_DIR="$RESUME_DIR"
    echo "Resuming into $OUTPUT_DIR"
else
    OUTPUT_DIR="$OUTPUT_ROOT/${RUN_TAG}_j${SLURM_JOB_ID}"
fi
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# LAUNCH — batching is by sequence count (32/device), not token budget, so the
# token-budget fields from training_args/rec.yaml must be cleared.
# =============================================================================
NUM_PROCESSES=$((SLURM_JOB_NUM_NODES * SLURM_GPUS_ON_NODE))
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_STEPS="${MAX_STEPS:-1000000}"
LR="${LR:-1e-4}"
COMPILE="${COMPILE:-false}"

echo "Processes: $NUM_PROCESSES   batch/device: $BATCH_SIZE seq   grad_accum: $GRAD_ACCUM"
echo "max_steps: $MAX_STEPS   lr: $LR   compile: $COMPILE"
echo "Overrides: $VARIANT_OVERRIDES $COMMON_OVERRIDES"
echo "----------------------------------------------"

cmd=(
    accelerate launch
    --config_file=conf/amplify/accelerate_ddp.yaml
    --machine_rank=\$SLURM_NODEID
    --num_machines=$SLURM_JOB_NUM_NODES
    --num_processes=$NUM_PROCESSES
    --num_cpu_threads_per_process=$((SLURM_CPUS_PER_TASK / SLURM_GPUS_ON_NODE))
    --main_process_ip=$MASTER_ADDR
    --main_process_port=$MASTER_PORT
    --mixed_precision=bf16
    scripts/pretrain_rec.py
    +data.train_files.uniref100="$TRAIN_FILE"
    +data.validation_files.uniprot="$VALID_FILE"
    training_args.output_dir="$OUTPUT_DIR"
    training_args.optimizer.lr=$LR
    training_args.max_steps=$MAX_STEPS
    training_args.scheduler.num_training_steps=900000
    training_args.max_length=512
    training_args.packing=true
    training_args.per_device_train_batch_size=$BATCH_SIZE
    training_args.per_device_eval_batch_size=$BATCH_SIZE
    training_args.per_device_train_batch_tokens=null
    training_args.per_device_eval_batch_tokens=null
    training_args.gradient_accumulation_steps=$GRAD_ACCUM
    training_args.compile=$COMPILE
    $VARIANT_OVERRIDES
    $COMMON_OVERRIDES
    $EXTRA_OVERRIDES
    wandb.name="${RUN_TAG}_j${SLURM_JOB_ID}"
    wandb.mode=${WANDB_MODE:-online}
)

srun --kill-on-bad-exit=1 bash -c "${cmd[*]}"

echo "----------------------------------------------"
echo "End: $(date)"
