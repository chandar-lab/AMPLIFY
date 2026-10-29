#!/bin/bash

#SBATCH --job-name=rec-eval
##SBATCH --partition=<partition>   # set for your cluster
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# =============================================================================
# recAMPLIFY evaluation suite: 3 zero-shot probes + 18 supervised probes.
#
#   CHECKPOINT=<hf_checkpoints/checkpoint_N> MODEL_NAME=halt_only sbatch eval/jobs/run_eval.sh
#
# STAGE=1    zero-shot only (CASP16 contacts, ProteinGym DMS, KL calibration)
# STAGE=2    supervised only (the 18 probes)
# STAGE=all  both (default)
# TASKS="fold_prediction stability_prediction"   restrict stage 2
#
# Evaluation driver of recamplify. Unlike the original AMPLIFY driver, the model is imported through
# apperitif._amplify (no sys.path walking), and the KL probe uses the paper's divergence
# direction (see eval/scripts/eval_calibration/kl_consensus.py).
# =============================================================================

set -eo pipefail

# Default to the directory the job was submitted from (the repository root); validate
# before cd-ing under `set -e`.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -d "$REPO/src/amplify" ]; then
    echo "ERROR: REPO=$REPO is not the repository root (no src/amplify)." >&2
    echo "       Pass it explicitly: REPO=<repo> CHECKPOINT=... sbatch eval/jobs/run_eval.sh" >&2
    exit 2
fi
EVAL="$REPO/eval"
cd "$REPO"

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to an hf_checkpoints/checkpoint_N directory}"
MODEL_NAME="${MODEL_NAME:-$(basename "$(dirname "$(dirname "$CHECKPOINT")")")}"
STAGE="${STAGE:-all}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUTPUT_DIR:-/path/to/outputs}/recamplify_eval_results}"
OUT="$OUTPUT_BASE/$MODEL_NAME"

mkdir -p "$REPO/logs" "$OUT"
exec 1>"$REPO/logs/eval_${MODEL_NAME}_j${SLURM_JOB_ID}.out" \
     2>"$REPO/logs/eval_${MODEL_NAME}_j${SLURM_JOB_ID}.err"

echo "=============================================="
echo "recAMPLIFY evaluation — $MODEL_NAME"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT"
echo "Output:     $OUT"
echo "Stage:      $STAGE"
echo "Start:      $(date)"
echo "----------------------------------------------"

# Activate the training environment (torch 2.10+cu128, flash-attn 2.8.3) — set for your cluster.
source "${ENV_ACTIVATE:-/path/to/env/bin/activate}"

# src/ for the model, eval/ for the apperitif package.
export PYTHONPATH="$REPO/src:$EVAL:$PYTHONPATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))
TMP_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
export HF_HOME="${HF_HOME:-$TMP_DIR/hf_cache}"
export HF_DATASETS_CACHE="${HF_ARROW_CACHE:-/path/to/hf_datasets_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# Read the carry mode off the checkpoint so eval matches training.
CARRY_MODE=$(python - "$CHECKPOINT" <<'PY'
import json, sys, pathlib
cfg = json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text())
print("ponder_act" if cfg.get("use_ponder_act") else "none")
PY
)
L_CYCLES_ARG=""
[ -n "$EVAL_L_CYCLES" ] && L_CYCLES_ARG="model.l_cycles=$EVAL_L_CYCLES"
echo "carry_mode=$CARRY_MODE   ${L_CYCLES_ARG:-(checkpoint L_cycles)}"

CALIB_DATA_DIR="${CALIB_DATA_DIR:-${DATA_DIR:-/path/to/data}/evaluation}"
DMS_DATA_DIR="${DMS_DATA_DIR:-${DATA_DIR:-/path/to/data}/ProteinGym/DMS_ProteinGym_substitutions}"
PG_REF="$EVAL/scripts/proteingym_ref"
BS="${PRECOMPUTE_BS:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"

# ---------------------------------------------------------------- stage 1
if [ "$STAGE" = "1" ] || [ "$STAGE" = "all" ]; then
    S1="$OUT/stage1"; mkdir -p "$S1"

    echo "STAGE1_TASK | name=casp16_jacobian | start"
    python "$EVAL/scripts/eval_categorical_jacobian/categorical_jacobian.py" \
        --checkpoint-dir "$CHECKPOINT" --split casp16 \
        --batch-size "$BS" --max-len "$MAX_LENGTH" \
        --results "$S1/casp16_jacobian.json" \
        ${EVAL_L_CYCLES:+--l-cycles $EVAL_L_CYCLES} \
        && echo "STAGE1_TASK | name=casp16_jacobian | status=OK" \
        || echo "STAGE1_TASK | name=casp16_jacobian | status=FAILED"

    # Recursive models overflow in bf16 on this path, so force fp32 (as upstream did).
    echo "STAGE1_TASK | name=proteingym_dms | start"
    python "$EVAL/scripts/compute_fitness_amplify.py" \
        --model_name_or_path "$CHECKPOINT" --load-from auto \
        --dms-input "$DMS_DATA_DIR" --dms-output "$S1/proteingym_dms" \
        --dms_mapping "$PG_REF/DMS_substitutions.csv" --dms_index all \
        --scoring-strategy masked-marginals --scoring-window optimal \
        --bos-offset 1 --batch-size "$BS" --autocast-dtype fp32 \
        --max-length 2048 --overwrite-prior-scores \
        ${EVAL_L_CYCLES:+--l-cycles $EVAL_L_CYCLES} \
        && echo "STAGE1_TASK | name=proteingym_dms | status=OK" \
        || echo "STAGE1_TASK | name=proteingym_dms | status=FAILED"

    # compute_fitness_amplify.py writes one CSV per assay but no summary, so aggregate the
    # per-assay Spearman here (the headline DMS number).
    set +e
    python - "$S1/proteingym_dms" "$MODEL_NAME" > "$S1/proteingym_dms_summary.json" <<'AGG'
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

d, model = sys.argv[1], sys.argv[2]
META = {"mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"}
per = []
for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
    df = pd.read_csv(f)
    score_cols = [c for c in df.columns if c not in META]
    if not score_cols or "DMS_score" not in df.columns or df[score_cols[0]].isna().all():
        continue
    r = spearmanr(df[score_cols[0]], df["DMS_score"]).correlation
    if np.isfinite(r):
        per.append({"assay": os.path.basename(f)[:-4], "spearman": float(r)})
json.dump({"model": model, "n_assays": len(per),
           "mean_spearman": float(np.mean([p["spearman"] for p in per])) if per else None,
           "per_assay": per}, sys.stdout, indent=2)
AGG
    rc=$?; set -e
    [ $rc -eq 0 ] \
        && echo "STAGE1_TASK | name=proteingym_dms_summary | status=OK" \
        || echo "STAGE1_TASK | name=proteingym_dms_summary | status=FAILED"

    echo "STAGE1_TASK | name=kl_calibration | start"
    python "$EVAL/scripts/eval_calibration/kl_consensus.py" \
        --checkpoint_dir "$CHECKPOINT" --model_name "$MODEL_NAME" \
        --batch_size "$BS" \
        --proteome_path "$CALIB_DATA_DIR/human_proteome.csv" \
        --consensus_path "$CALIB_DATA_DIR/human_consensus_positions.csv" \
        --dataset_path "$HF_DATASETS_CACHE/kl_calibration_dataset" \
        --output_path "$S1/calibration" --compute_scores ${KL_FORCE:+--force-recompute} \
        ${EVAL_L_CYCLES:+--l_cycles $EVAL_L_CYCLES} \
        && echo "STAGE1_TASK | name=kl_calibration | status=OK" \
        || echo "STAGE1_TASK | name=kl_calibration | status=FAILED"
fi

# ---------------------------------------------------------------- stage 2
DEFAULT_TASKS="stability_prediction fluorescence_prediction fitness_prediction \
enzyme_catalytic_efficiency optimal_ph optimal_temperature \
antibiotic_resistance cloning_clf fold_prediction localization_prediction \
material_production metal_ion_binding peptide_hla_mhc_affinity solubility_prediction \
tcr_pmhc_affinity temperature_stability ssp_q3 ssp_q8"

if [ "$STAGE" = "2" ] || [ "$STAGE" = "all" ]; then
    TASK_LIST="${TASKS:-$DEFAULT_TASKS}"
    NGPU=$(python -c "import torch;print(torch.cuda.device_count())")
    i=0
    for TASK in $TASK_LIST; do
        # ssp_* are token-classification and only the MLP head is wired for them.
        HTASK="mlp"
        GPU=$((i % NGPU)); i=$((i + 1))
        echo "TASK_START | task=$TASK | gpu=$GPU"
        (
          export CUDA_VISIBLE_DEVICES=$GPU
          python "$EVAL/scripts/main.py" \
            --config-path "$EVAL/conf" --config-name config \
            model=trm \
            model.checkpoint_path="$CHECKPOINT" \
            model.carry_mode="$CARRY_MODE" \
            model.embed_mode=dec \
            model.model_name="$MODEL_NAME" \
            model.max_length="$MAX_LENGTH" \
            model.use_packed_sequence=true \
            model.attention_mask_type=multiplicative \
            model.ponder_eval_decode=weighted \
            $L_CYCLES_ARG \
            data_config="$TASK" head_type="$HTASK" grid=mlp_bayes \
            base_folder="$OUT" \
            +precomputing_batch_size="$BS" \
            device=cuda do_test=true return_all_results=true \
            hydra.run.dir="$TMP_DIR/hydra/$TASK" \
          && echo "TASK_END | task=$TASK | status=OK" \
          || echo "TASK_END | task=$TASK | status=FAILED"
        ) &
        # Keep at most NGPU tasks resident.
        [ $((i % NGPU)) -eq 0 ] && wait
    done
    wait
fi

echo "----------------------------------------------"
echo "End: $(date)"
