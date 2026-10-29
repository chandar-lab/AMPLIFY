# Shared variant -> hydra-override table for the recAMPLIFY launchers.
# Sourced by run_slurm.sh and run_node.sh; sets VARIANT_OVERRIDES and RUN_TAG.
#
#   VARIANT=baseline      fixed L cycles, no halting, no gating
#   VARIANT=halt_only     halt-only GMR: a scalar halt head alone decides halting
#   VARIANT=gate_scalar   gated GMR, 1 channel
#   VARIANT=gate_lowrank  gated GMR, ponder_gate_rank channels
#   VARIANT=gate_perdim   gated GMR, one channel per feature
#
# See README.md for the full description of each variant.

VARIANT="${VARIANT:-halt_only}"

case "$VARIANT" in
    baseline)
        VARIANT_OVERRIDES="model.use_ponder_act=false"
        ;;
    halt_only)
        VARIANT_OVERRIDES="model.use_ponder_act=true model.ponder_gate_mode=none"
        ;;
    gate_scalar)
        VARIANT_OVERRIDES="model.use_ponder_act=true model.ponder_gate_mode=scalar"
        ;;
    gate_lowrank)
        VARIANT_OVERRIDES="model.use_ponder_act=true model.ponder_gate_mode=lowrank model.ponder_gate_rank=16"
        ;;
    gate_perdim)
        VARIANT_OVERRIDES="model.use_ponder_act=true model.ponder_gate_mode=perdim model.ponder_decode_perdim=true model.ponder_budget_perchannel=true"
        ;;
    *)
        echo "ERROR: unknown VARIANT '$VARIANT' (expected one of: baseline halt_only gate_scalar gate_lowrank gate_perdim)" >&2
        exit 1
        ;;
esac

# Architecture / objective knobs shared by all five variants.
L_CYCLES="${L_CYCLES:-6}"
CONTEXT_TYPE="${CONTEXT_TYPE:-zero}"
NUM_CORE_LAYERS="${NUM_CORE_LAYERS:-12}"
NO_RECUR_LAYERS="${NO_RECUR_LAYERS:-2}"
NO_EXIT_LAYERS="${NO_EXIT_LAYERS:-2}"
MIN_STEPS="${MIN_STEPS:-1}"
BETA="${BETA:-0.05}"

COMMON_OVERRIDES="\
model.L_cycles=$L_CYCLES \
model.context_type=$CONTEXT_TYPE \
model.num_core_layers=$NUM_CORE_LAYERS \
model.no_recur_layers=$NO_RECUR_LAYERS \
model.no_exit_layers=$NO_EXIT_LAYERS \
model.ponder_act_min_steps=$MIN_STEPS \
training_args.beta=$BETA"

RUN_TAG="${RUN_TAG:-rec_nc${NUM_CORE_LAYERS}_nr${NO_RECUR_LAYERS}_ne${NO_EXIT_LAYERS}_L${L_CYCLES}_ct${CONTEXT_TYPE}_${VARIANT}}"
