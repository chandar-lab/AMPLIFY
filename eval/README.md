# recAMPLIFY evaluation suite

Self-contained evaluation for recAMPLIFY: **3 zero-shot probes + 18 supervised probes**.
Part of the `recamplify` branch of [chandar-lab/AMPLIFY](https://github.com/chandar-lab/AMPLIFY);
it depends on nothing outside `recamplify`.

```bash
CHECKPOINT=<run>/hf_checkpoints/checkpoint_100000 MODEL_NAME=halt_only \
  sbatch eval/jobs/run_eval.sh                      # all 21
STAGE=1 ... sbatch eval/jobs/run_eval.sh            # zero-shot only
TASKS="fold_prediction" STAGE=2 ... sbatch ...      # one supervised probe
```

Needs `PYTHONPATH=<repo>/src:<repo>/eval` (the driver sets it) plus `requirements.txt`
on top of the training environment (torch 2.10+cu128, transformers 4.57, flash-attn 2.8.3).

## Layout

| path | role |
|---|---|
| `apperitif/` | the evaluator/task/model package |
| `apperitif/_amplify.py` | **the single import seam** to `src/amplify` |
| `conf/` | hydra configs: 24 task definitions, 9 HPO grids, model + metric configs |
| `scripts/main.py` | entrypoint for all 18 supervised probes |
| `scripts/eval_categorical_jacobian/` | CASP16 contact prediction |
| `scripts/compute_fitness_amplify.py` | ProteinGym DMS, 217 assays |
| `scripts/eval_calibration/kl_consensus.py` | KL calibration |
| `jobs/run_eval.sh` | driver |

## The 21 evaluations

**Zero-shot (3):** CASP16 categorical-Jacobian contacts (`P@L`, `P@L/2`, `P@L/5`, AUC) on
long-range pairs `|i−j| ≥ 24` (CASP convention; `--minsep`); ProteinGym DMS over 217 assays
(Spearman, mean over assays; `scripts/proteingym_ref/performance_DMS_benchmarks.py` gives the ProteinGym hierarchical aggregate); KL
calibration to human MSA consensus (`calibration_T` / `calibration_NT`).

**Supervised (18),** all via `main.py` with an MLP head and Optuna TPE over 3 seeds:
6 regression (`stability_prediction`, `fluorescence_prediction`, `fitness_prediction`,
`optimal_ph`, `optimal_temperature` — Spearman; `enzyme_catalytic_efficiency` — Pearson) ·
10 sequence classification (`antibiotic_resistance`, `fold_prediction`,
`localization_prediction`, `metal_ion_binding`, `solubility_prediction` — accuracy;
`cloning_clf`, `material_production`, `peptide_hla_mhc_affinity`, `tcr_pmhc_affinity` —
AUROC; `temperature_stability` — MCC) ·
2 token classification (`ssp_q3`, `ssp_q8` — per-residue accuracy).
Each task reports, and selects its hyperparameters by, the metric of its source benchmark
(xTrimoPGLM, Chen et al. 2024); per-task overrides live in `conf/data_config/*.yaml` under
`evaluator_overrides`.

## Two deliberate differences from upstream

**1. The KL metric direction is fixed.** The paper (appendix, "Zero-shot tasks" item A)
defines `cal_i = 100·exp(−KL(f_i ‖ p_θ))` with the MSA consensus profile `f_i` first.
Upstream computed the opposite, `KL(p_θ ‖ f_i)`. That direction is also numerically unsound
here: `f_i` has structural zeros, so the `1e-10` clip became load-bearing and the score
tracked the clip constant rather than the model —

| eps | `KL(p‖f)` (old) | `KL(f‖p)` (paper) |
|---|---|---|
| 1e-6 | 0.68 | 0.0939 |
| 1e-10 | 1.51 | 0.0940 |
| 1e-14 | 2.33 | 0.0940 |

The old direction scales with `−log(eps)`; the paper's is eps-invariant because `p_θ` is a
softmax. The superseded direction is still computed per position as `kl_forward` and
aggregated as `calibration_forward_{T,NT}` so corrected and published numbers can be
compared from one inference pass. **The two directions are not a monotone relabelling —
rankings can change, so all previously reported KL values must be regenerated.**

**2. `embed_mode="z"` is unavailable.** It needs the pre-exit recursive latent
(`carry_out.Z`), which the refactored model no longer returns. Use `embed_mode="dec"` — the
driver default, and what every published recAMPLIFY number used (the upstream result
directories are all suffixed `_emdec`). It hooks `lm_head` and is post-exit, post-norm.

## Corrections to the upstream suite

- **Early stopping** is reset at the start of every training run. Upstream never reset it in
  the regression HPO path, so patience carried across trials and seed re-runs and later runs
  stopped after one validation check.
- **SSP scoring** ignores the BOS/EOS positions (label `-100`); upstream scored them as errors.
- **Truncation is deterministic** (left-most 510 residues + BOS/EOS) and token labels are cropped
  to exactly the residues kept; upstream took a random window while labels came from the start.
- **Ambiguous residues** (`<unk>`, X, B, O, U, Z, J and out-of-vocabulary characters such as `.`) are
  stripped by the collator *before* truncation, together with their per-residue labels. The
  tokenizer used to drop them after truncating, so such sequences kept fewer than 510 residues and
  token labels would shift.
- **Regression targets are standardised** with train-split mean/std for training; predictions are
  mapped back to label units before any metric. Raw targets span up to ~10² (optimal_temperature),
  which left MSE-trained heads unconverged.
- **Pooling is over residues only** (BOS/EOS excluded from the mean), as in xTrimoPGLM's probing
  head, which averages residue embeddings after dropping `<eos>`; upstream averaged BOS/EOS in.
- **Validation split:** tasks without an HF `valid` split use a random 10% of train (seed 42).
  xTrimoPGLM defines no validation split for these (e.g. antibiotic_resistance: "same split
  (2,072 training samples and 1,344 test samples)"), so this is our choice, not a deviation.

## Data

| asset | path |
|---|---|
| human proteome + MSA consensus | `$CALIB_DATA_DIR` (default `$DATA_DIR/evaluation/`) |
| ProteinGym DMS (217 CSVs) | `$DMS_DATA_DIR` (default `$DATA_DIR/ProteinGym/DMS_ProteinGym_substitutions`) |
| 19 HF datasets | `$HF_ARROW_CACHE` (exported as `HF_DATASETS_CACHE`) |

Set `REPO`, `ENV_ACTIVATE`, `DATA_DIR` (or the per-asset variables above), `HF_ARROW_CACHE`
and `OUTPUT_BASE` (or `OUTPUT_DIR`) for your cluster before `sbatch`, and the `#SBATCH --partition`
line in `jobs/run_eval.sh` if your scheduler needs one.

ProteinGym is an external data dependency only — no eval code imports it. Nothing outside
`recamplify` is needed.

## Notes

- ProteinGym DMS runs in **fp32**: recursive layers overflow in bf16 and scores collapse.
- Supervised probes are resumable via `optuna_study.db` in each task's output folder.
- Outputs: `<OUTPUT_BASE>/<model>/stage1/…` and
  `<OUTPUT_BASE>/<model>/mlp_bayes/biomap_research_<task>/<model>/results.json`.
