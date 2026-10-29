# recAMPLIFY

<p align="center">
  <a href="https://arxiv.org/abs/2610.26747"><img src="https://img.shields.io/badge/arXiv-2610.26747-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv:2610.26747"></a>
  <a href="https://huggingface.co/chandar-lab/recamplify"><img src="https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face models"></a>
</p>

**Recursive pretraining enables test-time scaling in protein language models.**

Official code for *recAMPLIFY*: a compact masked protein language model that scales
**computation per sequence** rather than parameter count. A non-recurrent *stem* and *exit*
wrap a weight-tied **recurrent core** that refines a residue-aligned latent state over `L`
cycles, raising effective depth at a fixed parameter budget. On top of this,
**Gated Mixture Recurrence (GMR)** decodes each protein from a probability-weighted mixture
of the hidden states visited during recursion, gating every cycle's residual update by a
learned survival signal — making recursion depth adaptive per sequence at inference time.

At full scale (1 M pretraining steps on the same 2 T tokens as AMPLIFY), a 155 M-parameter
recAMPLIFY is competitive with AMPLIFY-350 M while using 56% fewer parameters, improving fold
accuracy by 22.8%, stability Spearman by 8.7% and localization accuracy by 2.3%. The GMR study
uses a 100 K-step (210 B-token) budget at 133 M parameters, where GMR reduces expected recursion
depth by 32–40% at a task-dependent accuracy trade-off. The released checkpoints below are
these 100 K-step models.

## Model variants

| variant | gating | description |
|---|---|---|
| `baseline` | — | fixed-depth recursion, `L` cycles for every sequence |
| `halt_only` | none | a halt head alone allocates depth, recovering PonderNet-style halting as the no-gate special case of GMR |
| `gate_scalar` | scalar | one survival channel per sequence |
| `gate_lowrank` | rank-16 | 16 channels mapped to features by a row-stochastic assignment |
| `gate_perdim` | per-dim | one survival channel per feature |

Each GMR cycle produces a conditional halt probability from a stop-gradient latent, so the
compute cost of adaptivity cannot corrupt the language-modeling features, while the task loss
still shapes the depth policy through the mixture weights.

## Installation

```bash
pip install -e .                      # training
pip install -r eval/requirements.txt  # evaluation
pip install -e '.[flash-attn]'        # FlashAttention, required for packed sequences
```

Requires CUDA and bf16 for packed-sequence training.

## Pretraining

```bash
VARIANT=halt_only sbatch --partition=<partition> jobs/run_slurm.sh   # SLURM, 2 GPUs
VARIANT=gate_perdim bash jobs/run_node.sh                             # 8 GPUs, no scheduler
```

Site-specific locations are read from the environment: `ENV_ACTIVATE`, `DATA_DIR` and
`OUTPUT_ROOT` for `run_slurm.sh`; `PATH_TO_ENV`, `PATH_TO_DATA` and `PATH_TO_LOGS` for
`run_node.sh`. Set `WANDB_ENTITY` to log to your Weights & Biases team.

`VARIANT` selects one of the five models above; `jobs/variant.sh` holds the table. Any knob is
overridable from the environment, e.g. `L_CYCLES=8 BETA=0.1`. Architecture, optimizer and data
settings are in `conf/amplify/`.

The objective is masked-language-modeling cross-entropy plus, for the GMR variants, an entropy
regulariser on the halting distribution weighted by `beta`, warmed up over the learning-rate
warmup horizon.

## Evaluation

21 probes: 3 zero-shot and 18 supervised.

```bash
CHECKPOINT=<run>/hf_checkpoints/checkpoint_100000 MODEL_NAME=halt_only \
  sbatch eval/jobs/run_eval.sh
```

**Zero-shot** — CASP16 contact recovery via the categorical Jacobian; ProteinGym deep
mutational scanning over 217 assays; KL calibration against human MSA consensus profiles.
**Supervised** — 18 probing tasks (6 regression, 10 sequence classification, 2 token
classification) with an MLP head and Bayesian hyperparameter search.

See [eval/README.md](eval/README.md) for the task list, metrics and data paths.

## Repository layout

| path | contents |
|---|---|
| `src/amplify/model/recamplify.py` | the recursive model and GMR |
| `src/amplify/model/amplify.py` | the AMPLIFY encoder block, RoPE and tokenizer primitives |
| `src/amplify/trainer/` | pretraining and checkpoint evaluation loops |
| `src/amplify/dataset/` | MLM collator, sequence packing, CSV/FASTA loading |
| `conf/amplify/` | Hydra configuration |
| `jobs/` | cluster launchers |
| `eval/` | the 21-probe evaluation suite |
| `scripts/` | entrypoints |

Checkpoints are standard Hugging Face directories:

```python
from amplify.model import RecAMPLIFY
model = RecAMPLIFY.from_pretrained("<run>/hf_checkpoints/checkpoint_100000")
```

## Pretrained checkpoints

All five models share the architecture: 2 stem + 8 weight-tied core + 2 exit blocks, hidden
size 960, 15 heads, trained with `L = 6` cycles for 100 K steps (effective depth 60, counting
the read-out core pass). They load with plain `transformers`, no install of this repo needed:

| variant | Hub repo | unique parameters |
|---|---|---|
| `baseline` | `chandar-lab/recAMPLIFY-baseline` | 132,795,872 |
| `halt_only` | `chandar-lab/recAMPLIFY-halt-only` | 132,796,833 |
| `gate_scalar` | `chandar-lab/recAMPLIFY-gate-scalar` | 132,797,794 |
| `gate_lowrank` | `chandar-lab/recAMPLIFY-gate-lowrank` | 132,827,569 |
| `gate_perdim` | `chandar-lab/recAMPLIFY-gate-perdim` | 133,719,393 |

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

repo = "chandar-lab/recAMPLIFY-gate-perdim"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(repo, trust_remote_code=True).eval()

inputs = tokenizer(["MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTT"], return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)        # out.logits; out.avg_cycles = expected cycles (GMR variants)

model.set_num_cycles(9)          # test-time depth: more (or fewer) cycles than the 6 used in training
```

A training checkpoint becomes such a repo with
`python scripts/export_hf.py --checkpoint_dir <ckpt> --output_dir <dir> --repo_id <org>/<name>`
(which also writes the model card), then `python scripts/upload_to_hf.py --checkpoint_dir <dir>
--repo_id <org>/<name>`.

## Citation

```bibtex
@misc{recamplify,
  title         = {recAMPLIFY: Large-Scale Pretraining of Recursive Protein Language Models},
  author        = {Jha, Saurav and Liao, Qianli and Vernon, Robert McCoy and Chandar, Sarath and Fournier, Quentin},
  year          = {2026},
  eprint        = {2610.26747},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2610.26747},
}
```

## License

See [LICENSE](LICENSE).
