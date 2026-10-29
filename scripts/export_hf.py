"""Export a recAMPLIFY training checkpoint as a self-contained Hugging Face model repo.

Usage:
    python scripts/export_hf.py \
        --checkpoint_dir <run>/hf_checkpoints/checkpoint_100000 \
        --output_dir exports/recAMPLIFY-halt-only --repo_id <org>/recAMPLIFY-halt-only

The export loads with plain ``transformers`` (no install of this package):

    AutoModelForMaskedLM.from_pretrained(<dir or repo>, trust_remote_code=True)

What the export does on top of the training checkpoint:
  * rewrites ``config.json`` with ``model_type="recamplify"`` and only the fields the model
    reads, dropping the dead fields that the pre-release code serialised;
  * copies the model and tokenizer code next to the weights and sets ``auto_map``;
  * re-saves the weights as safetensors after a strict state-dict round trip, and checks
    that the exported model reproduces the original logits;
  * writes a model card (``README.md``).

Upload the result with ``scripts/upload_to_hf.py``.
"""

import argparse
import inspect
import os
import re

import torch

from amplify.model import AMPLIFYConfig, ProteinTokenizer, RecAMPLIFY, RecAMPLIFYConfig

CODE_URL = "https://github.com/chandar-lab/AMPLIFY/tree/recamplify"

_VARIANTS = {"none": "halt_only", "scalar": "gate_scalar", "lowrank": "gate_lowrank", "perdim": "gate_perdim"}
_DESCRIPTIONS = {
    "baseline": "fixed-depth recursion: every sequence runs `L` cycles (no GMR)",
    "halt_only": "halt-only GMR: a halt head alone allocates depth (PonderNet-style halting)",
    "gate_scalar": "GMR with a scalar gate: one survival channel per sequence",
    "gate_lowrank": "GMR with a rank-16 gate: 16 survival channels mapped to features",
    "gate_perdim": "GMR with a per-dimension gate: one survival channel per feature",
}


def variant_of(config: RecAMPLIFYConfig) -> str:
    """Name of the recAMPLIFY variant a config describes."""
    return _VARIANTS[config.ponder_gate_mode] if config.use_ponder_act else "baseline"


def clean_config(config: RecAMPLIFYConfig) -> RecAMPLIFYConfig:
    """Rebuild *config* from the constructor arguments only, dropping any other field."""
    fields = set(inspect.signature(RecAMPLIFYConfig.__init__).parameters)
    fields |= set(inspect.signature(AMPLIFYConfig.__init__).parameters)
    fields -= {"self", "kwargs"}
    values = config.to_dict()
    return RecAMPLIFYConfig(**{k: values[k] for k in fields if k in values})


def model_card(config: RecAMPLIFYConfig, repo_id: str, num_params: int, train_steps: str) -> str:
    """Hugging Face model card for one exported variant."""
    variant = variant_of(config)
    recur = config.num_core_layers - config.no_recur_layers - config.no_exit_layers
    depth = config.no_recur_layers + (config.L_cycles + 1) * recur + config.no_exit_layers
    adaptive = (
        "\n`out.avg_cycles` is the expected number of cycles under the learned halting distribution."
        if config.use_ponder_act
        else ""
    )
    return f"""---
license: mit
library_name: transformers
pipeline_tag: fill-mask
tags:
  - protein
  - protein-language-model
  - recAMPLIFY
---

# recAMPLIFY — `{variant}`

recAMPLIFY is a masked protein language model whose weight-tied recurrent core refines a
residue-aligned latent state over `L` cycles, so compute per sequence can grow at test time
without adding parameters. This checkpoint is the **{_DESCRIPTIONS[variant]}** variant.

| | |
|---|---|
| unique parameters | {num_params:,} |
| layers (stem / tied core / exit) | {config.no_recur_layers} / {recur} / {config.no_exit_layers} |
| training cycles `L` | {config.L_cycles} (effective depth {depth}, including the read-out core pass) |
| hidden size / heads | {config.hidden_size} / {config.num_attention_heads} |
| pretraining | {train_steps} steps on UniRef100 + SCOP + OAS (AMPLIFY recipe) |

## Usage

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

repo = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(repo, trust_remote_code=True).eval()

inputs = tokenizer(["MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTT"], return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)
logits = out.logits{"  # out.avg_cycles: expected cycles (GMR)" if config.use_ponder_act else ""}

# Test-time depth: run more (or fewer) recursive cycles than the {config.L_cycles} used in training.
model.set_num_cycles(9)
```
{adaptive}
Padded batches run on CPU or GPU with PyTorch SDPA. Packed (FlashAttention varlen) batches
need `flash-attn`, CUDA and half precision.

Training and evaluation code: <{CODE_URL}>.

## Citation

```bibtex
@misc{{recamplify,
  title  = {{recAMPLIFY: Large-Scale Pretraining of Recursive Protein Language Models}},
  author = {{Jha, Saurav and Liao, Qianli and Vernon, Robert McCoy and Chandar, Sarath and Fournier, Quentin}},
  year   = {{2026}},
  eprint = {{2610.26747}},
  archivePrefix = {{arXiv}},
  url    = {{https://arxiv.org/abs/2610.26747}},
}}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a recAMPLIFY checkpoint for the Hugging Face Hub.")
    parser.add_argument("--checkpoint_dir", required=True, help="Training checkpoint (config.json + weights).")
    parser.add_argument("--output_dir", required=True, help="Where to write the self-contained export.")
    parser.add_argument("--repo_id", default="<org>/<model>", help="Hub repo id quoted in the model card.")
    args = parser.parse_args()

    source = RecAMPLIFY.from_pretrained(args.checkpoint_dir).eval()
    tokenizer = ProteinTokenizer.from_pretrained(args.checkpoint_dir)

    # Strict round trip into a model built from the cleaned config: proves the dropped fields
    # were dead and that every weight and buffer is accounted for.
    model = RecAMPLIFY(clean_config(source.config)).eval()
    model.load_state_dict(source.state_dict(), strict=True)

    ids = torch.randint(5, 25, (2, 48), generator=torch.Generator().manual_seed(0))
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[1, 30:] = False
    with torch.no_grad():
        diff = (source(ids, attention_mask=mask).logits - model(ids, attention_mask=mask).logits).abs().max().item()
    # Float32 kernels may differ in the last bits between two module instances (measured up to
    # ~1e-5 on logits of magnitude ~10); anything larger is a real mismatch.
    if diff > 1e-4:
        raise RuntimeError(f"exported model does not reproduce the checkpoint's logits (max |diff| {diff:.2e})")

    # Copies recamplify.py (with its relative import amplify.py) and tokenizer.py into the
    # export and records them in auto_map, so trust_remote_code loading needs no install.
    RecAMPLIFYConfig.register_for_auto_class()
    RecAMPLIFY.register_for_auto_class("AutoModelForMaskedLM")
    tokenizer.register_for_auto_class("AutoTokenizer")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    step = re.search(r"checkpoint_(\d+)", os.path.abspath(args.checkpoint_dir))
    train_steps = f"{int(step.group(1)):,}" if step else "unknown"
    num_params = sum(p.numel() for p in model.parameters())
    with open(os.path.join(args.output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card(model.config, args.repo_id, num_params, train_steps))

    print(f"Exported {variant_of(model.config)} ({num_params:,} params, max |logit diff| {diff:.1e}) to {args.output_dir}")


if __name__ == "__main__":
    main()
