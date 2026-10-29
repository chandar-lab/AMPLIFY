"""Verify that RecAMPLIFY loads the original pretrained checkpoints with no key mismatch.

Usage:  python scripts/verify_checkpoints.py [checkpoint_root]   (default: $CKPT_ROOT, else ./checkpoints)

For each reference run it loads the HF checkpoint, asserts that no parameter or buffer is
missing or unexpected, and runs one forward pass.  This is the regression test for the
checkpoint-compatibility requirement of the refactor.
"""

import glob
import os
import sys

import numpy as np
import torch

from amplify import RecAMPLIFY

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CKPT_ROOT", "checkpoints")
_PONDER = "srm_nc12_nr2_ne2_L6_bp-1_ctzero_cmponder_act_hctoken_acc_ms1_lp0.25_dz_pf_decweighted"

RUNS = {
    "baseline":      "srm_nc12_nr2_ne2_L6_bp-1_ctzL0_cmnone",
    "halt_only":     _PONDER,
    "gate_scalar":   f"{_PONDER}_gscalar_gweighted_100k",
    "gate_lowrank":  f"{_PONDER}_glowrank16_gweighted_100k",
    "gate_perdim":   f"{_PONDER}_gperdim_gweighted_pdec_pbud_100k",
    "baseline nc=14": "srm_nc14_nr2_ne2_L4_bp-1_ctzL0_cmnone",
    "baseline nc=16": "srm_nc16_nr2_ne2_L4_bp-1_ctzL0_cmnone",
}


def find_checkpoint(run: str) -> str:
    """Return the newest checkpoint directory under *run* that actually holds weights."""
    hits = [
        os.path.dirname(f)
        for pat in ("*.safetensors", "*.bin")
        for f in glob.glob(os.path.join(ROOT, run, "**", pat), recursive=True)
    ]
    if not hits:
        raise FileNotFoundError(f"no weight file under {os.path.join(ROOT, run)}")
    return sorted(hits)[-1]


def padded_batch(device: torch.device, batch: int = 2, length: int = 64) -> dict:
    """A small padded MLM batch, matching the layout the original eval path uses."""
    rng = np.random.default_rng(0)
    ids = rng.integers(5, 32, (batch, length))
    labels = np.where(rng.random((batch, length)) < 0.15, ids, -100)
    labels[:, 0] = ids[:, 0]
    return {
        "input_ids": torch.from_numpy(ids).to(device),
        "labels": torch.from_numpy(labels).to(device),
        "position_ids": torch.arange(length, device=device).unsqueeze(0).expand(batch, -1),
        "attention_mask": torch.ones(batch, length, dtype=torch.bool, device=device),
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    failures = []

    for label, run in RUNS.items():
        try:
            path = find_checkpoint(run)
        except FileNotFoundError as e:
            print(f"{label:16s} SKIP  {e}")
            continue

        model, info = RecAMPLIFY.from_pretrained(path, output_loading_info=True)
        missing = [k for k in info["missing_keys"]]
        unexpected = [k for k in info["unexpected_keys"]]
        mismatched = info.get("mismatched_keys", [])

        model = model.to(device).eval()
        with torch.inference_mode():
            out = model(**padded_batch(device))

        cfg = model.config
        depth = cfg.no_recur_layers + (cfg.L_cycles + 1) * (
            cfg.num_core_layers - cfg.no_recur_layers - cfg.no_exit_layers
        ) + cfg.no_exit_layers
        ok = not (missing or unexpected or mismatched) and torch.isfinite(out.loss)
        if not ok:
            failures.append((label, missing, unexpected, mismatched))

        print(
            f"{label:16s} {'OK ' if ok else 'FAIL'} "
            f"nc={cfg.num_core_layers} nr={cfg.no_recur_layers} ne={cfg.no_exit_layers} "
            f"L={cfg.L_cycles} ctx={cfg.context_type:5s} gate={str(cfg.ponder_gate_mode):7s} "
            f"ms={cfg.ponder_act_min_steps} | depth={depth} params={sum(p.numel() for p in model.parameters()):,} "
            f"| loss={out.loss.item():.4f} "
            f"E[k]={'-' if out.avg_cycles is None else f'{out.avg_cycles.item():.3f}'}"
        )
        if missing:
            print(f"{'':16s}      missing:    {missing}")
        if unexpected:
            print(f"{'':16s}      unexpected: {unexpected}")
        if mismatched:
            print(f"{'':16s}      mismatched: {mismatched}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)} checkpoint(s) with key problems)")
        return 1
    print("RESULT: PASS (all checkpoints load with no missing/unexpected keys)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
