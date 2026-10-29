import argparse
import os
import pathlib
import inspect
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def _summarize_perchannel(pmf_pc, gate_traj_sum, n_forwards: int) -> dict:
    """Compact JSON-friendly summary of gated-branch per-channel halt diagnostics.

    pmf_pc: ndarray (L_max, C) — forward-averaged halt PMF per channel.
        Each column already ~sums to 1 over depths (model normalizes); we
        re-normalize defensively here.
    gate_traj_sum: ndarray (L_max, C) or None — summed gate-openness trajectory
        across forwards; we divide by ``n_forwards`` to get the average.
    n_forwards: int — number of forward passes accumulated.

    For small C (≤ 64; covers scalar + lowrank-up-to-64) we write the full
    (L_max, C) matrices for downstream visualization.  For larger C (perdim,
    C=D≈960) the matrices balloon the sidecar so we keep only summary stats.
    """
    L_max, C = pmf_pc.shape
    col_sums = pmf_pc.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums > 0, col_sums, 1.0)
    pmf_norm = pmf_pc / col_sums                                  # (L_max, C)
    depths   = np.arange(1, L_max + 1, dtype=float)
    e_n_per_channel = (pmf_norm * depths[:, None]).sum(axis=0)    # (C,)

    out: dict = {"n_channels": int(C)}
    out["per_channel_E_n_mean"] = round(float(e_n_per_channel.mean()), 4)
    out["per_channel_E_n_std"]  = round(float(e_n_per_channel.std()), 4)
    qs = np.quantile(e_n_per_channel, [0.0, 0.25, 0.5, 0.75, 1.0])
    out["per_channel_E_n_quantiles"] = {
        "min":    round(float(qs[0]), 4),
        "q25":    round(float(qs[1]), 4),
        "median": round(float(qs[2]), 4),
        "q75":    round(float(qs[3]), 4),
        "max":    round(float(qs[4]), 4),
    }
    if gate_traj_sum is not None and n_forwards > 0:
        gate_avg = gate_traj_sum / max(n_forwards, 1)
        out["gate_openness_mean_per_cycle"] = [round(float(v), 4) for v in gate_avg.mean(axis=1).tolist()]
        out["gate_openness_std_per_cycle"]  = [round(float(v), 4) for v in gate_avg.std(axis=1).tolist()]
    else:
        gate_avg = None
    if C <= 64:
        out["halt_pmf_perchannel"] = [[round(float(v), 6) for v in row] for row in pmf_pc.tolist()]
        if gate_avg is not None:
            out["gate_openness_trajectory"] = [[round(float(v), 6) for v in row] for row in gate_avg.tolist()]
    return out


# --- Small wrapper to normalize AMPLIFY model API differences ---
class AmplifyForwardWrapper(torch.nn.Module):
    """Unifies forward arg names across AMPLIFY variants (src vs input_ids).

    - We only pass token ids (no attention mask) since batches are rectangular
    - Exposes a `get_logits(input_ids)` method returning [B, T, V] tensor.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        # PonderNet halt-step accumulator (None for non-ponder ckpts).
        # forward() appends every batch's avg_halt_step / halt_pmf / ponder_cost
        # here; pop_halt_stats() returns the mean + count and resets, for
        # sidecar JSON logging.
        self._halt_sum: float = 0.0
        self._halt_n: int = 0
        self._halt_pmf_sum: Optional[np.ndarray] = None  # (L_max,) — lazy-init on first capture
        self._pc_sum: float = 0.0
        self._pc_n: int = 0
        # Gated-branch only: per-channel halt PMF (L_max, C) + gate openness
        # trajectory (L_max, C).  Stay None for non-gated ponder / cm=none ckpts.
        self._halt_pmf_perchannel_sum: Optional[np.ndarray] = None
        self._gate_traj_sum: Optional[np.ndarray] = None
        # Per-forward halt_depth_per_seq (B,) arrays — buffered so the caller
        # can snapshot ``snap_halt_buf()`` before an assay and ``pop_persseq_halt(start)``
        # after to get per-assay halt-depth stats.  None on non-ponder ckpts.
        self._halt_depth_buf: list = []
        # Per-forward halt_pmf_entropy_per_seq (B,) and cycle_z_bos_cos_sim (B, L-1).
        # Parallel to _halt_depth_buf (same forward passes, same index space).
        self._halt_pmf_entropy_buf: list = []
        self._cycle_cos_sim_buf: list = []
        # Per-forward cycle_argmax_flips_per_seq (B,): prediction-stability signal.
        # Parallel to _halt_depth_buf (same slicing per assay / per position).
        self._cycle_argmax_flips_buf: list = []
        cfg = getattr(model, "config", None)
        self._L_max: Optional[int] = int(getattr(cfg, "L_cycles", 0)) if cfg is not None else None
        self._gate_mode: str = str(getattr(cfg, "ponder_gate_mode", "none")) if cfg is not None else "none"
        # Try to ensure non-packed inference
        try:
            if hasattr(self.model, "pack_sequences"):
                self.model.pack_sequences = False
            if hasattr(getattr(self.model, "module", None), "pack_sequences"):
                self.model.module.pack_sequences = False
            if hasattr(getattr(self.model, "config", None), "pack_sequences"):
                self.model.config.pack_sequences = False
        except Exception:
            pass
        # Inspect forward signature once to decide the expected arg names.
        try:
            sig = inspect.signature(model.forward)
        except (ValueError, AttributeError):
            sig = inspect.signature(model.__call__)

        params = set(sig.parameters.keys())
        if "src" in params and "input_ids" not in params:
            self._ids_kw = "src"
        else:
            self._ids_kw = "input_ids"

        # Discover optional flags we can propagate
        self._supports_hidden = "output_hidden_states" in params
        self._supports_attn = "output_attentions" in params
        self._supports_pos = "position_ids" in params
        self._supports_attn_mask = "attention_mask" in params

        print(
            f"AmplifyForwardWrapper defined with ids_kw='{self._ids_kw}', "
            f"supports_hidden={self._supports_hidden}, supports_attn={self._supports_attn}, "
            f"supports_pos={self._supports_pos}, supports_attn_mask={self._supports_attn_mask}"
            f" for model {type(model).__name__}"
            f"Base model API: {sig}"
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        kwargs = {self._ids_kw: input_ids}
        if self._supports_hidden:
            kwargs["output_hidden_states"] = output_hidden_states
        if self._supports_attn:
            kwargs["output_attentions"] = output_attentions
        if self._supports_pos and position_ids is not None:
            kwargs["position_ids"] = position_ids
        if self._supports_attn_mask and attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        out = self.model(**kwargs)
        # PonderNet halt-step capture (None for non-ponder modes — silently ignored).
        ahs = getattr(out, "avg_halt_step", None)
        if ahs is None:
            ahs = getattr(out, "avg_cycles", None)  # recAMPLIFY naming
        if ahs is not None:
            try:
                self._halt_sum += float(ahs.item() if hasattr(ahs, "item") else ahs)
                self._halt_n += 1
            except Exception:
                pass
        # PonderNet halt PMF capture: model returns (L_max,) batch-averaged.
        pmf = getattr(out, "halt_pmf", None)
        if pmf is not None:
            try:
                _pmf_np = pmf.detach().float().cpu().numpy()
                if self._halt_pmf_sum is None:
                    self._halt_pmf_sum = _pmf_np.copy()
                else:
                    self._halt_pmf_sum += _pmf_np
            except Exception:
                pass
        # PonderNet per-sequence ponder cost (shape (B,)); accumulate batch-mean.
        pc = getattr(out, "ponder_cost", None)
        if pc is not None:
            try:
                self._pc_sum += float(pc.detach().float().mean().item())
                self._pc_n += 1
            except Exception:
                pass
        # Gated-branch per-channel halt PMF (L_max, C) — sum across forwards.
        pmf_pc = getattr(out, "halt_pmf_perchannel", None)
        if pmf_pc is not None:
            try:
                arr = pmf_pc.detach().float().cpu().numpy()
                self._halt_pmf_perchannel_sum = (
                    arr.copy() if self._halt_pmf_perchannel_sum is None
                    else self._halt_pmf_perchannel_sum + arr
                )
            except Exception:
                pass
        # Gated-branch gate-openness trajectory (L_max, C) — sum across forwards.
        gt = getattr(out, "gate_openness_trajectory", None)
        if gt is not None:
            try:
                arr = gt.detach().float().cpu().numpy()
                self._gate_traj_sum = (
                    arr.copy() if self._gate_traj_sum is None
                    else self._gate_traj_sum + arr
                )
            except Exception:
                pass
        # Per-forward halt_depth_per_seq (B,).  Buffered so caller can slice by
        # assay via ``snap_halt_buf()`` before + ``pop_persseq_halt(start)`` after.
        hdps = getattr(out, "halt_depth_per_seq", None)
        if hdps is not None:
            try:
                self._halt_depth_buf.append(hdps.detach().float().cpu().numpy())
            except Exception:
                pass
        # Per-forward halt_pmf_entropy_per_seq (B,) — H(p_k) per sequence.
        hent = getattr(out, "halt_pmf_entropy_per_seq", None)
        if hent is not None:
            try:
                self._halt_pmf_entropy_buf.append(hent.detach().float().cpu().numpy())
            except Exception:
                pass
        # Per-forward cycle_z_bos_cos_sim (B, L-1) — BOS rep stability across cycles.
        czcs = getattr(out, "cycle_z_bos_cos_sim", None)
        if czcs is not None:
            try:
                self._cycle_cos_sim_buf.append(czcs.detach().float().cpu().numpy())
            except Exception:
                pass
        # Per-forward cycle_argmax_flips_per_seq (B,): prediction-stability signal.
        caf = getattr(out, "cycle_argmax_flips_per_seq", None)
        if caf is not None:
            try:
                self._cycle_argmax_flips_buf.append(caf.detach().float().cpu().numpy())
            except Exception:
                pass
        return out

    def snap_halt_buf(self) -> int:
        """Return current length of the per-forward halt-depth buffer.
        Call BEFORE processing an assay so ``pop_persseq_halt(returned_idx)``
        can compute assay-scoped halt-depth stats."""
        return len(self._halt_depth_buf)

    def pop_persseq_halt(self, start_idx: int) -> dict:
        """Aggregate per-forward halt depths from _halt_depth_buf[start_idx:]
        into a JSON-friendly summary dict (min/q25/median/q75/max/mean/std/n).
        Also aggregates halt_pmf_entropy_per_seq and cycle_z_bos_cos_sim if captured.
        Empty dict on non-ponder ckpts."""
        arrs = self._halt_depth_buf[start_idx:]
        if not arrs:
            return {}
        try:
            hd = np.concatenate(arrs)
        except Exception:
            return {}
        if hd.size == 0:
            return {}
        out = {
            "mean":   round(float(hd.mean()),   4),
            "std":    round(float(hd.std()),    4),
            "min":    round(float(hd.min()),    4),
            "q25":    round(float(np.quantile(hd, 0.25)), 4),
            "median": round(float(np.median(hd)),         4),
            "q75":    round(float(np.quantile(hd, 0.75)), 4),
            "max":    round(float(hd.max()),    4),
            "n_forwards_persseq": int(hd.size),
        }
        # halt PMF entropy per sequence — H(p_k), spread of halt distribution.
        ent_arrs = self._halt_pmf_entropy_buf[start_idx:]
        if ent_arrs:
            try:
                ent = np.concatenate(ent_arrs)
                out["halt_pmf_entropy_mean"] = round(float(ent.mean()), 4)
                out["halt_pmf_entropy_std"]  = round(float(ent.std()),  4)
            except Exception:
                pass
        # Cycle BOS cos-sim — how much the BOS rep changes between consecutive cycles.
        # Each entry is (B, L-1); nonzero mask removes padding.
        cos_arrs = self._cycle_cos_sim_buf[start_idx:]
        if cos_arrs:
            try:
                cos = np.concatenate(cos_arrs, axis=0)   # (N_seqs, L-1)
                nonzero = cos != 0.0
                if nonzero.any():
                    out["cycle_cos_sim_mean"] = round(float(cos[nonzero].mean()), 4)
                    last_vals = [float(row[row != 0.0][-1]) for row in cos if (row != 0.0).any()]
                    if last_vals:
                        out["cycle_cos_sim_last_mean"] = round(float(np.mean(last_vals)), 4)
                    # Prediction-stability proxies (item 3):
                    #   min across cycles → worst-case per-cycle disagreement.
                    #   drift = last − first → does stability trend up (converge) or down (drift)?
                    min_vals = [float(row[row != 0.0].min()) for row in cos if (row != 0.0).any()]
                    if min_vals:
                        out["cycle_cos_sim_min_mean"] = round(float(np.mean(min_vals)), 4)
                    drifts = [float(row[row != 0.0][-1] - row[row != 0.0][0])
                              for row in cos if (row != 0.0).sum() >= 2]
                    if drifts:
                        out["cycle_cos_sim_drift_mean"] = round(float(np.mean(drifts)), 4)
            except Exception:
                pass
        return out

    def pop_halt_stats(self) -> dict:
        """Return aggregated PonderNet halt stats since last pop, then reset.

        Returns:
            {"mean_avg_halt_step": float, "n_forwards": int, "L_max": int,
             "compute_efficiency": float, "compute_savings_fraction": float,
             "mean_ponder_cost": float, "halt_pmf": [..L_max..],
             "halt_pmf_entropy": float, "halt_pmf_std": float, "halt_pmf_mode": int}
            or an empty dict if no halt info was captured (non-ponder ckpt).
        """
        if self._halt_n == 0:
            return {}
        mean = self._halt_sum / self._halt_n
        L = self._L_max or 0
        out = {"mean_avg_halt_step": round(mean, 4), "n_forwards": self._halt_n, "L_max": L}
        if L > 0:
            out["compute_efficiency"] = round(mean / L, 4)
            out["compute_savings_fraction"] = round(1.0 - mean / L, 4)
        if self._pc_n > 0:
            out["mean_ponder_cost"] = round(self._pc_sum / self._pc_n, 6)
        if self._halt_pmf_sum is not None:
            pmf = self._halt_pmf_sum / max(self._halt_n, 1)
            s = float(pmf.sum())
            if s > 0:
                pmf = pmf / s
            out["halt_pmf"] = [round(float(p), 6) for p in pmf.tolist()]
            depths = np.arange(1, len(pmf) + 1, dtype=float)
            e_n = float((pmf * depths).sum())
            var = float((pmf * (depths - e_n) ** 2).sum())
            ent = float(-(pmf * np.log(np.clip(pmf, 1e-12, 1.0))).sum())
            out["halt_pmf_entropy"] = round(ent, 4)
            out["halt_pmf_std"] = round(var ** 0.5, 4)
            out["halt_pmf_mode"] = int(np.argmax(pmf)) + 1
        # Gated-branch per-channel summary (only fires when model has
        # ponder_gate_mode != "none" and emitted the per-channel fields).
        if self._halt_pmf_perchannel_sum is not None:
            pmf_pc = self._halt_pmf_perchannel_sum / max(self._halt_n, 1)
            out["gate_mode"] = self._gate_mode
            out.update(_summarize_perchannel(pmf_pc, self._gate_traj_sum, self._halt_n))
        # Global per-forward halt-depth histogram (over all DMS assays' mutants).
        if self._halt_depth_buf:
            try:
                hd = np.concatenate(self._halt_depth_buf)
                out["halt_depth_quantiles"] = {
                    "min":    round(float(hd.min()),    4),
                    "q05":    round(float(np.quantile(hd, 0.05)), 4),
                    "q25":    round(float(np.quantile(hd, 0.25)), 4),
                    "median": round(float(np.median(hd)), 4),
                    "q75":    round(float(np.quantile(hd, 0.75)), 4),
                    "q95":    round(float(np.quantile(hd, 0.95)), 4),
                    "max":    round(float(hd.max()),    4),
                }
                out["halt_depth_std_across_seqs"] = round(float(hd.std()), 4)
                out["halt_depth_n"] = int(hd.size)
            except Exception:
                pass
        # Global halt PMF entropy per sequence (across all forwards).
        if self._halt_pmf_entropy_buf:
            try:
                ent_all = np.concatenate(self._halt_pmf_entropy_buf)
                out["halt_pmf_entropy_per_seq_mean"] = round(float(ent_all.mean()), 4)
                out["halt_pmf_entropy_per_seq_std"]  = round(float(ent_all.std()),  4)
            except Exception:
                pass
        # Global cycle BOS cos-sim (across all forwards).
        if self._cycle_cos_sim_buf:
            try:
                cos_all = np.concatenate(self._cycle_cos_sim_buf, axis=0)
                nonzero = cos_all != 0.0
                if nonzero.any():
                    out["cycle_cos_sim_global_mean"] = round(float(cos_all[nonzero].mean()), 4)
            except Exception:
                pass
        self._halt_sum, self._halt_n = 0.0, 0
        self._halt_pmf_sum = None
        self._pc_sum, self._pc_n = 0.0, 0
        self._halt_pmf_perchannel_sum = None
        self._gate_traj_sum = None
        self._halt_depth_buf = []
        self._halt_pmf_entropy_buf = []
        self._cycle_cos_sim_buf = []
        self._cycle_argmax_flips_buf = []
        return out

    @torch.no_grad()
    def get_logits(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.forward(
            input_ids,
            output_hidden_states=False,
            output_attentions=False,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        if hasattr(out, "logits"):
            out = out.logits
        if isinstance(out, dict) and "logits" in out:
            out = out["logits"]
        if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            out = out[0]
        return out


def create_parser():
    parser = argparse.ArgumentParser(description="Compute mutation fitness using Hugging Face AMPLIFY models.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for masked-marginals computation (default: 16)",
    )
    parser.add_argument(
        "--autocast-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type for autocast computations: bf16 (default), fp16, or fp32.",
    )
    # removed --model-dtype; rely on autocast only
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more HF model identifiers (for hub). For local, ignore this and provide "
            "--local-checkpoint (.pt) and --local-config (config.yaml)."
        ),
    )
    parser.add_argument(
        "--load-from",
        type=str,
        choices=["auto", "hub", "local", "hf_local"],
        default="auto",
        help=(
            "Where to load models from: auto (default), hub, or local. For local, provide --local-checkpoint (.pt) "
            "and --local-config (config.yaml)."
        ),
    )
    parser.add_argument("--local-checkpoint", type=str, default=None, help="Path to local .pt checkpoint file")
    parser.add_argument("--local-config", type=str, default=None, help="Path to local Hydra config.yaml file")
    parser.add_argument(
        "--use_auth_token",
        type=str,
        default=None,
        help=("Hugging Face auth token."),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional HF revision (branch, tag, or commit) to load models/tokenizers from.",
    )
    parser.add_argument("--sequence", type=str, help="Base WT sequence (overridden by --dms_index mapping if set)")
    parser.add_argument("--target_seq", type=str, default=None, help="Alias for --sequence when not using mapping")
    parser.add_argument("--dms-input", type=pathlib.Path, required=True, help="CSV file or folder (with mapping)")
    parser.add_argument("--dms-output", type=pathlib.Path, required=True, help="Output folder for scored CSV")
    parser.add_argument("--mutation-col", type=str, default="mutant", help="Column with mutation strings")
    parser.add_argument(
        "--dms_index",
        type=str,
        nargs="+",
        default=None,
        help="Indices of DMS in mapping file, or 'all'",
    )
    parser.add_argument("--dms_mapping", type=str, default=None, help="CSV with mapping rows when using --dms_index")
    parser.add_argument("--offset-idx", type=int, default=1, help="Offset of mutation positions in mutation column")
    parser.add_argument(
        "--bos-offset",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Index offset to account for a leading BOS token and trailing EOS in tokenized inputs. "
            "Use 1 when tokenizer adds BOS/EOS (default), 0 when no special tokens are added."
        ),
    )
    parser.add_argument(
        "--scoring-strategy",
        type=str,
        default="wt-marginals",
        choices=["wt-marginals", "masked-marginals", "pseudo-ppl"],
        help="Scoring strategy using only sequences",
    )
    parser.add_argument(
        "--scoring-window",
        type=str,
        default="optimal",
        choices=["full", "optimal"],
        help="If sequences exceed max length, use an optimal local window around positions",
    )
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    parser.add_argument("--overwrite-prior-scores", action="store_true", help="Overwrite existing columns if present")
    parser.add_argument(
        "--max-length", type=int, default=2048,
        help="Context window for masked-marginals / pseudo-ppl windowing. Sequences longer "
             "than this are windowed around each position. Default 2048 (= model "
             "max_position_embeddings). Do NOT confuse with config.max_length (HF generation "
             "default of 20, which previously caused 18-residue windowing and near-zero DMS).",
    )
    parser.add_argument(
        "--l-cycles", type=int, default=None,
        help="Override SRM/TRM model.config.L_cycles after load (recursive inference passes). "
             "Has no effect for vanilla AMPLIFY checkpoints.",
    )
    # Eval-only stochastic init-perturbation (SRM only; deterministic by default).
    parser.add_argument("--init-perturb-std", type=float, default=0.0, dest="init_perturb_std",
                        help="Init-perturbation noise std on SRM recurrence-start z. 0 = deterministic.")
    parser.add_argument("--init-perturb-samples", type=int, default=1, dest="init_perturb_samples",
                        help="Number of perturbation samples N to ensemble (1 = deterministic).")
    parser.add_argument("--init-perturb-agg", type=str, default="average", dest="init_perturb_agg",
                        choices=["average", "vote"], help="Aggregation across samples: average | vote.")
    parser.add_argument("--init-perturb-seed", type=int, default=0, dest="init_perturb_seed",
                        help="Base seed for per-sample noise (sample s uses seed+s).")
    parser.add_argument("--ponder-eval-decode", type=str, default=None, dest="ponder_eval_decode",
                        choices=["weighted", "greedy"],
                        help="Ponder_act eval decode (use_ponder_act ckpts only): weighted = Sum_n p_n*z_n "
                             "(default), greedy = z at argmax-halt step. None keeps the model default.")
    parser.add_argument("--init-perturb-init-only", action="store_true", dest="init_perturb_init_only",
                        help="Legacy: noise the recurrence-start z once instead of every step (default off = per-step PTRM-style).")
    parser.add_argument("--verifier", type=str, default="none", choices=["none", "red"],
                        help="Best-of-N trajectory selector: 'red' = RED via frozen external pLM (PTRM-style); 'none' = average/vote.")
    parser.add_argument("--verifier-checkpoint-dir", type=str, default=None, dest="verifier_checkpoint_dir",
                        help="Vanilla AMPLIFY checkpoint dir used as the frozen RED scorer (required when --verifier red).")
    parser.add_argument("--verifier-autocast", type=str, default="fp32", choices=["bf16", "fp16", "fp32"],
                        dest="verifier_autocast", help="Autocast dtype for the RED scorer forward.")
    return parser


def _window_bounds(center: int, seq_len: int, window_size: int) -> Tuple[int, int]:
    """Simple symmetric window [start, end) around center token index (0-based over raw sequence)."""
    if window_size >= seq_len:
        return 0, seq_len
    half = window_size // 2
    start = max(0, center - half)
    end = start + window_size
    if end > seq_len:
        end = seq_len
        start = max(0, end - window_size)
    return start, end


def _device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_hf_model_and_tokenizer(
    model_name: str, device: torch.device, revision: Optional[str] = None, auth_token: Optional[str] = None
):
    from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

    if auth_token == "":
        auth_token = None
    # AMPLIFY models on HF expose a compatible masked LM head for token-level scoring
    kwargs = {"trust_remote_code": True}
    if revision is not None:
        kwargs["revision"] = revision
    if auth_token is not None:
        kwargs["token"] = auth_token

    print(f"Loading HF model {model_name} on device {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    # AMPLIFY's remote code returns logits straight from AutoModel; every other
    # HF encoder (ESM-2, ESM-1b, ...) exposes the MLM head only through
    # AutoModelForMaskedLM. Loading those with AutoModel silently yields
    # last_hidden_state where get_logits() expects [B, T, V] logits.
    _is_amplify = ("AMPLIFY" in model_name) or model_name.lower().startswith("lolalb/")
    _loader = AutoModel if _is_amplify else AutoModelForMaskedLM
    print(f"  loader = {_loader.__name__}")
    model = _loader.from_pretrained(model_name, **kwargs)
    model.eval()
    model.to(device)
    return tokenizer, model


def _load_local_amplify_model_and_tokenizer(
    checkpoint_path: str,
    config_path: str,
    device: torch.device,
):
    """Load an AMPLIFY model/tokenizer from explicit local paths.

    Args:
        checkpoint_path: Path to a .pt checkpoint file.
        config_path: Path to a Hydra config.yaml file.
        device: torch device.
    """
    try:
        from amplify.model import AMPLIFY  # type: ignore
    except Exception as e:
        raise ImportError(
            "Could not import amplify.model.AMPLIFY. Ensure AMPLIFY is installed (pip install -e .) or on PYTHONPATH."
        ) from e

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Local AMPLIFY checkpoint (.pt) not found at: {checkpoint_path}")
    if not checkpoint_path.endswith(".pt"):
        raise ValueError(f"--local-checkpoint must point to a .pt file, got: {checkpoint_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Local AMPLIFY config.yaml not found at: {config_path}")

    print(f"Loading local AMPLIFY model from checkpoint {checkpoint_path} with config {config_path} on device {device}")
    model, tokenizer = AMPLIFY.load(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )
    model.eval()
    model.to(device)
    return tokenizer, model


def _load_local_srm_hf_dir(checkpoint_dir: str, device: torch.device):
    """Load recAMPLIFY from a HuggingFace-format local checkpoint directory."""
    import json, sys as _sys
    from pathlib import Path

    # The model is imported through apperitif._amplify; no sys.path walking.

    cfg_file = os.path.join(checkpoint_dir, "config.json")
    with open(cfg_file) as fh:
        saved_cfg = json.load(fh)

    mtype = saved_cfg.get("model_type", "amplify_recursive")

    if mtype == "amplify":
        # Vanilla AMPLIFY baseline (no recursive core) — load via transformers AutoModelForMaskedLM.
        # tie_word_embeddings=False: these checkpoints have a separately-trained lm_head; the configs
        # omit the flag (HF default True), which would re-tie lm_head to the embeddings and discard the
        # trained decoder -> uniform predictions / DMS~0. Force untied so the trained decoder is used.
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        model = AutoModelForMaskedLM.from_pretrained(checkpoint_dir, trust_remote_code=True, tie_word_embeddings=False)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
        model.eval()
        model.to(device)
        print(f"Loaded vanilla AMPLIFY checkpoint from {checkpoint_dir} (model_type={mtype}) on {device}")
        return tokenizer, model

    from apperitif._amplify import RecAMPLIFY  # noqa: F401
    from amplify.model.tokenizer import ProteinTokenizer  # noqa: F401

    # tie_word_embeddings=False: SRM/Recursive checkpoints also ship a separately-trained lm_head;
    # without this the default tie re-initializes/overwrites the decoder -> uniform predictions.
    if mtype in ("recamplify", "amplify_srm"):
        # See load_srm.py: an untied lm_head is required for correct MLM logits.
        model = RecAMPLIFY.from_pretrained(checkpoint_dir, tie_word_embeddings=False)

    model.eval()
    model.to(device)

    tokenizer = ProteinTokenizer.from_pretrained(checkpoint_dir)
    print(f"Loaded local SRM/TRM checkpoint from {checkpoint_dir} (model_type={mtype}) on {device}")
    return tokenizer, model


def _char_to_id(tokenizer, aa: str) -> int:
    tid = tokenizer.convert_tokens_to_ids(aa)
    if tid is None or tid == tokenizer.unk_token_id:
        raise ValueError(f"Unknown residue token '{aa}' for tokenizer {type(tokenizer).__name__}")
    return tid


def _tokenize_seq(tokenizer, seq: str, device: torch.device, add_special_tokens: bool = True):
    """Tokenize a single sequence. We only use input_ids and drop attention masks. Ensures batch dim."""
    enc = tokenizer(seq, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = enc["input_ids"].to(device)
    # Be robust: ensure shape [B, T]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    return {"input_ids": input_ids}


# Removed special-token alignment; we assume a single BOS at index 0 and EOS at the end.


def _log_softmax_logits(model_wrapper: AmplifyForwardWrapper, inputs, autocast_dtype: str = "bf16") -> torch.Tensor:
    # Use autocast for computation (default bf16, fp16 if weights are fp16)
    autocast_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    autocast_t = autocast_map.get(autocast_dtype, torch.bfloat16)
    with torch.no_grad():
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, dtype=autocast_t, enabled=True):
            input_ids = inputs["input_ids"]
            # Build per-batch position_ids to satisfy RoPE shape (B, T)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            B, T = input_ids.shape[0], input_ids.shape[1]
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
            logits = model_wrapper.get_logits(input_ids, position_ids=pos)  # [B, T, V]
            return torch.log_softmax(logits, dim=-1)


def _label_row_hf(
    row_mut_str: str,
    sequence: str,
    token_log_probs: torch.Tensor,
    tokenizer,
    offset_idx: int,
    bos_offset: int,
) -> float:
    # token_log_probs expected shape: [1, T, V] with special tokens at positions 0 and T-1
    score = 0.0
    for mutation in row_mut_str.split(":"):
        wt, idx, mt = mutation[0], int(mutation[1:-1]) - offset_idx, mutation[-1]
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
        wt_id = _char_to_id(tokenizer, wt)
        mt_id = _char_to_id(tokenizer, mt)
        # Offset by BOS when present
        score += (token_log_probs[0, bos_offset + idx, mt_id] - token_log_probs[0, bos_offset + idx, wt_id]).item()
    return float(score)


def _per_mutant_stat(row_mut_str: str, pos_stat_list: list, offset_idx: int):
    """Attribute a per-position stat (halt depth / entropy / cos-sim) to a
    mutant string like ``A123G`` or ``A123G:C456T``.  For multi-mutations we
    average across the mutation positions.  Returns ``None`` when no data is
    available for any position (non-ponder ckpt or windowed fallback)."""
    vals = []
    try:
        for mutation in row_mut_str.split(":"):
            idx = int(mutation[1:-1]) - offset_idx
            if 0 <= idx < len(pos_stat_list):
                v = pos_stat_list[idx]
                if v is not None:
                    vals.append(v)
    except Exception:
        return None
    if not vals:
        return None
    return round(float(sum(vals) / len(vals)), 4)


def _write_permutant_sidecar(dms_output_dir: str, DMS_id: str, mutants: list,
                             pos_stats: dict, offset_idx: int) -> None:
    """Emit a compact parquet sidecar ``<DMS_id>_halt_permutant.parquet`` with
    per-mutant halt / entropy / cos-sim stats in fp32.  Skips the write if all
    per-position stats are empty (non-ponder ckpt) so no bloat on baselines.

    Storage: parquet + snappy compression + fp32 → ~5-10× smaller than CSV
    columns.  For 217 assays × avg 10K mutants this totals ~10 MB per model.
    """
    # Skip when the model didn't populate any of the ponder fields.
    if not any(v is not None
               for lst in pos_stats.values() for v in lst):
        return
    try:
        import pandas as pd
        cols = {"mutant": mutants}
        # Attribute each per-position stat vector to per-mutant means.
        for key, vec in pos_stats.items():
            cols[key] = [
                _per_mutant_stat(m, vec, offset_idx) for m in mutants
            ]
        _df = pd.DataFrame(cols)
        # Downcast float columns to fp32 for compactness.
        for c in _df.columns:
            if c != "mutant":
                _df[c] = _df[c].astype("float32")
        _path = os.path.join(dms_output_dir, f"{DMS_id}_halt_permutant.parquet")
        _df.to_parquet(_path, index=False, compression="snappy")
    except Exception as _e:
        # Non-fatal — parquet may not be installed in older envs.
        print(f"[WARN] per-mutant sidecar write failed for {DMS_id}: {_e}")


def _effective_max_len(cfg, tokenizer, override=None) -> int:
    """Resolve the model's CONTEXT length for windowing.

    IMPORTANT: do NOT use cfg.max_length — on HF configs that is the *generation*
    default (20) and has nothing to do with the encoder context window. Use
    max_position_embeddings (2048 for AMPLIFY/SRM). An explicit override (e.g.
    the --max-length CLI flag) takes precedence.
    """
    if override:
        return int(override)
    return int(
        getattr(cfg, "max_position_embeddings", None)
        or getattr(tokenizer, "model_max_length", 2048)
        or 2048
    )


def _compute_pseudo_ppl(
    seq: str,
    tokenizer,
    model_wrapper: AmplifyForwardWrapper,
    device: torch.device,
    autocast_dtype: str = "bf16",
    dms_idx=None,
    bos_offset: int = 1,
    max_length: int = None,
) -> float:
    # Sum of log p(x_i | x_\i) by masking one token at a time
    max_len_cfg = getattr(getattr(model_wrapper, "model", None), "config", None)
    max_len = _effective_max_len(max_len_cfg, tokenizer, max_length)
    # Reserve space for BOS/EOS when present
    reserved_special = 2 * int(bos_offset == 1)
    window_cap = max_len - reserved_special if max_len and max_len > reserved_special else len(seq)

    log_probs = []
    bar_desc = f"pppl DMS={dms_idx}" if dms_idx is not None else "pppl"
    for pos in tqdm(range(len(seq)), desc=bar_desc):
        if window_cap < len(seq):
            start, end = _window_bounds(pos, len(seq), window_cap)
            subseq = seq[start:end]
            local_pos = pos - start
            inputs = _tokenize_seq(tokenizer, subseq, device, add_special_tokens=(bos_offset == 1))
            # Mask position (offset for BOS if present)
            inputs["input_ids"][0, bos_offset + local_pos] = tokenizer.mask_token_id
        else:
            inputs = _tokenize_seq(tokenizer, seq, device, add_special_tokens=(bos_offset == 1))
            inputs["input_ids"][0, bos_offset + pos] = tokenizer.mask_token_id

        logp = _log_softmax_logits(model_wrapper, inputs, autocast_dtype=autocast_dtype)
        aa_id = _char_to_id(tokenizer, seq[pos])
        if window_cap < len(seq):
            log_probs.append(logp[0, bos_offset + local_pos, aa_id].item())
        else:
            log_probs.append(logp[0, bos_offset + pos, aa_id].item())
    return float(np.sum(log_probs))


def _masked_marginals(
    sequence: str,
    tokenizer,
    model_wrapper: AmplifyForwardWrapper,
    device: torch.device,
    scoring_window: str,
    batch_size: int = 16,
    autocast_dtype: str = "bf16",
    dms_idx=None,
    bos_offset: int = 1,
    max_length: int = None,
) -> torch.Tensor:
    # For each position, mask it, run model, and collect the full vocab log-prob vector.
    _cfg = getattr(getattr(model_wrapper, "model", None), "config", None)
    max_len = _effective_max_len(_cfg, tokenizer, max_length)
    reserved_special = 2 * int(bos_offset == 1)
    window_cap = max_len - reserved_special if max_len and max_len > reserved_special else len(sequence)
    use_windows = scoring_window == "optimal" and window_cap < len(sequence)
    all_rows = []  # list of [V] tensors per position
    # Per-position interpretability metrics.  Aligned with `positions` — index
    # `pos` in each list corresponds to sequence[pos].  Stay Nones for
    # non-ponder ckpts.
    all_pos_halt:    list = [None] * len(sequence)
    all_pos_entropy: list = [None] * len(sequence)
    all_pos_cos_mean: list = [None] * len(sequence)
    all_pos_cos_last: list = [None] * len(sequence)
    all_pos_cos_min:  list = [None] * len(sequence)
    all_pos_cos_drift: list = [None] * len(sequence)
    # Analysis F: per-position prediction stability across recursion cycles.
    all_pos_argmax_flips: list = [None] * len(sequence)
    bar_desc = f"mm DMS={dms_idx}" if dms_idx is not None else "mm"
    positions = list(range(len(sequence)))
    for batch_start in tqdm(range(0, len(positions), batch_size), desc=bar_desc, disable=True):
        batch_positions = positions[batch_start : batch_start + batch_size]
        # Snapshot wrapper's per-forward buffer indices — so we can pull out
        # this batch's per-position stats right after the forward call and
        # attribute them position-by-position.
        _hbuf_before = (model_wrapper.snap_halt_buf()
                        if hasattr(model_wrapper, "snap_halt_buf") else None)
        input_ids = []
        for pos in batch_positions:
            if use_windows:
                start, end = _window_bounds(pos, len(sequence), window_cap)
                subseq = sequence[start:end]
                local_pos = pos - start
                enc = tokenizer(subseq, return_tensors="pt", add_special_tokens=(bos_offset == 1))
                ids = enc["input_ids"].to(device)
                # Ensure ids has a batch dimension for consistent indexing
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                ids[0, bos_offset + local_pos] = tokenizer.mask_token_id
                input_ids.append(ids[0])
            else:
                enc = tokenizer(sequence, return_tensors="pt", add_special_tokens=(bos_offset == 1))
                ids = enc["input_ids"].to(device)
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                ids[0, bos_offset + pos] = tokenizer.mask_token_id
                input_ids.append(ids[0])
        # Stack batch
        input_ids = torch.stack(input_ids, dim=0)
        batch_inputs = {"input_ids": input_ids}
        logp = _log_softmax_logits(model_wrapper, batch_inputs, autocast_dtype=autocast_dtype)
        # logp shape: [batch, T, V]
        for i, pos in enumerate(batch_positions):
            if use_windows:
                start, end = _window_bounds(pos, len(sequence), window_cap)
                local_pos = pos - start
                all_rows.append(logp[i, bos_offset + local_pos].detach().cpu())
            else:
                all_rows.append(logp[i, bos_offset + pos].detach().cpu())
        # Per-position attribution: the wrapper appended one array to each of
        # _halt_depth_buf, _halt_pmf_entropy_buf, _cycle_cos_sim_buf during
        # this forward.  Extract per-item values and align with batch_positions.
        if _hbuf_before is not None:
            try:
                _hd_arrs = model_wrapper._halt_depth_buf[_hbuf_before:]
                _hd_flat = np.concatenate(_hd_arrs) if _hd_arrs else None
                _ent_arrs = model_wrapper._halt_pmf_entropy_buf[_hbuf_before:]
                _ent_flat = np.concatenate(_ent_arrs) if _ent_arrs else None
                _cos_arrs = model_wrapper._cycle_cos_sim_buf[_hbuf_before:]
                _cos_stack = np.concatenate(_cos_arrs, axis=0) if _cos_arrs else None  # (B, L-1)
                _caf_arrs = getattr(model_wrapper, "_cycle_argmax_flips_buf", [])[_hbuf_before:] if hasattr(model_wrapper, "_cycle_argmax_flips_buf") else []
                _caf_flat = np.concatenate(_caf_arrs) if _caf_arrs else None
                for i, pos in enumerate(batch_positions):
                    if _hd_flat is not None and i < _hd_flat.size:
                        all_pos_halt[pos] = float(_hd_flat[i])
                    if _ent_flat is not None and i < _ent_flat.size:
                        all_pos_entropy[pos] = float(_ent_flat[i])
                    if _cos_stack is not None and i < _cos_stack.shape[0]:
                        _row = _cos_stack[i]
                        _nz = _row[_row != 0.0]
                        if _nz.size:
                            all_pos_cos_mean[pos] = float(_nz.mean())
                            all_pos_cos_last[pos] = float(_nz[-1])
                            all_pos_cos_min[pos]  = float(_nz.min())
                            if _nz.size >= 2:
                                all_pos_cos_drift[pos] = float(_nz[-1] - _nz[0])
                    if _caf_flat is not None and i < _caf_flat.size:
                        all_pos_argmax_flips[pos] = float(_caf_flat[i])
            except Exception:
                pass
    # Shape [1, L + reserved_special, V]
    token_probs = torch.zeros((1, len(sequence) + reserved_special, all_rows[0].size(-1)))
    for i, row in enumerate(all_rows):
        token_probs[0, bos_offset + i] = row
    pos_stats = {
        "halt_depth":       all_pos_halt,
        "halt_pmf_entropy": all_pos_entropy,
        "cycle_cos_sim_mean":  all_pos_cos_mean,
        "cycle_cos_sim_last":  all_pos_cos_last,
        "cycle_cos_sim_min":   all_pos_cos_min,
        "cycle_cos_sim_drift": all_pos_cos_drift,
        "cycle_argmax_flips":  all_pos_argmax_flips,
    }
    return token_probs, pos_stats


def main(args):
    if not os.path.exists(args.dms_output):
        os.mkdir(args.dms_output)

    # Resolve device
    device = _device(use_gpu=not args.nogpu)

    # Load single model once
    # Detect HF-format local checkpoint dir: --load-from hf_local, or auto-detect from path
    _model_name_or_path = (
        args.model_name_or_path[0]
        if isinstance(args.model_name_or_path, list)
        else args.model_name_or_path
    )
    _is_local_hf_dir = (
        args.load_from == "hf_local"
        or (args.load_from == "auto" and os.path.isdir(_model_name_or_path) and os.path.isfile(os.path.join(_model_name_or_path, "config.json")))
    )

    if _is_local_hf_dir:
        tokenizer, model = _load_local_srm_hf_dir(_model_name_or_path, device)
        if getattr(args, "l_cycles", None) is not None and hasattr(model, "config") and hasattr(model.config, "L_cycles"):
            new_L = int(args.l_cycles)
            model.set_num_cycles(new_L)
            print(f"[L_cycles override] model.config.L_cycles = {model.config.L_cycles}")
        # Eval-only stochastic init-perturbation knobs (SRM forward reads via getattr).
        if hasattr(model, "config"):
            model.config.init_perturb_std = float(getattr(args, "init_perturb_std", 0.0) or 0.0)
            model.config.init_perturb_samples = int(getattr(args, "init_perturb_samples", 1) or 1)
            model.config.init_perturb_agg = str(getattr(args, "init_perturb_agg", "average") or "average")
            model.config.init_perturb_seed = int(getattr(args, "init_perturb_seed", 0) or 0)
            model.config.init_perturb_per_step = not bool(getattr(args, "init_perturb_init_only", False))
            # Ponder_act eval decode strategy (weighted | greedy); only meaningful for
            # use_ponder_act/full_ponder_act ckpts. None preserves the model default.
            _ped = getattr(args, "ponder_eval_decode", None)
            if _ped is not None:
                model.config.ponder_eval_decode = str(_ped)
                if bool(getattr(model.config, "use_ponder_act", False)) or \
                   bool(getattr(model.config, "use_full_ponder_act", False)):
                    print(f"[ponder] eval decode = {model.config.ponder_eval_decode}")
            if str(getattr(args, "verifier", "none")).lower() == "red":
                raise NotImplementedError("the RED best-of-N verifier is not part of this release")
            if model.config.init_perturb_std > 0.0 and model.config.init_perturb_samples > 1:
                print(f"[init-perturb] std={model.config.init_perturb_std} "
                      f"samples={model.config.init_perturb_samples} "
                      f"agg={model.config.init_perturb_agg} seed={model.config.init_perturb_seed} "
                      f"per_step={model.config.init_perturb_per_step}")
        model_wrapper = AmplifyForwardWrapper(model)
        model_label = os.path.basename(_model_name_or_path.rstrip("/"))
    elif args.load_from == "local":
        if not args.local_checkpoint or not args.local_config:
            raise ValueError(
                "For local loading, provide both --local-checkpoint (.pt) and --local-config (config.yaml)"
            )
        tokenizer, model = _load_local_amplify_model_and_tokenizer(
            checkpoint_path=args.local_checkpoint, config_path=args.local_config, device=device
        )
        model.to(device)
        model_wrapper = AmplifyForwardWrapper(model)
        model_label = os.path.basename(args.local_checkpoint)
    else:
        tokenizer, model = _load_hf_model_and_tokenizer(
            _model_name_or_path, device, revision=args.revision, auth_token=args.use_auth_token
        )
        model.to(device)
        model_wrapper = AmplifyForwardWrapper(model)
        model_label = _model_name_or_path

    # Handle DMS indices (single, multiple, or 'all')
    if args.dms_index is None:
        dms_indices = [None]
    elif any(str(x).lower() == "all" for x in args.dms_index):
        if args.dms_mapping is None:
            raise ValueError("--dms_mapping is required when using --dms_index all")
        mapping = pd.read_csv(args.dms_mapping)
        dms_indices = list(range(len(mapping)))
    else:
        # Convert all entries to int
        dms_indices = [int(x) for x in args.dms_index]

    # Aggregate per-assay halt stats (one entry per DMS ID) if the wrapper
    # buffers per-forward halt depths.
    _per_assay_halt: dict = {}

    for dms_idx in tqdm(dms_indices, desc="DMS"):
        # Snapshot halt-depth buffer index BEFORE processing this assay so we
        # can slice its per-forward halt depths at the end of the loop body.
        _assay_hdps_start = (model_wrapper.snap_halt_buf()
                             if hasattr(model_wrapper, "snap_halt_buf") else 0)
        mutant_col = args.mutation_col
        if dms_idx is not None:
            if args.dms_mapping is None:
                raise ValueError("--dms_mapping is required when using --dms_index")
            mapping = pd.read_csv(args.dms_mapping)
            DMS_id = mapping["DMS_id"][dms_idx]
            row = mapping[mapping["DMS_id"] == DMS_id]
            if len(row) != 1:
                raise ValueError(f"Expected exactly one mapping row for DMS_id={DMS_id}, got {len(row)}")
            row = row.iloc[0].replace(np.nan, "")

            sequence = row["target_seq"].upper()
            dms_input_path = str(args.dms_input) + os.sep + row["DMS_filename"]
            mutant_col = (
                row["DMS_mutant_column"]
                if "DMS_mutant_column" in mapping.columns and row["DMS_mutant_column"]
                else mutant_col
            )
            dms_output_file = str(args.dms_output) + os.sep + DMS_id + ".csv"

            target_seq_start_index = (
                row["start_idx"] if "start_idx" in mapping.columns and row["start_idx"] != "" else 1
            )
            offset_idx = int(target_seq_start_index)
        else:
            if args.sequence is None and args.target_seq is None:
                raise ValueError("Provide --sequence/--target_seq or use --dms_index with --dms_mapping")
            sequence = (args.sequence or args.target_seq).upper()
            dms_input_path = str(args.dms_input)
            DMS_id = pathlib.Path(dms_input_path).name.replace(".csv", "")
            dms_output_file = str(args.dms_output) + os.sep + DMS_id + ".csv"
            offset_idx = args.offset_idx

        df = pd.read_csv(dms_input_path)
        if len(df) == 0:
            raise ValueError("No rows found in the dataframe")

        # Score with the single model and append column
        col_name = os.path.basename(str(model_label).rstrip("/"))

        # Skip if already computed and not overwriting
        if os.path.exists(dms_output_file) and not args.overwrite_prior_scores:
            prior_df = pd.read_csv(dms_output_file)
            if col_name in prior_df.columns:
                df = prior_df
                print(f"Skipping {col_name} (already present). Use --overwrite-prior-scores to recompute.")
            else:
                # Fall through to compute and merge below
                pass

        # Snapshot halt counters before this assay so we can attribute E[n] per assay.
        _assay_halt_sum0 = getattr(model_wrapper, "_halt_sum", 0.0)
        _assay_halt_n0 = getattr(model_wrapper, "_halt_n", 0)

        if col_name not in df.columns:
            if args.scoring_strategy == "wt-marginals":
                raise NotImplementedError("Not sure of the implemenation here")
                inputs = _tokenize_seq(tokenizer, sequence, device, add_special_tokens=(args.bos_offset == 1))
                _cfg = getattr(getattr(model_wrapper, "model", None), "config", None)
                max_len = getattr(_cfg, "max_length", None) or getattr(tokenizer, "model_max_length", 2048) or 2048
                bar_desc = f"wtm DMS={dms_idx}" if dms_idx is not None else "wtm"
                pos_stats = None
                if inputs["input_ids"].shape[1] > max_len and args.scoring_window == "optimal":
                    token_probs, pos_stats = _masked_marginals(
                        sequence,
                        tokenizer,
                        model_wrapper,
                        device,
                        scoring_window="optimal",
                        batch_size=args.batch_size,
                        autocast_dtype=args.autocast_dtype,
                        dms_idx=dms_idx,
                        bos_offset=args.bos_offset,
                    )
                else:
                    for pos in tqdm([0], desc=bar_desc):
                        token_probs = (
                            _log_softmax_logits(model_wrapper, inputs, autocast_dtype=args.autocast_dtype)
                            .detach()
                            .cpu()
                        )

                df[col_name] = df.apply(
                    lambda row: _label_row_hf(
                        row[mutant_col], sequence, token_probs, tokenizer, offset_idx, args.bos_offset
                    ),
                    axis=1,
                )
                # Per-mutant interpretability columns.  Write ALL to a compact
                # companion parquet sidecar (much smaller than adding to CSV).
                if pos_stats is not None:
                    _write_permutant_sidecar(
                        args.dms_output, DMS_id, df[mutant_col].tolist(),
                        pos_stats, offset_idx,
                    )

            elif args.scoring_strategy == "masked-marginals":
                token_probs, pos_stats = _masked_marginals(
                    sequence,
                    tokenizer,
                    model_wrapper,
                    device,
                    args.scoring_window,
                    batch_size=args.batch_size,
                    autocast_dtype=args.autocast_dtype,
                    dms_idx=dms_idx,
                    bos_offset=args.bos_offset,
                    max_length=args.max_length,
                )
                df[col_name] = df.apply(
                    lambda row: _label_row_hf(
                        row[mutant_col], sequence, token_probs, tokenizer, offset_idx, args.bos_offset
                    ),
                    axis=1,
                )
                # Per-mutant interpretability columns → compact parquet sidecar.
                if pos_stats is not None:
                    _write_permutant_sidecar(
                        args.dms_output, DMS_id, df[mutant_col].tolist(),
                        pos_stats, offset_idx,
                    )

            elif args.scoring_strategy == "pseudo-ppl":
                tqdm.pandas()
                if "mutated_sequence" not in df:
                    df["mutated_sequence"] = df.progress_apply(
                        lambda row: _get_mutated_sequence(row[mutant_col], sequence, offset_idx), axis=1
                    )
                df[col_name] = df.progress_apply(
                    lambda row: _compute_pseudo_ppl(
                        row["mutated_sequence"],
                        tokenizer,
                        model_wrapper,
                        device,
                        autocast_dtype=args.autocast_dtype,
                        dms_idx=dms_idx,
                        bos_offset=args.bos_offset,
                        max_length=args.max_length,
                    ),
                    axis=1,
                )

        # Per-assay E[n] — delta halt counters since snapshot above.
        _assay_dn = getattr(model_wrapper, "_halt_n", 0) - _assay_halt_n0
        if _assay_dn > 0:
            _assay_en = (getattr(model_wrapper, "_halt_sum", 0.0) - _assay_halt_sum0) / _assay_dn
            _per_assay_halt[DMS_id] = round(float(_assay_en), 4)

        # Write out incrementally if prior file exists
        if os.path.exists(dms_output_file) and not args.overwrite_prior_scores:
            prior_df = pd.read_csv(dms_output_file)
            if col_name not in prior_df.columns:
                prior_df = prior_df.merge(df[[col_name, mutant_col]], on=mutant_col)
            prior_df.to_csv(dms_output_file, index=False)
            df = prior_df
        else:
            df.to_csv(dms_output_file, index=False)

        # Per-assay halt-depth stats: slice the wrapper buffer that grew during
        # this assay's forwards.  No-op for non-ponder ckpts.
        if hasattr(model_wrapper, "pop_persseq_halt"):
            _assay_hd = model_wrapper.pop_persseq_halt(_assay_hdps_start)
            if _assay_hd:
                _per_assay_halt[DMS_id] = _assay_hd

    # ── PonderNet halt-step sidecar ─────────────────────────────────────────
    # If the model is a ponder_act / full_ponder_act ckpt, the forward wrapper
    # accumulated avg_halt_step on every call.  Dump a sidecar JSON next to the
    # per-assay CSVs so eval_results.json can pick it up.  No-op for non-ponder.
    halt_stats = model_wrapper.pop_halt_stats() if hasattr(model_wrapper, "pop_halt_stats") else {}
    if halt_stats:
        import json as _json
        sidecar = os.path.join(args.dms_output, "halt_stats.json")
        with open(sidecar, "w") as _f:
            _json.dump(halt_stats, _f, indent=2)
        print(f"Wrote PonderNet halt stats -> {sidecar}: {halt_stats}")

    if _per_assay_halt:
        import json as _json
        sidecar_pa = os.path.join(args.dms_output, "halt_stats_per_assay.json")
        with open(sidecar_pa, "w") as _f:
            _json.dump({"expected_halt_depth_per_assay": _per_assay_halt}, _f, indent=2)
        print(f"Wrote per-assay E[n] -> {sidecar_pa} ({len(_per_assay_halt)} assays)")


def _get_mutated_sequence(row: str, wt_sequence: str, offset_idx: int) -> str:
    seq = list(wt_sequence)
    for mutation in row.split(":"):
        wt, idx, mt = mutation[0], int(mutation[1:-1]) - offset_idx, mutation[-1]
        assert seq[idx] == wt, "The listed wildtype does not match the provided sequence"
        seq[idx] = mt
    return "".join(seq)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
