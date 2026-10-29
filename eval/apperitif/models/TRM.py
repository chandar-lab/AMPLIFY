"""TRM / SRM encoder wrapper for the AMPLIFY-evaluation (apperitif) suite.

Carry modes at evaluation time
-------------------------------
``none``
    Single forward pass with fixed ``h_cycles`` (or the model's configured
    H_cycles when ``h_cycles=None``).  No Q-halt head, no cross-sequence
    carry.  Equivalent to using vanilla AMPLIFY with extra reasoning depth.

``inference_act``
    ``eval_steps`` separate forward passes are chained: the ``carry_out``
    from pass *k* is fed as ``carry_in`` to pass *k+1*.  Each outer pass
    runs with ``h_cycles=1`` (one H-cycle of the recur_core); the model's
    Q-halt head (trained during ``inference_act`` pre-training) is exposed
    in the output but is **not** used to cut the loop short here — the loop
    always runs all ``eval_steps`` iterations for a clean apples-to-apples
    comparison.  Use this mode to measure whether more "outer thinking steps"
    improve downstream task performance.

``ponder_act``
    Single forward pass.  The PonderNet geometric ACT loop runs *inside*
    ``model.forward()`` (requires the checkpoint to have been trained with
    ``use_ponder_act=True`` in the SRM config).  The model internally
    computes ``z_weighted = Σ_n p_n · z_n`` and stores it in
    ``carry_out.Z`` — so we get a halt-probability-weighted representation
    automatically without any outer loop.

``full_ponder_act``
    Identical eval behaviour to ``ponder_act``.  Use this when the
    checkpoint was trained with ``use_full_ponder_act=True`` (per-step full
    decode variant).  The model selects Path A'' internally; the eval-side
    representation is still ``carry_out.Z = z_weighted``.

Embedding modes
---------------
``z``
    Use ``carry_out.Z`` directly — the recursive state **before** the exit
    head and output norm.  This is the raw "thought" state.

``dec``
    Capture the input to ``lm_head`` via a forward hook — i.e. the
    representation **after** exit head + output norm, right before the
    vocabulary projection.  This is the most semantically polished
    representation and typically performs better on downstream tasks.

Difference between ``inference_act`` and ``none`` at eval
----------------------------------------------------------
Both ``none`` and ``inference_act`` models are trained **statelessly**
(single forward pass per batch, no cross-batch carry during training).
The difference is purely at *evaluation* time:

* ``none``: one forward pass → fixed H×L recur_core applications.  The
  model has no Q-halt head.  Compute is determined by the checkpoint's
  ``H_cycles × L_cycles`` config.

* ``inference_act``: ``eval_steps`` forward passes, each feeding
  ``carry_out → carry_in`` for the next.  The model **has** a Q-halt head,
  trained on intermediate Z states from the inner L-cycle loop to predict
  when the representation has converged.  At each outer step the model
  "reviews" the carry from the previous step, allowing it to refine its
  representation across passes.  The Q-halt head's signal can be used
  for adaptive stopping (not implemented here — fixed ``eval_steps`` for
  reproducibility).  This is the test-time scaling axis: more outer ACT
  steps → better representations, at the cost of proportionally more
  compute.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from .Encoder import EncoderConfig, SequenceEncoder, TokenEncoder, StructureEncoder, residue_tokens
from apperitif.collator import DataCollatorForSeqTasks
from apperitif.utils.dtype_utils import get_torch_dtype


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TRMConfig(EncoderConfig):
    """Configuration for TRM / SRM encoder.

    Required
    --------
    checkpoint_path
        Absolute path to the HF checkpoint directory produced by the
        recursive trainer (contains ``config.json`` + ``model.safetensors``
        + tokenizer files).

    Key optional fields
    -------------------
    carry_mode
        Eval-time carry strategy: ``"none"``, ``"inference_act"``,
        ``"ponder_act"``, or ``"full_ponder_act"``.  Must match how the
        checkpoint was trained.
    eval_steps
        Number of outer ACT passes for ``inference_act`` mode.  Ignored for
        ``"none"`` and ``"ponder_act"``.
    h_cycles
        Override the model's ``H_cycles`` config for each forward pass.
        ``None`` preserves the checkpoint's trained value.  For
        ``inference_act``, each outer step uses ``h_cycles=1`` regardless of
        this setting (the "depth" per step is fixed; the "breadth" is
        ``eval_steps``).
    l_cycles
        Override L_cycles (max ACT steps for ponder_act; reasoning depth for
        none/inference_act).  ``None`` preserves the checkpoint's trained value.
        When overriding for ponder_act/full_ponder_act the geometric prior
        buffer is rebuilt to match the new length.
    embed_mode
        ``"z"`` — use ``carry_out.Z`` (pre-exit recursive state).
        ``"dec"`` — hook ``lm_head`` input (post-exit + post-norm).
    ponder_eval_decode
        Decode strategy for ponder_act / full_ponder_act only.
        ``"weighted"`` — z_weighted = Σ_n p_n · z_n (default, matches training).
        ``"greedy"``   — z at the step with highest halt probability.
        Has no effect for other carry modes.
    """

    # --- identity ---
    model_type: str = "trm"
    model_name: str = "trm"           # used for output-folder naming in main.py
    local_name: Optional[str] = None  # override display name; None → model_name

    # --- checkpoint ---
    checkpoint_path: Optional[str] = None   # full path to HF checkpoint dir

    # --- eval carry strategy ---
    carry_mode: str = "none"                # "none" | "inference_act" | "ponder_act" | "full_ponder_act"
    eval_steps: int = 4                     # outer ACT steps (inference_act only)
    h_cycles: Optional[int] = None         # per-pass H_cycles override (None = ckpt default)
    l_cycles: Optional[int] = None         # L_cycles override (None = ckpt default)

    # --- embedding extraction ---
    embed_mode: str = "dec"                 # "dec" (lm_head input); "z" is not supported

    # --- ponder_act / full_ponder_act decode ---
    ponder_eval_decode: str = "weighted"    # "weighted" | "greedy"

    # --- tokenizer / mask ---
    vocab_path: Optional[str] = None        # fallback vocab file if not in checkpoint
    attention_mask_type: str = "multiplicative"  # "additive" | "multiplicative" | "bool"; RecAMPLIFY casts it with .bool()
    attention_mask_dtype: str = "fp16"

    # --- compatibility with main.py output-folder logic ---
    randomize: bool = False
    randomize_mode: str = "distribution"

    # Inherited from EncoderConfig but set sane defaults here
    max_length: int = 2048
    use_packed_sequence: bool = False       # eval suite uses padded sequences


# ---------------------------------------------------------------------------
# Shared mixin: model loading + forward helpers
# ---------------------------------------------------------------------------

class _TRMModelMixin:
    """Load TRM/SRM checkpoint and expose shared forward helpers."""

    # set by subclass __init__ via _init_trm()
    model: Any
    tokenizer: Any
    carry_mode: str
    eval_steps: int
    h_cycles: Optional[int]
    embed_mode: str
    attention_mask_type: str
    attention_mask_dtype: Any  # torch.dtype

    def _init_trm(self, config: TRMConfig) -> None:
        """Load checkpoint and configure carry / embed settings."""
        if config.checkpoint_path is None:
            raise ValueError(
                "TRMConfig.checkpoint_path must be set to the HF checkpoint directory."
            )

        from apperitif._amplify import (  # noqa: F401
            ProteinTokenizer,
            RecAMPLIFY,
            TokenizerConfig,
        )

        # ── Detect model type from saved config.json ──────────────────────
        cfg_json = os.path.join(config.checkpoint_path, "config.json")
        if not os.path.isfile(cfg_json):
            raise FileNotFoundError(
                f"config.json not found in checkpoint directory: {config.checkpoint_path}"
            )
        with open(cfg_json) as fh:
            saved_cfg = json.load(fh)

        # One recursive class; pre-release checkpoints say "amplify_srm", released ones "recamplify".
        _mtype = saved_cfg.get("model_type", "recamplify")
        if _mtype not in ("recamplify", "amplify_srm"):
            raise ValueError(
                f"unsupported model_type {_mtype!r}; this suite evaluates recAMPLIFY "
                '(model_type="recamplify" or the pre-release "amplify_srm")'
            )
        self.model = RecAMPLIFY.from_pretrained(config.checkpoint_path)
        self._is_srm = True

        self.model.eval()

        # ── Tokenizer ─────────────────────────────────────────────────────
        try:
            self.tokenizer = ProteinTokenizer.from_pretrained(config.checkpoint_path)
        except Exception:
            if config.vocab_path is not None:
                # Build from raw vocab file (fallback for older checkpoints).
                _vocab = _read_vocab(config.vocab_path)
                self.tokenizer = ProteinTokenizer(
                    TokenizerConfig(
                        vocab=_vocab,
                        pad_token="<pad>",
                        mask_token="<mask>",
                        bos_token="<cls>",
                        eos_token="<eos>",
                        unk_token="<unk>",
                    )
                )
            else:
                raise RuntimeError(
                    "Tokenizer not found in checkpoint.  Pass vocab_path in TRMConfig "
                    "as a fallback."
                )

        # ── L_cycles override ─────────────────────────────────────────────
        if config.l_cycles is not None:
            self.model.set_num_cycles(config.l_cycles)

        # ── Eval settings ─────────────────────────────────────────────────
        self.carry_mode           = config.carry_mode
        self.eval_steps           = config.eval_steps
        self.h_cycles             = config.h_cycles
        self.embed_mode           = config.embed_mode
        self.ponder_eval_decode   = config.ponder_eval_decode
        self.use_packed_sequence  = config.use_packed_sequence
        self.attention_mask_type  = config.attention_mask_type
        self.attention_mask_dtype = get_torch_dtype(config.attention_mask_dtype)

        # Correct hidden_size (overrides the placeholder "1" from EncoderConfig.__init__)
        self.hidden_size = self.model.config.hidden_size

        # PonderNet diagnostic accumulators (populated only in ponder_act /
        # full_ponder_act modes when the model exposes the relevant fields on
        # ``RecursiveMaskedLMOutput``).  pop_halt_stats() returns the means +
        # derived distribution stats and resets — evaluators call it after a
        # forward sweep to write a sidecar json next to results.
        self._halt_sum: float = 0.0
        self._halt_n: int = 0
        self._halt_pmf_sum = None       # np.ndarray (L_max,), lazy-init
        self._pc_sum: float = 0.0
        self._pc_n: int = 0
        # Gated-branch only: per-channel halt PMF (L_max, C) + gate openness
        # trajectory (L_max, C).  Stay None for non-gated ponder / cm=none.
        self._halt_pmf_perchannel_sum = None
        self._gate_traj_sum = None
        self._L_max: int = int(getattr(self.model.config, "L_cycles", 0) or 0)
        self._gate_mode: str = str(getattr(self.model.config, "ponder_gate_mode", "none"))
        # Per-forward (B,) halt_depth_per_seq buffer — parallel to entropy +
        # cos-sim buffers below.  All three populated in ponder_act mode from
        # RecursiveMaskedLMOutput fields exposed by amplify_srm.py.
        self._halt_depth_buf: list = []
        # Per-forward (B,) PMF entropy and (B, L-1) BOS cos-sim buffers.
        # Parallel to the halt_depth accumulator; populated in ponder_act mode.
        self._halt_pmf_entropy_buf: list = []
        self._cycle_cos_sim_buf: list = []
        # Per-forward (B,) argmax-flip count: prediction-stability signal.
        # Parallel slicing to halt_depth (populated in eval ponder_act mode).
        self._cycle_argmax_flips_buf: list = []

    def pop_halt_stats(self) -> Dict[str, Any]:
        """Return aggregated PonderNet halt stats since last pop, then reset.

        Returns empty dict for non-ponder modes (no avg_halt_step ever seen).
        """
        if self._halt_n == 0:
            return {}
        mean = self._halt_sum / self._halt_n
        out: Dict[str, Any] = {
            "mean_avg_halt_step": round(mean, 4),
            "n_forwards": self._halt_n,
            "L_max": self._L_max,
        }
        if self._L_max > 0:
            out["compute_efficiency"] = round(mean / self._L_max, 4)
            out["compute_savings_fraction"] = round(1.0 - mean / self._L_max, 4)
        if self._pc_n > 0:
            out["mean_ponder_cost"] = round(self._pc_sum / self._pc_n, 6)
        if self._halt_pmf_sum is not None:
            import numpy as _np
            pmf = self._halt_pmf_sum / max(self._halt_n, 1)
            s = float(pmf.sum())
            if s > 0:
                pmf = pmf / s
            out["halt_pmf"] = [round(float(p), 6) for p in pmf.tolist()]
            depths = _np.arange(1, len(pmf) + 1, dtype=float)
            e_n = float((pmf * depths).sum())
            var = float((pmf * (depths - e_n) ** 2).sum())
            ent = float(-(pmf * _np.log(_np.clip(pmf, 1e-12, 1.0))).sum())
            out["halt_pmf_entropy"] = round(ent, 4)
            out["halt_pmf_std"] = round(var ** 0.5, 4)
            out["halt_pmf_mode"] = int(_np.argmax(pmf)) + 1
        # Gated-branch per-channel summary.
        if self._halt_pmf_perchannel_sum is not None:
            import numpy as _np
            pmf_pc = self._halt_pmf_perchannel_sum / max(self._halt_n, 1)
            L_pc, C_pc = pmf_pc.shape
            col_sums = pmf_pc.sum(axis=0, keepdims=True)
            col_sums = _np.where(col_sums > 0, col_sums, 1.0)
            pmf_pc_norm = pmf_pc / col_sums
            depths_pc = _np.arange(1, L_pc + 1, dtype=float)
            e_n_pc = (pmf_pc_norm * depths_pc[:, None]).sum(axis=0)
            out["gate_mode"] = self._gate_mode
            out["n_channels"] = int(C_pc)
            out["per_channel_E_n_mean"] = round(float(e_n_pc.mean()), 4)
            out["per_channel_E_n_std"]  = round(float(e_n_pc.std()), 4)
            qs = _np.quantile(e_n_pc, [0.0, 0.25, 0.5, 0.75, 1.0])
            out["per_channel_E_n_quantiles"] = {
                "min": round(float(qs[0]),4), "q25": round(float(qs[1]),4),
                "median": round(float(qs[2]),4), "q75": round(float(qs[3]),4),
                "max": round(float(qs[4]),4),
            }
            if self._gate_traj_sum is not None:
                gt = self._gate_traj_sum / max(self._halt_n, 1)
                out["gate_openness_mean_per_cycle"] = [round(float(v),4) for v in gt.mean(axis=1).tolist()]
                out["gate_openness_std_per_cycle"]  = [round(float(v),4) for v in gt.std(axis=1).tolist()]
            if C_pc <= 64:
                out["halt_pmf_perchannel"] = [[round(float(v),6) for v in row] for row in pmf_pc.tolist()]
                if self._gate_traj_sum is not None:
                    gt = self._gate_traj_sum / max(self._halt_n, 1)
                    out["gate_openness_trajectory"] = [[round(float(v),6) for v in row] for row in gt.tolist()]
        if self._halt_depth_buf:
            import numpy as _np
            try:
                _hd_all = _np.concatenate(self._halt_depth_buf)
                out["halt_depth_per_seq_mean"]   = round(float(_hd_all.mean()),   4)
                out["halt_depth_per_seq_std"]    = round(float(_hd_all.std()),    4)
                out["halt_depth_per_seq_min"]    = round(float(_hd_all.min()),    4)
                out["halt_depth_per_seq_q25"]    = round(float(_np.quantile(_hd_all, 0.25)), 4)
                out["halt_depth_per_seq_median"] = round(float(_np.median(_hd_all)), 4)
                out["halt_depth_per_seq_q75"]    = round(float(_np.quantile(_hd_all, 0.75)), 4)
                out["halt_depth_per_seq_max"]    = round(float(_hd_all.max()),    4)
                out["halt_depth_per_seq_n"]      = int(_hd_all.size)
            except Exception:
                pass
        if self._halt_pmf_entropy_buf:
            import numpy as _np
            try:
                _ent_all = _np.concatenate(self._halt_pmf_entropy_buf)
                out["halt_pmf_entropy_per_seq_mean"] = round(float(_ent_all.mean()), 4)
                out["halt_pmf_entropy_per_seq_std"]  = round(float(_ent_all.std()),  4)
            except Exception:
                pass
        if self._cycle_cos_sim_buf:
            import numpy as _np
            try:
                _cos_all = _np.concatenate(self._cycle_cos_sim_buf, axis=0)
                _nonzero = _cos_all != 0.0
                if _nonzero.any():
                    out["cycle_cos_sim_global_mean"] = round(float(_cos_all[_nonzero].mean()), 4)
                    _last_vals = [float(row[row != 0.0][-1]) for row in _cos_all if (row != 0.0).any()]
                    if _last_vals:
                        out["cycle_cos_sim_last_mean"] = round(float(_np.mean(_last_vals)), 4)
                    # Prediction-stability proxies (item 3): worst-case + drift.
                    _min_vals = [float(row[row != 0.0].min()) for row in _cos_all if (row != 0.0).any()]
                    if _min_vals:
                        out["cycle_cos_sim_min_mean"] = round(float(_np.mean(_min_vals)), 4)
                    _drifts = [float(row[row != 0.0][-1] - row[row != 0.0][0])
                               for row in _cos_all if (row != 0.0).sum() >= 2]
                    if _drifts:
                        out["cycle_cos_sim_drift_mean"] = round(float(_np.mean(_drifts)), 4)
            except Exception:
                pass
        # Snapshot per-forward arrays into a class attribute *before* resetting
        # buffers.  Callers (e.g., stage-2 evaluators) can read
        # ``self._last_persseq_arrays`` after get_embeddings_and_labels(test)
        # to align per-seq halt/entropy/cos-sim with the test_predictions.npz
        # produced by the item-1 patch.  Empty dict for non-ponder ckpts.
        import numpy as _np
        self._last_persseq_arrays: dict = {}
        try:
            if self._halt_depth_buf:
                self._last_persseq_arrays["halt_depth"] = (
                    _np.concatenate(self._halt_depth_buf).astype(_np.float32)
                )
            if self._halt_pmf_entropy_buf:
                self._last_persseq_arrays["halt_pmf_entropy"] = (
                    _np.concatenate(self._halt_pmf_entropy_buf).astype(_np.float32)
                )
            if self._cycle_cos_sim_buf:
                _cos = _np.concatenate(self._cycle_cos_sim_buf, axis=0).astype(_np.float32)
                _mean  = _np.zeros(_cos.shape[0], dtype=_np.float32)
                _last  = _np.zeros(_cos.shape[0], dtype=_np.float32)
                _minv  = _np.zeros(_cos.shape[0], dtype=_np.float32)
                _drift = _np.zeros(_cos.shape[0], dtype=_np.float32)
                for i in range(_cos.shape[0]):
                    _row = _cos[i]
                    _nz  = _row[_row != 0.0]
                    if _nz.size:
                        _mean[i] = _nz.mean()
                        _last[i] = _nz[-1]
                        _minv[i] = _nz.min()
                        if _nz.size >= 2:
                            _drift[i] = _nz[-1] - _nz[0]
                self._last_persseq_arrays["cycle_cos_sim_mean"]  = _mean
                self._last_persseq_arrays["cycle_cos_sim_last"]  = _last
                self._last_persseq_arrays["cycle_cos_sim_min"]   = _minv
                self._last_persseq_arrays["cycle_cos_sim_drift"] = _drift
            if self._cycle_argmax_flips_buf:
                self._last_persseq_arrays["cycle_argmax_flips"] = (
                    _np.concatenate(self._cycle_argmax_flips_buf).astype(_np.float32)
                )
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

    # ------------------------------------------------------------------
    # Core forward: returns (B, L, D) features for padded batches,
    #               or (1, total_tokens, D) for packed batches
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _run_forward(self, batch: Dict[str, Tensor]) -> Tensor:
        """Run the model forward pass(es) and return feature tensor.

        Padded mode (``use_packed_sequence=False``):
            Returns ``(B, L, D)`` — same shape as the padded input.

        Packed mode (``use_packed_sequence=True``):
            Returns ``(1, total_tokens, D)`` — all sequences concatenated.
            The caller can unpack using ``batch["cu_seqlens"]`` (N+1 entries).

        Dispatches to the correct carry-mode path and optionally hooks
        ``lm_head`` to capture pre-projection features for ``embed_mode="dec"``.
        """
        input_ids      = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        # ── packed-sequence kwargs ─────────────────────────────────────────
        # The eval collator produces cu_seqlens with N+1 entries [0…total].
        # The model's forward expects N+2 entries (training convention: the
        # last entry is a dummy boundary for the padding region).  We append
        # total_tokens as that dummy so numel() - 2 == N inside the model.
        _packed_kwargs: dict = {}
        if self.use_packed_sequence:
            _cu_n1 = batch.get("cu_seqlens")          # (N+1,) int32
            _pos   = batch.get("position_ids")         # (1, total_tokens) long
            _msql  = batch.get("max_seqlen")           # int
            if _cu_n1 is None:
                raise RuntimeError(
                    "use_packed_sequence=True but batch has no 'cu_seqlens'. "
                    "Make sure get_collate_fn() is called with pack_sequences=True."
                )
            _total = int(_cu_n1[-1].item())
            _cu_n2 = torch.cat([_cu_n1, _cu_n1.new_tensor([_total])])  # N+2
            _packed_kwargs = {
                "cu_seqlens":  _cu_n2,
                "position_ids": _pos,
                "max_seqlen":   _msql,
            }
            attention_mask = None   # flash_attn uses cu_seqlens, not mask

        # ── hook lm_head input for "dec" embed mode ───────────────────────
        _dec_feat: List[Optional[Tensor]] = [None]
        _hook_handle = None
        if self.embed_mode == "dec":
            def _capture(module, inp, out):
                # inp is a tuple; inp[0] is the Linear's input tensor (B, L, D)
                _dec_feat[0] = inp[0]
            _hook_handle = self.model.lm_head.register_forward_hook(_capture)

        # Packed mode requires half precision; wrap in bfloat16 autocast.
        _autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.use_packed_sequence
            else contextlib.nullcontext()
        )

        try:
            with _autocast_ctx:
                if self.carry_mode == "none":
                    # ── single pass, fixed h_cycles ───────────────────────
                    _fwd_kwargs: dict = {"return_carry": True, **_packed_kwargs}
                    if not self._is_srm:
                        _fwd_kwargs["h_cycles"] = self.h_cycles
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **_fwd_kwargs,
                    )
                    carry = getattr(output, "carry_out", None)

                elif self.carry_mode == "inference_act":
                    # ── chain eval_steps outer ACT passes with carry ───────
                    _h1_kwargs: dict = {"return_carry": True, **_packed_kwargs}
                    if not self._is_srm:
                        _h1_kwargs["h_cycles"] = 1
                    carry = None
                    for _ in range(self.eval_steps):
                        output = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            carry_in=carry,
                            **_h1_kwargs,
                        )
                        carry = getattr(output, "carry_out", None)

                elif self.carry_mode in ("ponder_act", "full_ponder_act"):
                    # ── PonderNet ACT runs inside forward() ───────────────
                    self.model.config.ponder_eval_decode = self.ponder_eval_decode
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_carry=True,
                        **_packed_kwargs,
                    )
                    carry = getattr(output, "carry_out", None)
                    # Accumulate PonderNet halt stats (see pop_halt_stats()).
                    # recAMPLIFY names this avg_cycles; the original called it avg_halt_step.
                    _ahs = getattr(output, "avg_halt_step", None)
                    if _ahs is None:
                        _ahs = getattr(output, "avg_cycles", None)
                    if _ahs is not None:
                        try:
                            self._halt_sum += float(_ahs.item() if hasattr(_ahs, "item") else _ahs)
                            self._halt_n += 1
                        except Exception:
                            pass
                    _pmf = getattr(output, "halt_pmf", None)
                    if _pmf is not None:
                        try:
                            _pmf_np = _pmf.detach().float().cpu().numpy()
                            self._halt_pmf_sum = (
                                _pmf_np.copy() if self._halt_pmf_sum is None
                                else self._halt_pmf_sum + _pmf_np
                            )
                        except Exception:
                            pass
                    _pc = getattr(output, "ponder_cost", None)
                    if _pc is not None:
                        try:
                            self._pc_sum += float(_pc.detach().float().mean().item())
                            self._pc_n += 1
                        except Exception:
                            pass
                    _pmf_pc = getattr(output, "halt_pmf_perchannel", None)
                    if _pmf_pc is not None:
                        try:
                            _arr = _pmf_pc.detach().float().cpu().numpy()
                            self._halt_pmf_perchannel_sum = (
                                _arr.copy() if self._halt_pmf_perchannel_sum is None
                                else self._halt_pmf_perchannel_sum + _arr
                            )
                        except Exception:
                            pass
                    _gt = getattr(output, "gate_openness_trajectory", None)
                    if _gt is not None:
                        try:
                            _arr = _gt.detach().float().cpu().numpy()
                            self._gate_traj_sum = (
                                _arr.copy() if self._gate_traj_sum is None
                                else self._gate_traj_sum + _arr
                            )
                        except Exception:
                            pass
                    _hdps = getattr(output, "halt_depth_per_seq", None)
                    if _hdps is not None:
                        try:
                            self._halt_depth_buf.append(_hdps.detach().float().cpu().numpy())
                        except Exception:
                            pass
                    _hent = getattr(output, "halt_pmf_entropy_per_seq", None)
                    if _hent is not None:
                        try:
                            self._halt_pmf_entropy_buf.append(_hent.detach().float().cpu().numpy())
                        except Exception:
                            pass
                    _czcs = getattr(output, "cycle_z_bos_cos_sim", None)
                    if _czcs is not None:
                        try:
                            self._cycle_cos_sim_buf.append(_czcs.detach().float().cpu().numpy())
                        except Exception:
                            pass
                    _caf = getattr(output, "cycle_argmax_flips_per_seq", None)
                    if _caf is not None:
                        try:
                            self._cycle_argmax_flips_buf.append(_caf.detach().float().cpu().numpy())
                        except Exception:
                            pass

                else:
                    raise ValueError(
                        f"Unknown carry_mode={self.carry_mode!r}. "
                        "Valid values: 'none', 'inference_act', 'ponder_act', 'full_ponder_act'."
                    )
        finally:
            if _hook_handle is not None:
                _hook_handle.remove()

        # ── select feature tensor ─────────────────────────────────────────
        if self.embed_mode == "z":
            if carry is None:
                raise NotImplementedError(
                    'embed_mode="z" needs the pre-exit recursive latent (carry_out.Z), which '
                    "recAMPLIFY does not return -- the refactor dropped the carry machinery. "
                    'Use embed_mode="dec" (the driver default, and what every published '
                    "recAMPLIFY/GMR number was produced with: the result directories are all "
                    'suffixed "_emdec"). It hooks lm_head and is strictly more informative, '
                    "being post-exit and post-norm."
                )
            feat = carry.Z  # (B, L, D) padded  |  (1, total_tokens, D) packed
        else:  # "dec"
            if _dec_feat[0] is None:
                raise RuntimeError(
                    "lm_head forward hook did not fire.  Ensure the model has a "
                    "'lm_head' attribute (AMPLIFYRecursive / AMPLIFYSRM both do)."
                )
            feat = _dec_feat[0]

        # Cast back to float32 if autocast produced bfloat16.
        if feat.dtype != torch.float32:
            feat = feat.float()
        return feat

    # ------------------------------------------------------------------
    # Pooling helpers
    # ------------------------------------------------------------------

    def _pool_features(
        self,
        features: Tensor,                   # (B, L, D)
        attention_mask: Optional[Tensor],   # (B, L)
        reduction_method: str = "mean",
    ) -> Tensor:
        """Pool (B, L, D) features → (B, D), masking pad tokens."""
        out: List[Tensor] = []
        for i in range(features.shape[0]):
            emb = self._select_valid_tokens(features[i], attention_mask, i)

            if self.normalize_embeddings:
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

            if reduction_method == "mean":
                out.append(residue_tokens(emb).mean(dim=0))
            elif reduction_method in ("first", "cls"):
                out.append(emb[0])
            elif reduction_method == "sum":
                out.append(residue_tokens(emb).sum(dim=0))
            else:
                out.append(residue_tokens(emb).mean(dim=0))

        return torch.stack(out)  # (B, D)

    def _pool_packed(
        self,
        features: Tensor,               # (1, total_tokens, D)
        batch: Dict[str, Tensor],
        reduction_method: str = "mean",
    ) -> Tensor:
        """Pool packed features (1, T, D) → (B, D) using cu_seqlens for boundaries."""
        cu       = batch["cu_seqlens"].long()   # (N+1,)
        flat     = features.squeeze(0)           # (T, D)
        B        = cu.numel() - 1
        out: List[Tensor] = []
        for i in range(B):
            emb = flat[cu[i]:cu[i + 1]]         # (seq_len_i, D)
            if self.normalize_embeddings:
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            if reduction_method == "mean":
                out.append(residue_tokens(emb).mean(dim=0))
            elif reduction_method in ("first", "cls"):
                out.append(emb[0])
            elif reduction_method == "sum":
                out.append(residue_tokens(emb).sum(dim=0))
            else:
                out.append(residue_tokens(emb).mean(dim=0))
        return torch.stack(out)  # (B, D)

    def _select_valid_tokens(
        self,
        token_features: Tensor,             # (L, D)
        attention_mask: Optional[Tensor],   # (B, L) or None
        seq_idx: int,
    ) -> Tensor:
        """Return (n_valid, D) slice for valid (non-pad) positions."""
        if attention_mask is None:
            return token_features

        row = attention_mask[seq_idx]
        if self.attention_mask_type == "additive":
            valid = row == 0        # additive: 0 → attend, negative → ignore
        elif self.attention_mask_type == "multiplicative":
            valid = row == 1
        else:                       # "bool"
            valid = row.bool()

        return token_features[valid]  # (n_valid, D)


# ---------------------------------------------------------------------------
# Sequence encoder (sequence-level tasks: classification, regression)
# ---------------------------------------------------------------------------

class TRMSequenceEncoder(SequenceEncoder, _TRMModelMixin):
    """TRM / SRM encoder for sequence-level downstream tasks.

    Returns mean-pooled (or CLS/first-token) embeddings of shape
    ``(batch_size, hidden_size)`` — the same contract as
    :class:`~apperitif.models.Amplify.AmplifySequenceEncoder`.
    """

    def __init__(self, config: TRMConfig, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._init_trm(config)

    def forward(self, batch: Dict[str, Tensor], reduction_method: str = "mean") -> Tensor:
        features = self._run_forward(batch)            # (B, L, D) | (1, T, D) packed
        if self.use_packed_sequence:
            return self._pool_packed(features, batch, reduction_method)  # (B, D)
        return self._pool_features(
            features,
            batch.get("attention_mask"),
            reduction_method=reduction_method,
        )                                               # (B, D)

    def get_collate_fn(
        self,
        sequence_column: str = "seq",
        label_column: str = "label",
        label_dtype: torch.dtype = torch.long,
        device=None,
        **kwargs,
    ):
        return DataCollatorForSeqTasks(
            tokenizer=self.tokenizer,
            pack_sequences=self.use_packed_sequence,
            device=device,
            sequence_column=sequence_column,
            max_length=self.max_length,
            label_column=label_column,
            label_dtype=label_dtype,
            token_level_task=False,
            attention_mask_type=self.attention_mask_type,
            attention_mask_dtype=self.attention_mask_dtype,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Token encoder (token-level tasks: secondary structure prediction)
# ---------------------------------------------------------------------------

class TRMTokenEncoder(TokenEncoder, _TRMModelMixin):
    """TRM / SRM encoder for token-level downstream tasks (e.g. SS3/SS8).

    Returns per-token embeddings with padding removed but special tokens
    (BOS/EOS) KEPT, matching the :class:`~apperitif.models.Amplify.AmplifyTokenEncoder`
    contract. The collator marks special-token label positions with -100 and the
    downstream classifier ignores them. Dropping special tokens here (while the
    labels still carry the -100 sentinels) shifts every residue's label and
    silently corrupts the offset-based alignment in get_token_embeddings_and_labels.
    """

    def __init__(self, config: TRMConfig, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._init_trm(config)

    def forward(
        self,
        batch: Dict[str, Tensor],
        reduction_method: str = None,
        return_individual_sequences: bool = False,
    ) -> Union[Tensor, List[Tensor]]:
        features = self._run_forward(batch)   # (B, L, D) | (1, T, D) packed

        embeddings: List[Tensor] = []

        if self.use_packed_sequence:
            # Unpack flat sequence by cu_seqlens. Special tokens (BOS/EOS) are KEPT;
            # their label positions carry -100 (set by the collator) and are ignored
            # by the token classifier. This matches AmplifyTokenEncoder and keeps the
            # offset-based label alignment in get_token_embeddings_and_labels valid.
            cu  = batch["cu_seqlens"].long()       # (N+1,)
            flat_feat = features.squeeze(0)             # (T, D)
            B = cu.numel() - 1
            for i in range(B):
                tok_feats = flat_feat[cu[i]:cu[i + 1]]
                if self.normalize_embeddings:
                    tok_feats = torch.nn.functional.normalize(tok_feats, p=2, dim=-1)
                embeddings.append(tok_feats)
        else:
            attention_mask = batch.get("attention_mask")
            for i in range(features.shape[0]):
                # Keep special tokens; only padding is dropped (see note above).
                tok_feats = self._select_valid_tokens(features[i], attention_mask, i)
                if self.normalize_embeddings:
                    tok_feats = torch.nn.functional.normalize(tok_feats, p=2, dim=-1)
                embeddings.append(tok_feats)

        if return_individual_sequences:
            return embeddings
        return torch.cat(embeddings, dim=0)   # (total_tokens, D)

    def get_collate_fn(
        self,
        sequence_column: str = "seq",
        label_column: str = "label",
        label_dtype: torch.dtype = torch.long,
        device=None,
        **kwargs,
    ):
        return DataCollatorForSeqTasks(
            tokenizer=self.tokenizer,
            pack_sequences=self.use_packed_sequence,
            device=device,
            sequence_column=sequence_column,
            max_length=self.max_length,
            label_column=label_column,
            label_dtype=label_dtype,
            token_level_task=True,
            attention_mask_type=self.attention_mask_type,
            attention_mask_dtype=self.attention_mask_dtype,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Structure encoder (contact map prediction)
# ---------------------------------------------------------------------------

class TRMStructureEncoder(StructureEncoder, _TRMModelMixin):
    """TRM / SRM encoder for structure-prediction tasks (contact maps).

    Returns per-residue embeddings as a nested tensor of shape
    ``(batch_size, [seq_len_i], hidden_size)`` — matching
    :class:`~apperitif.models.Amplify.AmplifyStructureEncoder`.
    """

    def __init__(self, config: TRMConfig, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._init_trm(config)

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        features = self._run_forward(batch)   # (B, L, D) | (1, T, D) packed

        per_seq: List[Tensor] = []
        if self.use_packed_sequence:
            cu       = batch["cu_seqlens"].long()   # (N+1,)
            flat_feat = features.squeeze(0)          # (T, D)
            B = cu.numel() - 1
            for i in range(B):
                emb = flat_feat[cu[i]:cu[i + 1]]
                if self.normalize_embeddings:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                per_seq.append(emb)
        else:
            attention_mask = batch.get("attention_mask")
            for i in range(features.shape[0]):
                emb = self._select_valid_tokens(features[i], attention_mask, i)
                if self.normalize_embeddings:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                per_seq.append(emb)

        return torch.nested.as_nested_tensor(per_seq)   # (B, [L_i], D)

    def get_collate_fn(
        self,
        sequence_column: str = "seq",
        label_column: str = "label",
        label_dtype: torch.dtype = torch.long,
        device=None,
        **kwargs,
    ):
        return DataCollatorForSeqTasks(
            tokenizer=self.tokenizer,
            pack_sequences=self.use_packed_sequence,
            device=device,
            sequence_column=sequence_column,
            max_length=self.max_length,
            label_column=label_column,
            label_dtype=label_dtype,
            token_level_task=False,
            attention_mask_type=self.attention_mask_type,
            attention_mask_dtype=self.attention_mask_dtype,
            structure_task=True,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _read_vocab(vocab_path: str) -> List[str]:
    """Read an AMPLIFY vocabulary file (one token per line)."""
    with open(vocab_path) as fh:
        return [line.strip() for line in fh if line.strip()]
