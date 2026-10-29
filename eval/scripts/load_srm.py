"""Shared helper: load a local recAMPLIFY checkpoint + tokenizer.

Usage
-----
    from load_srm import load_srm_checkpoint
    model, tokenizer, bos_offset = load_srm_checkpoint(ckpt_dir, device="cuda")

The model class is imported through ``apperitif._amplify`` so that
can be imported from any script inside ``AMPLIFY-evaluation/scripts/``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _ensure_amplify_on_path() -> None:
    """Retained as a no-op: the model is imported through ``apperitif._amplify``."""


def load_srm_checkpoint(
    checkpoint_dir: str,
    device: str = "cuda",
    l_cycles: int | None = None,
    init_perturb_std: float = 0.0,
    init_perturb_samples: int = 1,
    init_perturb_agg: str = "average",
    init_perturb_seed: int = 0,
    init_perturb_per_step: bool = True,
    ponder_eval_decode: str | None = None,
    verifier: str = "none",
    verifier_checkpoint_dir: str | None = None,
    verifier_autocast: str = "fp32",
):
    """Load a local HuggingFace-format SRM/TRM checkpoint.

    Parameters
    ----------
    checkpoint_dir : str
        Path to the checkpoint directory containing ``config.json``,
        ``model.safetensors`` (or ``pytorch_model.bin``), and tokenizer files.
    device : str
        Target device, e.g. ``"cuda"`` or ``"cpu"``.
    l_cycles : int | None
        If set, override ``model.config.L_cycles`` (and ``model.config.h_cycles``
        is left untouched). Vanilla AMPLIFY checkpoints ignore this parameter.
    init_perturb_std : float
        Eval-only initialization-perturbation noise std. When >0 together with
        ``init_perturb_samples > 1``, AMPLIFYSRM runs a stochastic ensemble
        (Gaussian noise on the recurrence-start z), drawing N samples and
        aggregating into one logit output. 0 disables (deterministic). Vanilla
        AMPLIFY / TRM ignore these.
    init_perturb_samples : int
        Number of perturbation samples N to ensemble (1 = deterministic).
    init_perturb_agg : str
        Aggregation mode: ``"average"`` (mean softmax) or ``"vote"`` (max-voting
        histogram of per-sample argmaxes). Output is ``log(p_bar)`` either way.
    init_perturb_seed : int
        Base seed for the per-sample Gaussian noise (sample s uses seed+s).

    Returns
    -------
    model : RecAMPLIFY
        Model in eval mode, moved to ``device``.
    tokenizer : ProteinTokenizer
        Corresponding tokenizer.
    bos_offset : int
        ``1`` when the tokenizer prepends a BOS/CLS token (standard), else ``0``.
    """
    _ensure_amplify_on_path()

    from apperitif._amplify import RecAMPLIFY  # noqa: F401
    from amplify.model.tokenizer import ProteinTokenizer  # noqa: F401

    cfg_file = os.path.join(checkpoint_dir, "config.json")
    if not os.path.isfile(cfg_file):
        raise FileNotFoundError(f"config.json not found in checkpoint dir: {checkpoint_dir}")
    with open(cfg_file) as fh:
        saved_cfg = json.load(fh)

    mtype = saved_cfg.get("model_type", "recamplify")
    # tie_word_embeddings=False: these checkpoints ship a separately-trained lm_head. The configs
    # omit the flag (HF default True), which re-ties lm_head to the embeddings and discards the
    # trained decoder -> uniform MLM predictions (DMS~0, KL blows up). Force untied at load.
    if mtype in ("recamplify", "amplify_srm"):
        model = RecAMPLIFY.from_pretrained(checkpoint_dir, tie_word_embeddings=False)
    else:
        # Vanilla AMPLIFY (model_type="amplify") — use AutoModelForMaskedLM so lm_head is present
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        model = AutoModelForMaskedLM.from_pretrained(checkpoint_dir, trust_remote_code=True, tie_word_embeddings=False)
        model.eval()
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
        probe_ids = tokenizer("A")["input_ids"]
        bos_offset = 1 if (len(probe_ids) > 1 and probe_ids[0] == tokenizer.cls_token_id) else 0
        return model, tokenizer, bos_offset

    if l_cycles is not None:
        model.set_num_cycles(l_cycles)

    # Eval-only stochastic init-perturbation knobs (read via getattr in the model
    # forward; harmless on TRM/AMPLIFY which never check them).
    model.config.init_perturb_std = float(init_perturb_std)
    model.config.init_perturb_samples = int(init_perturb_samples)
    model.config.init_perturb_agg = str(init_perturb_agg)
    model.config.init_perturb_seed = int(init_perturb_seed)
    model.config.init_perturb_per_step = bool(init_perturb_per_step)

    # Ponder_act eval decode strategy ("weighted" = Σ_n p_n·z_n, the default;
    # "greedy" = z at argmax-halt step). Read via getattr in the SRM forward; only
    # meaningful for use_ponder_act / use_full_ponder_act checkpoints. Leaving it None
    # preserves the model's built-in default ("weighted").
    if ponder_eval_decode is not None:
        model.config.ponder_eval_decode = str(ponder_eval_decode)
        if bool(getattr(model.config, "use_ponder_act", False)) or \
           bool(getattr(model.config, "use_full_ponder_act", False)):
            print(f"[ponder] eval decode = {model.config.ponder_eval_decode}")

    model.eval()
    model.to(device)

    if str(verifier).lower() in ("red", "verifier_red"):
        raise NotImplementedError("the RED best-of-N verifier is not part of this release")

    try:
        tokenizer = ProteinTokenizer.from_pretrained(checkpoint_dir)
    except Exception as exc:
        raise RuntimeError(
            f"ProteinTokenizer not found in {checkpoint_dir}. "
            "Ensure the checkpoint directory contains tokenizer files."
        ) from exc

    # Detect BOS offset: tokenize a single-char sequence and check for prepended special token.
    probe_ids = tokenizer("A")["input_ids"]
    bos_offset = 1 if (len(probe_ids) > 1 and probe_ids[0] == tokenizer.cls_token_id) else 0

    return model, tokenizer, bos_offset


def add_perturb_args(parser) -> None:
    """Register the shared eval-only init-perturbation CLI flags on an argparse parser.

    Used by the zero-shot scripts (DMS, KL, categorical Jacobian) so they share one
    flag spec. All default to the deterministic (no-perturbation) behaviour.
    """
    parser.add_argument(
        "--init-perturb-std", type=float, default=0.0, dest="init_perturb_std",
        help="Init-perturbation Gaussian noise std on the SRM recurrence-start z "
             "(eval-only stochastic inference). 0 = deterministic (default).",
    )
    parser.add_argument(
        "--init-perturb-samples", type=int, default=1, dest="init_perturb_samples",
        help="Number of perturbation samples N to ensemble (1 = deterministic).",
    )
    parser.add_argument(
        "--init-perturb-agg", type=str, default="average", dest="init_perturb_agg",
        choices=["average", "vote"],
        help="Aggregation across N samples: 'average' (mean softmax) or 'vote' "
             "(max-voting). Output logits = log(p_bar) either way.",
    )
    parser.add_argument(
        "--init-perturb-seed", type=int, default=0, dest="init_perturb_seed",
        help="Base seed for per-sample Gaussian noise (sample s uses seed+s).",
    )


def apply_perturb_config(model, args) -> None:
    """Set the eval-only init_perturb_* attrs on ``model.config`` from parsed args.

    Read via getattr inside AMPLIFYSRM.forward; harmless on TRM / vanilla AMPLIFY
    which never check them. No-op when the model has no ``config``.
    """
    if not hasattr(model, "config"):
        return
    model.config.init_perturb_std = float(getattr(args, "init_perturb_std", 0.0) or 0.0)
    model.config.init_perturb_samples = int(getattr(args, "init_perturb_samples", 1) or 1)
    model.config.init_perturb_agg = str(getattr(args, "init_perturb_agg", "average") or "average")
    model.config.init_perturb_seed = int(getattr(args, "init_perturb_seed", 0) or 0)
    if model.config.init_perturb_std > 0.0 and model.config.init_perturb_samples > 1:
        print(
            f"[init-perturb] std={model.config.init_perturb_std} "
            f"samples={model.config.init_perturb_samples} "
            f"agg={model.config.init_perturb_agg} seed={model.config.init_perturb_seed}"
        )
