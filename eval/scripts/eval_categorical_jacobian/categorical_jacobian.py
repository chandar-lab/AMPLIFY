import argparse
import contextlib
import json
import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM


def _no_flash_sdp_ctx():
    """Disable flash-attention SDPA backend so PyTorch falls back to math/mem-efficient.

    On H100, PyTorch's SDPA dispatcher can route FP32 inputs through flash-attention,
    which computes internally in BF16. For recursive models (SRM/TRM) this causes NaN
    after multiple cycles. Forcing non-flash backends keeps everything in FP32.
    """
    try:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=True, enable_math=True
        )
    except Exception:
        return contextlib.nullcontext()


# Module-level PonderNet accumulators (populated by get_logits when the
# underlying model exposes RecursiveMaskedLMOutput.avg_halt_step /
# halt_pmf / ponder_cost).  No-op for non-ponder ckpts.  The last two
# accumulators fire only for gated ckpts (ponder_gate_mode != "none").
_HALT_SUM: float = 0.0
_HALT_N: int = 0
_HALT_PMF_SUM = None  # np.ndarray (L_max,), lazy-init on first capture
_PC_SUM: float = 0.0
_PC_N: int = 0
_HALT_PMF_PERCHANNEL_SUM = None   # np.ndarray (L_max, C), lazy-init
_GATE_TRAJ_SUM = None             # np.ndarray (L_max, C), lazy-init
# Per-forward halt_depth_per_seq (B,) arrays.  The main loop snapshots
# len(_HALT_DEPTH_BUF) before/after processing each CASP16 sequence, so the
# slice for that sequence covers all L*20 mutant forwards → per-seq halt-depth
# stats (mean/std/quantiles across mutants) are attached to per_sequence[i].
_HALT_DEPTH_BUF: list = []
# Per-forward halt_pmf_entropy_per_seq (B,) and cycle_z_bos_cos_sim (B, L-1).
# Same slicing convention as _HALT_DEPTH_BUF.
_HALT_PMF_ENTROPY_BUF: list = []
_CYCLE_COS_SIM_BUF: list = []
# Per-forward cycle_argmax_flips_per_seq (B,).  Prediction-stability signal:
# per-sequence count of positions whose vocabulary argmax changed at least once
# across the recursion cycles.  Same slicing convention as _HALT_DEPTH_BUF.
_CYCLE_ARGMAX_FLIPS_BUF: list = []


def get_logits(model, input_ids):
    out = model(input_ids)
    # PonderNet halt-step / halt_pmf / ponder_cost capture (tensor output -> skip).
    if not isinstance(out, torch.Tensor):
        ahs = getattr(out, "avg_halt_step", None)
        if ahs is None:
            ahs = getattr(out, "avg_cycles", None)  # recAMPLIFY naming
        if ahs is not None:
            global _HALT_SUM, _HALT_N
            try:
                _HALT_SUM += float(ahs.item() if hasattr(ahs, "item") else ahs)
                _HALT_N += 1
            except Exception:
                pass
        _pmf = getattr(out, "halt_pmf", None)
        if _pmf is not None:
            global _HALT_PMF_SUM
            try:
                _pmf_np = _pmf.detach().float().cpu().numpy()
                _HALT_PMF_SUM = _pmf_np.copy() if _HALT_PMF_SUM is None else _HALT_PMF_SUM + _pmf_np
            except Exception:
                pass
        _pc = getattr(out, "ponder_cost", None)
        if _pc is not None:
            global _PC_SUM, _PC_N
            try:
                _PC_SUM += float(_pc.detach().float().mean().item())
                _PC_N += 1
            except Exception:
                pass
        _pmf_pc = getattr(out, "halt_pmf_perchannel", None)
        if _pmf_pc is not None:
            global _HALT_PMF_PERCHANNEL_SUM
            try:
                _arr = _pmf_pc.detach().float().cpu().numpy()
                _HALT_PMF_PERCHANNEL_SUM = _arr.copy() if _HALT_PMF_PERCHANNEL_SUM is None else _HALT_PMF_PERCHANNEL_SUM + _arr
            except Exception:
                pass
        _gt = getattr(out, "gate_openness_trajectory", None)
        if _gt is not None:
            global _GATE_TRAJ_SUM
            try:
                _arr = _gt.detach().float().cpu().numpy()
                _GATE_TRAJ_SUM = _arr.copy() if _GATE_TRAJ_SUM is None else _GATE_TRAJ_SUM + _arr
            except Exception:
                pass
        _hdps = getattr(out, "halt_depth_per_seq", None)
        if _hdps is not None:
            try:
                _HALT_DEPTH_BUF.append(_hdps.detach().float().cpu().numpy())
            except Exception:
                pass
        _hent = getattr(out, "halt_pmf_entropy_per_seq", None)
        if _hent is not None:
            try:
                _HALT_PMF_ENTROPY_BUF.append(_hent.detach().float().cpu().numpy())
            except Exception:
                pass
        _czcs = getattr(out, "cycle_z_bos_cos_sim", None)
        if _czcs is not None:
            try:
                _CYCLE_COS_SIM_BUF.append(_czcs.detach().float().cpu().numpy())
            except Exception:
                pass
        _caf = getattr(out, "cycle_argmax_flips_per_seq", None)
        if _caf is not None:
            try:
                _CYCLE_ARGMAX_FLIPS_BUF.append(_caf.detach().float().cpu().numpy())
            except Exception:
                pass
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits
    # AutoModel (no LM head in output): apply lm_head manually if available
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is not None and hasattr(model, "lm_head"):
        return model.lm_head(hidden)
    raise ValueError(f"Cannot extract logits from {type(out).__name__}. Use AutoModelForMaskedLM or a model with lm_head.")


def get_categorical_jacobian(model, tokenizer, sequence, batch_size=32, device="cuda"):
    """
    Categorical Jacobian J[i,a,j,b] = log P(x_i=a | x_j=b) - log P(x_i=a | wt)
    as defined in "Protein language models learn evolutionary statistics of
    interacting sequence motifs" (Sgarbossa et al.).

    Returns jacobian of shape (L, 20, L, 20).
    """
    model.eval()
    L = len(sequence)

    inputs = tokenizer(sequence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    if len(input_ids.shape) == 1:
        input_ids = input_ids.unsqueeze(0)  # [1, L+2] with cls/eos tokens

    standard_aa = "ARNDCQEGHILKMFPSTWYV"
    vocab_ids = torch.tensor(tokenizer.convert_tokens_to_ids(list(standard_aa)), device=device)
    num_aa = len(vocab_ids)  # 20

    # Determine offset: some tokenizers prepend a BOS token, others don't
    token_offset = input_ids.shape[1] - L  # total extra tokens; BOS adds 1 at start if present
    has_bos = (
        token_offset > 0
        and input_ids[0, 0].item() == tokenizer.cls_token_id
        or token_offset > 0
        and tokenizer.bos_token_id is not None
        and input_ids[0, 0].item() == tokenizer.bos_token_id
    )
    seq_offset = 1 if has_bos else 0

    with torch.no_grad(), _no_flash_sdp_ctx():
        wt_logits = get_logits(model, input_ids)[0, seq_offset : seq_offset + L, :]  # [L, V]
        wt_log_probs = F.log_softmax(wt_logits, dim=-1)[:, vocab_ids]  # [L, 20]

    # Build all L*20 mutant sequences
    batch_input_ids = input_ids.repeat(L * num_aa, 1)
    mutation_positions = torch.arange(seq_offset, seq_offset + L, device=device).repeat_interleave(num_aa)
    mutation_tokens = vocab_ids.repeat(L)
    row_indices = torch.arange(L * num_aa, device=device)
    batch_input_ids[row_indices, mutation_positions] = mutation_tokens

    jacobian = torch.zeros((L * num_aa, L, num_aa), device=device, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(batch_input_ids)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    offset = 0
    with torch.no_grad(), _no_flash_sdp_ctx():
        for (b_ids,) in tqdm(loader, desc=f"Jacobian L={L}", leave=False):
            b_logits = get_logits(model, b_ids)[:, seq_offset : seq_offset + L, :]  # [B, L, V]
            b_log_probs = F.log_softmax(b_logits, dim=-1)[:, :, vocab_ids]  # [B, L, 20]
            B = b_ids.shape[0]
            jacobian[offset : offset + B] = b_log_probs - wt_log_probs.unsqueeze(0)
            offset += B

    return jacobian.view(L, num_aa, L, num_aa)


def jacobian_to_contacts(jacobian):
    """
    Converts Jacobian (L, 20, L, 20) to a contact map (L, L) via:
    4-pass centering -> symmetrization -> Frobenius norm -> APC correction.
    """
    # 4-pass sequential centering (subtract mean over each dimension in order)
    j = jacobian
    for i in range(4):
        j = j - j.mean(i, keepdim=True)

    # Symmetrize
    j = (j + j.permute(2, 3, 0, 1)) / 2.0

    # Frobenius norm over amino acid axes
    contacts = torch.sqrt((j**2).sum(dim=(1, 3)))  # [L, L]

    # Zero diagonal before APC
    contacts.fill_diagonal_(0)

    # APC
    a1 = contacts.sum(dim=0, keepdim=True)
    a2 = contacts.sum(dim=1, keepdim=True)
    contacts = contacts - (a1 * a2) / contacts.sum()

    contacts.fill_diagonal_(0)
    return contacts


def pairs_to_matrix(pairs, L):
    """Converts a list of [i, j] contact pairs to a symmetric binary matrix."""
    mat = np.zeros((L, L), dtype=np.float32)
    for i, j in pairs:
        mat[i, j] = 1.0
        mat[j, i] = 1.0
    return mat


def compute_metrics(contact_map, true_contacts, L, minsep=24):
    """
    Computes P@L, P@L/2, P@L/5, ESM-style AUC (mean binned precision), and ROC-AUC
    on contacts with sequence separation |i-j| >= minsep, following the ESM implementation.
    The default minsep=24 is the CASP long-range convention (also Zhang et al., PNAS 2024).

    contact_map: (L, L) numpy array of predicted scores
    true_contacts: (L, L) binary numpy array (negative values treated as invalid)
    """
    predictions = torch.from_numpy(contact_map).unsqueeze(0)  # [1, L, L]
    targets = torch.from_numpy(true_contacts).unsqueeze(0)  # [1, L, L]

    seqlen_range = torch.arange(L)
    sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)  # [L, L]
    valid_mask = (sep >= minsep) & (targets[0] >= 0)  # [L, L]

    # Mask invalid positions to -inf so they sink in ranking
    predictions = predictions.masked_fill(~valid_mask.unsqueeze(0), float("-inf"))

    # Upper triangle with minsep offset (avoids double-counting and short-range pairs)
    x_ind, y_ind = np.triu_indices(L, minsep)
    scores = predictions[0, x_ind, y_ind].numpy()  # 1D
    labels = targets[0, x_ind, y_ind].numpy()  # 1D

    if labels.sum() == 0:
        return None

    # bf16 flash-attn on H100 can produce NaN logits for some sequences.
    # Replace non-finite scores with -inf so they rank below real predictions
    # in P@L (consistent with the masked_fill above), and exclude them from ROC-AUC.
    nonfinite = ~np.isfinite(scores)
    if nonfinite.any():
        print(f"  [warn] {int(nonfinite.sum())}/{len(scores)} non-finite scores; replacing with -inf")
        scores = np.where(nonfinite, -np.inf, scores)

    # ESM-style binned precision: top L*0.1, L*0.2, ..., L*1.0
    indices = np.argsort(scores)[::-1][:L]
    topk_targets = labels[indices]
    if len(topk_targets) < L:
        topk_targets = np.pad(topk_targets, (0, L - len(topk_targets)))
    cumulative_dist = topk_targets.cumsum()
    gather_indices = (np.arange(0.1, 1.1, 0.1) * L).astype(int) - 1
    gather_indices = np.clip(gather_indices, 0, len(cumulative_dist) - 1)
    binned_precisions = cumulative_dist[gather_indices] / (gather_indices + 1)

    finite = np.isfinite(scores)
    if finite.sum() >= 2 and 0 < labels[finite].sum() < finite.sum():
        roc_auc = float(roc_auc_score(labels[finite], scores[finite]))
    else:
        roc_auc = float("nan")

    metrics = {
        "P@L": float(binned_precisions[9]),
        "P@L/2": float(binned_precisions[4]),
        "P@L/5": float(binned_precisions[1]),
        "AUC": float(binned_precisions.mean()),
        "ROC-AUC": roc_auc,
    }
    return metrics


def load_model(model_name, device, hf_token=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    if "AMPLIFY" in model_name or "lolalb" in model_name.lower():
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    return model.to(device), tokenizer


def main():
    parser = argparse.ArgumentParser(description="Categorical Jacobian contact prediction on CASP16 dataset.")
    parser.add_argument("--model", default=None, help="HuggingFace hub model ID (mutually exclusive with --checkpoint-dir).")
    parser.add_argument("--checkpoint-dir", default=None, help="Local AMPLIFYSRM/TRM checkpoint dir (HF format). Takes priority over --model.")
    parser.add_argument("--split", default="casp16", choices=["train", "valid", "test", "casp16"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--output-dir", default=None,
                        help="Where to save per-sequence contact maps (.npy). Default: a casp16_contacts/ "
                             "folder next to --results, so metrics can be recomputed without the model.")
    parser.add_argument("--minsep", type=int, default=24,
                        help="Minimum sequence separation |i-j| for scored contacts (24 = CASP long-range).")
    parser.add_argument("--results", default="results.json", help="Path to save aggregated metrics JSON.")
    parser.add_argument("--token", default=None, help="HuggingFace access token.")
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
    args = parser.parse_args()

    if args.checkpoint_dir is None and args.model is None:
        parser.error("Provide either --checkpoint-dir (local SRM/TRM) or --model (HF hub ID).")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.checkpoint_dir is not None:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from load_srm import load_srm_checkpoint
        model, tokenizer, _bos_offset = load_srm_checkpoint(
            args.checkpoint_dir, device=device, l_cycles=args.l_cycles,
            init_perturb_std=getattr(args, "init_perturb_std", 0.0),
            init_perturb_samples=getattr(args, "init_perturb_samples", 1),
            init_perturb_agg=getattr(args, "init_perturb_agg", "average"),
            init_perturb_seed=getattr(args, "init_perturb_seed", 0),
            init_perturb_per_step=not getattr(args, "init_perturb_init_only", False),
            ponder_eval_decode=getattr(args, "ponder_eval_decode", None),
            verifier=getattr(args, "verifier", "none"),
            verifier_checkpoint_dir=getattr(args, "verifier_checkpoint_dir", None),
            verifier_autocast=getattr(args, "verifier_autocast", "fp32"),
        )
        model_id = args.checkpoint_dir
        if args.l_cycles is not None:
            print(f"[L_cycles override] model.config.L_cycles = {getattr(model.config, 'L_cycles', None)}")
    else:
        model, tokenizer = load_model(args.model, device, hf_token=args.token)
        model_id = args.model

    ds = load_dataset(
        "davidhd/contact_prediction_binary_casp16",
        split=args.split,
        token=args.token,
    )

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(args.results)), "casp16_contacts")
    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = []
    # Per-mutant sidecar bookkeeping — flat arrays keyed by (seq_idx, mutant_idx)
    # where mutant_idx = position * 20 + aa_idx (natural iteration order inside
    # get_categorical_jacobian).  Enables downstream join with contact-map
    # scores by seq.
    _permutant_seq_idx: list = []      # int32 per mutant row
    _permutant_mut_idx: list = []      # int32 per mutant row (0 .. L*20-1)

    for i, sample in enumerate(tqdm(ds, desc="Sequences")):
        seq = sample["seq"]
        L = len(seq)

        if L > args.max_len:
            print(f"Skipping sequence {i} (len={L} > {args.max_len})")
            continue

        # ── Snapshot global halt counters so we can attribute the delta to
        #    THIS sequence's L*20 mutant forwards (no per-batch attribution
        #    needed — all forwards run by get_categorical_jacobian for this
        #    seq share the same WT context).
        _seq_halt_sum0 = _HALT_SUM
        _seq_halt_n0   = _HALT_N
        _seq_pc_sum0   = _PC_SUM
        _seq_pc_n0     = _PC_N
        _seq_hdps_start = len(_HALT_DEPTH_BUF)

        jacobian = get_categorical_jacobian(model, tokenizer, seq, args.batch_size, device)
        contact_map = jacobian_to_contacts(jacobian).cpu().numpy()

        # Record per-mutant seq_idx + mutant_idx for the L*20 mutant forwards
        # this seq just produced (indexed by the delta on _HALT_DEPTH_BUF).
        _seq_delta_arrs = _HALT_DEPTH_BUF[_seq_hdps_start:]
        if _seq_delta_arrs:
            try:
                _n_new = sum(a.shape[0] for a in _seq_delta_arrs)
                _permutant_seq_idx.extend([i] * _n_new)
                _permutant_mut_idx.extend(range(_n_new))   # 0..L*20-1 in loader order
            except Exception:
                pass

        true_contacts = pairs_to_matrix(sample["label"], L)
        metrics = compute_metrics(contact_map, true_contacts, L, minsep=args.minsep)

        if metrics is not None:
            metrics["seq_idx"] = i
            metrics["length"] = L
            # AA-composition entropy — model-independent difficulty proxy.
            # Low-diversity sequences (long repeats, biased composition) tend
            # to give a pLM less information per position and are useful for
            # stratifying halt-depth vs performance correlations.
            try:
                _cnt = {aa: 0 for aa in "ARNDCQEGHILKMFPSTWYV"}
                for _aa in seq:
                    if _aa in _cnt:
                        _cnt[_aa] += 1
                _tot = sum(_cnt.values())
                if _tot > 0:
                    _p = np.array([v / _tot for v in _cnt.values() if v > 0])
                    metrics["aa_composition_entropy"] = round(float(-(_p * np.log(_p)).sum()), 4)
                    # Normalized to [0, 1] (max is log 20 ≈ 2.996 for uniform).
                    metrics["aa_composition_entropy_norm"] = round(float(-(_p * np.log(_p)).sum() / np.log(20)), 4)
            except Exception:
                pass
            # Per-sequence halt depth (E[n] averaged over this seq's mutant forwards).
            # None for non-ponder ckpts (no halt counter increments).
            _dn = _HALT_N - _seq_halt_n0
            if _dn > 0:
                metrics["mean_avg_halt_step"] = round((_HALT_SUM - _seq_halt_sum0) / _dn, 4)
                metrics["expected_halt_depth"] = metrics["mean_avg_halt_step"]  # alias for consistency with KL parquets
                metrics["n_forwards"] = _dn
            _dpc = _PC_N - _seq_pc_n0
            if _dpc > 0:
                metrics["mean_ponder_cost"] = round((_PC_SUM - _seq_pc_sum0) / _dpc, 6)
            # Per-sequence halt-depth distribution across this CASP16 seq's
            # L*20 mutant forwards.  Concatenate per-batch (B,) arrays for the
            # slice of forwards that belong to this seq.
            _seq_arrs = _HALT_DEPTH_BUF[_seq_hdps_start:]
            if _seq_arrs:
                try:
                    _hd = np.concatenate(_seq_arrs)
                    metrics["halt_depth_stats"] = {
                        "mean":   round(float(_hd.mean()),   4),
                        "std":    round(float(_hd.std()),    4),
                        "min":    round(float(_hd.min()),    4),
                        "q25":    round(float(np.quantile(_hd, 0.25)), 4),
                        "median": round(float(np.median(_hd)), 4),
                        "q75":    round(float(np.quantile(_hd, 0.75)), 4),
                        "max":    round(float(_hd.max()),    4),
                        "n_mutants": int(_hd.size),
                    }
                except Exception:
                    pass
            # Per-sequence halt PMF entropy H(p_k) across this seq's mutant forwards.
            _seq_ent_arrs = _HALT_PMF_ENTROPY_BUF[_seq_hdps_start:]
            if _seq_ent_arrs:
                try:
                    _he = np.concatenate(_seq_ent_arrs)
                    metrics["halt_pmf_entropy_stats"] = {
                        "mean": round(float(_he.mean()), 4),
                        "std":  round(float(_he.std()),  4),
                    }
                except Exception:
                    pass
            # Per-sequence cycle BOS cos-sim (across cycles) for this seq's mutants.
            # Shape per batch: (B, L-1) padded with zeros for cycles beyond halt.
            _seq_cos_arrs = _CYCLE_COS_SIM_BUF[_seq_hdps_start:]
            if _seq_cos_arrs:
                try:
                    _cs = np.concatenate(_seq_cos_arrs, axis=0)   # (N_mutants, L-1)
                    _nz = _cs != 0.0
                    if _nz.any():
                        _last_vals = [float(row[row != 0.0][-1]) for row in _cs if (row != 0.0).any()]
                        _min_vals  = [float(row[row != 0.0].min()) for row in _cs if (row != 0.0).any()]
                        _drifts    = [float(row[row != 0.0][-1] - row[row != 0.0][0])
                                      for row in _cs if (row != 0.0).sum() >= 2]
                        metrics["cycle_cos_sim_stats"] = {
                            "mean":       round(float(_cs[_nz].mean()),         4),
                            "last_mean":  round(float(np.mean(_last_vals)),      4),
                            "min_mean":   round(float(np.mean(_min_vals)),       4) if _min_vals else None,
                            "drift_mean": round(float(np.mean(_drifts)),         4) if _drifts   else None,
                        }
                except Exception:
                    pass
            all_metrics.append(metrics)

        if args.output_dir:
            np.save(f"{args.output_dir}/contacts_{i:04d}.npy", contact_map)

    # Aggregate
    keys = ["P@L", "P@L/2", "P@L/5", "AUC", "ROC-AUC"]
    summary = {k: float(np.nanmean([m[k] for m in all_metrics])) for k in keys}
    summary["n_sequences"] = len(all_metrics)

    print("\nResults:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    output = {
        "model": model_id,
        "split": args.split,
        "l_cycles": getattr(model.config, "L_cycles", None) if hasattr(model, "config") else None,
        "minsep": args.minsep,
        "summary": summary,
        "per_sequence": all_metrics,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)
    with open(args.results, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved results to {args.results}")

    # PonderNet halt-step sidecar (only if ponder ckpt).
    if _HALT_N > 0:
        mean = _HALT_SUM / _HALT_N
        L_max = getattr(model.config, "L_cycles", 0) if hasattr(model, "config") else 0
        halt_stats = {"mean_avg_halt_step": round(mean, 4), "n_forwards": _HALT_N, "L_max": int(L_max or 0)}
        if L_max:
            halt_stats["compute_efficiency"] = round(mean / L_max, 4)
            halt_stats["compute_savings_fraction"] = round(1.0 - mean / L_max, 4)
        if _PC_N > 0:
            halt_stats["mean_ponder_cost"] = round(_PC_SUM / _PC_N, 6)
        # Global per-forward halt-depth distribution (concatenated across all
        # CASP16 seqs).  Also dumped to a per-mutant NumPy sidecar for post-hoc
        # analysis — one row per masked-position mutant across all CASP16 seqs
        # (≈164K rows total: 41 seqs × avg L × 20 AAs).
        if _HALT_DEPTH_BUF:
            try:
                _hd_all = np.concatenate(_HALT_DEPTH_BUF).astype(np.float32)
                halt_stats["halt_depth_quantiles"] = {
                    "min":    round(float(_hd_all.min()),    4),
                    "q05":    round(float(np.quantile(_hd_all, 0.05)), 4),
                    "q25":    round(float(np.quantile(_hd_all, 0.25)), 4),
                    "median": round(float(np.median(_hd_all)), 4),
                    "q75":    round(float(np.quantile(_hd_all, 0.75)), 4),
                    "q95":    round(float(np.quantile(_hd_all, 0.95)), 4),
                    "max":    round(float(_hd_all.max()),    4),
                }
                halt_stats["halt_depth_std_across_seqs"] = round(float(_hd_all.std()), 4)
                halt_stats["halt_depth_n"] = int(_hd_all.size)
                # Per-mutant compact npz: join with entropy + cos-sim in the
                # same file so downstream loading is a single call.
                _ent_all = (np.concatenate(_HALT_PMF_ENTROPY_BUF).astype(np.float32)
                            if _HALT_PMF_ENTROPY_BUF else None)
                _cos_stack = (np.concatenate(_CYCLE_COS_SIM_BUF, axis=0).astype(np.float32)
                              if _CYCLE_COS_SIM_BUF else None)
                # Derive per-mutant cos-sim summary stats (mean/last/min/drift)
                # from the (N, L-1) matrix, zero-padded.
                if _cos_stack is not None:
                    _cos_mean  = np.zeros(_cos_stack.shape[0], dtype=np.float32)
                    _cos_last  = np.zeros(_cos_stack.shape[0], dtype=np.float32)
                    _cos_min   = np.zeros(_cos_stack.shape[0], dtype=np.float32)
                    _cos_drift = np.zeros(_cos_stack.shape[0], dtype=np.float32)
                    for i in range(_cos_stack.shape[0]):
                        _row = _cos_stack[i]
                        _nz = _row[_row != 0.0]
                        if _nz.size:
                            _cos_mean[i]  = _nz.mean()
                            _cos_last[i]  = _nz[-1]
                            _cos_min[i]   = _nz.min()
                            if _nz.size >= 2:
                                _cos_drift[i] = _nz[-1] - _nz[0]
                _npz_sidecar = os.path.join(os.path.dirname(os.path.abspath(args.results)),
                                            "casp16_halt_permutant.npz")
                _payload = {
                    "seq_idx":    np.asarray(_permutant_seq_idx, dtype=np.int32),
                    "mutant_idx": np.asarray(_permutant_mut_idx, dtype=np.int32),
                    "halt_depth": _hd_all,
                }
                if _ent_all is not None:
                    _payload["halt_pmf_entropy"] = _ent_all
                if _cos_stack is not None:
                    _payload["cycle_cos_sim_mean"]  = _cos_mean
                    _payload["cycle_cos_sim_last"]  = _cos_last
                    _payload["cycle_cos_sim_min"]   = _cos_min
                    _payload["cycle_cos_sim_drift"] = _cos_drift
                # Analysis F: per-mutant prediction-stability (argmax-flip count).
                _caf_all = (np.concatenate(_CYCLE_ARGMAX_FLIPS_BUF).astype(np.float32)
                            if _CYCLE_ARGMAX_FLIPS_BUF else None)
                if _caf_all is not None and _caf_all.size == _hd_all.size:
                    _payload["cycle_argmax_flips"] = _caf_all
                np.savez_compressed(_npz_sidecar, **_payload)
                print(f"Wrote per-mutant halt sidecar ({_hd_all.size} rows, {len(_payload)} arrays) -> {_npz_sidecar}")
            except Exception as _e:
                print(f"[WARN] per-mutant halt sidecar failed: {_e}")
        # Global halt_pmf_entropy_per_seq aggregates.
        if _HALT_PMF_ENTROPY_BUF:
            try:
                _ent_all = np.concatenate(_HALT_PMF_ENTROPY_BUF)
                halt_stats["halt_pmf_entropy_per_seq_mean"] = round(float(_ent_all.mean()), 4)
                halt_stats["halt_pmf_entropy_per_seq_std"]  = round(float(_ent_all.std()),  4)
            except Exception:
                pass
        # Global cycle BOS cos-sim aggregates.
        if _CYCLE_COS_SIM_BUF:
            try:
                _cos_all = np.concatenate(_CYCLE_COS_SIM_BUF, axis=0)
                _nz = _cos_all != 0.0
                if _nz.any():
                    halt_stats["cycle_cos_sim_global_mean"] = round(float(_cos_all[_nz].mean()), 4)
                    _last_vals = [float(row[row != 0.0][-1]) for row in _cos_all if (row != 0.0).any()]
                    if _last_vals:
                        halt_stats["cycle_cos_sim_last_mean"] = round(float(np.mean(_last_vals)), 4)
            except Exception:
                pass
        if _HALT_PMF_SUM is not None:
            _pmf_mean = _HALT_PMF_SUM / max(_HALT_N, 1)
            _s = float(_pmf_mean.sum())
            if _s > 0:
                _pmf_mean = _pmf_mean / _s
            halt_stats["halt_pmf"] = [round(float(p), 6) for p in _pmf_mean.tolist()]
            _depths = np.arange(1, len(_pmf_mean) + 1, dtype=float)
            _e_n = float((_pmf_mean * _depths).sum())
            _var = float((_pmf_mean * (_depths - _e_n) ** 2).sum())
            _ent = float(-(_pmf_mean * np.log(np.clip(_pmf_mean, 1e-12, 1.0))).sum())
            halt_stats["halt_pmf_entropy"] = round(_ent, 4)
            halt_stats["halt_pmf_std"] = round(_var ** 0.5, 4)
            halt_stats["halt_pmf_mode"] = int(np.argmax(_pmf_mean)) + 1
        # Gated-branch per-channel summary (fires only when model emitted the
        # per-channel fields, i.e. ponder_gate_mode != "none").
        if _HALT_PMF_PERCHANNEL_SUM is not None:
            _pmf_pc_mean = _HALT_PMF_PERCHANNEL_SUM / max(_HALT_N, 1)
            _L_max_pc, _C_pc = _pmf_pc_mean.shape
            _col_sums = _pmf_pc_mean.sum(axis=0, keepdims=True)
            _col_sums = np.where(_col_sums > 0, _col_sums, 1.0)
            _pmf_pc_norm = _pmf_pc_mean / _col_sums
            _depths_pc   = np.arange(1, _L_max_pc + 1, dtype=float)
            _e_n_pc      = (_pmf_pc_norm * _depths_pc[:, None]).sum(axis=0)
            _gm = str(getattr(model.config, "ponder_gate_mode", "none")) if hasattr(model, "config") else "none"
            halt_stats["gate_mode"] = _gm
            halt_stats["n_channels"] = int(_C_pc)
            halt_stats["per_channel_E_n_mean"] = round(float(_e_n_pc.mean()), 4)
            halt_stats["per_channel_E_n_std"]  = round(float(_e_n_pc.std()), 4)
            _qs = np.quantile(_e_n_pc, [0.0, 0.25, 0.5, 0.75, 1.0])
            halt_stats["per_channel_E_n_quantiles"] = {
                "min": round(float(_qs[0]),4), "q25": round(float(_qs[1]),4),
                "median": round(float(_qs[2]),4), "q75": round(float(_qs[3]),4),
                "max": round(float(_qs[4]),4),
            }
            if _GATE_TRAJ_SUM is not None:
                _gt_mean = _GATE_TRAJ_SUM / max(_HALT_N, 1)
                halt_stats["gate_openness_mean_per_cycle"] = [round(float(v), 4) for v in _gt_mean.mean(axis=1).tolist()]
                halt_stats["gate_openness_std_per_cycle"]  = [round(float(v), 4) for v in _gt_mean.std(axis=1).tolist()]
            if _C_pc <= 64:
                halt_stats["halt_pmf_perchannel"] = [[round(float(v), 6) for v in row] for row in _pmf_pc_mean.tolist()]
                if _GATE_TRAJ_SUM is not None:
                    halt_stats["gate_openness_trajectory"] = [[round(float(v), 6) for v in row] for row in _gt_mean.tolist()]
        sidecar = os.path.join(os.path.dirname(os.path.abspath(args.results)), "halt_stats.json")
        with open(sidecar, "w") as f:
            json.dump(halt_stats, f, indent=2)
        print(f"Wrote PonderNet halt stats -> {sidecar}: {halt_stats}")


if __name__ == "__main__":
    main()
