"""
Evaluate model logits against homolog AA frequency profiles via a calibration score.

For each consensus position (AA frequency >= threshold in homologs), the WT residue
is masked and the model predicts a distribution over the 20 standard AAs. We report:

  kl          = D_KL(homolog_freq || model_softmax)   # the paper's direction: MSA profile first
  calibration = 100 * exp(-kl)                         # in [0, 100]; 100 = perfect agreement

This matches the paper, appendix "Zero-shot tasks" item A:
    cal_i = 100 * exp(-KL(f_i || p_theta)),  f_i = MSA consensus profile.
Earlier versions of this script computed the OPPOSITE direction, D_KL(model || homolog).
That is retained per-position as `kl_forward` (and aggregated as calibration_forward_*)
so corrected numbers can be compared against previously published ones, but it is not the
headline metric. The two directions are not a monotone relabelling: rankings can change.

Calibration is computed PER POSITION, then averaged (the faithful summary is the
mean of per-position calibration, not 100*exp(-mean_kl), since exp is nonlinear).

Aggregates are also split by whether the WT residue is the consensus AA:
  calibration_T  : WT == consensus (target positions)
  calibration_NT : WT != consensus (non-target positions)

Output parquet columns per row (one per sequence×position):
  sequence_name, position, wt_aa, consensus_aa, pred_aa, kl, kl_forward, calibration
"""

import os
import re
import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from torch.utils.data import DataLoader

AA_COLS = list("ACDEFGHIKLMNPQRSTVWY")
TRUNCATION_LEN = 512
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def build_dataset(proteome_path: str, consensus_path: str) -> Dataset:
    """
    Build a flat dataset of (masked_sequence, metadata) rows, one per consensus position.
    Each sequence is truncated to TRUNCATION_LEN with the masked position centred.
    """
    proteome = pd.read_csv(proteome_path).set_index("id")
    consensus = pd.read_csv(consensus_path)  # sequence_name, position, wt_aa, A..Y

    rows = []
    missing = 0
    skipped_non_standard = 0
    for _, cons_row in consensus.iterrows():
        seq_name = cons_row["sequence_name"]
        pos = int(cons_row["position"])

        if seq_name not in proteome.index:
            missing += 1
            continue

        full_seq = proteome.loc[seq_name, "sequence"]
        seq_len = len(full_seq)

        if pos >= seq_len:
            continue

        if cons_row["wt_aa"] not in AA_COLS:
            skipped_non_standard += 1
            continue

        # Determine the consensus AA (highest frequency in profile)
        freqs = cons_row[AA_COLS].values.astype(float)
        consensus_aa = AA_COLS[int(np.argmax(freqs))]

        # Truncate window centred on pos
        if seq_len > TRUNCATION_LEN:
            half = TRUNCATION_LEN // 2
            start = max(0, pos - half)
            end = start + TRUNCATION_LEN
            if end > seq_len:
                end = seq_len
                start = max(0, end - TRUNCATION_LEN)

            local_pos = pos - start  # position within the truncated window

        else:
            start = 0
            end = seq_len
            local_pos = pos

        rows.append(
            {
                "sequence": full_seq[start:end],
                "id": seq_name,
                "position": pos,
                "local_position": local_pos,
                "wt_aa": cons_row["wt_aa"],
                "consensus_aa": consensus_aa,
                **{aa: float(cons_row[aa]) for aa in AA_COLS},
            }
        )

    if missing:
        logger.warning(f"{missing} sequence names from consensus file not found in proteome.")
    if skipped_non_standard:
        logger.info(f"Skipped {skipped_non_standard} consensus positions with non-standard wt_aa.")

    logger.info(f"Built dataset with {len(rows)} consensus positions.")
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# KL computation helpers
# ---------------------------------------------------------------------------


def get_aa_token_ids(tokenizer) -> list[int]:
    """Return token IDs for the 20 standard AAs, in AA_COLS order."""
    ids = tokenizer.convert_tokens_to_ids(AA_COLS)
    assert all(i != tokenizer.unk_token_id for i in ids), "Some AAs mapped to <unk> — check vocab."
    return ids


def softmax_over_aa(logits_np: np.ndarray, aa_token_ids: list[int]) -> np.ndarray:
    """Extract logits for 20 AAs and apply softmax (re-normalised over 20 AAs only)."""
    aa_logits = logits_np[aa_token_ids]
    aa_logits = aa_logits - aa_logits.max()  # numerical stability
    exp = np.exp(aa_logits)
    return exp / exp.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Generic D_KL(p || q) = sum_i p_i log(p_i / q_i).

    Only ``p`` is clipped, to make the 0*log 0 terms vanish. ``q`` is left alone: in the
    direction the paper defines, q is the model softmax and so is strictly positive, and
    clipping it would silently bound the divergence. (The reverse direction DOES need a
    clip on q, because the homolog profile has structural zeros -- which is exactly why
    the old forward-KL numbers were dominated by the clip constant: a model placing mass
    on a zero-frequency amino acid contributed up to -log(1e-10) ~ 23 nats.)
    """
    p = np.clip(p, eps, 1.0)
    return float(np.sum(p * np.log(p / np.clip(q, eps, 1.0))))


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


def evaluate(args):
    os.makedirs(args.output_path, exist_ok=True)

    # ---- Force-recompute: clear any cached predictions before resumability check ----
    # Without this, the resumability logic below silently reuses cached parquet files
    # from a previous run — which is wrong whenever the model config (e.g. L_cycles)
    # changes between runs targeting the same output dir.
    if getattr(args, "force_recompute", False):
        parquet_pattern_force = re.compile(r"data_(\d+)\.parquet")
        n_removed = 0
        for fname in os.listdir(args.output_path):
            if parquet_pattern_force.match(fname):
                os.remove(os.path.join(args.output_path, fname))
                n_removed += 1
        scores_file = os.path.join(args.output_path, "scores.json")
        if os.path.isfile(scores_file):
            os.remove(scores_file)
        logger.info(f"[--force-recompute] removed {n_removed} cached data_*.parquet files and scores.json")

    # ---- Resumability: find last saved parquet ----
    parquet_pattern = re.compile(r"data_(\d+)\.parquet")
    existing = [int(m.group(1)) for f in os.listdir(args.output_path) if (m := parquet_pattern.match(f))]
    start_idx = (max(existing) + 1) if existing else 0
    if start_idx > 0:
        logger.info(f"Resuming from batch {start_idx}")

    if not args.skip_predictions:
        # ---- Load model ----
        # Priority: local HF checkpoint dir > HF hub model name > legacy AMPLIFY.load
        _checkpoint_dir = getattr(args, "checkpoint_dir", None)
        from_local_hf = _checkpoint_dir is not None and os.path.isdir(_checkpoint_dir)
        from_hub = not from_local_hf and (
            "esm" in args.model_name
            or "AMPLIFY" in args.model_name
            or args.model_name.lower().startswith("lolalb/")
            or args.model_name.lower().startswith("flair-bio/")
            or args.model_name.startswith("chandar-lab/")
            or args.model_name.startswith("facebook/")
        )

        if from_local_hf:
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from load_srm import load_srm_checkpoint
            model, tokenizer, _bos_offset = load_srm_checkpoint(
                _checkpoint_dir, l_cycles=getattr(args, "l_cycles", None),
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
            if getattr(args, "l_cycles", None) is not None:
                print(f"[L_cycles override] model.config.L_cycles = {getattr(model.config, 'L_cycles', None)}")
        elif from_hub:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, token=args.token)
            if "AMPLIFY" in args.model_name or args.model_name.lower().startswith("lolalb/"):
                model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True, token=args.token)
            else:
                model = AutoModelForMaskedLM.from_pretrained(args.model_name, trust_remote_code=True, token=args.token)
        else:
            from amplify.model import AMPLIFY  # legacy non-HF format
            model, tokenizer = AMPLIFY.load(
                checkpoint_path=args.checkpoint_path,
                config_path=args.config_path,
                vocab_path=args.vocab_path,
                tag=args.tag,
            )
            if args.tag is not None:
                args.model_name = args.model_name + "_" + args.tag

        aa_token_ids = get_aa_token_ids(tokenizer)

        probe_ids = tokenizer("A", add_special_tokens=True)["input_ids"]
        # ESM-2 tokenizers set cls_token_id (0) and leave bos_token_id None, so a
        # bos-only probe silently returns offset 0 and every logit is read one
        # position early. Accept either leading special token.
        _lead_special = {
            t for t in (getattr(tokenizer, "bos_token_id", None), getattr(tokenizer, "cls_token_id", None))
            if t is not None
        }
        bos_offset = 1 if (len(probe_ids) > 1 and probe_ids[0] in _lead_special) else 0
        n_special = len(probe_ids) - 1
        logger.info(f"Tokenizer probe: bos_offset={bos_offset}, n_special_tokens={n_special}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        # ---- Build or load dataset ----
        if args.dataset_path and os.path.isdir(args.dataset_path):
            logger.info(f"Loading pre-built dataset from {args.dataset_path}")
            dataset = Dataset.load_from_disk(args.dataset_path)
        else:
            logger.info("Building dataset from proteome and consensus files...")
            dataset = build_dataset(args.proteome_path, args.consensus_path)
            if args.dataset_path:
                dataset.save_to_disk(args.dataset_path)
                logger.info(f"Saved dataset to {args.dataset_path}")

        max_length = TRUNCATION_LEN + n_special
        mask_token_id = tokenizer.mask_token_id
        assert mask_token_id is not None, "Tokenizer has no mask_token_id."

        def tokenize(batch):
            enc = tokenizer(batch["sequence"], truncation=True, max_length=max_length, padding=False, return_tensors=None)
            mask_token_indices = []
            for i, (seq, lp) in enumerate(zip(batch["sequence"], batch["local_position"])):
                target_pid = bos_offset + lp
                if "position_ids" in enc:
                    pids = enc["position_ids"][i]
                    matches = [j for j, p in enumerate(pids) if p == target_pid]
                    assert len(matches) == 1, (
                        f"Expected exactly one token with position_id={target_pid}, got {matches} "
                        f"(seq len {len(seq)}, lp {lp}, pids={pids})"
                    )
                    mask_idx = matches[0]
                else:
                    mask_idx = target_pid
                    assert mask_idx < len(enc["input_ids"][i]) - (n_special - bos_offset), (
                        f"mask_idx {mask_idx} out of range for tokenized length {len(enc['input_ids'][i])} "
                        f"(seq len {len(seq)}, lp {lp}, n_special {n_special})"
                    )
                enc["input_ids"][i][mask_idx] = mask_token_id
                mask_token_indices.append(mask_idx)
            enc["id"] = batch["id"]
            enc["position"] = batch["position"]
            enc["mask_token_index"] = mask_token_indices
            enc["wt_aa"] = batch["wt_aa"]
            enc["consensus_aa"] = batch["consensus_aa"]
            for aa in AA_COLS:
                enc[aa] = batch[aa]
            return enc

        dataset.set_transform(tokenize)

        input_ids_pad = tokenizer.pad_token_id

        def collate_fn(batch):
            result = {}
            for key in batch[0]:
                vals = [b[key] for b in batch]
                if key == "input_ids":
                    tensors = [torch.tensor(v) for v in vals]
                    result[key] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=input_ids_pad)
                elif key == "attention_mask":
                    tensors = [torch.tensor(v) for v in vals]
                    result[key] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)
                elif key == "position_ids":
                    tensors = [torch.tensor(v) for v in vals]
                    result[key] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)
                elif isinstance(vals[0], (int, float)):
                    result[key] = torch.tensor(vals)
                else:
                    result[key] = vals
            return result

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

        total_batches = len(dataloader)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        buffer = []
        # PonderNet halt-step + halt_pmf + ponder_cost accumulators
        # (silently no-ops on non-ponder ckpts).  Gated-branch adds two extra
        # per-channel accumulators that stay None unless gate_mode != "none".
        _halt_sum, _halt_n = 0.0, 0
        _halt_pmf_sum = None      # np.ndarray (L_max,), lazy-init on first capture
        _pc_sum, _pc_n = 0.0, 0
        _halt_pmf_perchannel_sum = None   # np.ndarray (L_max, C), lazy-init
        _gate_traj_sum = None             # np.ndarray (L_max, C), lazy-init
        _L_max = int(getattr(getattr(model, "config", None), "L_cycles", 0)) or None
        _gate_mode = str(getattr(getattr(model, "config", None), "ponder_gate_mode", "none"))

        with tqdm(desc=f"KL consensus eval — {args.model_name}", total=total_batches) as pbar:
            for idx, batch in enumerate(dataloader):
                if idx < start_idx:
                    pbar.update(1)
                    continue

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                position_ids = batch.get("position_ids", None)
                if position_ids is not None:
                    position_ids = position_ids.to(device)
                mask_token_indices = batch["mask_token_index"]

                with torch.no_grad():
                    try:
                        if position_ids is not None:
                            output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
                        else:
                            output = model(input_ids=input_ids, attention_mask=attention_mask)
                    except TypeError:
                        output = model(src=input_ids, pad_mask=attention_mask.bool())

                # PonderNet halt-step / halt_pmf / ponder_cost capture
                # (None for non-ponder modes — silently ignored).
                _ahs = getattr(output, "avg_halt_step", None)
                _hdps = getattr(output, "halt_depth_per_seq", None)  # (B,) per-seq E[n]
                _hdps_cpu = (
                    _hdps.detach().float().cpu().numpy() if _hdps is not None else None
                )
                # Analysis C: per-sequence PMF entropy H(p_k), shape (B,)
                _hent = getattr(output, "halt_pmf_entropy_per_seq", None)
                _hent_cpu = (
                    _hent.detach().float().cpu().numpy() if _hent is not None else None
                )
                # Analysis A: per-cycle BOS cosine similarity, shape (B, L-1)
                _czcs = getattr(output, "cycle_z_bos_cos_sim", None)
                _czcs_cpu = (
                    _czcs.detach().float().cpu().numpy() if _czcs is not None else None
                )
                # Analysis F (prediction stability): per-sequence argmax-flip count
                # across recursion cycles, shape (B,).  Sum over all positions in the
                # sequence of positions whose argmax changed at least once.
                _caf = getattr(output, "cycle_argmax_flips_per_seq", None)
                _caf_cpu = (
                    _caf.detach().float().cpu().numpy() if _caf is not None else None
                )
                if _ahs is not None:
                    try:
                        _halt_sum += float(_ahs.item() if hasattr(_ahs, "item") else _ahs)
                        _halt_n += 1
                    except Exception:
                        pass
                _pmf = getattr(output, "halt_pmf", None)
                if _pmf is not None:
                    try:
                        _pmf_np = _pmf.detach().float().cpu().numpy()
                        _halt_pmf_sum = _pmf_np.copy() if _halt_pmf_sum is None else _halt_pmf_sum + _pmf_np
                    except Exception:
                        pass
                _pc = getattr(output, "ponder_cost", None)
                if _pc is not None:
                    try:
                        _pc_sum += float(_pc.detach().float().mean().item())
                        _pc_n += 1
                    except Exception:
                        pass
                _pmf_pc = getattr(output, "halt_pmf_perchannel", None)
                if _pmf_pc is not None:
                    try:
                        _arr = _pmf_pc.detach().float().cpu().numpy()
                        _halt_pmf_perchannel_sum = _arr.copy() if _halt_pmf_perchannel_sum is None else _halt_pmf_perchannel_sum + _arr
                    except Exception:
                        pass
                _gt = getattr(output, "gate_openness_trajectory", None)
                if _gt is not None:
                    try:
                        _arr = _gt.detach().float().cpu().numpy()
                        _gate_traj_sum = _arr.copy() if _gate_traj_sum is None else _gate_traj_sum + _arr
                    except Exception:
                        pass

                if hasattr(output, "logits") and output.logits is not None:
                    logits = output.logits
                elif hasattr(output, "last_hidden_state") and hasattr(model, "lm_head"):
                    logits = model.lm_head(output.last_hidden_state)
                elif "logits" in output:
                    logits = output["logits"]
                else:
                    raise ValueError(f"Cannot extract logits from model output of type {type(output)}")
                # [B, seq_len, vocab]

                for i in range(logits.size(0)):
                    mask_idx = int(mask_token_indices[i].item())
                    logit_vec = logits[i, mask_idx].float().cpu().numpy()

                    model_probs = softmax_over_aa(logit_vec, aa_token_ids)
                    pred_aa = AA_COLS[int(np.argmax(model_probs))]
                    # Sharpening-control fields: how peaked is the model's per-position
                    # distribution?  pred_entropy = H(model_probs); pred_max_prob = top-1.
                    # Needed to decompose whether halt_depth ↔ KL is a genuine confidence
                    # signal or just an artifact of "more compute → sharper softmax".
                    _mp = model_probs.clip(1e-12, 1.0)
                    pred_entropy  = float(-(_mp * np.log(_mp)).sum())
                    pred_max_prob = float(model_probs.max())

                    wt_aa = batch["wt_aa"][i]
                    consensus_aa = batch["consensus_aa"][i]

                    homolog_freq = np.array([float(batch[aa][i].item()) for aa in AA_COLS])
                    homolog_freq = homolog_freq / homolog_freq.sum()

                    # The paper's definition (appendix, "Zero-shot tasks" item A):
                    #     cal_i = 100 * exp(-KL(f_i || p_theta))
                    # i.e. the MSA consensus profile f_i is the FIRST argument. This is the
                    # headline calibration metric.
                    kl = kl_divergence(homolog_freq, model_probs)
                    calibration = 100.0 * float(np.exp(-kl))
                    # The opposite direction, KL(p_theta || f_i), is what this script used to
                    # report. Kept so corrected and historical numbers can be compared from a
                    # single inference pass; it is not aggregated into the headline score.
                    kl_forward = kl_divergence(model_probs, homolog_freq)

                    # Analysis B: profile entropy and conservation score (no model output needed).
                    # Profile entropy H_profile = −Σ f_i log f_i measures how diverse the
                    # evolutionary consensus is at this position.  Low H = highly conserved
                    # (few homologs agree on one AA); high H = variable position.
                    # profile_max_freq = max(f_i) = conservation score ∈ (0,1].
                    _eps_prof = 1e-10
                    _prof = homolog_freq.clip(_eps_prof, 1.0)
                    profile_entropy  = float(-(_prof * np.log(_prof)).sum())
                    profile_max_freq = float(homolog_freq.max())

                    row = {
                        "sequence_name": batch["id"][i],
                        "position": int(batch["position"][i].item()),
                        "wt_aa": wt_aa,
                        "consensus_aa": consensus_aa,
                        "pred_aa": pred_aa,
                        "kl": kl,
                        "kl_forward": kl_forward,
                        "calibration": calibration,
                        # Sharpening control (Analysis F): model self-confidence per position.
                        "pred_entropy":  pred_entropy,
                        "pred_max_prob": pred_max_prob,
                        # Analysis B: evolutionary profile statistics (independent of model)
                        "profile_entropy":  profile_entropy,
                        "profile_max_freq": profile_max_freq,
                    }
                    # Analysis C: per-sequence halt PMF entropy H(p_k)
                    if _hent_cpu is not None and i < len(_hent_cpu):
                        row["halt_pmf_entropy"] = float(_hent_cpu[i])
                    # Analysis A (E[n]) — existing field kept for backward compatibility
                    if _hdps_cpu is not None and i < len(_hdps_cpu):
                        row["expected_halt_depth"] = float(_hdps_cpu[i])
                    # Analysis A (cycle consistency): per-cycle BOS cos-sim stats
                    # mean over all L-1 pairs; last-pair cos-sim (late-cycle stability).
                    if _czcs_cpu is not None and i < _czcs_cpu.shape[0]:
                        _cs_row = _czcs_cpu[i]           # (L-1,) padded with zeros
                        _nonzero = _cs_row[_cs_row != 0.0]
                        row["cycle_cos_sim_mean"] = float(_nonzero.mean()) if len(_nonzero) else float("nan")
                        row["cycle_cos_sim_last"] = float(_cs_row[_cs_row != 0.0][-1]) if len(_nonzero) else float("nan")
                        # Prediction-stability proxies (item 3): worst-case
                        # per-cycle change and monotonicity proxy.
                        row["cycle_cos_sim_min"] = float(_nonzero.min()) if len(_nonzero) else float("nan")
                        if len(_nonzero) >= 2:
                            row["cycle_cos_sim_drift"] = float(_nonzero[-1] - _nonzero[0])
                        else:
                            row["cycle_cos_sim_drift"] = float("nan")
                    # Analysis F: per-sequence argmax-flip count across recursion cycles.
                    if _caf_cpu is not None and i < len(_caf_cpu):
                        row["cycle_argmax_flips"] = float(_caf_cpu[i])
                    buffer.append(row)

                pbar.update(1)

                flush = (idx > 0 and idx % args.save_steps == 0) or (idx == total_batches - 1)
                if flush:
                    out_file = os.path.join(args.output_path, f"data_{idx:05d}.parquet")
                    pd.DataFrame(buffer).to_parquet(out_file, index=False, compression="gzip")
                    logger.info(f"Saved {out_file}")
                    buffer = []

        with open(os.path.join(args.output_path, "done.txt"), "w") as f:
            f.write("done")

        # PonderNet halt-step sidecar (only if ponder ckpt; silently skipped for cmnone).
        if _halt_n > 0:
            mean = _halt_sum / _halt_n
            halt_stats = {"mean_avg_halt_step": round(mean, 4), "n_forwards": _halt_n,
                          "L_max": _L_max or 0}
            if _L_max:
                halt_stats["compute_efficiency"] = round(mean / _L_max, 4)
                halt_stats["compute_savings_fraction"] = round(1.0 - mean / _L_max, 4)
            if _pc_n > 0:
                halt_stats["mean_ponder_cost"] = round(_pc_sum / _pc_n, 6)
            if _halt_pmf_sum is not None:
                _pmf_mean = _halt_pmf_sum / max(_halt_n, 1)
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
            if _halt_pmf_perchannel_sum is not None:
                _pmf_pc_mean = _halt_pmf_perchannel_sum / max(_halt_n, 1)
                _L_max_pc, _C_pc = _pmf_pc_mean.shape
                _col_sums = _pmf_pc_mean.sum(axis=0, keepdims=True)
                _col_sums = np.where(_col_sums > 0, _col_sums, 1.0)
                _pmf_pc_norm = _pmf_pc_mean / _col_sums
                _depths_pc   = np.arange(1, _L_max_pc + 1, dtype=float)
                _e_n_pc      = (_pmf_pc_norm * _depths_pc[:, None]).sum(axis=0)
                halt_stats["gate_mode"] = _gate_mode
                halt_stats["n_channels"] = int(_C_pc)
                halt_stats["per_channel_E_n_mean"] = round(float(_e_n_pc.mean()), 4)
                halt_stats["per_channel_E_n_std"]  = round(float(_e_n_pc.std()), 4)
                _qs = np.quantile(_e_n_pc, [0.0, 0.25, 0.5, 0.75, 1.0])
                halt_stats["per_channel_E_n_quantiles"] = {
                    "min": round(float(_qs[0]),4), "q25": round(float(_qs[1]),4),
                    "median": round(float(_qs[2]),4), "q75": round(float(_qs[3]),4),
                    "max": round(float(_qs[4]),4),
                }
                if _gate_traj_sum is not None:
                    _gt_mean = _gate_traj_sum / max(_halt_n, 1)
                    halt_stats["gate_openness_mean_per_cycle"] = [round(float(v), 4) for v in _gt_mean.mean(axis=1).tolist()]
                    halt_stats["gate_openness_std_per_cycle"]  = [round(float(v), 4) for v in _gt_mean.std(axis=1).tolist()]
                if _C_pc <= 64:
                    halt_stats["halt_pmf_perchannel"] = [[round(float(v), 6) for v in row] for row in _pmf_pc_mean.tolist()]
                    if _gate_traj_sum is not None:
                        halt_stats["gate_openness_trajectory"] = [[round(float(v), 6) for v in row] for row in _gt_mean.tolist()]
            with open(os.path.join(args.output_path, "halt_stats.json"), "w") as f:
                json.dump(halt_stats, f, indent=2)
            logger.info(f"Wrote PonderNet halt stats -> halt_stats.json: {halt_stats}")

    # ---- Aggregate scores ----
    if args.compute_scores:
        parquet_files = sorted(f for f in os.listdir(args.output_path) if f.startswith("data") and f.endswith(".parquet"))
        df = pd.concat([pd.read_parquet(os.path.join(args.output_path, f)) for f in parquet_files], ignore_index=True)

        wt_is_consensus = df[df["wt_aa"] == df["consensus_aa"]]
        wt_is_not_consensus = df[df["wt_aa"] != df["consensus_aa"]]

        scores = {
            # Calibration = 100*exp(-D_KL(model||homolog)), averaged per-position.
            # This is the headline metric. T = WT is consensus, NT = WT not consensus.
            "mean_calibration": float(df["calibration"].mean()),
            "calibration_T": float(wt_is_consensus["calibration"].mean()),
            "calibration_NT": float(wt_is_not_consensus["calibration"].mean()),
            # Raw forward KL (model||homolog) retained for transparency.
            "mean_kl": float(df["kl"].mean()),
            # The superseded direction, for comparison against pre-fix results.
            "mean_kl_forward": float(df["kl_forward"].mean()),
            "calibration_forward_T": float((100.0 * np.exp(-wt_is_consensus["kl_forward"])).mean()),
            "calibration_forward_NT": float((100.0 * np.exp(-wt_is_not_consensus["kl_forward"])).mean()),
            "std_kl": float(df["kl"].std()),
            "median_kl": float(df["kl"].median()),
            # Breakdown by whether WT == consensus
            "mean_kl_wt_is_consensus": float(wt_is_consensus["kl"].mean()),
            "mean_kl_wt_not_consensus": float(wt_is_not_consensus["kl"].mean()),
            # Recovery rates
            "pred_wt_consensus": float((wt_is_consensus["pred_aa"] == wt_is_consensus["wt_aa"]).mean()),
            "pred_other_t": float((wt_is_consensus["pred_aa"] != wt_is_consensus["wt_aa"]).mean()),
            "pred_wt_nt": float((wt_is_not_consensus["pred_aa"] == wt_is_not_consensus["wt_aa"]).mean()),
            "pred_consensus_nt": float((wt_is_not_consensus["pred_aa"] == wt_is_not_consensus["consensus_aa"]).mean()),
            "pred_other_nt": float(
                (
                    (wt_is_not_consensus["pred_aa"] != wt_is_not_consensus["wt_aa"])
                    & (wt_is_not_consensus["pred_aa"] != wt_is_not_consensus["consensus_aa"])
                ).mean()
            ),
            # Counts
            "n_wt_is_consensus": len(wt_is_consensus),
            "n_wt_is_not_consensus": len(wt_is_not_consensus),
            "n_positions": len(df),
        }

        out_scores = os.path.join(args.output_path, "scores.json")
        with open(out_scores, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"Scores saved to {out_scores}")
        print(json.dumps(scores, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)

    # Model
    parser.add_argument("--model_name", type=str, default="local")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Local AMPLIFYSRM/TRM HF checkpoint dir. Takes priority over --model_name.")
    parser.add_argument("--config_path", type=str, required=False)
    parser.add_argument("--vocab_path", type=str, required=False)
    parser.add_argument("--checkpoint_path", type=str, required=False)
    parser.add_argument("--tag", type=str, required=False)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--bos_token", action="store_true", help="Set if the tokenizer prepends a BOS token (offsets position index by 1).")

    # Data
    parser.add_argument("--proteome_path", type=str, required=False, help="CSV with columns: id, sequence")
    parser.add_argument("--consensus_path", type=str, required=False, help="CSV output of extract_consensus_positions.py")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to a pre-built HuggingFace Dataset on disk. If it exists, it is loaded directly "
        "(proteome_path and consensus_path are not needed). If it does not exist and this path "
        "is provided, the built dataset is saved here for future reuse.",
    )

    # Output / control
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--skip_predictions", action="store_true")
    parser.add_argument("--compute_scores", action="store_true")
    parser.add_argument(
        "--force-recompute", "--force_recompute", dest="force_recompute",
        action="store_true",
        help="Delete any existing data_*.parquet and scores.json under --output_path "
             "before running. Use whenever model config (e.g. L_cycles) changes between "
             "runs of the same output dir — otherwise the script silently reuses cached "
             "predictions from the previous run.",
    )
    parser.add_argument(
        "--l_cycles", type=int, default=None,
        help="Override SRM/TRM model.config.L_cycles after load. "
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
    evaluate(args)
