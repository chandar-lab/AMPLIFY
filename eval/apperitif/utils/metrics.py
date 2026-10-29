import torch
import numpy as np
import torch.nn.functional as F

# Loss function for contact map prediction (two-class logits)
# Takes as first input a (nested) tensor of contact map logits of shape (batch_size, [seq_len, seq_len, 2])
# Each element is a 3D tensor representing per-pair logits for (neg, pos) classes for a single sequence
# The first token is always BOS. The sequence can be truncated to max_length, otherwise it has the EOS token at the end
# Padding has been removed
# The second input is also a nested tensor of shape (batch_size, [num_contacts, 2])
# Each element is a 2D tensor representing the list of contact positions for a single sequence in the batch
# These positions have already been offset by 1, to account for the BOS token
# so they are in the range [1, seq_len]


def sequence_level_contact_map_loss(
    predicted_contact_map,
    true_contacts,
    shape=None,
    label_smoothing: float = 0.0,
    pos_weight: float = 30.0,
    minsep: int = 0,
):
    """
    Computes the sequence-level loss for contact map prediction.

    Args:
    predicted_contact_map (torch.Tensor): Predicted contact map of shape (seq_len, seq_len). (Deprecated path)
        true_contacts (torch.Tensor): True contact positions of shape (num_contacts, 2).
        shape (tuple, optional): Shape of the predicted contact map, used to determine sequence length.
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
        label_smoothing (float): If greater than 0, applies label smoothing to the loss.
        pos_weight (float): Positive weight for the loss function, used to balance the loss for positive and negative samples.
        minsep (int): Minimum separation between contacts to consider them valid.

    Returns:
        torch.Tensor: The computed loss.
    """
    # NOTE: This function implements a BCEWithLogits path for single-logit maps.
    # For the two-class setup (preferred), contact_map_loss below uses CrossEntropy directly
    # and does not call this function.
    seq_len = shape[0] if shape is not None else predicted_contact_map.shape[0]

    i_indices = true_contacts[:, 0]
    j_indices = true_contacts[:, 1]
    i_mask = i_indices < seq_len
    j_mask = j_indices < seq_len
    valid_mask = i_mask & j_mask
    # Apply minsep to positive pairs (drop easy contacts)
    if minsep > 0:
        valid_mask = valid_mask & (torch.abs(i_indices - j_indices) >= minsep)
    if not valid_mask.any():
        return torch.tensor(0.0, device=predicted_contact_map.device)
    i_indices = i_indices[valid_mask]
    j_indices = j_indices[valid_mask]

    target = torch.zeros_like(predicted_contact_map, dtype=torch.float32)
    target[i_indices, j_indices] = 1.0
    target[j_indices, i_indices] = 1.0
    if label_smoothing > 0.0:
        target = (1 - label_smoothing) * target + label_smoothing / 2

    # Mask out near-diagonal positions from loss entirely (remove easy negatives/positives)
    if minsep > 0:
        L = seq_len
        idx = torch.arange(L, device=predicted_contact_map.device)
        sep = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
        valid_loss_mask = (sep >= minsep).to(predicted_contact_map.dtype)
    else:
        valid_loss_mask = torch.ones_like(predicted_contact_map, dtype=predicted_contact_map.dtype)

    # Compute BCE with logits elementwise then mask and reduce
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        predicted_contact_map,
        target,
        reduction="none",
        pos_weight=(torch.tensor(pos_weight, device=predicted_contact_map.device) if pos_weight > 0 else None),
    )

    # Optional: auto pos_weight if pos_weight <= 0 (balance positives vs negatives under mask)
    if pos_weight <= 0:
        pos_count = (target * valid_loss_mask).sum()
        neg_count = (valid_loss_mask.numel() - valid_loss_mask.sum()) + ((1 - target) * valid_loss_mask).sum()
        # Avoid division by zero
        if pos_count > 0:
            dynamic_pw = (neg_count / pos_count).clamp(min=1.0)
            bce_pos = torch.nn.functional.binary_cross_entropy_with_logits(
                predicted_contact_map, target, reduction="none", pos_weight=dynamic_pw
            )
            bce = bce_pos

    masked_loss = (bce * valid_loss_mask).sum() / valid_loss_mask.sum().clamp(min=1.0)
    return masked_loss


def contact_map_loss(
    predicted_contact_maps: torch.Tensor, true_contacts: torch.Tensor, label_smoothing=0.0, pos_weight=30, minsep=0
) -> torch.Tensor:
    """
    Computes the loss for contact map prediction over a batch using a for loop.

    Args:
        predicted_contact_maps (torch.Tensor): Predicted contact maps of shape (batch_size, seq_len, seq_len) or nested tensor.
        true_contacts (list of torch.Tensor): List of true contact positions, each of shape (num_contacts, 2).
        label_smoothing (float): If greater than 0, applies label smoothing to the loss.
        pos_weight (float): Positive weight for the loss function, used to balance the loss for positive and negative samples.
        minsep (int): Minimum separation between contacts to consider them valid.
    Returns:
        torch.Tensor: The computed loss.
    """
    # Enforce two-class logits throughout
    losses = []
    for contact_map, contacts in zip(predicted_contact_maps, true_contacts):
        if contact_map.dim() != 3 or contact_map.shape[-1] != 2:
            raise ValueError(
                f"contact_map_loss expects (L, L, 2) logits per sequence. Got shape {tuple(contact_map.shape)}"
            )
        L = contact_map.shape[0]
        # Build dense integer targets (0/1)
        target = torch.zeros((L, L), dtype=torch.long, device=contact_map.device)
        if contacts.numel() > 0:
            i = contacts[:, 0].clamp(max=L - 1)
            j = contacts[:, 1].clamp(max=L - 1)
            target[i, j] = 1
            target[j, i] = 1
        # Min-sep valid mask
        if minsep > 0:
            idx = torch.arange(L, device=contact_map.device)
            sep = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
            valid_mask = sep >= minsep
        else:
            valid_mask = torch.ones((L, L), dtype=torch.bool, device=contact_map.device)
        # Compute per-pixel CE loss with class weights [neg=1.0, pos=pos_weight]
        logits_flat = contact_map.reshape(-1, 2)
        target_flat = target.reshape(-1)
        mask_flat = valid_mask.reshape(-1)
        if mask_flat.any():
            # Match the dtype of the logits to ensure compatibility with autocast
            weights = torch.tensor([1.0, float(pos_weight)], device=contact_map.device, dtype=logits_flat.dtype)
            ce = F.cross_entropy(
                logits_flat, target_flat, reduction="none", weight=weights, label_smoothing=float(label_smoothing)
            )
            loss = ce[mask_flat].mean()
        else:
            loss = torch.tensor(0.0, device=contact_map.device)
        losses.append(loss)
    return torch.stack(losses).mean()


# =====================
# New metrics (AUC, P@L, P@L2, P@L5) per distance range
# Adapted from https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb
# =====================


def _contact_scores_from_pred(predicted_contact_map: torch.Tensor) -> torch.Tensor:
    """
    Normalize predicted contact map to a 2D score/probability matrix of shape (L, L).

    Expects logits of shape (L, L, 2) and returns the probability for the contact class (index 1).
    """
    if predicted_contact_map.dim() != 3 or predicted_contact_map.shape[-1] != 2:
        raise ValueError(
            f"Expected (L, L, 2) logits for contact prediction, got shape {tuple(predicted_contact_map.shape)}"
        )
    probs = torch.softmax(predicted_contact_map, dim=-1)
    return probs[..., 1]


def _dense_targets_from_pairs(true_contacts: torch.Tensor, L: int, device: torch.device) -> torch.Tensor:
    """
    Build a dense target matrix of shape (L, L) with 1.0 at contact pairs and 0.0 elsewhere.
    """
    target = torch.zeros((L, L), dtype=torch.float32, device=device)
    if true_contacts.numel() > 0:
        i = true_contacts[:, 0].clamp(max=L - 1)
        j = true_contacts[:, 1].clamp(max=L - 1)
        target[i, j] = 1.0
        target[j, i] = 1.0
    return target


def compute_precisions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    src_lengths: torch.Tensor | None = None,
    minsep: int = 6,
    maxsep: int | None = None,
    override_length: int | None = None,
):
    """
    Compute binned precisions (P@L/5, P@L/2, P@L) and AUC over top-k ranked pairs for upper triangle, subject to
    a minimum and optional maximum sequence separation.

    predictions: (B, L, L) scores/probabilities
    targets:     (B, L, L) binary labels (>=0 valid)
    Returns dict of tensors of shape (B,)
    """
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)

    device = predictions.device
    batch_size, seqlen, _ = predictions.size()
    seqlen_range = torch.arange(seqlen, device=device)

    # sep[i, j] = j - i so that upper triangle has positive separations
    # seqlen_range.unsqueeze(0) broadcasts as row, unsqueeze(1) as column
    # sep[i, j] = seqlen_range[j] - seqlen_range[i] = j - i
    sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
    sep = sep.unsqueeze(0)
    valid_mask = sep >= minsep
    valid_mask = valid_mask & (targets >= 0)  # negative targets are invalid

    if maxsep is not None:
        valid_mask &= sep < maxsep

    if src_lengths is not None:
        valid = seqlen_range.unsqueeze(0) < src_lengths.unsqueeze(1)
        valid_mask &= valid.unsqueeze(1) & valid.unsqueeze(2)
    else:
        src_lengths = torch.full([batch_size], seqlen, device=device, dtype=torch.long)

    predictions = predictions.masked_fill(~valid_mask, float("-inf"))

    # Upper triangle indices with offset = minsep
    iu = torch.triu_indices(seqlen, seqlen, offset=minsep, device=device)
    x_ind, y_ind = iu[0], iu[1]
    predictions_upper = predictions[:, x_ind, y_ind]
    targets_upper = targets[:, x_ind, y_ind]

    # Determine ranking length
    if override_length is None:
        topk = seqlen
        gather_lengths = src_lengths.unsqueeze(1)
    else:
        topk = max(seqlen, override_length)
        gather_lengths = override_length * torch.ones((batch_size, 1), device=device, dtype=torch.long)

    indices = predictions_upper.argsort(dim=-1, descending=True)[:, :topk]
    topk_targets = targets_upper[torch.arange(batch_size, device=device).unsqueeze(1), indices]
    if topk_targets.size(1) < topk:
        topk_targets = F.pad(topk_targets, [0, topk - topk_targets.size(1)])

    cumulative_dist = topk_targets.type_as(predictions).cumsum(-1)

    gather_indices = (torch.arange(0.1, 1.1, 0.1, device=device).unsqueeze(0) * gather_lengths).type(torch.long) - 1
    gather_indices = gather_indices.clamp(min=0)

    binned_cumulative_dist = cumulative_dist.gather(1, gather_indices)
    binned_precisions = binned_cumulative_dist / (gather_indices + 1).type_as(binned_cumulative_dist)

    pl5 = binned_precisions[:, 1]
    pl2 = binned_precisions[:, 4]
    pl = binned_precisions[:, 9]
    auc = binned_precisions.mean(-1)

    return {"AUC": auc, "P@L": pl, "P@L2": pl2, "P@L5": pl5}


def evaluate_prediction(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Compute metrics across distance ranges for a batch or single sequence.
    Returns a dict mapping metric names (e.g., 'local_AUC') to tensors of shape (B,).
    """
    contact_ranges = [("local", 3, 6), ("short", 6, 12), ("medium", 12, 24), ("long", 24, None)]
    metrics: dict[str, torch.Tensor] = {}
    for name, minsep, maxsep in contact_ranges:
        rangemetrics = compute_precisions(predictions, targets, minsep=minsep, maxsep=maxsep)
        for key, val in rangemetrics.items():
            metrics[f"{name}_{key}"] = val
    return metrics


def compute_contact_metrics_for_sequence(pred_map: torch.Tensor, contacts: torch.Tensor) -> dict[str, float]:
    """
    Convenience wrapper: build dense target, compute local/short/medium/long metrics for a single sequence.
    Returns a dict of Python floats.
    """
    scores = _contact_scores_from_pred(pred_map)
    L = scores.shape[0]
    targets = _dense_targets_from_pairs(contacts, L=L, device=scores.device)
    batch_metrics = evaluate_prediction(scores, targets)
    # Convert to floats
    return {k: float(v.item()) for k, v in batch_metrics.items()}


def compute_contact_metric_value(pred_map: torch.Tensor, contacts: torch.Tensor, metric_name: str) -> float:
    """
    Compute a single metric value by name for one sequence, computing only that range.
    Supported names: '{range}_AUC' | '{range}_P@L' | '{range}_P@L2' | '{range}_P@L5'
    where range in {local, short, medium, long}.
    """
    try:
        range_name, stat = metric_name.split("_", 1)
    except ValueError:
        raise ValueError(f"Invalid metric name '{metric_name}'. Expected format '<range>_<stat>' e.g. 'medium_P@L5'")

    range_to_bounds = {
        "local": (3, 6),
        "short": (6, 12),
        "medium": (12, 24),
        "long": (24, None),
    }
    if range_name not in range_to_bounds:
        raise ValueError(f"Invalid range '{range_name}'. Expected one of {sorted(range_to_bounds.keys())}")
    if stat not in {"AUC", "P@L", "P@L2", "P@L5"}:
        raise ValueError("Invalid stat '{stat}'. Expected one of {'AUC','P@L','P@L2','P@L5'}")

    scores = _contact_scores_from_pred(pred_map)
    L = scores.shape[0]
    targets = _dense_targets_from_pairs(contacts, L=L, device=scores.device)

    minsep, maxsep = range_to_bounds[range_name]
    res = compute_precisions(scores, targets, minsep=minsep, maxsep=maxsep)
    value = res[stat]
    return float(value.item())
