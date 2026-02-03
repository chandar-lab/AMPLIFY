"""Metrics related to contact prediction"""

import torch
from torchmetrics import Metric


class LongRangePrecisionAtL(Metric):
    """
    Custom metric for long-range Precision at L for contact prediction.
    This metric computes the precision over the top_k most confident long-range predictions
    for each sample.
    """

    def __init__(self, top_factor: int = 5, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # States: we accumulate total precision sum and count of samples.
        self.add_state("precision_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sample_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.top_factor = top_factor

    def update(self, preds: torch.Tensor, target: torch.Tensor, effective_L: int):
        """
        Update metric state for a single sample.
        Args:
            preds (torch.Tensor): Predicted contact map of shape (L_pad, L_pad， 2)
            target (torch.Tensor): Ground truth contact map of shape (L_pad, L_pad)
            effective_L (int): Effective sequence length.
        """
        # Crop to the effective region.
        pred_valid = preds[1 : effective_L + 1, 1 : effective_L + 1]  # bos and eos
        pred_score = torch.softmax(pred_valid, dim=-1)[..., 1]  # obtain the score of 1

        target_valid = target[1 : effective_L + 1, 1 : effective_L + 1]  # bos and eos

        # Create a mask for long-range contacts: only consider pairs with row < col and |row-col| >= 7.
        long_range_mask = torch.triu(
            torch.ones((effective_L, effective_L), dtype=torch.bool), diagonal=7
        )
        pred_long = pred_score[long_range_mask]
        target_long = target_valid[long_range_mask]

        top_k = (
            effective_L // self.top_factor
            if effective_L >= self.top_factor
            else effective_L
        )
        if pred_long.numel() == 0 or top_k == 0:
            sample_precision = torch.tensor(0.0, device=preds.device)
        else:
            sorted_indices = torch.argsort(pred_long, descending=True)
            top_indices = sorted_indices[:top_k]
            top_target = target_long[top_indices]
            true_positives = top_target.sum().float()
            sample_precision = true_positives / top_k

        self.precision_sum += sample_precision
        self.sample_count += 1

    def compute(self) -> torch.Tensor:
        return (
            self.precision_sum / self.sample_count
            if self.sample_count > 0
            else torch.tensor(0.0)
        )

    def reset(self):
        self.precision_sum.zero_()
        self.sample_count.zero_()


class Fmax(Metric):
    """
    https://github.com/westlake-repl/SaProt/blob/main/utils/metrics.py
    """

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # Accumulate predictions and targets from batches.
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update the state with predictions and targets from the current batch.

        Args:
            preds (Tensor): Prediction scores of shape (B, N).
            targets (Tensor): Binary targets of shape (B, N).
        """
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self) -> torch.Tensor:
        """
        Compute the maximum F1 score (Fmax) over all possible thresholds.

        Returns:
            Tensor: The maximum F1 score computed on the accumulated predictions and targets.
        """
        # Concatenate the stored tensors along the batch dimension.
        preds = torch.cat(self.preds, dim=0)
        targets = torch.cat(self.targets, dim=0)
        return self._count_f1_max(preds, targets)

    @staticmethod
    def _count_f1_max(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Fmax score by implicitly considering all possible thresholds.
        This implementation is adapted from TorchDrug's version.

        Args:
            pred (Tensor): Predictions of shape (B, N).
            target (Tensor): Binary targets of shape (B, N).

        Returns:
            Tensor: The maximum F1 score.
        """
        # Sort predictions in descending order along each row.
        order = pred.argsort(descending=True, dim=1)
        # Rearrange target tensor according to the sorted order.
        target = target.gather(1, order)
        precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
        recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)

        # Create a boolean mask indicating the start of each sorted row.
        is_start = torch.zeros_like(target).bool()
        is_start[:, 0] = True
        is_start = torch.scatter(is_start, 1, order, is_start)

        # Flatten the predictions and obtain a global descending order.
        all_order = pred.flatten().argsort(descending=True)
        # Adjust indices for flattening.
        order = (
            order
            + torch.arange(order.shape[0], device=order.device).unsqueeze(1)
            * order.shape[1]
        )
        order = order.flatten()
        inv_order = torch.zeros_like(order)
        inv_order[order] = torch.arange(order.shape[0], device=order.device)

        # Reorder is_start and compute cumulative precision and recall.
        is_start = is_start.flatten()[all_order]
        all_order = inv_order[all_order]
        precision = precision.flatten()
        recall = recall.flatten()

        all_precision = precision[all_order] - torch.where(
            is_start, torch.zeros_like(precision), precision[all_order - 1]
        )
        all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
        all_recall = recall[all_order] - torch.where(
            is_start, torch.zeros_like(recall), recall[all_order - 1]
        )
        all_recall = all_recall.cumsum(0) / pred.shape[0]

        all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
        return all_f1.max()
