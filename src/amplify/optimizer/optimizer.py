import torch
from torch.optim import AdamW
from typing import Any


def get_optimizer(model: torch.nn.Module, **kwargs: Any) -> torch.optim.Optimizer:
    """Create an optimizer for the model.

    Currently only supports AdamW optimizer.

    Args:
        model: Model whose parameters will be optimized.
        **kwargs: Optimizer configuration including:
            - type: Optimizer type (currently only "AdamW" is supported).
            - lr: Learning rate.
            - weight_decay: Weight decay coefficient.
            - betas: Adam beta parameters.
            - eps: Adam epsilon for numerical stability.

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: If optimizer type is not "AdamW".
    """
    match kwargs.pop("type"):
        case "AdamW":
            return AdamW(model.parameters(), **kwargs)
        case other:
            raise ValueError(f"Unsupported optimizer type: {other}. Only 'AdamW' is supported.")
