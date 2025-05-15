import torch
import torch.optim as optim


def get_optimizer(model: torch.nn.Module, **kwargs) -> torch.optim.Optimizer:
    """Optimizer.

    Args:
        model (torch.nn.Module): Model.

    Returns:
        torch.optim.Optimizer: Initialized optimizer.
    """
    match kwargs.pop("_name_"):
        case "Adam":
            return optim.Adam(model.parameters(), **kwargs)
        case "AdamW":
            return optim.AdamW(model.parameters(), **kwargs)
