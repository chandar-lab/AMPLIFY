import math
import torch
from functools import partial
from torch.optim.lr_scheduler import LambdaLR


def learning_rate_fn(
    current_step: int,
    algorithm: str,
    warmup_steps: int,
    final_step: int,
    final_ratio: float,
    warm_restart_steps: int | None = None,
) -> float:
    """Return the factor we multiply learning rate with.

    Args:
        current_step (int): Current optimizer step.
        algorithm (str): Name of the algorithm. Only "LinearDecay", "LinearDecayWarmRestart", "CosineDecay",
        and "CosineDecayWarmRestart" are supported.
        warmup_steps (int): Number of warmup steps (over which to linearly increase the learning rate from 0 to the peak
        learning rate).
        final_step (int): Number of decay steps (over which to decay the learning rate from the peak learning rate to
        the final ratio).
        final_ratio (float): Number we multiply learning rate at the end of linear changing process.
        warm_restart_steps (int | None): Number of steps to periodically restart the learning rate.

    Returns:
        float: Factor with which to multiply the learning rate at the current step.
    """
    if current_step < warmup_steps:
        return float(current_step) / float(warmup_steps)
    elif current_step < final_step:
        steps_remaining = final_step - current_step
        steps_after_warmup = current_step - warmup_steps
        total_steps_after_warmup = final_step - warmup_steps

        if "LinearDecay" in algorithm:
            if warm_restart_steps is None or warm_restart_steps == 0:
                return (steps_remaining + steps_after_warmup * final_ratio) / (final_step - warmup_steps)
            else:
                return final_ratio + (1 - final_ratio) * (
                    1 - (steps_after_warmup // warm_restart_steps * warm_restart_steps / total_steps_after_warmup)
                ) * ((warm_restart_steps - (steps_after_warmup % warm_restart_steps)) / warm_restart_steps)

        elif "CosineDecay" in algorithm:
            factor = (
                (1 - final_ratio) * (steps_remaining + steps_after_warmup * final_ratio) / (final_step - warmup_steps)
            )
            if warm_restart_steps is None or warm_restart_steps == 0:
                return (
                    final_ratio + factor * (1 + math.cos(math.pi * steps_after_warmup / total_steps_after_warmup)) / 2
                )
            else:
                return (
                    final_ratio
                    + factor
                    * (1 + math.cos(math.pi * (steps_after_warmup % warm_restart_steps) / warm_restart_steps))
                    / 2
                )

    return final_ratio


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    _name_: str,
    warmup_steps: int,
    final_step: int,
    final_ratio: float,
    warm_restart_steps: int | None = None,
    **kwargs,
) -> torch.optim.lr_scheduler:
    """Scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer.
        _name_ (str): Name of the algorithm. Only "LinearDecay", "LinearDecayWarmRestart", "CosineDecay",
        and "CosineDecayWarmRestart" are supported.
        warmup_steps (int): Number of warmup steps (over which to linearly increase the learning rate from 0 to the peak
        learning rate).
        final_step (int): Number of decay steps (over which to decay the learning rate from the peak learning rate to
        the final ratio).
        final_ratio (float): Number we multiply learning rate at the end of linear changing process.
        warm_restart_steps (int | None, optional): _description_. Number of steps to periodically restart the learning
        rate.

    Returns:
        torch.optim.lr_scheduler: Initialized scheduler.
    """
    return LambdaLR(
        optimizer,
        partial(
            learning_rate_fn,
            algorithm=_name_,
            warmup_steps=warmup_steps,
            final_step=final_step,
            final_ratio=final_ratio,
            warm_restart_steps=warm_restart_steps,
        ),
    )
