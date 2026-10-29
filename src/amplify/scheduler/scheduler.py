from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from transformers import get_scheduler as _hf_get_scheduler


def get_scheduler(
    optimizer: Optimizer,
    lr_scheduler_type: str,
    num_warmup_steps: int,
    num_training_steps: int,
    scheduler_specific_kwargs: dict | None = None,
    **kwargs,
) -> LRScheduler:
    """Create a learning rate scheduler using HuggingFace Transformers.

    Thin wrapper around :func:`transformers.get_scheduler` that clamps the
    learning rate to its final value beyond ``num_training_steps``. Some HF
    schedulers (e.g. ``cosine_with_min_lr``) let the LR oscillate past the
    minimum when stepped beyond the configured horizon; the clamping ensures
    the LR stays constant instead.

    Common scheduler types:
        - ``"cosine_with_min_lr"``: Cosine decay with warmup. Supports
          ``min_lr_rate`` in *scheduler_specific_kwargs*.
        - ``"linear"``: Linear decay from peak LR to 0 after warmup.
        - ``"cosine"``: Cosine decay from peak LR to 0 after warmup.
        - ``"warmup_stable_decay"``: Warmup → constant → decay. Supports
          ``num_stable_steps``, ``num_decay_steps``, ``min_lr_ratio``, and
          ``decay_type`` in *scheduler_specific_kwargs*.

    Args:
        optimizer: Optimizer whose learning rate will be scheduled.
        lr_scheduler_type: HuggingFace scheduler name (see above).
        num_warmup_steps: Number of warmup steps (linear ramp from 0 to peak LR).
        num_training_steps: Total number of training steps.
        scheduler_specific_kwargs: Extra kwargs forwarded to the underlying
            HuggingFace scheduler (e.g. ``{"min_lr_rate": 0.1}``).
        **kwargs: Ignored (absorbs extra config keys for forward compatibility).

    Returns:
        A PyTorch LR scheduler.
    """
    scheduler = _hf_get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs=scheduler_specific_kwargs,
    )

    # Some schedulers (e.g. constant) don't use lr_lambdas — return as-is.
    if not hasattr(scheduler, "lr_lambdas"):
        return scheduler

    # Clamp: wrap each lr_lambda so the LR holds at its final value beyond num_training_steps.
    clamped_lambdas = []
    for lr_lambda in scheduler.lr_lambdas:
        final_factor = lr_lambda(num_training_steps)

        def clamped_lr_lambda(current_step, fn=lr_lambda, final=final_factor):
            if current_step >= num_training_steps:
                return final
            return fn(current_step)

        clamped_lambdas.append(clamped_lr_lambda)

    return LambdaLR(optimizer, clamped_lambdas, last_epoch=scheduler.last_epoch)
