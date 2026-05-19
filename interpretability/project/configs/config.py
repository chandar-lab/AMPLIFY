"""Structured configuration schema for Hydra-based experiment management.

This module defines the Config dataclass that serves as the schema for all Hydra
configuration files. It ensures type safety and provides documentation for all
experiment parameters.
"""
import random
from dataclasses import dataclass, field
from logging import getLogger as get_logger
from typing import Any, Optional

logger = get_logger(__name__)


@dataclass
class Config:
    """Structured configuration schema for interpretability experiments.

    This dataclass defines all possible configuration options for experiments,
    including algorithm settings, datamodule configuration, trainer parameters,
    and experiment metadata. It acts as a type-safe schema for Hydra configs.

    For more info, see https://hydra.cc/docs/tutorials/structured_config/schema/
    """

    algorithm: Any | None = None  #
    """Configuration for the algorithm (a
    [LightningModule][lightning.pytorch.core.module.LightningModule]).

    It is suggested for this class to accept a `datamodule` and `network` as arguments. The
    instantiated datamodule and network will be passed to the algorithm's constructor.

    For more info, see the [instantiate_algorithm][project.main.instantiate_algorithm] function.
    """

    datamodule: Optional[Any] = None  # noqa
    """Configuration for the datamodule (dataset + transforms + dataloader creation).

    This should normally create a [LightningDataModule][lightning.pytorch.core.datamodule.LightningDataModule].
    See the [MNISTDataModule][project.datamodules.image_classification.mnist.MNISTDataModule] for an example.
    """

    trainer: dict = field(default_factory=dict)
    """Configuration for the 'Trainer'."""

    log_level: str = "info"
    """Logging level."""

    # Random seed.
    seed: int = field(default_factory=lambda: random.randint(0, int(1e5)))
    """Random seed for reproducibility.

    If None, a random seed is generated.
    """

    name: str = "default"
    """Name for the experiment."""

    debug: bool = False

    verbose: bool = False

    ckpt_path: str | None = None
    """Path to a checkpoint to load the training state and resume the training run.

    This is the same as the `ckpt_path` argument in the `lightning.Trainer.fit` method.
    """
