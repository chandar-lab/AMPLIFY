"""PyTorch Lightning callback for measuring training throughput.

This callback tracks and logs samples per second and updates per second metrics
during training, validation, and testing. Useful for performance monitoring and
benchmarking.
"""
import time
from typing import Any, Generic, Literal, override

import lightning
import optree
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing_extensions import TypeVar

from project.utils.typing_utils import NestedMapping, is_sequence_of

BatchType = TypeVar(
    "BatchType",
    bound=torch.Tensor | tuple[torch.Tensor, ...] | NestedMapping[str, torch.Tensor],
    contravariant=True,
)


class MeasureSamplesPerSecondCallback(lightning.Callback, Generic[BatchType]):
    def __init__(self, num_optimizers: int | None = None):
        super().__init__()
        self.last_step_times: dict[Literal["train", "val", "test"], float] = {}
        self.last_update_time: dict[int, float | None] = {}
        self.num_optimizers: int | None = num_optimizers

    @override
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_epoch_start(trainer, pl_module)
        self.on_shared_epoch_start(trainer, pl_module, phase="train")

    @override
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_epoch_start(trainer, pl_module)
        self.on_shared_epoch_start(trainer, pl_module, phase="val")

    @override
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_test_epoch_start(trainer, pl_module)
        self.on_shared_epoch_start(trainer, pl_module, phase="test")

    def on_shared_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        phase: Literal["train", "val", "test"],
    ) -> None:
        self.last_update_time.clear()
        self.last_step_times.pop(phase, None)
        if self.num_optimizers is None:
            optimizer_or_optimizers = pl_module.optimizers()
            if not isinstance(optimizer_or_optimizers, list):
                self.num_optimizers = 1
            else:
                self.num_optimizers = len(optimizer_or_optimizers)

    @override
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: BatchType,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
        )
        self.on_shared_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_index=batch_idx,
            phase="train",
        )

    @override
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: BatchType,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,  # type: ignore
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )
        self.on_shared_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_index=batch_idx,
            phase="val",
            dataloader_idx=dataloader_idx,
        )

    @override
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: BatchType,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_test_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,  # type: ignore
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )
        self.on_shared_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_index=batch_idx,
            dataloader_idx=dataloader_idx,
            phase="test",
        )

    def on_shared_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: BatchType,
        batch_index: int,
        phase: Literal["train", "val", "test"],
        dataloader_idx: int | None = None,
    ):
        now = time.perf_counter()
        if phase in self.last_step_times:
            elapsed = now - self.last_step_times[phase]
            batch_size = self.get_num_samples(batch)
            pl_module.log(
                f"{phase}/samples_per_second",
                batch_size / elapsed,
                # module=pl_module,
                # trainer=trainer,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
            # todo: support other kinds of batches
        self.last_step_times[phase] = now

    def log(
        self,
        name: str,
        value: Any,
        module: LightningModule | Any,
        trainer: Trainer | Any,
        **kwargs,
    ):
        # Used to possibly customize how the values are logged (e.g. for non-LightningModules).
        # By default, uses the LightningModule.log method.
        return module.log(
            name,
            value,
            **kwargs,
        )

    def get_num_samples(self, batch: BatchType) -> int:
        if isinstance(batch, Tensor):
            return batch.shape[0]
        if is_sequence_of(batch, Tensor):
            return batch[0].shape[0]
        if isinstance(batch, dict):
            return next(
                v.shape[0]
                for v in optree.tree_leaves(batch)  # type: ignore
                if isinstance(v, torch.Tensor)
            )
        raise NotImplementedError(
            f"Don't know how many 'samples' there are in batch of type {type(batch)}"
        )

    @override
    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: Optimizer,
        opt_idx: int = 0,
    ) -> None:
        if opt_idx not in self.last_update_time or self.last_update_time[opt_idx] is None:
            self.last_update_time[opt_idx] = time.perf_counter()
            return
        last_update_time = self.last_update_time[opt_idx]
        assert last_update_time is not None
        now = time.perf_counter()
        elapsed = now - last_update_time
        updates_per_second = 1 / elapsed
        if self.num_optimizers == 1:
            key = "ups"
        else:
            key = f"optimizer_{opt_idx}/ups"
        pl_module.log(
            key,
            updates_per_second,
            # module=pl_module,
            # trainer=trainer,
            prog_bar=False,
            on_step=True,
        )
