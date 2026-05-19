"""PyTorch Lightning callback for automatically computing classification metrics.

This callback attaches classification metrics (accuracy, top-5 accuracy, cross-entropy)
to Lightning modules during training, validation, and testing phases. It automatically
detects classification tasks and adds appropriate metrics.
"""
import warnings
from logging import getLogger as get_logger
from typing import Literal, NotRequired, Required, TypedDict, override

import lightning
import torch
import torchmetrics
from lightning import LightningModule, Trainer
from torch import Tensor
from torchmetrics.classification import MulticlassAccuracy

from project.utils.typing_utils.protocols import ClassificationDataModule

logger = get_logger(__name__)


class ClassificationOutputs(TypedDict, total=False):
    """The outputs that should be minimally returned from the training/val/test_step of
    classification LightningModules so that metrics can be added aumatically by the
    `ClassificationMetricsCallback`."""

    loss: NotRequired[torch.Tensor | float]
    """The loss at this step."""

    logits: Required[Tensor]
    """The un-normalized logits."""

    y: Required[Tensor]
    """The class labels."""


class ClassificationMetricsCallback(lightning.Callback):
    """Callback that adds classification metrics to a LightningModule."""

    def __init__(self) -> None:
        super().__init__()
        self.disabled = False

    @classmethod
    def attach_to(cls, algorithm: LightningModule, num_classes: int):
        callback = cls()
        callback.add_metrics_to(algorithm, num_classes=num_classes)
        return callback

    def add_metrics_to(self, pl_module: LightningModule, num_classes: int) -> None:
        # IDEA: Could use a dict of metrics from torchmetrics instead of just accuracy:
        # self.supervised_metrics: dist[str, Metrics]
        # NOTE: Need to have one per phase! Not 100% sure that I'm not forgetting a phase here.

        # Slightly ugly. Need to set the metrics on the pl module for things to be logged / synced
        # easily.
        metrics_to_add = {
            "train_accuracy": MulticlassAccuracy(num_classes=num_classes),
            "val_accuracy": MulticlassAccuracy(num_classes=num_classes),
            "test_accuracy": MulticlassAccuracy(num_classes=num_classes),
            "train_top5_accuracy": MulticlassAccuracy(num_classes=num_classes, top_k=5),
            "val_top5_accuracy": MulticlassAccuracy(num_classes=num_classes, top_k=5),
            "test_top5_accuracy": MulticlassAccuracy(num_classes=num_classes, top_k=5),
        }
        if all(
            hasattr(pl_module, name) and isinstance(getattr(pl_module, name), type(metric))
            for name, metric in metrics_to_add.items()
        ):
            logger.info("Not adding metrics to the pl module because they are already present.")
            return

        for metric_name, metric in metrics_to_add.items():
            self._set_metric(pl_module, metric_name, metric)

    # todo: change these two if we end up putting metrics in a ModuleDict.
    @staticmethod
    def _set_metric(pl_module: LightningModule, name: str, metric: torchmetrics.Metric):
        if hasattr(pl_module, name):
            raise RuntimeError(f"The pl module already has an attribute with the name {name}.")
        logger.info(f"Setting a new metric on the pl module at attribute {name}.")
        setattr(pl_module, name, metric)

    @staticmethod
    def _get_metric(pl_module: LightningModule, name: str):
        return getattr(pl_module, name)

    @override
    def setup(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: Literal["fit", "validate", "test", "predict", "tune"],
    ) -> None:
        if self.disabled:
            return
        datamodule = pl_module.datamodule
        if not isinstance(datamodule, ClassificationDataModule):
            warnings.warn(
                RuntimeWarning(
                    f"Disabling the {type(self).__name__} callback because it only works with "
                    f"classification datamodules, but {pl_module.datamodule=} isn't a "
                    f"{ClassificationDataModule.__name__}."
                )
            )
            self.disabled = True
            return

        num_classes = datamodule.num_classes
        self.add_metrics_to(pl_module, num_classes=num_classes)

    @override
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: ClassificationOutputs,
        batch: tuple[Tensor, Tensor],
        batch_index: int,
    ) -> None:
        super().on_train_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_index,
        )
        self.on_shared_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_index=batch_index,
            phase="train",
        )

    @override
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: ClassificationOutputs,
        batch: tuple[Tensor, Tensor],
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
        outputs: ClassificationOutputs,
        batch: tuple[Tensor, Tensor],
        batch_index: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_test_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,  # type: ignore
            batch=batch,
            batch_idx=batch_index,
            dataloader_idx=dataloader_idx,
        )
        self.on_shared_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_index=batch_index,
            dataloader_idx=dataloader_idx,
            phase="test",
        )

    def on_shared_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: ClassificationOutputs,
        batch: tuple[Tensor, Tensor],
        batch_index: int,
        phase: Literal["train", "val", "test"],
        dataloader_idx: int | None = None,
    ):
        if self.disabled:
            return
        step_output = outputs
        required_entries = ClassificationOutputs.__required_keys__
        if not isinstance(outputs, dict):
            warnings.warn(
                RuntimeWarning(
                    f"Expected the {phase} step method to output a dictionary with at least the "
                    f"{required_entries} keys, but got an output of type {type(step_output)} instead!\n"
                    f"Disabling the {type(self).__name__} callback."
                )
            )
            self.disabled = True
            return
        if not all(k in step_output for k in required_entries):
            warnings.warn(
                RuntimeWarning(
                    f"Expected all the following keys to be in the output of the {phase} step "
                    f"method: {required_entries}. Disabling the {type(self).__name__} callback."
                )
            )
            self.disabled = True
            return

        logits = step_output["logits"]
        y = step_output["y"]

        probs = torch.softmax(logits, -1)

        accuracy = self._get_metric(pl_module, f"{phase}_accuracy")
        top5_accuracy = self._get_metric(pl_module, f"{phase}_top5_accuracy")
        assert isinstance(accuracy, MulticlassAccuracy)
        assert isinstance(top5_accuracy, MulticlassAccuracy)

        # TODO: It's a bit confusing, not sure if this is the right way to use this:
        accuracy(probs, y)
        top5_accuracy(probs, y)
        prog_bar = phase == "train"
        pl_module.log(f"{phase}/accuracy", accuracy, prog_bar=prog_bar, sync_dist=True)
        pl_module.log(f"{phase}/top5_accuracy", top5_accuracy, prog_bar=prog_bar, sync_dist=True)

        if "cross_entropy" not in step_output:
            # Add the cross entropy loss as a metric.
            with torch.no_grad():
                ce_loss = torch.nn.functional.cross_entropy(logits.detach(), y, reduction="mean")
            pl_module.log(f"{phase}/cross_entropy", ce_loss, prog_bar=prog_bar, sync_dist=True)

        loss: Tensor | float | None = step_output.get("loss", None)
        if loss is not None:
            # note: Perhaps we should be careful not to overwrite the logged value if its already been logged?
            pl_module.log(
                f"{phase}/loss", torch.as_tensor(loss).mean(), prog_bar=prog_bar, sync_dist=True
            )

        # This part isn't necessary here: Average out the losses properly.
        # fused_output = step_output.copy()
        # if isinstance(loss, Tensor) and loss.shape:
        #     # Replace the loss with its mean. This is useful when automatic
        #     # optimization is enabled, for example in the baseline (backprop), where each replica
        #     # returns the un-reduced cross-entropy loss. Here we need to reduce it to a scalar.
        #     fused_output["loss"] = loss.mean()

        # return fused_output
