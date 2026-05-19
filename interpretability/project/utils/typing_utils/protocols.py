from __future__ import annotations

import typing
from typing import Literal, ParamSpec, Protocol, TypeVar, runtime_checkable

if typing.TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader

P = ParamSpec("P")
OutT = TypeVar("OutT", covariant=True)


@runtime_checkable
class Module(Protocol[P, OutT]):
    """Small protocol that can be used to annotate the input/output types of `torch.nn.Module`s."""

    def forward(self, *args: P.args, **kwargs: P.kwargs) -> OutT:
        raise NotImplementedError

    if typing.TYPE_CHECKING:
        # note: Only define this for typing purposes so that we don't actually override anything.
        def __call__(self, *args: P.args, **kwagrs: P.kwargs) -> OutT: ...

        modules = nn.Module.modules
        named_modules = nn.Module.named_modules
        state_dict = nn.Module.state_dict
        zero_grad = nn.Module.zero_grad
        parameters = nn.Module.parameters
        named_parameters = nn.Module.named_parameters
        cuda = nn.Module.cuda
        cpu = nn.Module.cpu
        # note: the overloads on nn.Module.to cause a bug with missing `self`.
        # This shouldn't be a problem.
        to = nn.Module().to


BatchType = TypeVar("BatchType", covariant=True)


@runtime_checkable
class DataModule(Protocol[BatchType]):
    """Protocol that shows the minimal attributes / methods of the `LightningDataModule` class.

    This is used to type hint the batches that are yielded by the DataLoaders.
    """

    def prepare_data(self) -> None: ...

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None: ...

    def train_dataloader(self) -> DataLoader[BatchType]: ...


@runtime_checkable
class ClassificationDataModule(DataModule[BatchType], Protocol):
    """Protocol that matches "datamodules with a 'num_classes' int attribute."""

    num_classes: int
