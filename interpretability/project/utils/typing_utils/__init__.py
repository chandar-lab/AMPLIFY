"""Utilities to help annotate the types of values in the project."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NewType, TypeGuard

from hydra_zen.typing import Builds
from typing_extensions import TypeVar

from .protocols import DataModule

# These are used to show which dim is which.
C = NewType("C", int)
H = NewType("H", int)
W = NewType("W", int)


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

HydraConfigFor = Builds[type[T]]
"""Type annotation to say "a hydra config that returns an object of type T when instantiated"."""


NestedMapping = Mapping[K, V | "NestedMapping[K, V]"]
PyTree = T | Iterable["PyTree[T]"] | Mapping[Any, "PyTree[T]"]


def is_sequence_of(
    object: Any, item_type: type[V] | tuple[type[V], ...]
) -> TypeGuard[Sequence[V]]:
    """Used to check (and tell the type checker) that `object` is a sequence of items of this
    type."""
    return isinstance(object, Sequence) and all(isinstance(value, item_type) for value in object)


def is_mapping_of(object: Any, key_type: type[K], value_type: type[V]) -> TypeGuard[Mapping[K, V]]:
    """Used to check (and tell the type checker) that `object` is a mapping with keys and values of
    the given types."""
    return isinstance(object, Mapping) and all(
        isinstance(key, key_type) and isinstance(value, value_type)
        for key, value in object.items()
    )


__all__ = [
    "DataModule",
]
