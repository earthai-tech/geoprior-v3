# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# https://lkouadio.com
# Copyright (c) 2026-present Laurent Kouadio
# Author: LKouadio <etanoyau@gmail.com>

"""
Provides a compatibility layer for Python typing
features across supported Python versions.

The module re-exports common typing-related names
used across the codebase. Deprecated container
aliases from ``typing`` are mapped to their modern
built-in or stdlib equivalents.
"""

from collections import deque
from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import AbstractContextManager
from re import Pattern
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    SupportsInt,
    TypedDict,
    TypeVar,
    Union,
)

try:
    from typing import TypeGuard
except ImportError:
    from typing import TypeGuard


if TYPE_CHECKING:
    # Keep conservative and lightweight.
    # we later settle on one canonical backend type,
    # we can refine this.
    TensorLike = Any
    DatasetLike = Any
else:
    TensorLike = Any
    DatasetLike = Any

# Modern aliases kept for backward compatibility.
List = list
Tuple = tuple
Dict = dict
Set = set
FrozenSet = frozenset
Type = type
Text = str
Deque = deque
ContextManager = AbstractContextManager


__all__ = [
    "List",
    "Tuple",
    "Sequence",
    "Dict",
    "Iterable",
    "Callable",
    "Union",
    "Any",
    "Generic",
    "Optional",
    "Type",
    "Mapping",
    "Text",
    "TypeVar",
    "Iterator",
    "SupportsInt",
    "Set",
    "ContextManager",
    "Deque",
    "FrozenSet",
    "NamedTuple",
    "NewType",
    "TypedDict",
    "Generator",
    "TypeGuard",
    "Pattern",
    "Literal",
    "TensorLike",
    "DatasetLike",
]
