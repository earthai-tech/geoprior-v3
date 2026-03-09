# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# https://lkouadio.com
# Copyright (c) 2026-present Laurent Kouadio
# Author: LKouadio <etanoyau@gmail.com>

"""
Provides a compatibility layer for Python typing features,
ensuring support across different Python versions.

It imports various typing constructs from the built-in `typing` module.
For the `TypeGuard` feature, which is available in Python 3.10 and later,
it attempts to import from `typing_extensions` if not found in the built-in module.
"""

from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from re import Pattern

# Check if Python version is 3.10 or higher
from typing import (
    Any,
    ContextManager,
    Deque,
    Dict,
    FrozenSet,
    Generic,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Set,
    SupportsInt,
    Text,
    Tuple,
    Type,
    TypedDict,
    TypeGuard,
    TypeVar,
    Union,
)

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
]
