# SPDX-License-Identifier: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>
# Adapted from: earthai-tech/gofast
# Modified for GeoPrior-v3 API conventions.

"""
Common type aliases used across GeoPrior-v3.

This module provides lightweight, runtime-safe aliases for
tabular data, array-like inputs, optional deep-learning
backends, and small callable helpers.

Notes
-----
This file is designed to be safe for runtime imports during
Sphinx autodoc and autosummary builds.

The guiding rule is:

- keep precise unions under ``TYPE_CHECKING``
- keep runtime aliases simple and import-safe

This avoids failures when optional backends such as PyTorch,
TensorFlow, or JAX are absent or partially initialized.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from collections.abc import Iterator as AbcIterator
from re import Pattern
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    SupportsInt,
    TypeAlias,
    TypeVar,
)

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Core tabular / array aliases
# ------------------------------------------------------------------
DataFrame = pd.DataFrame
Series = pd.Series
NDArray = np.ndarray

_T = TypeVar("_T")
_V = TypeVar("_V")

_ARRAYLIKE_TYPES = (
    np.ndarray,
    pd.Series,
    list,
    tuple,
)

# ------------------------------------------------------------------
# Static-only backend typing
# ------------------------------------------------------------------
if TYPE_CHECKING:
    try:
        import jax.numpy as jnp
    except Exception:
        jnp = None

    try:
        import torch
        import torch.nn as torch_nn
        from torch import Tensor as TorchTensor
        from torch.optim import Optimizer as TorchOptimizer
        from torch.utils.data import Dataset as TorchDataset

        TorchModel = torch_nn.Module
        TorchSequential = torch_nn.Sequential
    except Exception:
        torch = None
        TorchTensor = Any
        TorchDataset = Any
        TorchOptimizer = Any
        TorchModel = Any
        TorchSequential = Any

    try:
        import tensorflow as tf
        from tensorflow.data import Dataset as TFDataset
        from tensorflow.keras import Model as TFModel
        from tensorflow.keras import (
            Sequential as TFSequential,
        )
        from tensorflow.keras.callbacks import (
            Callback as TFCallback,
        )
        from tensorflow.keras.optimizers import (
            Optimizer as TFOptimizer,
        )

        TFTensor = tf.Tensor
    except Exception:
        tf = None
        TFTensor = Any
        TFDataset = Any
        TFOptimizer = Any
        TFCallback = Any
        TFModel = Any
        TFSequential = Any

    JNPNDArray = jnp.ndarray if jnp is not None else Any

    ArrayLike: TypeAlias = (
        NDArray | Series | list[Any] | tuple[Any, ...]
    )

    _Sub: TypeAlias = Callable[[Any], Any]
    _F: TypeAlias = Callable[[ArrayLike], Any]

    _Tensor: TypeAlias = TorchTensor | TFTensor | JNPNDArray
    _Dataset: TypeAlias = TorchDataset | TFDataset
    _Optimizer: TypeAlias = TorchOptimizer | TFOptimizer
    _Callback: TypeAlias = TFCallback
    _Model: TypeAlias = TorchModel | TFModel
    _Sequential: TypeAlias = TorchSequential | TFSequential

# ------------------------------------------------------------------
# Runtime-safe aliases
# ------------------------------------------------------------------
else:
    TorchTensor = Any
    TorchDataset = Any
    TorchOptimizer = Any
    TorchModel = Any
    TorchSequential = Any

    TFTensor = Any
    TFDataset = Any
    TFOptimizer = Any
    TFCallback = Any
    TFModel = Any
    TFSequential = Any

    JNPNDArray = Any

    from collections.abc import Callable as _RuntimeCallable

    ArrayLike = _ARRAYLIKE_TYPES
    _Sub = _RuntimeCallable
    _F = _RuntimeCallable

    _Tensor = Any
    _Dataset = Any
    _Optimizer = Any
    _Callback = Any
    _Model = Any
    _Sequential = Any

# ------------------------------------------------------------------
# Extra public typing helpers
# ------------------------------------------------------------------
Iterator = AbcIterator[Any]

MultioutputLiteral = Literal[
    "raw_values",
    "uniform_average",
]
NanPolicyLiteral = Literal[
    "omit",
    "propagate",
    "raise",
]
MetricFunctionType = Callable[..., float | np.ndarray]
MetricType = Literal[
    "mae",
    "accuracy",
    "interval_score",
]
PlotKind = Literal[
    "time_profile",
    "summary_bar",
]
PlotKindWIS = Literal[
    "scores_histogram",
    "summary_bar",
]
PlotKindTheilU = Literal["summary_bar"]


# ------------------------------------------------------------------
# Small predicates
# ------------------------------------------------------------------
def is_dataframe(obj: Any) -> bool:
    """
    Check whether an object is a pandas DataFrame.

    Parameters
    ----------
    obj : Any
        Object to test.

    Returns
    -------
    bool
        True when ``obj`` is a DataFrame.
    """
    return isinstance(obj, pd.DataFrame)


def is_series(obj: Any) -> bool:
    """
    Check whether an object is a pandas Series.

    Parameters
    ----------
    obj : Any
        Object to test.

    Returns
    -------
    bool
        True when ``obj`` is a Series.
    """
    return isinstance(obj, pd.Series)


def is_ndarray(obj: Any) -> bool:
    """
    Check whether an object is a NumPy ndarray.

    Parameters
    ----------
    obj : Any
        Object to test.

    Returns
    -------
    bool
        True when ``obj`` is an ndarray.
    """
    return isinstance(obj, np.ndarray)


def is_array_like(obj: Any) -> bool:
    """
    Check whether an object is array-like.

    Parameters
    ----------
    obj : Any
        Object to test.

    Returns
    -------
    bool
        True when ``obj`` is a NumPy array, pandas Series,
        list, or tuple.
    """
    return isinstance(obj, _ARRAYLIKE_TYPES)


# ------------------------------------------------------------------
# Small callable helpers
# ------------------------------------------------------------------
def apply_function(
    f: _F,
    data: ArrayLike,
) -> Any:
    """
    Apply a callable to an array-like input.

    Parameters
    ----------
    f : _F
        Callable to apply.

    data : ArrayLike
        Input data.

    Returns
    -------
    Any
        Result returned by ``f(data)``.
    """
    return f(data)


def transform_data(
    f: _Sub,
    data: Any,
) -> Any:
    """
    Apply a transformation callable to any input.

    Parameters
    ----------
    f : _Sub
        Transformation callable.

    data : Any
        Input data.

    Returns
    -------
    Any
        Transformed output.
    """
    return f(data)


__all__ = [
    "DataFrame",
    "Series",
    "NDArray",
    "ArrayLike",
    "is_dataframe",
    "is_series",
    "is_ndarray",
    "is_array_like",
    "_Sub",
    "_F",
    "apply_function",
    "transform_data",
    "_Tensor",
    "_Dataset",
    "_Optimizer",
    "_Callback",
    "_Model",
    "_Sequential",
    "_T",
    "_V",
    "Any",
    "Callable",
    "Optional",
    "Iterable",
    "Pattern",
    "SupportsInt",
    "Iterator",
    "MultioutputLiteral",
    "NanPolicyLiteral",
    "MetricFunctionType",
    "MetricType",
    "PlotKind",
    "PlotKindWIS",
    "PlotKindTheilU",
]
