# SPDX-License-Identifier: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>
# Adapted from: earthai-tech/gofast — https://github.com/earthai-tech/gofast
# Modified for GeoPrior-v3 API conventions.

"""
This module defines common type aliases used throughout the FusionLab package.
It includes types for handling pandas DataFrames, Series, numpy NDArray, and
Array-like objects, which are frequently encountered in data science and machine
learning tasks.

These types are designed to aid static type checking, ensuring compatibility
across different functions and classes within the package.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterable,
)
from collections.abc import (
    Iterator as AbcIterator,
)
from re import Pattern
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    SupportsInt,
    TypeVar,
)

import numpy as np
import pandas as pd

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

# --- PyTorch types ---
try:
    import torch
    import torch.nn as torch_nn
    from torch import Tensor as TorchTensor
    from torch.optim import Optimizer as TorchOptimizer
    from torch.utils.data import Dataset as TorchDataset

    TorchModel = torch_nn.Module
    TorchSequential = torch_nn.Sequential

except ImportError:
    torch = None

    class TorchTensor:
        def __init__(*args, **kwargs):
            raise ImportError(
                "PyTorch is required for TorchTensor"
                " but is not installed."
            )

    TorchDataset = None
    TorchOptimizer = None

    class TorchModel:
        def __init__(*args, **kwargs):
            raise ImportError(
                "PyTorch is required for TorchModel"
                " but is not installed."
            )

    class TorchSequential:
        def __init__(*args, **kwargs):
            raise ImportError(
                "PyTorch is required for TorchSequential"
                " but is not installed."
            )


# --- TensorFlow types ---
try:
    import tensorflow as tf
    from tensorflow.data import Dataset as TFDataset
    from tensorflow.keras import Model as TFModel
    from tensorflow.keras import Sequential as TFSequential
    from tensorflow.keras.callbacks import (
        Callback as TFCallback,
    )
    from tensorflow.keras.optimizers import (
        Optimizer as TFOptimizer,
    )

    TFTensor = tf.Tensor

except ImportError:
    tf = None

    class TFTensor:
        def __init__(*args, **kwargs):
            raise ImportError(
                "TensorFlow is required for TFTensor"
                " but is not installed."
            )

    TFDataset = None
    TFOptimizer = None
    TFCallback = None

    class TFModel:
        def __init__(*args, **kwargs):
            raise ImportError(
                "TensorFlow is required for TFModel"
                " but is not installed."
            )

    class TFSequential:
        def __init__(*args, **kwargs):
            raise ImportError(
                "TensorFlow is required for TFSequential but"
                " is not installed."
            )


# Type aliases for common data structures
DataFrame = pd.DataFrame
Series = pd.Series
NDArray = np.ndarray
_T = TypeVar("_T")
_V = TypeVar("_V")

# Type aliases for callable functions and operations
if TYPE_CHECKING:
    ArrayLike = NDArray | Series | list[Any] | tuple[Any, ...]
    _Sub = Callable[[Any], Any]
    _F = Callable[[ArrayLike], Any]
else:
    # At runtime we avoid the subscript, so
    # Callable[[...], ...] is not evaluated.
    from collections.abc import Callable as _RuntimeCallable

    ArrayLike = (
        list,
        tuple,
        np.ndarray,
        pd.Series,
    )  # just a marker
    _Sub = _RuntimeCallable
    _F = _RuntimeCallable


# --- Multi-framework type aliases ---
JNPNDArray = jnp.ndarray if jnp else None
_Tensor = TorchTensor | TFTensor | JNPNDArray
_Dataset = TorchDataset | TFDataset
_Optimizer = TorchOptimizer | TFOptimizer
_Callback = (
    TFCallback if TFCallback is not None else type(None)
)
_Model = TorchModel | TFModel
_Sequential = TorchSequential | TFSequential

# Type aliases for additional Python built-in types
Iterator = AbcIterator[Any]

# Define MultioutputLiteral for type hinting
# if not using StrOptions directly in hints
MultioutputLiteral = Literal["raw_values", "uniform_average"]
NanPolicyLiteral = Literal["omit", "propagate", "raise"]
MetricFunctionType = Callable[..., float | np.ndarray]
MetricType = Literal["mae", "accuracy", "interval_score"]
PlotKind = Literal["time_profile", "summary_bar"]
PlotKindWIS = Literal["scores_histogram", "summary_bar"]
PlotKindTheilU = Literal["summary_bar"]


def is_dataframe(obj: Any) -> bool:
    """
    Check if an object is a pandas DataFrame.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    bool
        True if the object is a pandas DataFrame, False otherwise.
    """
    return isinstance(obj, pd.DataFrame)


def is_series(obj: Any) -> bool:
    """
    Check if an object is a pandas Series.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    bool
        True if the object is a pandas Series, False otherwise.
    """
    return isinstance(obj, pd.Series)


def is_ndarray(obj: Any) -> bool:
    """
    Check if an object is a numpy ndarray.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    bool
        True if the object is a numpy ndarray, False otherwise.
    """
    return isinstance(obj, np.ndarray)


def is_array_like(obj: Any) -> bool:
    """
    Check if an object is array-like
    (e.g., numpy ndarray, pandas Series, list, or tuple).

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    bool
        True if the object is array-like, False otherwise.
    """
    return isinstance(
        obj, np.ndarray | pd.Series | list | tuple
    )


def apply_function(f: _F, data: ArrayLike) -> Any:
    """
    Apply a callable function (e.g., np.mean, np.sum)
    to an array-like structure.

    Parameters
    ----------
    f : _F
        The callable function (e.g., np.mean, np.sum).

    data : ArrayLike
        The data to which the function is applied.

    Returns
    -------
    Any
        The result of applying the function to the data.
    """
    return f(data)


def transform_data(f: _Sub, data: Any) -> Any:
    """
    Apply a transformation to the data
    using a callable function.

    Parameters
    ----------
    f : _Sub
        The callable function to transform the data
        (e.g., a lambda function).

    data : Any
        The data to which the transformation is applied.

    Returns
    -------
    Any
        The transformed data.
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
]
