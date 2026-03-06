from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest


MODULE_CANDIDATES = {
    "identifiability": [
        "geoprior.models.subsidence.identifiability",
    ],
    "utils": [
        "geoprior.models.subsidence.utils",
    ],
    "scaling": [
        "geoprior.models.subsidence.scaling",
    ],
    "payloads": [
        "geoprior.models.subsidence.payloads",
    ],
    "losses": [
        "geoprior.models.subsidence.losses",
    ],
    "maths": [
        "geoprior.models.subsidence.maths",
    ],
    "models": [
        "geoprior.models.subsidence.models",
    ],
}


def import_module_group(key: str):
    errs: list[str] = []
    for name in MODULE_CANDIDATES[key]:
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover
            errs.append(f"{name}: {type(exc).__name__}: {exc}")
    pytest.skip(
        f"Could not import module group {key!r}. Tried: "
        + " | ".join(errs)
    )


def to_scalar(x):
    if hasattr(x, "numpy"):
        x = x.numpy()
    arr = np.asarray(x)
    if arr.shape == ():
        return arr.item()
    return arr


class DummyLayer:
    def __init__(self, trainable: bool = True, name: str = "layer"):
        self.trainable = trainable
        self.name = name


class DummyModel(SimpleNamespace):
    pass
