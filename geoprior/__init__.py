# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026-present

"""
┌─────────────────────────────────────────────────────────────┐
│                         GeoPrior-v3                         │
├─────────────────────────────────────────────────────────────┤
│ Physics-guided AI for geohazards & risk analytics           │
├─────────────┬───────────────────────────────────────────────┤
│ Today       │ Land subsidence (GeoPriorSubsNet v3.x)        │
│ Next        │ Landslides and broader geohazard modeling     │
├─────────────┼───────────────────────────────────────────────┤
│ Quick Start │ >>> import geoprior as gp                     │
│             │ >>> # gp.cli / scripts drive the paper pipeline│
├─────────────┼───────────────────────────────────────────────┤
│ Source      │ https://github.com/earthai-tech/geoprior-v3   │
│ Web         │ https://lkouadio.com                          │
└─────────────┴───────────────────────────────────────────────┘
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import types
import warnings


# Keep import-time noise minimal (mirrors fusionlab init pattern).
logging.basicConfig(level=logging.WARNING)
logging.getLogger("matplotlib.font_manager").disabled = True


def _lazy_import(module_name: str, alias: str | None = None):
    """
    Lazily import a module to reduce initial package load time.
    """
    def _lazy_loader():
        return importlib.import_module(module_name)

    if alias:
        globals()[alias] = _lazy_loader
    else:
        globals()[module_name] = _lazy_loader


# Package version
try:
    from ._version import version as __version__
except Exception:
    __version__ = "3.2.0"


# Core dependencies (mirrors fusionlab-learn list; adjusted for PyYAML).
_required_dependencies: list[tuple[str, str | None]] = [
    ("numpy", None),
    ("pandas", None),
    ("scipy", None),
    ("matplotlib", None),
    ("tqdm", None),
    ("sklearn", "scikit-learn"),
    ("joblib", None),
    ("tensorflow", None),
    ("statsmodels", None),
    ("yaml", "pyyaml"),
    ("platformdirs", None),
    ("click", None),
    ("lz4", None),
    ("psutil", None),
]

_missing: list[str] = []
for import_name, pkg_name in _required_dependencies:
    try:
        _lazy_import(import_name)
    except Exception as e:
        name = pkg_name or import_name
        _missing.append(f"{name}: {e}")

if _missing:
    warnings.warn(
        "Some GeoPrior dependencies are missing; "
        "functionality may be limited:\n"
        + "\n".join(_missing),
        ImportWarning,
        stacklevel=2,
    )


# Warning controls .
_WARNING_CATEGORIES = {
    "FutureWarning": FutureWarning,
    "SyntaxWarning": SyntaxWarning,
}
_WARN_ACTIONS = {
    "FutureWarning": "ignore",
    "SyntaxWarning": "ignore",
}


def suppress_warnings(suppress: bool = True) -> None:
    """
    Globally suppress or re-enable FutureWarning and SyntaxWarning.
    """
    for name, cat in _WARNING_CATEGORIES.items():
        action = _WARN_ACTIONS.get(name, "default")
        warnings.filterwarnings(
            action if suppress else "default",
            category=cat,
        )


# Suppress by default on import
suppress_warnings()


# Reduce TensorFlow import logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# Initialize structured logging for GeoPrior if available.
try:
    from ._util import initialize_logging  # optional glue
    try:
        initialize_logging()
    except Exception:
        pass
except Exception:
    pass


__all__ = [
    "__version__",
    "suppress_warnings",
]


# Optional: bridge to k-diagram if installed
try:
    _kd = importlib.import_module("kdiagram")
    sys.modules[__name__ + ".kdiagram"] = _kd
    kdiagram = _kd
    __all__.append("kdiagram")
except Exception:
    pass

if __name__ + ".kdiagram" not in sys.modules:
    _dummy = types.ModuleType(__name__ + ".kdiagram")
    sys.modules[__name__ + ".kdiagram"] = _dummy


def __getattr__(name: str):
    if name == "kdiagram":
        hint = (
            "The optional submodule 'geoprior.kdiagram' is unavailable "
            "because `k-diagram` is not installed.\n\n"
            "Install it with:\n\n"
            "    pip install geoprior-v3[kdiagram]\n\n"
            "Or:\n"
            "    pip install k-diagram\n"
        )
        warnings.warn(hint, ImportWarning, stacklevel=2)
        raise AttributeError(
            "geoprior.kdiagram is not available. "
            "See warning above for install instructions."
        )

    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'"
    )


__doc__ = (__doc__ or "") + f"\nVersion: {__version__}\n"