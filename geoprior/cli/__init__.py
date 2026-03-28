# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""Command-line interface for GeoPrior-v3.

This subpackage exposes one versatile root entry point and
its public wrapper modules.

Examples
--------
Use the root dispatcher with an explicit family::

    geoprior run stage1-preprocess
    geoprior build full-inputs-npz --stage1-dir results/foo_stage1

Use the family-specific console scripts directly::

    geoprior-run stage4-infer --help
    geoprior-build full-inputs
    geoprior-plot <command>
    geoprior-init --yes

Notes
-----
The package lazily exposes public CLI wrapper modules so
``geoprior.cli.<name>`` works with Sphinx autosummary
without importing private execution backends such as
``geoprior.cli._stage2`` at package import time.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Final

_MAIN_EXPORTS: Final = {
    "main": (".__main__", "main"),
    "run_main": (".__main__", "run_main"),
    "build_main": (".__main__", "build_main"),
    "plot_main": (".__main__", "plot_main"),
}

_PUBLIC_MODULES: Final = {
    "config",
    "init_config",
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "run_sensitivity",
    "run_sm3_suite",
    "sensitivity_lib",
    "sm3_collect_summaries",
    "sm3_log_offsets_diagnostics",
    "sm3_synthetic_identifiability",
    "build_add_zsurf_from_coords",
    "build_assign_boreholes",
    "build_external_validation_fullcity",
    "build_external_validation_metrics",
    "build_full_inputs_npz",
    "build_physics_payload_npz",
    "build_sm3_collect_summaries",
}

__all__ = [
    "main",
    "run_main",
    "build_main",
    "plot_main",
    "config",
    "init_config",
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "run_sensitivity",
    "run_sm3_suite",
    "sensitivity_lib",
    "sm3_collect_summaries",
    "sm3_log_offsets_diagnostics",
    "sm3_synthetic_identifiability",
    "build_add_zsurf_from_coords",
    "build_assign_boreholes",
    "build_external_validation_fullcity",
    "build_external_validation_metrics",
    "build_full_inputs_npz",
    "build_physics_payload_npz",
    "build_sm3_collect_summaries",
]


def _load_module(name: str) -> ModuleType:
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module


def _load_export(name: str):
    mod_name, attr_name = _MAIN_EXPORTS[name]
    module = import_module(mod_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __getattr__(name: str):
    if name in _MAIN_EXPORTS:
        return _load_export(name)
    if name in _PUBLIC_MODULES:
        return _load_module(name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
