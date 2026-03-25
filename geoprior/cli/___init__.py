# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""Command-line interface for GeoPrior-v3.

This subpackage exposes one versatile root entry point and dedicated
family entry points.

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
"""

from __future__ import annotations

from .__main__ import build_main, main, plot_main, run_main

__all__ = [
    "main",
    "run_main",
    "build_main",
    "plot_main",
]
