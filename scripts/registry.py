# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""Compatibility registry for legacy ``python -m scripts`` runs.

The authoritative reproducibility registry now lives under
``geoprior.scripts``. This module re-exports the public objects so
legacy dispatch continues to work without duplicating the registry.
"""

from __future__ import annotations

from geoprior.scripts.registry import (
    SCRIPT_COMMANDS,
    SCRIPT_GROUPS,
    ScriptSpec,
)

__all__ = [
    "ScriptSpec",
    "SCRIPT_COMMANDS",
    "SCRIPT_GROUPS",
]
