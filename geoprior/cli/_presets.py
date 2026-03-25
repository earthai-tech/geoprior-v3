# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""Reusable CLI preset definitions.

This module keeps named preset bundles separate from the command
orchestration code so they can be reused by multiple CLI workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SM3_REGIMES: tuple[str, ...] = (
    "none",
    "base",
    "anchored",
    "closure_locked",
    "data_relaxed",
)


@dataclass(frozen=True)
class SM3Preset:
    """Named SM3 suite preset."""

    name: str
    identify: str
    suite_prefix: str
    params: dict[str, Any] = field(default_factory=dict)

    def merged(self, **overrides: Any) -> dict[str, Any]:
        """Return preset parameters with non-``None`` overrides."""
        out = dict(self.params)
        out["identify"] = self.identify
        out["suite_prefix"] = self.suite_prefix
        for key, value in overrides.items():
            if value is not None:
                out[key] = value
        return out


SM3_PRESETS: dict[str, SM3Preset] = {
    "tau50": SM3Preset(
        name="tau50",
        identify="tau",
        suite_prefix="sm3_tau_suite",
        params={
            "n_realizations": 50,
            "n_years": 25,
            "time_steps": 5,
            "forecast_horizon": 3,
            "val_tail": 5,
            "epochs": 40,
            "batch": 64,
            "lr": 1e-3,
            "patience": 5,
            "noise_std": 0.02,
            "load_type": "step",
            "tau_min": 0.3,
            "tau_max": 10.0,
            "tau_spread_dex": 0.35,
            "Ss_spread_dex": 0.45,
            "K_spread_dex": None,
            "alpha": 1.0,
            "hd_factor": 0.6,
            "thickness_cap": 30.0,
            "kappa_b": 1.0,
            "gamma_w": 9810.0,
            "scenario": "base",
            "nx": 21,
            "Lx_m": 5000.0,
            "h_right": 0.0,
            "device": "auto",
            "fast": 1,
            "seed": 123,
            "start_realisation": 1,
        },
    ),
    "both50": SM3Preset(
        name="both50",
        identify="both",
        suite_prefix="sm3_both_suite",
        params={
            "n_realizations": 50,
            "n_years": 25,
            "time_steps": 5,
            "forecast_horizon": 3,
            "val_tail": 5,
            "epochs": 40,
            "batch": 64,
            "lr": 1e-3,
            "patience": 5,
            "noise_std": 0.02,
            "load_type": "step",
            "tau_min": 0.3,
            "tau_max": 10.0,
            "tau_spread_dex": 0.35,
            "Ss_spread_dex": 0.45,
            "K_spread_dex": 0.6,
            "alpha": 1.0,
            "hd_factor": 0.6,
            "thickness_cap": 30.0,
            "kappa_b": 1.0,
            "gamma_w": 9810.0,
            "scenario": "base",
            "nx": 21,
            "Lx_m": 5000.0,
            "h_right": 0.0,
            "device": "auto",
            "fast": 1,
            "seed": 123,
            "start_realisation": 1,
        },
    ),
}


def get_sm3_preset(name: str) -> SM3Preset:
    """Return a registered SM3 preset by name."""
    key = str(name).strip().lower()
    try:
        return SM3_PRESETS[key]
    except KeyError as exc:
        known = ", ".join(sorted(SM3_PRESETS))
        raise KeyError(
            f"Unknown SM3 preset: {name!r}. Known presets: {known}."
        ) from exc
