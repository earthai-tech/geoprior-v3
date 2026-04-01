# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Run-manifest generation and inspection helpers.

This module focuses on the lightweight
``run_manifest.json`` artifact written during
Stage-2 training. It provides a compact workflow-
level snapshot that downstream steps can inspect
without reopening larger artifacts.

A run manifest typically captures:

- run identity (stage, city, model),
- a small training/config snapshot,
- the most important Stage-2 output paths,
- direct artifact pointers such as the training
  summary and CSV log.

The helpers here are intended for two common uses:

1. Sphinx-Gallery examples that need a realistic
   Stage-2 run-manifest without re-running training.
2. Real workflow inspection when a user wants to
   verify that a Stage-2 run exported the expected
   bundle, paths, and lightweight config.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .utils import (
    ArtifactRecord,
    clone_artifact,
    load_artifact,
    nested_get,
    plot_boolean_checks,
    plot_metric_bars,
    read_json,
    write_json,
)

PathLike = str | Path
RunManifestLike = (
    ArtifactRecord | Mapping[str, Any] | str | Path
)

__all__ = [
    "default_run_manifest_payload",
    "generate_run_manifest",
    "inspect_run_manifest",
    "load_run_manifest",
    "plot_run_manifest_boolean_summary",
    "plot_run_manifest_coord_ranges",
    "plot_run_manifest_feature_group_sizes",
    "plot_run_manifest_path_inventory",
    "run_manifest_artifacts_frame",
    "run_manifest_config_frame",
    "run_manifest_identity_frame",
    "run_manifest_paths_frame",
    "run_manifest_scaling_overview_frame",
    "summarize_run_manifest",
]

_CONFIG_KEYS = [
    "TIME_STEPS",
    "FORECAST_HORIZON_YEARS",
    "MODE",
    "PDE_MODE_CONFIG",
    "identifiability_regime",
]

_SCALING_KEYS = [
    "time_units",
    "coords_normalized",
    "coords_in_degrees",
    "gwl_dyn_index",
    "gwl_kind",
    "gwl_sign",
    "use_head_proxy",
    "z_surf_col",
]


def _as_payload(manifest: RunManifestLike) -> dict[str, Any]:
    """Return a plain run-manifest payload."""
    if isinstance(manifest, ArtifactRecord):
        return dict(manifest.payload)

    if isinstance(manifest, Mapping):
        return dict(manifest)

    return dict(read_json(manifest))


def _path_count(payload: dict[str, Any]) -> int:
    """Return the number of exported path entries."""
    paths = payload.get("paths", None)
    if not isinstance(paths, dict):
        return 0
    return len(paths)


def _artifact_count(payload: dict[str, Any]) -> int:
    """Return the number of direct artifact pointers."""
    artifacts = payload.get("artifacts", None)
    if not isinstance(artifacts, dict):
        return 0
    return len(artifacts)


def _feature_count(payload: dict[str, Any], key: str) -> int:
    """Return the size of a scaling feature-name list."""
    names = nested_get(
        payload,
        "config",
        "scaling_kwargs",
        key,
        default=[],
    )
    if isinstance(names, list):
        return len(names)
    return 0


def _coord_ranges(
    payload: dict[str, Any],
) -> dict[str, float]:
    """Return numeric coordinate ranges from nested scaling kwargs."""
    src = nested_get(
        payload,
        "config",
        "scaling_kwargs",
        "coord_ranges",
        default={},
    )
    if not isinstance(src, dict):
        return {}

    out: dict[str, float] = {}
    for key, value in src.items():
        try:
            if value is None:
                continue
            out[str(key)] = float(value)
        except Exception:
            continue
    return out


def default_run_manifest_payload(
    *,
    city: str = "nansha",
    model: str = "GeoPriorSubsNet",
    stage: str = "stage-2-train",
    time_steps: int = 5,
    forecast_horizon_years: int = 3,
    mode: str = "tft_like",
    pde_mode_config: str = "on",
    quantiles: list[float] | None = None,
    identifiability_regime: str | None = None,
) -> dict[str, Any]:
    """
    Build a realistic default run-manifest payload.

    The payload mirrors the lightweight Stage-2 run
    manifest saved after training:
    ``stage`` + ``city`` + ``model`` + ``config`` +
    ``paths`` + ``artifacts``.
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    run_dir = (
        f"results/{city}_{model}_stage1/train_20260222-141331"
    )
    prefix = f"{city}_{model}_H{forecast_horizon_years}"

    payload = {
        "stage": str(stage),
        "city": str(city),
        "model": str(model),
        "config": {
            "TIME_STEPS": int(time_steps),
            "FORECAST_HORIZON_YEARS": int(
                forecast_horizon_years
            ),
            "MODE": str(mode),
            "ATTENTION_LEVELS": [
                "cross",
                "hierarchical",
                "memory",
            ],
            "PDE_MODE_CONFIG": str(pde_mode_config),
            "QUANTILES": list(quantiles),
            "scaling_kwargs": {
                "bounds": {
                    "H_min": 0.1,
                    "H_max": 30.0,
                    "K_min": 1e-12,
                    "K_max": 1e-7,
                    "Ss_min": 1e-6,
                    "Ss_max": 1e-3,
                    "tau_min": 604800.0,
                    "tau_max": 9467085600.0,
                    "logK_min": -27.6310211159,
                    "logK_max": -16.1180956510,
                    "logSs_min": -13.8155105580,
                    "logSs_max": -6.9077552790,
                    "logTau_min": 13.3126531038,
                    "logTau_max": 22.9710869460,
                },
                "time_units": "year",
                "subs_scale_si": 1.0,
                "subs_bias_si": 0.0,
                "head_scale_si": 1.0,
                "head_bias_si": 0.0,
                "H_scale_si": 1.0,
                "H_bias_si": 0.0,
                "coords_normalized": True,
                "coord_ranges": {
                    "t": 7.0,
                    "x": 44447.0,
                    "y": 39275.0,
                },
                "coords_in_degrees": False,
                "deg_to_m_lon": None,
                "deg_to_m_lat": None,
                "dynamic_feature_names": [
                    "GWL_depth_bgs_m__si",
                    "subsidence_cum__si",
                    "rainfall_mm",
                    "urban_load_global",
                    "soil_thickness_censored",
                ],
                "future_feature_names": ["rainfall_mm"],
                "static_feature_names": [
                    "lithology_Conglomerate–Sandstone",
                    "lithology_Limestone–Sandstone",
                    "lithology_Mudstone–Siltstone",
                    "lithology_Sandstone–Siltstone",
                    "lithology_Shale–Limestone",
                    "lithology_Siltstone–Sandstone",
                    "lithology_Siltstone–Shale",
                    "lithology_Tuff–Sandstone",
                    "lithology_class_Coarse-Grained Soil",
                    "lithology_class_Fine-Grained Soil",
                    "lithology_class_Mixed Clastics",
                    "z_surf_m__si",
                ],
                "gwl_dyn_name": "GWL_depth_bgs_m__si",
                "gwl_dyn_index": 0,
                "coord_order": ["t", "x", "y"],
                "gwl_kind": "depth_bgs",
                "gwl_sign": "down_positive",
                "use_head_proxy": True,
                "z_surf_col": "z_surf_m__si",
            },
            "identifiability_regime": identifiability_regime,
        },
        "paths": {
            "run_dir": run_dir,
            "weights_h5": f"{run_dir}/{prefix}.weights.h5",
            "arch_json": f"{run_dir}/{city}_{model}_architecture.json",
            "csv_log": f"{run_dir}/{city}_{model}_train_log.csv",
            "best_keras": f"{run_dir}/{prefix}_best.keras",
            "best_weights": f"{run_dir}/{prefix}_best.weights.h5",
            "model_init_manifest": f"{run_dir}/model_init_manifest.json",
            "final_keras": f"{run_dir}/{prefix}_final.keras",
        },
        "artifacts": {
            "training_summary_json": (
                f"{run_dir}/{city}_{model}_training_summary.json"
            ),
            "train_log_csv": f"{run_dir}/{city}_{model}_train_log.csv",
        },
    }
    return payload


def generate_run_manifest(
    path: PathLike,
    *,
    template: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Path:
    """
    Generate a run-manifest JSON artifact.

    Parameters
    ----------
    path : str or pathlib.Path
        Output JSON path.
    template : mapping, optional
        Base payload. If omitted, uses
        :func:`default_run_manifest_payload`.
    overrides : mapping, optional
        Nested overrides applied on top of the template.
    """
    base = (
        dict(template)
        if template is not None
        else default_run_manifest_payload()
    )
    payload = clone_artifact(
        base, overrides=dict(overrides or {})
    )
    return write_json(payload, path)


def load_run_manifest(
    path: PathLike,
) -> ArtifactRecord:
    """Load a run-manifest as ``ArtifactRecord``."""
    return load_artifact(path, kind="run_manifest")


def run_manifest_identity_frame(
    manifest: RunManifestLike,
) -> pd.DataFrame:
    """Return a compact run-identity frame."""
    payload = _as_payload(manifest)
    rows = [
        {
            "section": "identity",
            "key": "stage",
            "value": payload.get("stage"),
        },
        {
            "section": "identity",
            "key": "city",
            "value": payload.get("city"),
        },
        {
            "section": "identity",
            "key": "model",
            "value": payload.get("model"),
        },
    ]
    return pd.DataFrame(rows)


def run_manifest_config_frame(
    manifest: RunManifestLike,
) -> pd.DataFrame:
    """Return a tidy frame for the lightweight config block."""
    payload = _as_payload(manifest)
    cfg = nested_get(payload, "config", default={})

    rows: list[dict[str, Any]] = []
    for key in _CONFIG_KEYS:
        rows.append(
            {
                "section": "config",
                "key": str(key),
                "value": (cfg or {}).get(key),
            }
        )

    rows.append(
        {
            "section": "config",
            "key": "ATTENTION_LEVELS",
            "value": (cfg or {}).get("ATTENTION_LEVELS"),
        }
    )
    rows.append(
        {
            "section": "config",
            "key": "QUANTILES",
            "value": (cfg or {}).get("QUANTILES"),
        }
    )
    return pd.DataFrame(rows)


def run_manifest_scaling_overview_frame(
    manifest: RunManifestLike,
) -> pd.DataFrame:
    """Return a compact frame for nested scaling overview."""
    payload = _as_payload(manifest)
    sk = nested_get(
        payload, "config", "scaling_kwargs", default={}
    )

    rows: list[dict[str, Any]] = []
    for key in _SCALING_KEYS:
        rows.append(
            {
                "section": "scaling_overview",
                "key": str(key),
                "value": (sk or {}).get(key),
            }
        )

    rows.append(
        {
            "section": "scaling_overview",
            "key": "dynamic_feature_count",
            "value": _feature_count(
                payload, "dynamic_feature_names"
            ),
        }
    )
    rows.append(
        {
            "section": "scaling_overview",
            "key": "future_feature_count",
            "value": _feature_count(
                payload, "future_feature_names"
            ),
        }
    )
    rows.append(
        {
            "section": "scaling_overview",
            "key": "static_feature_count",
            "value": _feature_count(
                payload, "static_feature_names"
            ),
        }
    )
    rows.append(
        {
            "section": "scaling_overview",
            "key": "bounds_count",
            "value": len((sk or {}).get("bounds", {}) or {}),
        }
    )
    return pd.DataFrame(rows)


def run_manifest_paths_frame(
    manifest: RunManifestLike,
) -> pd.DataFrame:
    """Return a frame describing exported run paths."""
    payload = _as_payload(manifest)
    paths = payload.get("paths", {})

    rows: list[dict[str, Any]] = []
    for key, value in (paths or {}).items():
        rows.append(
            {
                "section": "paths",
                "key": str(key),
                "value": value,
                "basename": Path(str(value)).name
                if value
                else None,
            }
        )
    return pd.DataFrame(rows)


def run_manifest_artifacts_frame(
    manifest: RunManifestLike,
) -> pd.DataFrame:
    """Return a frame describing direct artifact pointers."""
    payload = _as_payload(manifest)
    artifacts = payload.get("artifacts", {})

    rows: list[dict[str, Any]] = []
    for key, value in (artifacts or {}).items():
        rows.append(
            {
                "section": "artifacts",
                "key": str(key),
                "value": value,
                "basename": Path(str(value)).name
                if value
                else None,
            }
        )
    return pd.DataFrame(rows)


def summarize_run_manifest(
    manifest: RunManifestLike,
) -> dict[str, Any]:
    """
    Summarize a run manifest in a compact dictionary.

    The summary focuses on what a user usually wants at a
    glance: which Stage-2 run this is, how many paths were
    exported, whether feature groups are present, and the
    essential scaling/coord context.
    """
    payload = _as_payload(manifest)
    cfg = nested_get(payload, "config", default={})
    sk = nested_get(
        payload, "config", "scaling_kwargs", default={}
    )
    coord_ranges = _coord_ranges(payload)
    paths = payload.get("paths", {}) or {}

    summary = {
        "stage": payload.get("stage"),
        "city": payload.get("city"),
        "model": payload.get("model"),
        "time_steps": cfg.get("TIME_STEPS"),
        "forecast_horizon_years": cfg.get(
            "FORECAST_HORIZON_YEARS"
        ),
        "mode": cfg.get("MODE"),
        "pde_mode_config": cfg.get("PDE_MODE_CONFIG"),
        "identifiability_regime": cfg.get(
            "identifiability_regime"
        ),
        "quantiles": cfg.get("QUANTILES"),
        "attention_levels": cfg.get("ATTENTION_LEVELS"),
        "time_units": (sk or {}).get("time_units"),
        "coords_normalized": (sk or {}).get(
            "coords_normalized"
        ),
        "coords_in_degrees": (sk or {}).get(
            "coords_in_degrees"
        ),
        "coord_ranges": coord_ranges,
        "dynamic_feature_count": _feature_count(
            payload,
            "dynamic_feature_names",
        ),
        "future_feature_count": _feature_count(
            payload,
            "future_feature_names",
        ),
        "static_feature_count": _feature_count(
            payload,
            "static_feature_names",
        ),
        "bounds_count": len(
            (sk or {}).get("bounds", {}) or {}
        ),
        "path_count": _path_count(payload),
        "artifact_count": _artifact_count(payload),
        "has_final_keras": bool(
            (paths or {}).get("final_keras")
        ),
        "has_model_init_manifest": bool(
            (paths or {}).get("model_init_manifest")
        ),
        "has_best_keras": bool(
            (paths or {}).get("best_keras")
        ),
        "has_best_weights": bool(
            (paths or {}).get("best_weights")
        ),
    }
    return summary


def plot_run_manifest_path_inventory(
    ax: plt.Axes,
    manifest: RunManifestLike,
    *,
    title: str = "Run-manifest inventory",
    **plot_kws: Any,
) -> plt.Axes:
    """
    Plot path and artifact counts for the run manifest.
    """
    payload = _as_payload(manifest)
    metrics = {
        "paths": _path_count(payload),
        "artifacts": _artifact_count(payload),
        "dynamic_features": _feature_count(
            payload,
            "dynamic_feature_names",
        ),
        "future_features": _feature_count(
            payload,
            "future_feature_names",
        ),
        "static_features": _feature_count(
            payload,
            "static_feature_names",
        ),
    }
    return plot_metric_bars(
        ax,
        metrics,
        title=title,
        sort_by_value=False,
        **plot_kws,
    )


def plot_run_manifest_feature_group_sizes(
    ax: plt.Axes,
    manifest: RunManifestLike,
    *,
    title: str = "Feature-group sizes",
    **plot_kws: Any,
) -> plt.Axes:
    """Plot feature-group sizes from nested scaling kwargs."""
    payload = _as_payload(manifest)
    metrics = {
        "dynamic": _feature_count(
            payload, "dynamic_feature_names"
        ),
        "future": _feature_count(
            payload, "future_feature_names"
        ),
        "static": _feature_count(
            payload, "static_feature_names"
        ),
    }
    return plot_metric_bars(
        ax,
        metrics,
        title=title,
        sort_by_value=False,
        **plot_kws,
    )


def plot_run_manifest_coord_ranges(
    ax: plt.Axes,
    manifest: RunManifestLike,
    *,
    title: str = "Coordinate ranges",
    **plot_kws: Any,
) -> plt.Axes:
    """Plot coordinate ranges from nested scaling kwargs."""
    metrics = _coord_ranges(_as_payload(manifest))
    return plot_metric_bars(
        ax,
        metrics,
        title=title,
        sort_by_value=False,
        **plot_kws,
    )


def plot_run_manifest_boolean_summary(
    ax: plt.Axes,
    manifest: RunManifestLike,
    *,
    title: str = "Run-manifest checks",
    **plot_kws: Any,
) -> plt.Axes:
    """Plot simple boolean checks for expected run outputs."""
    payload = _as_payload(manifest)
    paths = payload.get("paths", {}) or {}
    checks = {
        "has_stage": bool(payload.get("stage")),
        "has_city": bool(payload.get("city")),
        "has_model": bool(payload.get("model")),
        "has_config": isinstance(payload.get("config"), dict),
        "has_paths": isinstance(payload.get("paths"), dict),
        "has_artifacts": isinstance(
            payload.get("artifacts"),
            dict,
        ),
        "has_best_keras": bool(paths.get("best_keras")),
        "has_best_weights": bool(paths.get("best_weights")),
        "has_final_keras": bool(paths.get("final_keras")),
        "has_model_init_manifest": bool(
            paths.get("model_init_manifest")
        ),
    }
    return plot_boolean_checks(
        ax,
        checks,
        title=title,
        **plot_kws,
    )


def inspect_run_manifest(
    manifest: RunManifestLike,
) -> dict[str, Any]:
    """
    Return a bundle of useful inspection outputs.

    Returns
    -------
    dict
        Dictionary containing the normalized payload,
        compact summary, and the main tidy frames.
    """
    payload = _as_payload(manifest)
    return {
        "payload": payload,
        "summary": summarize_run_manifest(payload),
        "identity": run_manifest_identity_frame(payload),
        "config": run_manifest_config_frame(payload),
        "scaling_overview": run_manifest_scaling_overview_frame(
            payload
        ),
        "paths": run_manifest_paths_frame(payload),
        "artifacts": run_manifest_artifacts_frame(payload),
    }
