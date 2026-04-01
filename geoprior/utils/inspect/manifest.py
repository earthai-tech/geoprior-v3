# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Stage-1 manifest generation and inspection helpers.

This module focuses on the broader ``manifest.json``
artifact exported by Stage-1 preprocessing.
Compared with the lighter Stage-2 ``run_manifest``
artifact, the Stage-1 manifest is a richer workflow
handshake that preserves:

- dataset and split identity,
- resolved columns and feature groups,
- conventions and scaling payload,
- holdout and sequence counts,
- artifact paths for CSV, encoders, NPZ files,
  and shape summaries,
- runtime/version metadata.

The helpers here support two common uses:

1. Sphinx-Gallery examples that need a realistic
   Stage-1 manifest without rerunning preprocessing.
2. Real workflow inspection when a user wants to
   verify that downstream stages can trust the
   preprocessing handshake.
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
ManifestLike = ArtifactRecord | Mapping[str, Any] | str | Path

__all__ = [
    "default_manifest_payload",
    "generate_manifest",
    "inspect_manifest",
    "load_manifest",
    "manifest_artifacts_frame",
    "manifest_config_frame",
    "manifest_feature_groups_frame",
    "manifest_holdout_frame",
    "manifest_identity_frame",
    "manifest_paths_frame",
    "manifest_shapes_frame",
    "manifest_versions_frame",
    "plot_manifest_artifact_inventory",
    "plot_manifest_boolean_summary",
    "plot_manifest_coord_ranges",
    "plot_manifest_feature_group_sizes",
    "plot_manifest_holdout_counts",
    "summarize_manifest",
]

_CONFIG_KEYS = [
    "TIME_STEPS",
    "FORECAST_HORIZON_YEARS",
    "MODE",
    "TRAIN_END_YEAR",
    "FORECAST_START_YEAR",
]

_HOLDOUT_KEYS = [
    "valid_for_train",
    "valid_for_forecast",
    "kept_for_processing",
    "train_groups",
    "val_groups",
    "test_groups",
    "train_rows",
    "val_rows",
    "test_rows",
    "train_seq",
    "val_seq",
    "test_seq",
]


def _as_payload(manifest: ManifestLike) -> dict[str, Any]:
    """Return a plain Stage-1 manifest payload."""
    if isinstance(manifest, ArtifactRecord):
        return dict(manifest.payload)

    if isinstance(manifest, Mapping):
        return dict(manifest)

    return dict(read_json(manifest))


def _try_int(value: Any) -> int | None:
    """Return ``value`` as int when possible."""
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _try_float(value: Any) -> float | None:
    """Return ``value`` as float when possible."""
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _feature_count(payload: dict[str, Any], key: str) -> int:
    """Return the size of a Stage-1 feature group."""
    vals = nested_get(
        payload, "config", "features", key, default=[]
    )
    if isinstance(vals, list):
        return len(vals)
    return 0


def _coord_ranges(
    payload: dict[str, Any],
) -> dict[str, float]:
    """Return numeric coord ranges from nested scaling kwargs."""
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
        num = _try_float(value)
        if num is None:
            continue
        out[str(key)] = num
    return out


def _walk_leaf_paths(
    obj: Any,
    *,
    prefix: str = "",
) -> list[dict[str, Any]]:
    """
    Return leaf rows for nested artifact mappings.

    String leaves are treated as path-like payload entries.
    List leaves are retained as values but not treated as paths.
    """
    rows: list[dict[str, Any]] = []

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_walk_leaf_paths(value, prefix=name))
        return rows

    value = obj
    str(value) if value is not None else None
    basename = None
    is_path = False

    if isinstance(value, str) and value:
        basename = Path(value).name
        is_path = any(
            token in value.lower()
            for token in (
                ".csv",
                ".json",
                ".joblib",
                ".npz",
                ".keras",
                ".h5",
                ".pkl",
            )
        )

    rows.append(
        {
            "key": prefix,
            "value": value,
            "basename": basename,
            "is_path": is_path,
        }
    )
    return rows


def _artifact_path_count(payload: dict[str, Any]) -> int:
    """Return the number of leaf artifact path entries."""
    rows = _walk_leaf_paths(payload.get("artifacts", {}))
    return sum(1 for row in rows if row.get("is_path"))


def _shape_rows(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract tidy rows from ``artifacts.shapes``."""
    src = nested_get(
        payload, "artifacts", "shapes", default={}
    )
    rows: list[dict[str, Any]] = []

    if not isinstance(src, dict):
        return rows

    for split_name, block in src.items():
        if not isinstance(block, dict):
            continue
        for tensor_name, shape in block.items():
            if not isinstance(shape, list):
                continue
            rows.append(
                {
                    "split": str(split_name),
                    "tensor": str(tensor_name),
                    "shape": shape,
                    "rank": len(shape),
                    "samples": _try_int(shape[0])
                    if shape
                    else None,
                    "time_dim": _try_int(shape[1])
                    if len(shape) > 1
                    else None,
                    "feature_dim": _try_int(shape[2])
                    if len(shape) > 2
                    else None,
                }
            )
    return rows


def default_manifest_payload(
    *,
    city: str = "nansha",
    model: str = "GeoPriorSubsNet",
    time_steps: int = 5,
    forecast_horizon_years: int = 3,
    mode: str = "tft_like",
) -> dict[str, Any]:
    """
    Build a realistic default Stage-1 manifest payload.

    The payload mirrors the stable Stage-1 handshake:
    ``schema_version`` + identity fields + ``config`` +
    ``artifacts`` + ``paths`` + ``versions``.
    """
    run_dir = f"results/{city}_{model}_stage1"
    artifacts_dir = f"{run_dir}/artifacts"

    payload = {
        "schema_version": "3.2",
        "timestamp": "2026-02-10 15:41:18",
        "city": str(city),
        "model": str(model),
        "stage": "stage1",
        "config": {
            "TIME_STEPS": int(time_steps),
            "FORECAST_HORIZON_YEARS": int(
                forecast_horizon_years
            ),
            "MODE": str(mode),
            "TRAIN_END_YEAR": 2022,
            "FORECAST_START_YEAR": 2023,
            "cols": {
                "time": "year",
                "lon": "longitude",
                "lat": "latitude",
                "time_numeric": "year_numeric_coord",
                "time_used": "year_numeric_coord",
                "x_base": "x_m",
                "y_base": "y_m",
                "x_used": "x_m",
                "y_used": "y_m",
                "subs_raw": "subsidence_cum",
                "subs_model": "subsidence_cum__si",
                "depth_raw": "GWL_depth_bgs_m",
                "head_raw": "head_m",
                "depth_model": "GWL_depth_bgs_m__si",
                "head_model": "head_m__si",
                "h_field_raw": "soil_thickness_eff",
                "h_field_model": "soil_thickness_eff__si",
                "z_surf_raw": "z_surf_m",
                "z_surf_static": "z_surf_m__si",
            },
            "features": {
                "static": [
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
                "dynamic": [
                    "GWL_depth_bgs_m__si",
                    "subsidence_cum__si",
                    "rainfall_mm",
                    "urban_load_global",
                    "soil_thickness_censored",
                ],
                "future": ["rainfall_mm"],
                "group_id_cols": ["longitude", "latitude"],
            },
            "indices": {
                "gwl_dyn_index": 0,
                "subs_dyn_index": 1,
                "gwl_dyn_name": "GWL_depth_bgs_m__si",
                "z_surf_static_index": 11,
            },
            "conventions": {
                "gwl_kind": "depth_bgs",
                "gwl_sign": "down_positive",
                "use_head_proxy": False,
                "time_units": "year",
                "gwl_driver_kind": "depth",
                "gwl_target_kind": "head",
                "gwl_driver_sign": "down_positive",
                "gwl_target_sign": "up_positive",
            },
            "scaler_info": {
                "rainfall_mm": {
                    "scaler_path": f"{artifacts_dir}/{city}_main_scaler.joblib",
                    "all_features": [
                        "rainfall_mm",
                        "soil_thickness_censored",
                    ],
                    "idx": 0,
                },
                "soil_thickness_censored": {
                    "scaler_path": f"{artifacts_dir}/{city}_main_scaler.joblib",
                    "all_features": [
                        "rainfall_mm",
                        "soil_thickness_censored",
                    ],
                    "idx": 1,
                },
            },
            "scaling_kwargs": {
                "subs_scale_si": 1.0,
                "subs_bias_si": 0.0,
                "head_scale_si": 1.0,
                "head_bias_si": 0.0,
                "H_scale_si": 1.0,
                "H_bias_si": 0.0,
                "subsidence_kind": "cumulative",
                "allow_subs_residual": True,
                "coords_normalized": True,
                "coord_order": ["t", "x", "y"],
                "coord_ranges": {
                    "t": 7.0,
                    "x": 44447.0,
                    "y": 39275.0,
                },
                "coord_mode": "degrees",
                "coord_src_epsg": 4326,
                "coord_target_epsg": 32649,
                "coord_epsg_used": 32649,
                "coords_in_degrees": False,
                "cons_residual_units": "second",
                "gw_residual_units": "second",
                "Q_wrt_normalized_time": False,
                "Q_in_si": False,
                "Q_in_per_second": False,
                "Q_kind": "per_volume",
                "Q_length_in_si": False,
                "drainage_mode": "double",
                "clip_global_norm": 5.0,
                "gwl_dyn_index": 0,
                "subs_dyn_index": 1,
                "gwl_kind": "depth_bgs",
                "gwl_sign": "down_positive",
                "use_head_proxy": True,
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
            },
            "feature_registry": {
                "optional_numeric_declared": [
                    "rainfall_mm",
                    "soil_thickness_censored",
                ],
                "optional_categorical_declared": [
                    "lithology",
                    "lithology_class",
                ],
                "already_normalized": [],
                "future_drivers_declared": ["rainfall_mm"],
                "resolved_optional_numeric": [
                    "rainfall_mm",
                    "soil_thickness_censored",
                ],
                "resolved_optional_categorical": [
                    "lithology",
                    "lithology_class",
                ],
            },
            "censoring": {
                "specs": [],
                "report": {},
                "use_effective_h_field": True,
                "flags_as_dynamic": True,
                "flags_as_future": False,
            },
            "units_provenance": {
                "subs_unit_to_si_applied_stage1": 0.001,
                "thickness_unit_to_si_applied_stage1": 1.0,
                "head_unit_to_si_assumed_stage1": 1.0,
                "z_surf_unit_to_si_assumed_stage1": 1.0,
            },
            "holdout": {
                "strategy": "random",
                "seed": 42,
                "val_frac": 0.2,
                "test_frac": 0.1,
                "group_cols": ["longitude", "latitude"],
                "coord_cols_for_blocking": {
                    "x_col": "x_m",
                    "y_col": "y_m",
                    "epsg_used": 32649,
                },
                "block_size_m": None,
                "requirements": {
                    "time_col": "year",
                    "train_end_year": 2022,
                    "forecast_start_year": 2023,
                    "time_steps": int(time_steps),
                    "horizon": int(forecast_horizon_years),
                    "years_required_for_train_windows": 8,
                    "note": (
                        "Group validity enforced by "
                        "compute_group_masks() before holdout "
                        "splitting."
                    ),
                },
                "group_counts": {
                    "valid_for_train": 194788,
                    "valid_for_forecast": 194788,
                    "kept_for_processing": 194788,
                    "train_groups": 136351,
                    "val_groups": 38958,
                    "test_groups": 19479,
                },
                "row_counts_hist": {
                    "train_rows": 1090808,
                    "val_rows": 311664,
                    "test_rows": 155832,
                },
                "sequence_counts": {
                    "train_seq": 136351,
                    "val_seq": 38958,
                    "test_seq": 19479,
                },
                "artifacts": {
                    "train_groups_csv": f"{artifacts_dir}/holdout/train_groups.csv",
                    "val_groups_csv": f"{artifacts_dir}/holdout/val_groups.csv",
                    "test_groups_csv": f"{artifacts_dir}/holdout/test_groups.csv",
                },
            },
        },
        "artifacts": {
            "csv": {
                "raw": f"{run_dir}/{city}_01_raw.csv",
                "clean": f"{run_dir}/{city}_02_clean.csv",
                "scaled": f"{run_dir}/{city}_03_scaled.csv",
            },
            "encoders": {
                "ohe": {
                    "lithology": f"{artifacts_dir}/{city}_ohe_lithology.joblib",
                    "lithology_class": (
                        f"{artifacts_dir}/{city}_ohe_lithology_class.joblib"
                    ),
                },
                "coord_scaler": f"{artifacts_dir}/{city}_coord_scaler.joblib",
                "main_scaler": f"{artifacts_dir}/{city}_main_scaler.joblib",
                "scaled_ml_numeric_cols": [
                    "rainfall_mm",
                    "soil_thickness_censored",
                ],
            },
            "sequences": {
                "joblib_train_sequences": (
                    f"{artifacts_dir}/{city}_train_sequences_"
                    f"T{time_steps}_H{forecast_horizon_years}.joblib"
                ),
                "dims": {
                    "output_subsidence_dim": 1,
                    "output_gwl_dim": 1,
                },
            },
            "numpy": {
                "train_inputs_npz": f"{artifacts_dir}/train_inputs.npz",
                "train_targets_npz": f"{artifacts_dir}/train_targets.npz",
                "val_inputs_npz": f"{artifacts_dir}/val_inputs.npz",
                "val_targets_npz": f"{artifacts_dir}/val_targets.npz",
                "test_inputs_npz": f"{artifacts_dir}/test_inputs.npz",
                "test_targets_npz": f"{artifacts_dir}/test_targets.npz",
            },
            "shapes": {
                "train_inputs": {
                    "H_field": [136351, 3, 1],
                    "coords": [136351, 3, 3],
                    "dynamic_features": [136351, 5, 5],
                    "future_features": [136351, 8, 1],
                    "static_features": [136351, 12],
                },
                "train_targets": {
                    "gwl_pred": [136351, 3, 1],
                    "subs_pred": [136351, 3, 1],
                },
                "val_inputs": {
                    "H_field": [38958, 3, 1],
                    "coords": [38958, 3, 3],
                    "dynamic_features": [38958, 5, 5],
                    "future_features": [38958, 8, 1],
                    "static_features": [38958, 12],
                },
                "val_targets": {
                    "gwl_pred": [38958, 3, 1],
                    "subs_pred": [38958, 3, 1],
                },
                "test_inputs": {
                    "H_field": [19479, 3, 1],
                    "coords": [19479, 3, 3],
                    "dynamic_features": [19479, 5, 5],
                    "future_features": [19479, 8, 1],
                    "static_features": [19479, 12],
                },
                "test_targets": {
                    "gwl_pred": [19479, 3, 1],
                    "subs_pred": [19479, 3, 1],
                },
            },
        },
        "paths": {
            "run_dir": run_dir,
            "artifacts_dir": artifacts_dir,
        },
        "versions": {
            "python": "3.10.19",
            "tensorflow": "2.20.0",
            "numpy": "2.0.2",
            "pandas": "2.3.3",
            "sklearn": "1.7.2",
        },
    }
    return payload


def generate_manifest(
    path: PathLike,
    *,
    template: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Path:
    """Generate a Stage-1 manifest JSON artifact."""
    base = (
        dict(template)
        if template is not None
        else default_manifest_payload()
    )
    payload = clone_artifact(
        base, overrides=dict(overrides or {})
    )
    return write_json(payload, path)


def load_manifest(path: PathLike) -> ArtifactRecord:
    """Load a Stage-1 manifest as ``ArtifactRecord``."""
    return load_artifact(path, kind="manifest")


def manifest_identity_frame(
    manifest: ManifestLike,
) -> pd.DataFrame:
    """Return a compact Stage-1 identity frame."""
    payload = _as_payload(manifest)
    rows = [
        {
            "section": "identity",
            "key": "schema_version",
            "value": payload.get("schema_version"),
        },
        {
            "section": "identity",
            "key": "timestamp",
            "value": payload.get("timestamp"),
        },
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


def manifest_config_frame(
    manifest: ManifestLike,
) -> pd.DataFrame:
    """Return a tidy frame for the compact config overview."""
    payload = _as_payload(manifest)
    cfg = nested_get(payload, "config", default={})
    conv = nested_get(
        payload, "config", "conventions", default={}
    )

    rows: list[dict[str, Any]] = []
    for key in _CONFIG_KEYS:
        rows.append(
            {
                "section": "config",
                "key": str(key),
                "value": (cfg or {}).get(key),
            }
        )

    for key in (
        "gwl_kind",
        "gwl_sign",
        "use_head_proxy",
        "time_units",
        "gwl_driver_kind",
        "gwl_target_kind",
    ):
        rows.append(
            {
                "section": "conventions",
                "key": str(key),
                "value": (conv or {}).get(key),
            }
        )

    rows.append(
        {
            "section": "config",
            "key": "coord_ranges",
            "value": _coord_ranges(payload),
        }
    )
    return pd.DataFrame(rows)


def manifest_feature_groups_frame(
    manifest: ManifestLike,
) -> pd.DataFrame:
    """Return feature-group names and counts."""
    payload = _as_payload(manifest)
    feats = nested_get(
        payload, "config", "features", default={}
    )

    rows: list[dict[str, Any]] = []
    for key in (
        "static",
        "dynamic",
        "future",
        "group_id_cols",
    ):
        values = (feats or {}).get(key, [])
        if not isinstance(values, list):
            values = []
        rows.append(
            {
                "section": "features",
                "group": str(key),
                "count": len(values),
                "values": values,
            }
        )
    return pd.DataFrame(rows)


def manifest_holdout_frame(
    manifest: ManifestLike,
) -> pd.DataFrame:
    """Return key holdout and split counts as a tidy frame."""
    payload = _as_payload(manifest)
    holdout = nested_get(
        payload, "config", "holdout", default={}
    )
    group_counts = (holdout or {}).get(
        "group_counts", {}
    ) or {}
    row_counts = (holdout or {}).get(
        "row_counts_hist", {}
    ) or {}
    seq_counts = (holdout or {}).get(
        "sequence_counts", {}
    ) or {}

    rows: list[dict[str, Any]] = []
    for key in _HOLDOUT_KEYS:
        value = None
        section = "holdout"
        if key in group_counts:
            value = group_counts.get(key)
            section = "group_counts"
        elif key in row_counts:
            value = row_counts.get(key)
            section = "row_counts_hist"
        elif key in seq_counts:
            value = seq_counts.get(key)
            section = "sequence_counts"

        if value is not None:
            rows.append(
                {
                    "section": section,
                    "key": str(key),
                    "value": value,
                }
            )

    rows.extend(
        [
            {
                "section": "holdout_meta",
                "key": "strategy",
                "value": (holdout or {}).get("strategy"),
            },
            {
                "section": "holdout_meta",
                "key": "seed",
                "value": (holdout or {}).get("seed"),
            },
            {
                "section": "holdout_meta",
                "key": "val_frac",
                "value": (holdout or {}).get("val_frac"),
            },
            {
                "section": "holdout_meta",
                "key": "test_frac",
                "value": (holdout or {}).get("test_frac"),
            },
        ]
    )
    return pd.DataFrame(rows)


def manifest_artifacts_frame(
    manifest: ManifestLike,
) -> pd.DataFrame:
    """Return a leaf-level artifact inventory frame."""
    payload = _as_payload(manifest)
    rows = _walk_leaf_paths(payload.get("artifacts", {}))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.insert(0, "section", "artifacts")
    return frame


def manifest_paths_frame(
    manifest: ManifestLike,
) -> pd.DataFrame:
    """Return a frame describing top-level manifest paths."""
    payload = _as_payload(manifest)
    rows: list[dict[str, Any]] = []
    for key, value in (
        payload.get("paths", {}) or {}
    ).items():
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


def manifest_versions_frame(
    manifest: ManifestLike,
) -> pd.DataFrame:
    """Return runtime/library versions saved in the manifest."""
    payload = _as_payload(manifest)
    rows: list[dict[str, Any]] = []
    for key, value in (
        payload.get("versions", {}) or {}
    ).items():
        rows.append(
            {
                "section": "versions",
                "key": str(key),
                "value": value,
            }
        )
    return pd.DataFrame(rows)


def manifest_shapes_frame(
    manifest: ManifestLike,
) -> pd.DataFrame:
    """Return a tidy tensor-shape summary frame."""
    payload = _as_payload(manifest)
    return pd.DataFrame(_shape_rows(payload))


def summarize_manifest(
    manifest: ManifestLike,
) -> dict[str, Any]:
    """
    Summarize the Stage-1 manifest in a compact dictionary.

    The summary focuses on what later workflow stages
    usually need to verify first.
    """
    payload = _as_payload(manifest)
    cfg = nested_get(payload, "config", default={})
    holdout = nested_get(
        payload, "config", "holdout", default={}
    )
    group_counts = (holdout or {}).get(
        "group_counts", {}
    ) or {}
    row_counts = (holdout or {}).get(
        "row_counts_hist", {}
    ) or {}
    seq_counts = (holdout or {}).get(
        "sequence_counts", {}
    ) or {}
    versions = payload.get("versions", {}) or {}

    summary = {
        "schema_version": payload.get("schema_version"),
        "timestamp": payload.get("timestamp"),
        "stage": payload.get("stage"),
        "city": payload.get("city"),
        "model": payload.get("model"),
        "time_steps": (cfg or {}).get("TIME_STEPS"),
        "forecast_horizon_years": (cfg or {}).get(
            "FORECAST_HORIZON_YEARS"
        ),
        "mode": (cfg or {}).get("MODE"),
        "train_end_year": (cfg or {}).get("TRAIN_END_YEAR"),
        "forecast_start_year": (cfg or {}).get(
            "FORECAST_START_YEAR"
        ),
        "static_feature_count": _feature_count(
            payload, "static"
        ),
        "dynamic_feature_count": _feature_count(
            payload, "dynamic"
        ),
        "future_feature_count": _feature_count(
            payload, "future"
        ),
        "group_id_cols_count": _feature_count(
            payload,
            "group_id_cols",
        ),
        "coord_ranges": _coord_ranges(payload),
        "artifact_path_count": _artifact_path_count(payload),
        "path_count": len(payload.get("paths", {}) or {}),
        "version_count": len(versions),
        "train_groups": group_counts.get("train_groups"),
        "val_groups": group_counts.get("val_groups"),
        "test_groups": group_counts.get("test_groups"),
        "train_rows": row_counts.get("train_rows"),
        "val_rows": row_counts.get("val_rows"),
        "test_rows": row_counts.get("test_rows"),
        "train_seq": seq_counts.get("train_seq"),
        "val_seq": seq_counts.get("val_seq"),
        "test_seq": seq_counts.get("test_seq"),
        "has_config": isinstance(payload.get("config"), dict),
        "has_artifacts": isinstance(
            payload.get("artifacts"), dict
        ),
        "has_paths": isinstance(payload.get("paths"), dict),
        "has_versions": isinstance(
            payload.get("versions"), dict
        ),
        "has_holdout": isinstance(
            (cfg or {}).get("holdout"), dict
        ),
        "has_scaling_kwargs": isinstance(
            (cfg or {}).get("scaling_kwargs"),
            dict,
        ),
    }
    return summary


def plot_manifest_artifact_inventory(
    ax: plt.Axes,
    manifest: ManifestLike,
    *,
    title: str = "Manifest artifact inventory",
) -> plt.Axes:
    """Plot artifact and metadata inventory counts."""
    payload = _as_payload(manifest)
    metrics = {
        "artifact_paths": _artifact_path_count(payload),
        "paths": len(payload.get("paths", {}) or {}),
        "versions": len(payload.get("versions", {}) or {}),
        "shape_blocks": len(
            nested_get(
                payload, "artifacts", "shapes", default={}
            )
            or {}
        ),
        "numpy_artifacts": len(
            nested_get(
                payload, "artifacts", "numpy", default={}
            )
            or {}
        ),
    }
    return plot_metric_bars(
        ax,
        metrics,
        title=title,
        sort_by_value=False,
    )


def plot_manifest_feature_group_sizes(
    ax: plt.Axes,
    manifest: ManifestLike,
    *,
    title: str = "Stage-1 feature groups",
) -> plt.Axes:
    """Plot Stage-1 feature-group sizes."""
    payload = _as_payload(manifest)
    metrics = {
        "static": _feature_count(payload, "static"),
        "dynamic": _feature_count(payload, "dynamic"),
        "future": _feature_count(payload, "future"),
        "group_id_cols": _feature_count(
            payload, "group_id_cols"
        ),
    }
    return plot_metric_bars(
        ax,
        metrics,
        title=title,
        sort_by_value=False,
    )


def plot_manifest_holdout_counts(
    ax: plt.Axes,
    manifest: ManifestLike,
    *,
    title: str = "Holdout split counts",
) -> plt.Axes:
    """Plot the main group and sequence split counts."""
    payload = _as_payload(manifest)
    holdout = nested_get(
        payload, "config", "holdout", default={}
    )
    groups = (holdout or {}).get("group_counts", {}) or {}
    seqs = (holdout or {}).get("sequence_counts", {}) or {}
    metrics = {
        "train_groups": groups.get("train_groups", 0),
        "val_groups": groups.get("val_groups", 0),
        "test_groups": groups.get("test_groups", 0),
        "train_seq": seqs.get("train_seq", 0),
        "val_seq": seqs.get("val_seq", 0),
        "test_seq": seqs.get("test_seq", 0),
    }
    return plot_metric_bars(
        ax,
        metrics,
        title=title,
        sort_by_value=False,
    )


def plot_manifest_coord_ranges(
    ax: plt.Axes,
    manifest: ManifestLike,
    *,
    title: str = "Coordinate ranges",
) -> plt.Axes:
    """Plot nested coordinate ranges from scaling kwargs."""
    return plot_metric_bars(
        ax,
        _coord_ranges(_as_payload(manifest)),
        title=title,
        sort_by_value=False,
    )


def plot_manifest_boolean_summary(
    ax: plt.Axes,
    manifest: ManifestLike,
    *,
    title: str = "Stage-1 manifest checks",
) -> plt.Axes:
    """Plot simple boolean checks for the handshake."""
    payload = _as_payload(manifest)
    cfg = payload.get("config", {}) or {}
    checks = {
        "has_schema_version": bool(
            payload.get("schema_version")
        ),
        "has_stage": bool(payload.get("stage")),
        "has_city": bool(payload.get("city")),
        "has_model": bool(payload.get("model")),
        "has_config": isinstance(payload.get("config"), dict),
        "has_artifacts": isinstance(
            payload.get("artifacts"), dict
        ),
        "has_paths": isinstance(payload.get("paths"), dict),
        "has_versions": isinstance(
            payload.get("versions"), dict
        ),
        "has_scaling_kwargs": isinstance(
            cfg.get("scaling_kwargs"),
            dict,
        ),
        "has_holdout": isinstance(cfg.get("holdout"), dict),
        "has_numpy_artifacts": bool(
            nested_get(
                payload, "artifacts", "numpy", default={}
            )
        ),
        "has_shape_summary": bool(
            nested_get(
                payload, "artifacts", "shapes", default={}
            )
        ),
    }
    return plot_boolean_checks(ax, checks, title=title)


def inspect_manifest(
    manifest: ManifestLike,
) -> dict[str, Any]:
    """
    Return a bundle of useful Stage-1 inspection outputs.

    Returns
    -------
    dict
        Normalized payload, compact summary, and
        the main tidy frames.
    """
    payload = _as_payload(manifest)
    return {
        "payload": payload,
        "summary": summarize_manifest(payload),
        "identity": manifest_identity_frame(payload),
        "config": manifest_config_frame(payload),
        "feature_groups": manifest_feature_groups_frame(
            payload
        ),
        "holdout": manifest_holdout_frame(payload),
        "artifacts": manifest_artifacts_frame(payload),
        "paths": manifest_paths_frame(payload),
        "versions": manifest_versions_frame(payload),
        "shapes": manifest_shapes_frame(payload),
    }
