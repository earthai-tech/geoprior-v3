# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Scaling-kwargs generation and inspection helpers.

This module focuses on the ``scaling_kwargs.json`` artifact.
It is one of the most important configuration sidecars in the
GeoPrior workflow because it preserves the resolved scaling,
units, coordinate conventions, feature-channel identities,
and physics-specific configuration used by the model.

The functions are designed for two common uses:

1. Sphinx-Gallery examples that need a realistic
   ``scaling_kwargs`` payload without rerunning the full
   Stage-1 / Stage-2 pipeline.
2. Real workflow inspection when a user wants to verify
   coordinate normalization, SI affine maps, groundwater
   conventions, forcing interpretation, MV-prior schedule,
   and bounds before training or diagnosis.
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
    deep_update,
    load_artifact,
    plot_boolean_checks,
    plot_metric_bars,
    read_json,
    write_json,
)

PathLike = str | Path
ScalingKwargsLike = (
    ArtifactRecord | Mapping[str, Any] | str | Path
)

__all__ = [
    "default_scaling_kwargs_payload",
    "generate_scaling_kwargs",
    "inspect_scaling_kwargs",
    "load_scaling_kwargs",
    "plot_scaling_kwargs_affine_maps",
    "plot_scaling_kwargs_boolean_summary",
    "plot_scaling_kwargs_bounds",
    "plot_scaling_kwargs_coord_ranges",
    "plot_scaling_kwargs_feature_group_sizes",
    "plot_scaling_kwargs_schedule_scalars",
    "scaling_kwargs_affine_frame",
    "scaling_kwargs_bounds_frame",
    "scaling_kwargs_coord_frame",
    "scaling_kwargs_feature_channels_frame",
    "scaling_kwargs_schedule_frame",
    "summarize_scaling_kwargs",
]

_AFFINE_KEYS = [
    "subs_scale_si",
    "subs_bias_si",
    "head_scale_si",
    "head_bias_si",
    "H_scale_si",
    "H_bias_si",
]

_SCHEDULE_KEYS = [
    "clip_global_norm",
    "cons_scale_floor",
    "gw_scale_floor",
    "dt_min_units",
    "mv_weight",
    "mv_alpha_disp",
    "mv_huber_delta",
    "mv_delay_epochs",
    "mv_warmup_epochs",
    "mv_delay_steps",
    "mv_warmup_steps",
]

_BOUND_ORDER = [
    "H_min",
    "H_max",
    "K_min",
    "K_max",
    "Ss_min",
    "Ss_max",
    "tau_min",
    "tau_max",
    "logK_min",
    "logK_max",
    "logSs_min",
    "logSs_max",
    "logTau_min",
    "logTau_max",
]

_BOOLEAN_KEYS = [
    "allow_subs_residual",
    "coords_normalized",
    "coords_in_degrees",
    "Q_wrt_normalized_time",
    "Q_in_si",
    "Q_in_per_second",
    "Q_length_in_si",
    "debug_physics_grads",
    "cons_stop_grad_ref",
    "use_head_proxy",
    "track_aux_metrics",
]


def _as_payload(
    payload: ScalingKwargsLike,
) -> dict[str, Any]:
    """Return a plain scaling-kwargs mapping."""
    if isinstance(payload, ArtifactRecord):
        return dict(payload.payload)

    if isinstance(payload, Mapping):
        return dict(payload)

    data = read_json(payload)
    return dict(data)


def _try_float(
    value: Any,
) -> float | None:
    """Return ``value`` as float when possible."""
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _string_or_none(
    value: Any,
) -> str | None:
    """Return a compact string or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _numeric_subset(
    mapping: dict[str, Any] | None,
    *,
    keys: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Return selected numeric scalar items."""
    src = mapping or {}
    keep = None if keys is None else {str(k) for k in keys}

    out: dict[str, float] = {}
    for key, value in src.items():
        if isinstance(value, bool):
            continue
        num = _try_float(value)
        if num is None:
            continue
        if keep is not None and str(key) not in keep:
            continue
        out[str(key)] = float(num)
    return out


def _feature_count(
    payload: dict[str, Any],
    key: str,
) -> int:
    """Return length of a feature-name list."""
    vals = payload.get(key, None)
    if isinstance(vals, list):
        return int(len(vals))
    return 0


def default_scaling_kwargs_payload(
    *,
    time_units: str = "year",
    coords_normalized: bool = True,
    coords_in_degrees: bool = False,
    coord_mode: str = "degrees",
    coord_order: list[str] | None = None,
    coord_ranges: dict[str, float] | None = None,
    bounds: dict[str, Any] | None = None,
    dynamic_feature_names: list[str] | None = None,
    future_feature_names: list[str] | None = None,
    static_feature_names: list[str] | None = None,
    gwl_dyn_name: str = "GWL_depth_bgs_m__si",
    gwl_dyn_index: int = 0,
    subs_dyn_index: int = 1,
    z_surf_static_index: int = 11,
    gwl_kind: str = "depth_bgs",
    gwl_sign: str = "down_positive",
    use_head_proxy: bool = True,
    q_kind: str = "per_volume",
    mv_prior_mode: str = "calibrate",
    mv_prior_units: str = "strict",
) -> dict[str, Any]:
    """
    Build a realistic default scaling-kwargs payload.

    The payload is intentionally template-based. It mirrors the
    resolved ``scaling_kwargs.json`` structure written by Stage-2
    and preserved in manifests, while staying lightweight enough
    for documentation examples.
    """
    return {
        "subs_scale_si": 1.0,
        "subs_bias_si": 0.0,
        "head_scale_si": 1.0,
        "head_bias_si": 0.0,
        "H_scale_si": 1.0,
        "H_bias_si": 0.0,
        "subsidence_kind": "cumulative",
        "allow_subs_residual": True,
        "coords_normalized": bool(coords_normalized),
        "coord_order": list(coord_order or ["t", "x", "y"]),
        "coord_ranges": dict(
            coord_ranges
            or {
                "t": 7.0,
                "x": 44447.0,
                "y": 39275.0,
            }
        ),
        "coord_mode": coord_mode,
        "coord_src_epsg": 4326,
        "coord_target_epsg": 32649,
        "coord_epsg_used": 32649,
        "coords_in_degrees": bool(coords_in_degrees),
        "cons_residual_units": "second",
        "cons_scale_floor": 3e-11,
        "gw_scale_floor": 1e-12,
        "dt_min_units": 1e-6,
        "Q_wrt_normalized_time": False,
        "Q_in_si": False,
        "Q_in_per_second": False,
        "Q_kind": q_kind,
        "Q_length_in_si": False,
        "drainage_mode": "double",
        "scaling_error_policy": "raise",
        "debug_physics_grads": False,
        "gw_residual_units": "second",
        "clip_global_norm": 5.0,
        "cons_drawdown_mode": "softplus",
        "cons_drawdown_rule": "ref_minus_mean",
        "cons_stop_grad_ref": True,
        "cons_drawdown_zero_at_origin": False,
        "cons_relu_beta": 20.0,
        "mv_prior_units": mv_prior_units,
        "mv_alpha_disp": 0.1,
        "mv_huber_delta": 1.0,
        "mv_prior_mode": mv_prior_mode,
        "mv_weight": 0.001,
        "mv_schedule_unit": "epoch",
        "mv_delay_epochs": 1,
        "mv_warmup_epochs": 2,
        "mv_delay_steps": 4261,
        "mv_warmup_steps": 8522,
        "track_aux_metrics": False,
        "gwl_dyn_index": int(gwl_dyn_index),
        "subs_dyn_index": int(subs_dyn_index),
        "gwl_dyn_name": gwl_dyn_name,
        "z_surf_static_index": int(z_surf_static_index),
        "gwl_col": gwl_dyn_name,
        "gwl_dyn_col": gwl_dyn_name,
        "gwl_target_col": "head_m__si",
        "subs_model_col": "subsidence_cum__si",
        "z_surf_col": "z_surf_m__si",
        "gwl_kind": gwl_kind,
        "gwl_sign": gwl_sign,
        "gwl_driver_kind": "depth",
        "gwl_driver_sign": "down_positive",
        "gwl_target_kind": "head",
        "gwl_target_sign": "up_positive",
        "use_head_proxy": bool(use_head_proxy),
        "time_units": time_units,
        "gwl_z_meta": {
            "raw_kind": "depth_bgs",
            "raw_sign": "down_positive",
            "driver_kind": "depth",
            "driver_sign": "down_positive",
            "target_kind": "head",
            "target_sign": "up_positive",
            "use_head_proxy": bool(use_head_proxy),
            "z_surf_col": "z_surf_m__si",
            "head_from_depth_rule": "z_surf - depth",
            "cols": {
                "depth_raw": "GWL_depth_bgs_m",
                "head_raw": "head_m",
                "z_surf_raw": "z_surf_m",
                "depth_model": "GWL_depth_bgs_m__si",
                "head_model": "head_m__si",
                "z_surf_static": "z_surf_m__si",
                "subs_model": "subsidence_cum__si",
            },
        },
        "bounds": dict(
            bounds
            or {
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
            }
        ),
        "dynamic_feature_names": list(
            dynamic_feature_names
            or [
                "GWL_depth_bgs_m__si",
                "subsidence_cum__si",
                "rainfall_mm",
                "urban_load_global",
                "soil_thickness_censored",
            ]
        ),
        "future_feature_names": list(
            future_feature_names or ["rainfall_mm"]
        ),
        "static_feature_names": list(
            static_feature_names
            or [
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
            ]
        ),
    }


def generate_scaling_kwargs(
    path: PathLike,
    *,
    template: ScalingKwargsLike | None = None,
    overrides: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Path:
    """
    Generate a scaling-kwargs JSON artifact.

    Parameters
    ----------
    path : path-like
        Output path for the JSON file.
    template : mapping or path-like, optional
        Existing payload used as a template.
    overrides : dict, optional
        Nested override values applied after the template.
    **kwargs : Any
        Convenience keyword overrides forwarded into the default
        payload builder when no explicit template is provided.

    Returns
    -------
    pathlib.Path
        Written JSON path.
    """
    if template is None:
        payload = default_scaling_kwargs_payload(**kwargs)
    else:
        payload = clone_artifact(_as_payload(template))
        if kwargs:
            payload = deep_update(payload, kwargs)

    payload = deep_update(payload, overrides)
    return write_json(payload, path)


def load_scaling_kwargs(
    path: PathLike,
) -> ArtifactRecord:
    """
    Load a scaling-kwargs artifact.

    Returns
    -------
    ArtifactRecord
        Normalized artifact wrapper.
    """
    return load_artifact(path, kind="scaling_kwargs")


def scaling_kwargs_affine_frame(
    payload: ScalingKwargsLike,
) -> pd.DataFrame:
    """Return affine SI-map rows."""
    data = _as_payload(payload)
    rows: list[dict[str, Any]] = []

    unit_map = {
        "subs_scale_si": "m / model-unit",
        "subs_bias_si": "m",
        "head_scale_si": "m / model-unit",
        "head_bias_si": "m",
        "H_scale_si": "m / model-unit",
        "H_bias_si": "m",
    }

    for key in _AFFINE_KEYS:
        rows.append(
            {
                "parameter": key,
                "value": _try_float(data.get(key)),
                "unit": unit_map.get(key),
            }
        )

    return pd.DataFrame(rows)


def scaling_kwargs_coord_frame(
    payload: ScalingKwargsLike,
) -> pd.DataFrame:
    """Return coordinate and convention rows."""
    data = _as_payload(payload)
    ranges = data.get("coord_ranges", {}) or {}

    rows = [
        {
            "section": "coord_meta",
            "name": "coord_mode",
            "value": _string_or_none(data.get("coord_mode")),
        },
        {
            "section": "coord_meta",
            "name": "coord_order",
            "value": ", ".join(
                map(str, data.get("coord_order", []) or [])
            ),
        },
        {
            "section": "coord_meta",
            "name": "coord_src_epsg",
            "value": data.get("coord_src_epsg"),
        },
        {
            "section": "coord_meta",
            "name": "coord_target_epsg",
            "value": data.get("coord_target_epsg"),
        },
        {
            "section": "coord_meta",
            "name": "coord_epsg_used",
            "value": data.get("coord_epsg_used"),
        },
        {
            "section": "coord_meta",
            "name": "time_units",
            "value": _string_or_none(data.get("time_units")),
        },
    ]

    for key in ["t", "x", "y"]:
        rows.append(
            {
                "section": "coord_ranges",
                "name": key,
                "value": _try_float(ranges.get(key)),
            }
        )

    return pd.DataFrame(rows)


def scaling_kwargs_bounds_frame(
    payload: ScalingKwargsLike,
) -> pd.DataFrame:
    """Return tidy bounds rows."""
    data = _as_payload(payload)
    bounds = data.get("bounds", {}) or {}

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for key in _BOUND_ORDER:
        seen.add(key)
        rows.append(
            {
                "bound": key,
                "value": _try_float(bounds.get(key)),
            }
        )

    for key in sorted(bounds):
        if key in seen:
            continue
        rows.append(
            {
                "bound": str(key),
                "value": _try_float(bounds.get(key)),
            }
        )

    return pd.DataFrame(rows)


def scaling_kwargs_feature_channels_frame(
    payload: ScalingKwargsLike,
) -> pd.DataFrame:
    """Return feature-group and channel rows."""
    data = _as_payload(payload)

    rows: list[dict[str, Any]] = [
        {
            "group": "feature_counts",
            "name": "dynamic_feature_names",
            "value": _feature_count(
                data,
                "dynamic_feature_names",
            ),
        },
        {
            "group": "feature_counts",
            "name": "future_feature_names",
            "value": _feature_count(
                data,
                "future_feature_names",
            ),
        },
        {
            "group": "feature_counts",
            "name": "static_feature_names",
            "value": _feature_count(
                data,
                "static_feature_names",
            ),
        },
        {
            "group": "channel_indices",
            "name": "gwl_dyn_index",
            "value": _try_float(data.get("gwl_dyn_index")),
        },
        {
            "group": "channel_indices",
            "name": "subs_dyn_index",
            "value": _try_float(data.get("subs_dyn_index")),
        },
        {
            "group": "channel_indices",
            "name": "z_surf_static_index",
            "value": _try_float(
                data.get("z_surf_static_index")
            ),
        },
        {
            "group": "channel_names",
            "name": "gwl_dyn_name",
            "value": _string_or_none(
                data.get("gwl_dyn_name")
            ),
        },
        {
            "group": "channel_names",
            "name": "gwl_col",
            "value": _string_or_none(data.get("gwl_col")),
        },
        {
            "group": "channel_names",
            "name": "gwl_target_col",
            "value": _string_or_none(
                data.get("gwl_target_col")
            ),
        },
        {
            "group": "channel_names",
            "name": "subs_model_col",
            "value": _string_or_none(
                data.get("subs_model_col")
            ),
        },
        {
            "group": "channel_names",
            "name": "z_surf_col",
            "value": _string_or_none(data.get("z_surf_col")),
        },
    ]

    return pd.DataFrame(rows)


def scaling_kwargs_schedule_frame(
    payload: ScalingKwargsLike,
) -> pd.DataFrame:
    """Return Q/MV schedule and runtime scalar rows."""
    data = _as_payload(payload)
    rows: list[dict[str, Any]] = []

    for key in _SCHEDULE_KEYS:
        rows.append(
            {
                "section": "scalar_schedule",
                "name": key,
                "value": _try_float(data.get(key)),
            }
        )

    for key in [
        "Q_kind",
        "mv_prior_mode",
        "mv_prior_units",
        "mv_schedule_unit",
        "cons_residual_units",
        "gw_residual_units",
        "drainage_mode",
        "scaling_error_policy",
        "gwl_kind",
        "gwl_sign",
    ]:
        rows.append(
            {
                "section": "modes",
                "name": key,
                "value": _string_or_none(data.get(key)),
            }
        )

    return pd.DataFrame(rows)


def summarize_scaling_kwargs(
    payload: ScalingKwargsLike,
) -> dict[str, Any]:
    """
    Return a compact high-level scaling summary.

    Returns
    -------
    dict
        Compact summary intended for logs or gallery prose.
    """
    data = _as_payload(payload)
    bounds = data.get("bounds", {}) or {}

    return {
        "time_units": _string_or_none(data.get("time_units")),
        "coord_mode": _string_or_none(data.get("coord_mode")),
        "coord_order": list(
            data.get("coord_order", []) or []
        ),
        "coords_normalized": bool(
            data.get("coords_normalized", False)
        ),
        "coords_in_degrees": bool(
            data.get("coords_in_degrees", False)
        ),
        "coord_ranges": dict(
            data.get("coord_ranges", {}) or {}
        ),
        "gwl_kind": _string_or_none(data.get("gwl_kind")),
        "gwl_sign": _string_or_none(data.get("gwl_sign")),
        "use_head_proxy": bool(
            data.get("use_head_proxy", False)
        ),
        "Q_kind": _string_or_none(data.get("Q_kind")),
        "mv_prior_mode": _string_or_none(
            data.get("mv_prior_mode")
        ),
        "mv_schedule_unit": _string_or_none(
            data.get("mv_schedule_unit")
        ),
        "dynamic_features": _feature_count(
            data,
            "dynamic_feature_names",
        ),
        "future_features": _feature_count(
            data,
            "future_feature_names",
        ),
        "static_features": _feature_count(
            data,
            "static_feature_names",
        ),
        "n_bounds": int(len(bounds)),
        "has_bounds": bool(bounds),
        "has_gwl_z_meta": isinstance(
            data.get("gwl_z_meta"),
            dict,
        ),
        "affine_maps": _numeric_subset(
            data,
            keys=_AFFINE_KEYS,
        ),
    }


def plot_scaling_kwargs_affine_maps(
    payload: ScalingKwargsLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Scaling affine maps",
) -> plt.Axes:
    """Plot subs/head/H affine map scalars."""
    data = _as_payload(payload)
    metrics = _numeric_subset(data, keys=_AFFINE_KEYS)

    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 4.2))

    return plot_metric_bars(
        ax,
        metrics,
        title=title,
        sort_by_value=False,
        annotate=True,
    )


def plot_scaling_kwargs_coord_ranges(
    payload: ScalingKwargsLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Coordinate ranges",
) -> plt.Axes:
    """Plot ``coord_ranges`` for t/x/y."""
    data = _as_payload(payload)
    metrics = _numeric_subset(
        data.get("coord_ranges", {}) or {}
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 3.8))

    return plot_metric_bars(
        ax,
        metrics,
        title=title,
        sort_by_value=False,
        annotate=True,
    )


def plot_scaling_kwargs_bounds(
    payload: ScalingKwargsLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Bounds overview",
    top_n: int | None = None,
) -> plt.Axes:
    """Plot numeric bounds as a compact bar chart."""
    frame = scaling_kwargs_bounds_frame(payload)
    frame = frame.dropna(subset=["value"]).rename(
        columns={"bound": "metric"}
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(8.4, 5.0))

    return plot_metric_bars(
        ax,
        frame,
        title=title,
        top_n=top_n,
        sort_by_value=False,
        annotate=False,
    )


def plot_scaling_kwargs_schedule_scalars(
    payload: ScalingKwargsLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Schedule and runtime scalars",
) -> plt.Axes:
    """Plot selected numeric schedule/runtime scalars."""
    data = _as_payload(payload)
    metrics = _numeric_subset(data, keys=_SCHEDULE_KEYS)

    if ax is None:
        _, ax = plt.subplots(figsize=(8.4, 4.8))

    return plot_metric_bars(
        ax,
        metrics,
        title=title,
        sort_by_value=False,
        annotate=False,
    )


def plot_scaling_kwargs_feature_group_sizes(
    payload: ScalingKwargsLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Feature group sizes",
) -> plt.Axes:
    """Plot dynamic/future/static feature-group counts."""
    data = _as_payload(payload)
    metrics = {
        "dynamic_feature_names": _feature_count(
            data,
            "dynamic_feature_names",
        ),
        "future_feature_names": _feature_count(
            data,
            "future_feature_names",
        ),
        "static_feature_names": _feature_count(
            data,
            "static_feature_names",
        ),
    }

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 3.6))

    return plot_metric_bars(
        ax,
        metrics,
        title=title,
        sort_by_value=False,
        annotate=True,
    )


def plot_scaling_kwargs_boolean_summary(
    payload: ScalingKwargsLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Scaling boolean checks",
) -> plt.Axes:
    """Plot common boolean config flags."""
    data = _as_payload(payload)
    checks = {
        key: bool(data.get(key, False))
        for key in _BOOLEAN_KEYS
        if key in data
    }

    if ax is None:
        _, ax = plt.subplots(figsize=(8.2, 4.2))

    return plot_boolean_checks(
        ax,
        checks,
        title=title,
    )


def inspect_scaling_kwargs(
    payload: ScalingKwargsLike,
) -> dict[str, Any]:
    """
    Return a compact multi-view inspection bundle.

    Returns
    -------
    dict
        Dictionary containing summary plus a set of tidy frames.
    """
    data = _as_payload(payload)

    return {
        "summary": summarize_scaling_kwargs(data),
        "affine": scaling_kwargs_affine_frame(data),
        "coords": scaling_kwargs_coord_frame(data),
        "bounds": scaling_kwargs_bounds_frame(data),
        "features": scaling_kwargs_feature_channels_frame(
            data
        ),
        "schedule": scaling_kwargs_schedule_frame(data),
    }
