# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Model-initialization manifest generation and inspection helpers.

This module focuses on the ``model_init_manifest``
artifact saved around model construction time. It
captures the initialized model class, input/output
dimensions, architecture choices, GeoPrior physics
initialization, and the resolved scaling payload used
at initialization.

The functions here support two common needs:

1. Sphinx-Gallery examples that need a realistic
   initialization manifest without rebuilding a model.
2. Real workflow inspection when a user wants to
   verify architecture, physics knobs, feature counts,
   and scaling conventions before training.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import (
    ArtifactRecord,
    as_path,
    clone_artifact,
    flatten_dict,
    load_artifact,
    nested_get,
    plot_boolean_checks,
    plot_metric_bars,
    read_json,
    write_json,
)

PathLike = str | Path
ModelInitManifestLike = (
    ArtifactRecord | Mapping[str, Any] | str | Path
)

__all__ = [
    "default_model_init_manifest_payload",
    "generate_model_init_manifest",
    "inspect_model_init_manifest",
    "load_model_init_manifest",
    "model_init_architecture_frame",
    "model_init_dims_frame",
    "model_init_feature_groups_frame",
    "model_init_geoprior_frame",
    "model_init_scaling_overview_frame",
    "plot_model_init_architecture",
    "plot_model_init_boolean_summary",
    "plot_model_init_dims",
    "plot_model_init_feature_group_sizes",
    "plot_model_init_geoprior",
    "summarize_model_init_manifest",
]

_ARCH_KEYS = [
    "embed_dim",
    "hidden_units",
    "lstm_units",
    "attention_units",
    "num_heads",
    "dropout_rate",
    "memory_size",
    "vsn_units",
]

_GEO_KEYS = [
    "gamma_w",
    "h_ref_value",
    "hd_factor",
    "init_kappa",
    "init_mv",
]

_SCALING_OVERVIEW_KEYS = [
    "time_units",
    "seconds_per_time_unit",
    "coords_normalized",
    "coords_in_degrees",
    "coord_epsg_used",
    "coord_src_epsg",
    "coord_target_epsg",
    "clip_global_norm",
    "cons_residual_units",
    "gw_residual_units",
    "cons_scale_floor",
    "gw_scale_floor",
    "Q_kind",
    "Q_in_si",
    "Q_in_per_second",
    "Q_length_in_si",
    "Q_wrt_normalized_time",
    "lambda_q",
    "mv_weight",
    "mv_schedule_unit",
    "mv_delay_epochs",
    "mv_warmup_epochs",
    "physics_ramp_steps",
    "physics_warmup_steps",
]


def _as_payload(
    manifest: ModelInitManifestLike,
) -> dict[str, Any]:
    """Return a plain model-init payload."""
    if isinstance(manifest, ArtifactRecord):
        return dict(manifest.payload)

    if isinstance(manifest, Mapping):
        return dict(manifest)

    payload = read_json(manifest)
    return dict(payload)


def _selected_numeric(
    mapping: dict[str, Any] | None,
    keys: list[str] | tuple[str, ...],
) -> dict[str, float]:
    """Extract selected numeric scalar items."""
    out: dict[str, float] = {}
    for key in keys:
        value = (mapping or {}).get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, np.number)):
            out[str(key)] = float(value)
    return out


def _selected_rows(
    mapping: dict[str, Any] | None,
    keys: list[str] | tuple[str, ...],
    *,
    section: str,
) -> list[dict[str, Any]]:
    """Build a row list for selected keys."""
    rows: list[dict[str, Any]] = []
    for key in keys:
        value = (mapping or {}).get(key)
        rows.append(
            {
                "section": section,
                "key": str(key),
                "value": value,
            }
        )
    return rows


def _feature_names(
    payload: dict[str, Any],
    kind: str,
) -> list[str]:
    """Return feature names from nested scaling kwargs."""
    names = nested_get(
        payload,
        "config",
        "scaling_kwargs",
        f"{kind}_feature_names",
        default=[],
    )
    if not isinstance(names, list):
        return []
    return [str(v) for v in names]


def _bounds_count(payload: dict[str, Any]) -> int:
    """Return the number of configured bounds entries."""
    bounds = nested_get(
        payload,
        "config",
        "scaling_kwargs",
        "bounds",
        default={},
    )
    if not isinstance(bounds, dict):
        return 0
    return len(bounds)


def default_model_init_manifest_payload(
    *,
    model_class: str = "GeoPriorSubsNet",
    forecast_horizon: int = 3,
    static_input_dim: int = 12,
    dynamic_input_dim: int = 5,
    future_input_dim: int = 1,
    output_subsidence_dim: int = 1,
    output_gwl_dim: int = 1,
    quantiles: list[float] | None = None,
    mode: str = "tft_like",
    pde_mode: str = "on",
    identifiability_regime: str | None = None,
    time_units: str = "year",
) -> dict[str, Any]:
    """
    Build a realistic default model-init manifest.

    The payload mirrors the saved structure used by the
    model-init manifest family:
    ``config`` + ``dims`` + ``model_class``.
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    payload = {
        "config": {
            "attention_units": 64,
            "dropout_rate": 0.1,
            "embed_dim": 32,
            "geoprior": {
                "gamma_w": 9810.0,
                "h_ref_mode": "auto",
                "h_ref_value": 0.0,
                "hd_factor": 0.6,
                "init_kappa": 1.0,
                "init_mv": 1e-7,
                "kappa_mode": "kb",
                "offset_mode": "mul",
                "use_effective_h": True,
            },
            "hidden_units": 64,
            "identifiability_regime": identifiability_regime,
            "lstm_units": 64,
            "memory_size": 50,
            "mode": mode,
            "num_heads": 2,
            "pde_mode": pde_mode,
            "quantiles": list(quantiles),
            "scales": [1, 2],
            "scaling_kwargs": {
                "H_bias_si": 0.0,
                "H_scale_si": 1.0,
                "Q_in_per_second": False,
                "Q_in_si": False,
                "Q_kind": "per_volume",
                "Q_length_in_si": False,
                "Q_wrt_normalized_time": False,
                "allow_subs_residual": True,
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
                "bounds_config": {
                    "beta": 20.0,
                    "guard": 5.0,
                    "include_tau": True,
                    "kind": "both",
                    "mode": "soft",
                    "tau_w": 1.0,
                    "w": 1.0,
                },
                "clip_global_norm": 5.0,
                "cons_drawdown_mode": "softplus",
                "cons_drawdown_rule": "ref_minus_mean",
                "cons_drawdown_zero_at_origin": False,
                "cons_relu_beta": 20.0,
                "cons_residual_units": "second",
                "cons_scale_floor": 3e-11,
                "cons_stop_grad_ref": True,
                "coord_epsg_used": 32649,
                "coord_inv_ranges_si": {
                    "t": 4.526962643830204e-09,
                    "x": 2.249870632438635e-05,
                    "y": 2.5461489497135583e-05,
                },
                "coord_mode": "degrees",
                "coord_order": ["t", "x", "y"],
                "coord_ranges": {
                    "t": 7.0,
                    "x": 44447.0,
                    "y": 39275.0,
                },
                "coord_ranges_si": {
                    "t": 220898664.0,
                    "x": 44447.0,
                    "y": 39275.0,
                },
                "coord_src_epsg": 4326,
                "coord_target_epsg": 32649,
                "coords_in_degrees": False,
                "coords_normalized": True,
                "debug_physics_grads": False,
                "drainage_mode": "double",
                "dt_min_units": 1e-6,
                "dynamic_feature_names": [
                    "GWL_depth_bgs_m__si",
                    "subsidence_cum__si",
                    "rainfall_mm",
                    "urban_load_global",
                    "soil_thickness_censored",
                ],
                "future_feature_names": ["rainfall_mm"],
                "gw_residual_units": "second",
                "gw_scale_floor": 1e-12,
                "gwl_col": "GWL_depth_bgs_m__si",
                "gwl_driver_kind": "depth",
                "gwl_driver_sign": "down_positive",
                "gwl_dyn_col": "GWL_depth_bgs_m__si",
                "gwl_dyn_index": 0,
                "gwl_dyn_name": "GWL_depth_bgs_m__si",
                "gwl_kind": "depth_bgs",
                "gwl_sign": "down_positive",
                "gwl_target_col": "head_m__si",
                "gwl_target_kind": "head",
                "gwl_target_sign": "up_positive",
                "head_bias_si": 0.0,
                "head_scale_si": 1.0,
                "lambda_q": 5e-4,
                "log_q_diagnostics": True,
                "loss_weight_gwl": 0.8,
                "mv_alpha_disp": 0.1,
                "mv_delay_epochs": 1,
                "mv_delay_steps": 4261,
                "mv_huber_delta": 1.0,
                "mv_prior_mode": "calibrate",
                "mv_prior_units": "strict",
                "mv_schedule_unit": "epoch",
                "mv_steps_per_epoch": 4261,
                "mv_warmup_epochs": 2,
                "mv_warmup_steps": 8522,
                "mv_weight": 0.001,
                "physics_ramp_steps": 500,
                "physics_warmup_steps": 500,
                "q_policy": "warmup_off",
                "q_ramp_epochs": 6,
                "q_ramp_steps": 25566,
                "q_warmup_epochs": 2,
                "q_warmup_steps": 8522,
                "scaling_error_policy": "raise",
                "seconds_per_time_unit": 31556952.0,
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
                "subs_bias_si": 0.0,
                "subs_dyn_index": 1,
                "subs_model_col": "subsidence_cum__si",
                "subs_scale_si": 1.0,
                "subsidence_kind": "cumulative",
                "time_units": time_units,
                "track_aux_metrics": False,
                "use_head_proxy": True,
                "z_surf_col": "z_surf_m__si",
                "z_surf_static_index": 11,
            },
            "time_units": time_units,
            "use_batch_norm": False,
            "use_residuals": True,
            "use_vsn": True,
            "vsn_units": 32,
        },
        "dims": {
            "dynamic_input_dim": int(dynamic_input_dim),
            "forecast_horizon": int(forecast_horizon),
            "future_input_dim": int(future_input_dim),
            "output_gwl_dim": int(output_gwl_dim),
            "output_subsidence_dim": int(
                output_subsidence_dim
            ),
            "static_input_dim": int(static_input_dim),
        },
        "model_class": str(model_class),
    }
    return payload


def generate_model_init_manifest(
    path: PathLike,
    *,
    template: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Path:
    """
    Generate a model-init manifest JSON file.

    Parameters
    ----------
    path : str or pathlib.Path
        Output JSON path.
    template : mapping, optional
        Base payload. If omitted, uses
        :func:`default_model_init_manifest_payload`.
    overrides : mapping, optional
        Nested overrides applied on top of the template.
    """
    base = (
        dict(template)
        if template is not None
        else default_model_init_manifest_payload()
    )
    payload = clone_artifact(
        base, overrides=dict(overrides or {})
    )
    return write_json(payload, path)


def load_model_init_manifest(
    path: PathLike,
) -> ArtifactRecord:
    """Load a model-init manifest as ``ArtifactRecord``."""
    return load_artifact(path, kind="model_init_manifest")


def model_init_dims_frame(
    manifest: ModelInitManifestLike,
) -> pd.DataFrame:
    """Return a tidy frame for input/output dimensions."""
    payload = _as_payload(manifest)
    dims = nested_get(payload, "dims", default={})
    rows: list[dict[str, Any]] = []
    for key, value in (dims or {}).items():
        rows.append(
            {
                "section": "dims",
                "name": str(key),
                "value": value,
            }
        )
    return pd.DataFrame(rows)


def model_init_architecture_frame(
    manifest: ModelInitManifestLike,
) -> pd.DataFrame:
    """Return a frame for architecture choices."""
    payload = _as_payload(manifest)
    cfg = nested_get(payload, "config", default={})
    rows = _selected_rows(
        cfg, _ARCH_KEYS, section="architecture"
    )
    rows.extend(
        [
            {
                "section": "architecture",
                "key": "mode",
                "value": cfg.get("mode"),
            },
            {
                "section": "architecture",
                "key": "pde_mode",
                "value": cfg.get("pde_mode"),
            },
            {
                "section": "architecture",
                "key": "identifiability_regime",
                "value": cfg.get("identifiability_regime"),
            },
            {
                "section": "architecture",
                "key": "quantiles",
                "value": cfg.get("quantiles"),
            },
            {
                "section": "architecture",
                "key": "scales",
                "value": cfg.get("scales"),
            },
        ]
    )
    return pd.DataFrame(rows)


def model_init_geoprior_frame(
    manifest: ModelInitManifestLike,
) -> pd.DataFrame:
    """Return a frame for GeoPrior-specific init settings."""
    payload = _as_payload(manifest)
    geoprior = nested_get(
        payload, "config", "geoprior", default={}
    )
    rows = _selected_rows(
        geoprior, _GEO_KEYS, section="geoprior"
    )
    rows.extend(
        [
            {
                "section": "geoprior",
                "key": "h_ref_mode",
                "value": geoprior.get("h_ref_mode"),
            },
            {
                "section": "geoprior",
                "key": "kappa_mode",
                "value": geoprior.get("kappa_mode"),
            },
            {
                "section": "geoprior",
                "key": "offset_mode",
                "value": geoprior.get("offset_mode"),
            },
            {
                "section": "geoprior",
                "key": "use_effective_h",
                "value": geoprior.get("use_effective_h"),
            },
        ]
    )
    return pd.DataFrame(rows)


def model_init_scaling_overview_frame(
    manifest: ModelInitManifestLike,
) -> pd.DataFrame:
    """Return a compact overview of resolved scaling kwargs."""
    payload = _as_payload(manifest)
    sk = nested_get(
        payload, "config", "scaling_kwargs", default={}
    )
    rows = _selected_rows(
        sk, _SCALING_OVERVIEW_KEYS, section="scaling_overview"
    )

    coord_ranges = (
        sk.get("coord_ranges", {})
        if isinstance(sk, dict)
        else {}
    )
    if isinstance(coord_ranges, dict):
        for key, value in coord_ranges.items():
            rows.append(
                {
                    "section": "coord_ranges",
                    "key": f"coord_range_{key}",
                    "value": value,
                }
            )

    bounds_cfg = (
        sk.get("bounds_config", {})
        if isinstance(sk, dict)
        else {}
    )
    if isinstance(bounds_cfg, dict):
        for key in [
            "mode",
            "kind",
            "include_tau",
            "beta",
            "guard",
            "w",
            "tau_w",
        ]:
            rows.append(
                {
                    "section": "bounds_config",
                    "key": f"bounds_{key}",
                    "value": bounds_cfg.get(key),
                }
            )

    return pd.DataFrame(rows)


def model_init_feature_groups_frame(
    manifest: ModelInitManifestLike,
) -> pd.DataFrame:
    """Return a tidy frame for nested feature-name groups."""
    payload = _as_payload(manifest)
    rows: list[dict[str, Any]] = []
    for group in ["static", "dynamic", "future"]:
        names = _feature_names(payload, group)
        for idx, name in enumerate(names):
            rows.append(
                {
                    "group": group,
                    "index": int(idx),
                    "feature_name": str(name),
                }
            )
    return pd.DataFrame(rows)


def summarize_model_init_manifest(
    manifest: ModelInitManifestLike,
) -> dict[str, Any]:
    """
    Return a compact summary of a model-init manifest.

    The goal is not to flatten every nested key, but to
    expose the most decision-relevant initialization facts.
    """
    payload = _as_payload(manifest)
    cfg = nested_get(payload, "config", default={})
    geoprior = nested_get(
        payload, "config", "geoprior", default={}
    )
    sk = nested_get(
        payload, "config", "scaling_kwargs", default={}
    )
    dims = nested_get(payload, "dims", default={})

    quantiles = cfg.get("quantiles")
    quantile_count = (
        len(quantiles) if isinstance(quantiles, list) else 0
    )

    summary = {
        "model_class": payload.get("model_class"),
        "forecast_horizon": dims.get("forecast_horizon"),
        "static_input_dim": dims.get("static_input_dim"),
        "dynamic_input_dim": dims.get("dynamic_input_dim"),
        "future_input_dim": dims.get("future_input_dim"),
        "output_subsidence_dim": dims.get(
            "output_subsidence_dim"
        ),
        "output_gwl_dim": dims.get("output_gwl_dim"),
        "mode": cfg.get("mode"),
        "pde_mode": cfg.get("pde_mode"),
        "identifiability_regime": cfg.get(
            "identifiability_regime"
        ),
        "quantile_count": quantile_count,
        "quantiles": quantiles,
        "use_batch_norm": cfg.get("use_batch_norm"),
        "use_residuals": cfg.get("use_residuals"),
        "use_vsn": cfg.get("use_vsn"),
        "time_units": cfg.get("time_units")
        or sk.get("time_units"),
        "coords_normalized": sk.get("coords_normalized"),
        "coords_in_degrees": sk.get("coords_in_degrees"),
        "coord_order": sk.get("coord_order"),
        "q_kind": sk.get("Q_kind"),
        "gwl_kind": sk.get("gwl_kind"),
        "gwl_sign": sk.get("gwl_sign"),
        "use_head_proxy": sk.get("use_head_proxy"),
        "mv_prior_mode": sk.get("mv_prior_mode"),
        "mv_schedule_unit": sk.get("mv_schedule_unit"),
        "n_bounds": _bounds_count(payload),
        "n_static_features": len(
            _feature_names(payload, "static")
        ),
        "n_dynamic_features": len(
            _feature_names(payload, "dynamic")
        ),
        "n_future_features": len(
            _feature_names(payload, "future")
        ),
        "kappa_mode": geoprior.get("kappa_mode"),
        "offset_mode": geoprior.get("offset_mode"),
        "use_effective_h": geoprior.get("use_effective_h"),
        "hd_factor": geoprior.get("hd_factor"),
        "gamma_w": geoprior.get("gamma_w"),
        "init_kappa": geoprior.get("init_kappa"),
        "init_mv": geoprior.get("init_mv"),
    }

    return summary


def plot_model_init_dims(
    manifest: ModelInitManifestLike,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot input/output dimensions."""
    payload = _as_payload(manifest)
    dims = nested_get(payload, "dims", default={})
    values = {
        str(k): float(v)
        for k, v in (dims or {}).items()
        if isinstance(v, (int, float, np.number))
    }
    return plot_metric_bars(
        values,
        ax=ax,
        title="Model-init dimensions",
        ylabel="size",
    )


def plot_model_init_architecture(
    manifest: ModelInitManifestLike,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot key architecture scalars."""
    payload = _as_payload(manifest)
    cfg = nested_get(payload, "config", default={})
    values = _selected_numeric(cfg, _ARCH_KEYS)
    return plot_metric_bars(
        values,
        ax=ax,
        title="Architecture scalars",
        ylabel="value",
    )


def plot_model_init_geoprior(
    manifest: ModelInitManifestLike,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot key GeoPrior physics-init scalars."""
    payload = _as_payload(manifest)
    geoprior = nested_get(
        payload, "config", "geoprior", default={}
    )
    values = _selected_numeric(geoprior, _GEO_KEYS)
    return plot_metric_bars(
        values,
        ax=ax,
        title="GeoPrior initialization",
        ylabel="value",
    )


def plot_model_init_feature_group_sizes(
    manifest: ModelInitManifestLike,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the sizes of static/dynamic/future feature groups."""
    payload = _as_payload(manifest)
    values = {
        "static": float(
            len(_feature_names(payload, "static"))
        ),
        "dynamic": float(
            len(_feature_names(payload, "dynamic"))
        ),
        "future": float(
            len(_feature_names(payload, "future"))
        ),
    }
    return plot_metric_bars(
        values,
        ax=ax,
        title="Feature-group sizes",
        ylabel="count",
    )


def plot_model_init_boolean_summary(
    manifest: ModelInitManifestLike,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot compact initialization checks as booleans."""
    payload = _as_payload(manifest)
    cfg = nested_get(payload, "config", default={})
    geoprior = nested_get(
        payload, "config", "geoprior", default={}
    )
    sk = nested_get(
        payload, "config", "scaling_kwargs", default={}
    )

    checks = {
        "coords_normalized": bool(
            sk.get("coords_normalized", False)
        ),
        "coords_not_degrees": not bool(
            sk.get("coords_in_degrees", True)
        ),
        "use_head_proxy": bool(
            sk.get("use_head_proxy", False)
        ),
        "use_batch_norm": bool(
            cfg.get("use_batch_norm", False)
        ),
        "use_residuals": bool(
            cfg.get("use_residuals", False)
        ),
        "use_vsn": bool(cfg.get("use_vsn", False)),
        "use_effective_h": bool(
            geoprior.get("use_effective_h", False)
        ),
        "track_aux_metrics": bool(
            sk.get("track_aux_metrics", False)
        ),
        "has_bounds": _bounds_count(payload) > 0,
        "has_static_features": len(
            _feature_names(payload, "static")
        )
        > 0,
        "has_dynamic_features": len(
            _feature_names(payload, "dynamic")
        )
        > 0,
        "has_future_features": len(
            _feature_names(payload, "future")
        )
        > 0,
    }
    return plot_boolean_checks(
        checks,
        ax=ax,
        title="Model-init checks",
    )


def inspect_model_init_manifest(
    manifest: ModelInitManifestLike,
    *,
    output_dir: PathLike | None = None,
    stem: str = "model_init_manifest",
    save_figures: bool = True,
) -> dict[str, Any]:
    """
    Inspect a model-init manifest and optionally save figures.

    Returns
    -------
    dict
        Bundle containing summary, tabular frames, and
        optionally written figure paths.
    """
    payload = _as_payload(manifest)
    summary_map = summarize_model_init_manifest(payload)

    bundle: dict[str, Any] = {
        "summary": summary_map,
        "frames": {
            "dims": model_init_dims_frame(payload),
            "architecture": model_init_architecture_frame(
                payload
            ),
            "geoprior": model_init_geoprior_frame(payload),
            "scaling_overview": model_init_scaling_overview_frame(
                payload
            ),
            "feature_groups": model_init_feature_groups_frame(
                payload
            ),
            "flattened": pd.DataFrame(
                [
                    {
                        "key": k,
                        "value": v,
                    }
                    for k, v in flatten_dict(payload).items()
                ]
            ),
        },
        "figure_paths": {},
    }

    if not (output_dir and save_figures):
        return bundle

    out_dir = as_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plots = {
        f"{stem}_dims.png": (plot_model_init_dims, {}),
        f"{stem}_architecture.png": (
            plot_model_init_architecture,
            {},
        ),
        f"{stem}_geoprior.png": (
            plot_model_init_geoprior,
            {},
        ),
        f"{stem}_feature_groups.png": (
            plot_model_init_feature_group_sizes,
            {},
        ),
        f"{stem}_checks.png": (
            plot_model_init_boolean_summary,
            {},
        ),
    }

    for name, (func, kwargs) in plots.items():
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        func(payload, ax=ax, **kwargs)
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        bundle["figure_paths"][name] = str(path)

    return bundle
