# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Stage-2 handshake audit generation and inspection helpers.

This module focuses on the Stage-2 handshake audit artifact.
It provides:

- robust loading,
- reproducible demo-audit generation,
- compact tabular summaries,
- quick visual inspection helpers.

The functions are designed for two common uses:

1. Sphinx-Gallery examples that need a realistic
   handshake payload without rerunning Stage-2.
2. Real workflow inspection when a user wants to
   verify that Stage-1 outputs and Stage-2 model
   expectations agree before training proceeds.
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
    deep_update,
    load_artifact,
    metrics_frame,
    nested_get,
    plot_boolean_checks,
    plot_metric_bars,
    read_json,
    write_json,
)

PathLike = str | Path
Stage2HandshakeLike = (
    ArtifactRecord | Mapping[str, Any] | str | Path
)

__all__ = [
    "default_stage2_handshake_payload",
    "generate_stage2_handshake",
    "inspect_stage2_handshake",
    "load_stage2_handshake",
    "plot_stage2_boolean_summary",
    "plot_stage2_coord_range_errors",
    "plot_stage2_coord_stats",
    "plot_stage2_finite_ratios",
    "plot_stage2_sample_sizes",
    "plot_stage2_scaling_summary",
    "stage2_coord_range_frame",
    "stage2_coord_stats_frame",
    "stage2_finite_frame",
    "stage2_layout_frame",
    "stage2_scaling_frame",
    "summarize_stage2_handshake",
]


# ------------------------------------------------------------------
# Small internal helpers
# ------------------------------------------------------------------


def _as_payload(
    audit: Stage2HandshakeLike,
) -> dict[str, Any]:
    """Return a plain Stage-2 handshake payload."""
    if isinstance(audit, ArtifactRecord):
        return dict(audit.payload)

    if isinstance(audit, Mapping):
        return dict(audit)

    payload = read_json(audit)
    return dict(payload)


def _shape_string(value: Any) -> str:
    """Render a shape-like payload as a stable string."""
    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple, np.ndarray)):
        vals = [
            str(int(v))
            if isinstance(v, (int, np.integer))
            else str(v)
            for v in list(value)
        ]
        return "(" + ",".join(vals) + ")"

    return str(value)


def _future_time_dim(
    *,
    time_steps: int,
    forecast_horizon: int,
    mode: str,
) -> int:
    """Return the future-feature time axis implied by mode."""
    if str(mode).strip().lower() == "tft_like":
        return int(time_steps) + int(forecast_horizon)
    return int(forecast_horizon)


def _coord_stat_block(
    *,
    prefix: str,
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
    means: tuple[float, float, float],
    stds: tuple[float, float, float],
) -> dict[str, Any]:
    """Build a coord stats section."""
    axes = ("t", "x", "y")
    out: dict[str, Any] = {}

    for axis, vmin, vmax, mean, std in zip(
        axes,
        mins,
        maxs,
        means,
        stds,
        strict=False,
    ):
        out[f"{axis}.{prefix}"] = {
            "min": float(vmin),
            "max": float(vmax),
            "mean": float(mean),
            "std": float(std),
        }
    return out


# ------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------


def default_stage2_handshake_payload(
    *,
    city: str = "demo_city",
    model: str = "GeoPriorSubsNet",
    time_steps: int = 5,
    forecast_horizon: int = 3,
    mode: str = "tft_like",
    n_train: int = 1200,
    n_val: int = 320,
    dynamic_dim: int = 5,
    future_dim: int = 1,
    static_dim: int = 12,
    coords_normalized: bool = True,
    coord_ranges: dict[str, float] | None = None,
    coords_in_degrees: bool = False,
    coord_epsg_used: int | None = 32649,
    time_units: str = "year",
    gwl_dyn_name: str = "GWL_depth_bgs_m__si",
    gwl_dyn_index: int = 0,
    subs_dyn_index: int = 1,
    z_surf_static_index: int = 11,
    use_head_proxy: bool = True,
    q_kind: str = "per_volume",
    drainage_mode: str = "double",
) -> dict[str, Any]:
    """
    Build a realistic default Stage-2 handshake payload.

    The payload is template-based. It is not meant to
    rerun Stage-2 logic. Instead, it creates a stable and
    inspectable audit artifact with the same broad
    structure as the real Stage-2 handshake file.
    """
    future_time_dim = _future_time_dim(
        time_steps=time_steps,
        forecast_horizon=forecast_horizon,
        mode=mode,
    )

    coord_ranges = dict(
        coord_ranges
        or {
            "t": 7.0,
            "x": 44447.0,
            "y": 39275.0,
        }
    )

    payload = {
        "city": city,
        "model": model,
        "stage": "stage2",
        "expected": {
            "TIME_STEPS": int(time_steps),
            "FORECAST_HORIZON": int(forecast_horizon),
            "MODE": str(mode),
            "coords.shape": (
                f"(N,{int(forecast_horizon)},3)"
            ),
            "dynamic_features.shape": (
                f"(N,{int(time_steps)},{int(dynamic_dim)})"
            ),
            "future_features.shape": (
                f"(N,{int(future_time_dim)},{int(future_dim)})"
            ),
            "static_features.shape": (
                f"(N,{int(static_dim)})"
            ),
            "H_field.shape": (
                f"(N,{int(forecast_horizon)},1)"
            ),
            "targets.shape": (
                f"(N,{int(forecast_horizon)},1)"
            ),
        },
        "got": {
            "N_train": int(n_train),
            "N_val": int(n_val),
            "coords": [
                int(n_train),
                int(forecast_horizon),
                3,
            ],
            "dynamic_features": [
                int(n_train),
                int(time_steps),
                int(dynamic_dim),
            ],
            "future_features": [
                int(n_train),
                int(future_time_dim),
                int(future_dim),
            ],
            "static_features": [
                int(n_train),
                int(static_dim),
            ],
            "H_field": [
                int(n_train),
                int(forecast_horizon),
                1,
            ],
            "y_subs_pred": [
                int(n_train),
                int(forecast_horizon),
                1,
            ],
            "y_gwl_pred": [
                int(n_train),
                int(forecast_horizon),
                1,
            ],
        },
        "finite": {
            "X_train.coords.finite_ratio": 1.0,
            "X_val.coords.finite_ratio": 1.0,
            "X_train.dynamic_features.finite_ratio": 1.0,
            "X_val.dynamic_features.finite_ratio": 1.0,
            "X_train.future_features.finite_ratio": 1.0,
            "X_val.future_features.finite_ratio": 1.0,
            "X_train.static_features.finite_ratio": 1.0,
            "X_val.static_features.finite_ratio": 1.0,
            "X_train.H_field.finite_ratio": 1.0,
            "X_val.H_field.finite_ratio": 1.0,
            "y_train.subs_pred.finite_ratio": 1.0,
            "y_val.subs_pred.finite_ratio": 1.0,
            "y_train.gwl_pred.finite_ratio": 1.0,
            "y_val.gwl_pred.finite_ratio": 1.0,
        },
        "coord_stats_norm": _coord_stat_block(
            prefix="norm",
            mins=(0.7143, 0.0, 0.0),
            maxs=(1.0, 1.0, 1.0),
            means=(0.8571, 0.4781, 0.6157),
            stds=(0.1166, 0.2116, 0.2323),
        ),
        "coord_checks": {
            "expected_in_[0,1]": bool(coords_normalized),
            "t_outside_01?": False,
            "x_outside_01?": False,
            "y_outside_01?": False,
        },
        "coord_stats_raw": _coord_stat_block(
            prefix="raw",
            mins=(2020.0, 735361.8, 2497507.5),
            maxs=(2022.0, 776965.9, 2535615.3),
            means=(2021.0, 756414.8, 2520612.2),
            stds=(0.8169, 9506.5, 9163.1),
        ),
        "coord_range_check": {
            "scaler_span": dict(coord_ranges),
            "coord_ranges_rel_err_t": 0.0,
            "coord_ranges_rel_err_x": 0.0,
            "coord_ranges_rel_err_y": 0.0,
        },
        "sk_summary": {
            "coords_normalized": bool(coords_normalized),
            "coord_order": "['t', 'x', 'y']",
            "coord_ranges": (
                "{ "
                f"t={coord_ranges['t']}, "
                f"x={coord_ranges['x']}, "
                f"y={coord_ranges['y']} "
                "}"
            ),
            "coords_in_degrees": bool(coords_in_degrees),
            "deg_to_m_lon": None,
            "deg_to_m_lat": None,
            "lat_ref_deg": None,
            "coord_epsg_used": coord_epsg_used,
            "coord_target_epsg": coord_epsg_used,
            "subs_scale_si": 1.0,
            "subs_bias_si": 0.0,
            "head_scale_si": 1.0,
            "head_bias_si": 0.0,
            "H_scale_si": 1.0,
            "H_bias_si": 0.0,
            "gwl_dyn_name": gwl_dyn_name,
            "gwl_dyn_index": int(gwl_dyn_index),
            "z_surf_static_index": int(z_surf_static_index),
            "subs_dyn_index": int(subs_dyn_index),
            "gwl_kind": "depth_bgs",
            "gwl_sign": "down_positive",
            "use_head_proxy": bool(use_head_proxy),
            "z_surf_col": "z_surf_m__si",
            "time_units": str(time_units),
            "cons_residual_units": "second",
            "gw_residual_units": "second",
            "cons_scale_floor": 3e-11,
            "gw_scale_floor": 1e-12,
            "dt_min_units": 1e-06,
            "clip_global_norm": 5.0,
            "scaling_error_policy": "raise",
            "debug_physics_grads": False,
            "Q_kind": str(q_kind),
            "Q_in_si": False,
            "Q_in_per_second": False,
            "Q_wrt_normalized_time": False,
            "Q_length_in_si": False,
            "drainage_mode": str(drainage_mode),
        },
        "sk_checks": {
            "has_coord_ranges": True,
            "coords_in_degrees": bool(coords_in_degrees),
            "deg_to_m_lon_needed": False,
            "deg_to_m_lat_needed": False,
            "gwl_dyn_index_in_bounds": True,
        },
    }
    return payload


def generate_stage2_handshake(
    *,
    output_path: PathLike | None = None,
    template: Stage2HandshakeLike | None = None,
    overrides: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | Path:
    """
    Generate a Stage-2 handshake payload or file.

    Parameters
    ----------
    output_path : path-like, optional
        Destination JSON path. If omitted, the payload
        is returned instead of written.
    template : mapping, ArtifactRecord, or path, optional
        Real or synthetic Stage-2 handshake template used
        as the generation base.
    overrides : dict, optional
        Nested overrides applied after template/default
        payload creation.
    **kwargs : dict
        Parameters forwarded to
        ``default_stage2_handshake_payload`` when no
        template is given.
    """
    if template is None:
        payload = default_stage2_handshake_payload(**kwargs)
    else:
        payload = clone_artifact(
            _as_payload(template),
            overrides=None,
        )

    if overrides:
        payload = deep_update(payload, overrides)

    if output_path is None:
        return payload

    return write_json(payload, output_path)


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------


def load_stage2_handshake(
    path: PathLike,
) -> ArtifactRecord:
    """
    Load a Stage-2 handshake artifact.

    Raises
    ------
    ValueError
        If the artifact does not look like a Stage-2
        handshake payload.
    """
    record = load_artifact(path, kind="stage2_handshake")
    needed = {
        "expected",
        "got",
        "finite",
    }
    if not needed.issubset(record.payload):
        raise ValueError(
            "The file does not contain the expected "
            "Stage-2 handshake sections."
        )
    return record


# ------------------------------------------------------------------
# Frames
# ------------------------------------------------------------------


def stage2_layout_frame(
    audit: Stage2HandshakeLike,
) -> pd.DataFrame:
    """
    Return expected vs observed layout rows.
    """
    payload = _as_payload(audit)
    expected = payload.get("expected", {}) or {}
    got = payload.get("got", {}) or {}

    rows: list[dict[str, Any]] = []

    for key, exp_val in expected.items():
        got_key = key.replace(".shape", "")
        got_val = got.get(got_key)
        rows.append(
            {
                "key": str(key),
                "expected": _shape_string(exp_val),
                "got": _shape_string(got_val),
            }
        )

    for key in ("N_train", "N_val"):
        if key in got:
            rows.append(
                {
                    "key": key,
                    "expected": None,
                    "got": _shape_string(got.get(key)),
                }
            )

    return pd.DataFrame(rows)


def stage2_finite_frame(
    audit: Stage2HandshakeLike,
) -> pd.DataFrame:
    """Return finite-ratio checks as a tidy frame."""
    payload = _as_payload(audit)
    frame = metrics_frame(
        payload.get("finite", {}),
        section="finite",
    )
    if not frame.empty:
        frame = frame.rename(columns={"metric": "name"})
    return frame


def stage2_coord_stats_frame(
    audit: Stage2HandshakeLike,
    *,
    section: str = "coord_stats_norm",
) -> pd.DataFrame:
    """
    Return a tidy frame for coord stat blocks.
    """
    payload = _as_payload(audit)
    src = payload.get(section, {}) or {}

    rows: list[dict[str, Any]] = []
    for name, stats in src.items():
        if not isinstance(stats, dict):
            continue
        for stat_name, value in stats.items():
            if isinstance(value, (int, float)):
                axis = str(name).split(".", 1)[0]
                rows.append(
                    {
                        "section": section,
                        "coord": axis,
                        "name": str(name),
                        "stat": str(stat_name),
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def stage2_coord_range_frame(
    audit: Stage2HandshakeLike,
) -> pd.DataFrame:
    """
    Return coord range spans and relative errors.
    """
    payload = _as_payload(audit)
    src = payload.get("coord_range_check", {}) or {}
    span = dict(src.get("scaler_span", {}) or {})

    rows: list[dict[str, Any]] = []
    for axis in ("t", "x", "y"):
        rows.append(
            {
                "coord": axis,
                "scaler_span": float(span.get(axis, np.nan)),
                "rel_err": float(
                    src.get(
                        f"coord_ranges_rel_err_{axis}", np.nan
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def stage2_scaling_frame(
    audit: Stage2HandshakeLike,
) -> pd.DataFrame:
    """
    Return a tidy frame for the compact scaling summary.
    """
    payload = _as_payload(audit)
    src = payload.get("sk_summary", {}) or {}

    rows: list[dict[str, Any]] = []
    for key, value in src.items():
        rows.append(
            {
                "key": str(key),
                "value": value,
                "is_numeric": isinstance(
                    value,
                    (int, float),
                )
                and not isinstance(value, bool),
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------


def summarize_stage2_handshake(
    audit: Stage2HandshakeLike,
) -> dict[str, Any]:
    """
    Build a compact semantic summary for inspection.
    """
    payload = _as_payload(audit)

    finite = payload.get("finite", {}) or {}
    coord_checks = payload.get("coord_checks", {}) or {}
    range_check = payload.get("coord_range_check", {}) or {}
    sk_summary = payload.get("sk_summary", {}) or {}
    sk_checks = payload.get("sk_checks", {}) or {}
    got = payload.get("got", {}) or {}
    expected = payload.get("expected", {}) or {}

    finite_vals = [
        float(v)
        for v in finite.values()
        if isinstance(v, (int, float))
    ]

    rel_errs = [
        float(
            range_check.get(
                f"coord_ranges_rel_err_{axis}", np.nan
            )
        )
        for axis in ("t", "x", "y")
        if isinstance(
            range_check.get(f"coord_ranges_rel_err_{axis}"),
            (int, float),
        )
    ]

    summary = {
        "brief": {
            "kind": "stage2_handshake",
            "city": payload.get("city"),
            "model": payload.get("model"),
            "stage": payload.get("stage", "stage2"),
        },
        "expected": {
            "time_steps": expected.get("TIME_STEPS"),
            "forecast_horizon": expected.get(
                "FORECAST_HORIZON"
            ),
            "mode": expected.get("MODE"),
        },
        "sample_sizes": {
            "N_train": got.get("N_train"),
            "N_val": got.get("N_val"),
        },
        "coord_ranges": dict(
            range_check.get("scaler_span", {}) or {}
        ),
        "checks": {
            "all_finite": bool(
                finite_vals and min(finite_vals) >= 1.0
            ),
            "coords_normalized": bool(
                sk_summary.get("coords_normalized", False)
            ),
            "coords_expected_in_unit_box": bool(
                coord_checks.get("expected_in_[0,1]", False)
            ),
            "coords_within_unit_box": bool(
                not coord_checks.get("t_outside_01?", False)
                and not coord_checks.get(
                    "x_outside_01?", False
                )
                and not coord_checks.get(
                    "y_outside_01?", False
                )
            ),
            "coord_ranges_match_scaler": bool(
                rel_errs and max(np.abs(rel_errs)) <= 1e-12
            ),
            "has_scaling_summary": bool(sk_summary),
            "has_coord_ranges": bool(
                sk_checks.get("has_coord_ranges", False)
            ),
            "gwl_dyn_index_in_bounds": bool(
                sk_checks.get(
                    "gwl_dyn_index_in_bounds", False
                )
            ),
            "degree_factors_not_needed": bool(
                not sk_checks.get(
                    "deg_to_m_lon_needed", False
                )
                and not sk_checks.get(
                    "deg_to_m_lat_needed", False
                )
            ),
        },
        "layout": {
            "coords": _shape_string(got.get("coords")),
            "dynamic_features": _shape_string(
                got.get("dynamic_features")
            ),
            "future_features": _shape_string(
                got.get("future_features")
            ),
            "static_features": _shape_string(
                got.get("static_features")
            ),
            "H_field": _shape_string(got.get("H_field")),
            "targets_subs": _shape_string(
                got.get("y_subs_pred")
            ),
            "targets_gwl": _shape_string(
                got.get("y_gwl_pred")
            ),
        },
    }
    return summary


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------


def plot_stage2_sample_sizes(
    audit: Stage2HandshakeLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Stage-2 sample sizes",
) -> plt.Axes:
    """Plot training and validation counts."""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(6.4, 4.0))

    payload = _as_payload(audit)
    got = payload.get("got", {}) or {}
    metrics = {
        "N_train": got.get("N_train", 0),
        "N_val": got.get("N_val", 0),
    }
    plot_metric_bars(
        ax,
        metrics,
        title=title,
        sort_by_value=False,
    )
    return ax


def plot_stage2_finite_ratios(
    audit: Stage2HandshakeLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Stage-2 finite ratios",
) -> plt.Axes:
    """Plot finite-ratio metrics."""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(9.0, 4.8))

    payload = _as_payload(audit)
    plot_metric_bars(
        ax,
        payload.get("finite", {}),
        title=title,
        sort_by_value=True,
    )
    return ax


def plot_stage2_coord_stats(
    audit: Stage2HandshakeLike,
    *,
    section: str = "coord_stats_norm",
    stat: str = "mean",
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    Plot one coord statistic across axes.
    """
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7.4, 4.0))

    frame = stage2_coord_stats_frame(audit, section=section)
    if frame.empty:
        ax.set_title(title or section)
        ax.text(
            0.5,
            0.5,
            "No coordinate stats",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    sub = frame.loc[frame["stat"].eq(stat)].copy()
    if sub.empty:
        ax.set_title(title or section)
        ax.text(
            0.5,
            0.5,
            f"Statistic {stat!r} not found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    metrics = dict(
        zip(sub["coord"], sub["value"], strict=False)
    )
    plot_metric_bars(
        ax,
        metrics,
        title=title or f"{section}: {stat}",
        sort_by_value=False,
    )
    ax.set_xlabel(stat)
    return ax


def plot_stage2_coord_range_errors(
    audit: Stage2HandshakeLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Stage-2 coord range relative errors",
) -> plt.Axes:
    """Plot coord range relative errors."""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7.0, 4.0))

    frame = stage2_coord_range_frame(audit)
    if frame.empty:
        ax.set_title(title)
        ax.text(
            0.5,
            0.5,
            "No coord range checks",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    ax.bar(frame["coord"], frame["rel_err"])
    ax.set_title(title)
    ax.set_ylabel("relative error")
    ax.grid(axis="y", alpha=0.25)

    for _, row in frame.iterrows():
        ax.text(
            row["coord"],
            row["rel_err"],
            f"{row['rel_err']:.3g}",
            ha="center",
            va="bottom",
        )
    return ax


def plot_stage2_scaling_summary(
    audit: Stage2HandshakeLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Stage-2 scaling summary",
    top_n: int | None = 12,
) -> plt.Axes:
    """Plot numeric items from ``sk_summary``."""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(8.4, 4.8))

    frame = stage2_scaling_frame(audit)
    if frame.empty:
        ax.set_title(title)
        ax.text(
            0.5,
            0.5,
            "No scaling summary",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    sub = frame.loc[frame["is_numeric"]].copy()
    if sub.empty:
        ax.set_title(title)
        ax.text(
            0.5,
            0.5,
            "No numeric scaling items",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    metrics = dict(
        zip(sub["key"], sub["value"], strict=False)
    )
    plot_metric_bars(
        ax,
        metrics,
        title=title,
        sort_by_value=True,
        top_n=top_n,
        absolute=True,
    )
    return ax


def plot_stage2_boolean_summary(
    audit: Stage2HandshakeLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Stage-2 handshake checks",
) -> plt.Axes:
    """Plot semantic pass/fail checks."""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(8.0, 4.6))

    checks = summarize_stage2_handshake(audit)["checks"]
    plot_boolean_checks(ax, checks, title=title)
    return ax


# ------------------------------------------------------------------
# Inspection bundle
# ------------------------------------------------------------------


def inspect_stage2_handshake(
    audit: Stage2HandshakeLike,
    *,
    output_dir: PathLike | None = None,
    stem: str = "stage2_handshake",
    save_figures: bool = True,
) -> dict[str, Any]:
    """
    Inspect a Stage-2 handshake and optionally save figures.

    Returns
    -------
    dict
        Bundle containing summary, tabular frames, and
        optionally written figure paths.
    """
    payload = _as_payload(audit)
    summary = summarize_stage2_handshake(payload)

    bundle: dict[str, Any] = {
        "summary": summary,
        "frames": {
            "layout": stage2_layout_frame(payload),
            "finite": stage2_finite_frame(payload),
            "coord_stats_norm": stage2_coord_stats_frame(
                payload,
                section="coord_stats_norm",
            ),
            "coord_stats_raw": stage2_coord_stats_frame(
                payload,
                section="coord_stats_raw",
            ),
            "coord_ranges": stage2_coord_range_frame(payload),
            "scaling_summary": stage2_scaling_frame(payload),
        },
        "figure_paths": {},
    }

    if not (output_dir and save_figures):
        return bundle

    out_dir = as_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plots = {
        f"{stem}_sample_sizes.png": (
            plot_stage2_sample_sizes,
            {},
        ),
        f"{stem}_finite_ratios.png": (
            plot_stage2_finite_ratios,
            {},
        ),
        f"{stem}_coord_norm_means.png": (
            plot_stage2_coord_stats,
            {
                "section": "coord_stats_norm",
                "stat": "mean",
            },
        ),
        f"{stem}_coord_raw_means.png": (
            plot_stage2_coord_stats,
            {
                "section": "coord_stats_raw",
                "stat": "mean",
            },
        ),
        f"{stem}_coord_range_errors.png": (
            plot_stage2_coord_range_errors,
            {},
        ),
        f"{stem}_scaling_summary.png": (
            plot_stage2_scaling_summary,
            {},
        ),
        f"{stem}_checks.png": (
            plot_stage2_boolean_summary,
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
