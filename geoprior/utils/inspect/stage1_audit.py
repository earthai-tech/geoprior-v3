# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Stage-1 audit generation and inspection helpers.

This module focuses on the Stage-1 scaling /
preprocessing audit artifact. It provides:

- robust loading,
- reproducible demo-audit generation,
- compact tabular summaries,
- quick visual inspection helpers.

The functions are designed for two common uses:

1. Sphinx-Gallery examples that need a realistic
   audit payload without rerunning the full
   pipeline.
2. Real workflow inspection when a user wants to
   check coordinate handling, feature splits, and
   Stage-1 scaling outputs before moving to the
   next stage.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .utils import (
    ArtifactRecord,
    as_path,
    clone_artifact,
    deep_update,
    empty_plot,
    filter_plot_kwargs,
    finalize_plot,
    load_artifact,
    nested_get,
    plot_boolean_checks,
    plot_metric_bars,
    prepare_plot,
    read_json,
    write_json,
)

PathLike = str | Path
Stage1AuditLike = (
    ArtifactRecord | Mapping[str, Any] | str | Path
)

__all__ = [
    "build_stage1_feature_split",
    "default_stage1_audit_payload",
    "generate_stage1_audit",
    "inspect_stage1_audit",
    "load_stage1_audit",
    "plot_stage1_boolean_summary",
    "plot_stage1_coord_ranges",
    "plot_stage1_feature_split",
    "plot_stage1_target_stats",
    "plot_stage1_variable_stats",
    "stage1_coord_ranges_frame",
    "stage1_feature_split_frame",
    "stage1_stats_frame",
    "summarize_stage1_audit",
]

_DEFAULT_DYNAMIC = [
    "GWL_depth_bgs_m__si",
    "subsidence_cum__si",
    "rainfall_mm",
    "urban_load_global",
    "soil_thickness_censored",
]

_DEFAULT_STATIC = [
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

_DEFAULT_FUTURE = ["rainfall_mm"]

_DEFAULT_SCALED_NUMERIC = [
    "rainfall_mm",
    "soil_thickness_censored",
]


def _as_payload(
    audit: Stage1AuditLike,
) -> dict[str, Any]:
    """Return a plain Stage-1 audit payload."""
    if isinstance(audit, ArtifactRecord):
        return dict(audit.payload)

    if isinstance(audit, Mapping):
        return dict(audit)

    payload = read_json(audit)
    return dict(payload)


def _feature_bucket(
    features: list[str],
    scaled_ml_numeric_cols: list[str],
) -> dict[str, Any]:
    """Classify features for the split summary."""
    scaled = [
        name
        for name in features
        if name in scaled_ml_numeric_cols
    ]
    si_unscaled = [
        name
        for name in features
        if name.endswith("__si") and name not in scaled
    ]
    other = [
        name
        for name in features
        if name not in scaled and name not in si_unscaled
    ]
    return {
        "count": int(len(features)),
        "scaled_by_main_scaler": scaled,
        "si_unscaled": si_unscaled,
        "other_unscaled_mixed": other,
    }


def build_stage1_feature_split(
    *,
    dynamic_features: list[str],
    static_features: list[str],
    future_features: list[str],
    scaled_ml_numeric_cols: list[str],
) -> dict[str, Any]:
    """
    Build the Stage-1 feature split section.

    Parameters
    ----------
    dynamic_features : list of str
        Dynamic input feature names.
    static_features : list of str
        Static input feature names.
    future_features : list of str
        Future-known feature names.
    scaled_ml_numeric_cols : list of str
        Features scaled by the main scaler.

    Returns
    -------
    dict
        Stage-1 style feature split mapping.
    """
    return {
        "dynamic_features": _feature_bucket(
            list(dynamic_features),
            list(scaled_ml_numeric_cols),
        ),
        "static_features": _feature_bucket(
            list(static_features),
            list(scaled_ml_numeric_cols),
        ),
        "future_features": _feature_bucket(
            list(future_features),
            list(scaled_ml_numeric_cols),
        ),
    }


def default_stage1_audit_payload(
    *,
    city: str = "demo_city",
    model: str = "GeoPriorSubsNet",
    normalize_coords: bool = True,
    keep_coords_raw: bool | None = None,
    shift_raw_coords: bool = True,
    coords_in_degrees: bool = False,
    coord_mode: str = "degrees",
    coord_epsg_used: int | None = 32649,
    coord_ranges: dict[str, float] | None = None,
    dynamic_features: list[str] | None = None,
    static_features: list[str] | None = None,
    future_features: list[str] | None = None,
    scaled_ml_numeric_cols: list[str] | None = None,
    scaler_path: str = "artifacts/main_scaler.joblib",
    n_sequences: int = 1200,
    forecast_horizon: int = 3,
    time_steps: int = 5,
) -> dict[str, Any]:
    """
    Build a realistic default Stage-1 audit payload.

    The payload is template-based. It is not meant to
    reproduce the full preprocessing mathematics.
    Instead, it creates a stable and inspectable audit
    artifact with the same broad structure as the real
    Stage-1 audit file.
    """
    dynamic = list(dynamic_features or _DEFAULT_DYNAMIC)
    static = list(static_features or _DEFAULT_STATIC)
    future = list(future_features or _DEFAULT_FUTURE)
    scaled = list(
        scaled_ml_numeric_cols or _DEFAULT_SCALED_NUMERIC
    )

    if keep_coords_raw is None:
        keep_coords_raw = not bool(normalize_coords)

    coord_ranges = dict(
        coord_ranges
        or {
            "t": 7.0,
            "x": 44447.0,
            "y": 39275.0,
        }
    )

    feature_split = build_stage1_feature_split(
        dynamic_features=dynamic,
        static_features=static,
        future_features=future,
        scaled_ml_numeric_cols=scaled,
    )

    scaler_info = {
        name: {
            "scaler_path": scaler_path,
            "all_features": scaled,
            "idx": idx,
        }
        for idx, name in enumerate(scaled)
    }

    payload = {
        "city": city,
        "model": model,
        "stage": "stage1",
        "provenance": {
            "COORD_MODE": coord_mode,
            "coords_in_degrees": bool(coords_in_degrees),
            "coord_epsg_used": coord_epsg_used,
            "COORD_X_COL_USED": "x_m",
            "COORD_Y_COL_USED": "y_m",
            "X_COL_USED": "x_m",
            "Y_COL_USED": "y_m",
            "TIME_COL_USED": "year_numeric_coord",
            "NORMALIZE_COORDS": bool(normalize_coords),
            "KEEP_COORDS_RAW": bool(keep_coords_raw),
            "SHIFT_RAW_COORDS": bool(shift_raw_coords),
        },
        "raw_stats": {
            "xy_units(heuristic)": "meters-like",
            "utm_plausibility": "UTM-like (OK)",
            "t_raw(df_train)": {
                "min": 2015.0,
                "max": 2022.0,
                "mean": 2018.5,
                "std": 2.2913,
            },
            "x_raw(df_train)": {
                "min": 735147.9,
                "max": 779594.9,
                "mean": 756399.0,
                "std": 9406.9,
            },
            "y_raw(df_train)": {
                "min": 2496397.8,
                "max": 2535672.8,
                "mean": 2520578.4,
                "std": 9125.1,
            },
        },
        "raw_samples": {
            "sample_df_train_coords(t,x,y)": (
                "- 2015, 735148, 2530470\n"
                "- 2015, 735149, 2530410\n"
                "- 2015, 735150, 2530350"
            )
        },
        "model_coords": {
            "coords.shape": (
                f"({int(n_sequences)},"
                f" {int(forecast_horizon)}, 3)"
            ),
            "t_in_model": {
                "min": 0.7143,
                "max": 1.0,
                "mean": 0.8571,
                "std": 0.1166,
            },
            "x_in_model": {
                "min": 0.0,
                "max": 1.0,
                "mean": 0.4781,
                "std": 0.2116,
            },
            "y_in_model": {
                "min": 0.0,
                "max": 1.0,
                "mean": 0.6157,
                "std": 0.2323,
            },
            "t_unique_norm": [
                0.7143,
                0.8571,
                1.0,
            ],
        },
        "coord_scaler": {
            "coord_scaler": (
                "MinMaxScaler fitted "
                "(coords normalized inside "
                "sequence builder)"
            ),
            "data_min_(t,x,y)": [
                2015.0,
                735147.9,
                2496397.8,
            ],
            "data_max_(t,x,y)": [
                2022.0,
                779594.9,
                2535672.8,
            ],
            "coord_ranges(chain_rule)": coord_ranges,
        },
        "physics_df_stats": {
            "subsidence_cum__si": {
                "min": 0.0,
                "max": 1.53,
                "mean": 0.069,
                "std": 0.098,
            },
            "GWL_depth_bgs_m__si": {
                "min": 0.0002,
                "max": 1.0,
                "mean": 0.474,
                "std": 0.275,
            },
            "head_m__si": {
                "min": -12.97,
                "max": 9.0,
                "mean": -1.43,
                "std": 2.89,
            },
            "soil_thickness_eff__si": {
                "min": 0.1,
                "max": 18.96,
                "mean": 5.81,
                "std": 2.85,
            },
        },
        "targets_stats": {
            "targets.subsidence": {
                "min": 0.0013,
                "max": 1.53,
                "mean": 0.136,
                "std": 0.14,
            },
            "targets.gwl(head)": {
                "min": -12.97,
                "max": 9.0,
                "mean": -1.42,
                "std": 2.88,
            },
        },
        "scaled_ml_numeric_cols": scaled,
        "feature_split": feature_split,
        "scalers": {
            "main_scaler_path": scaler_path,
            "scaled_ml_numeric_cols": scaled,
            "scaler_info": scaler_info,
        },
        "config": {
            "TIME_STEPS": int(time_steps),
            "FORECAST_HORIZON_YEARS": int(forecast_horizon),
        },
    }
    return payload


def generate_stage1_audit(
    *,
    output_path: PathLike | None = None,
    template: Stage1AuditLike | None = None,
    overrides: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | Path:
    """
    Generate a Stage-1 audit payload or file.

    Parameters
    ----------
    output_path : path-like, optional
        Destination JSON path. If omitted, the payload
        is returned instead of written.
    template : mapping, ArtifactRecord, or path, optional
        Real or synthetic Stage-1 audit template used as
        the generation base.
    overrides : dict, optional
        Nested overrides applied after template/default
        payload creation.
    **kwargs : dict
        Parameters forwarded to
        ``default_stage1_audit_payload`` when no template
        is given.
    """
    if template is None:
        payload = default_stage1_audit_payload(**kwargs)
    else:
        payload = clone_artifact(
            _as_payload(template),
            overrides=None,
        )

    if overrides:
        payload = deep_update(payload, overrides)

    feature_split = payload.get("feature_split")
    if not isinstance(feature_split, dict):
        feature_split = build_stage1_feature_split(
            dynamic_features=list(
                kwargs.get(
                    "dynamic_features", _DEFAULT_DYNAMIC
                )
            ),
            static_features=list(
                kwargs.get("static_features", _DEFAULT_STATIC)
            ),
            future_features=list(
                kwargs.get("future_features", _DEFAULT_FUTURE)
            ),
            scaled_ml_numeric_cols=list(
                payload.get(
                    "scaled_ml_numeric_cols",
                    _DEFAULT_SCALED_NUMERIC,
                )
            ),
        )
        payload["feature_split"] = feature_split

    if output_path is None:
        return payload

    return write_json(payload, output_path)


def load_stage1_audit(
    path: PathLike,
) -> ArtifactRecord:
    """
    Load a Stage-1 audit artifact.

    Raises
    ------
    ValueError
        If the artifact does not look like a Stage-1
        audit payload.
    """
    record = load_artifact(path, kind="stage1_audit")
    needed = {
        "provenance",
        "coord_scaler",
        "feature_split",
    }
    if not needed.issubset(record.payload):
        raise ValueError(
            "The file does not contain the expected "
            "Stage-1 audit sections."
        )
    return record


def stage1_coord_ranges_frame(
    audit: Stage1AuditLike,
) -> pd.DataFrame:
    """Return coord ranges as a tidy frame."""
    payload = _as_payload(audit)
    coord_ranges = nested_get(
        payload,
        "coord_scaler",
        "coord_ranges(chain_rule)",
        default={},
    )
    rows = [
        {
            "coord": str(key),
            "range": float(value),
        }
        for key, value in dict(coord_ranges).items()
        if isinstance(value, (int, float))
    ]
    return pd.DataFrame(rows)


def stage1_feature_split_frame(
    audit: Stage1AuditLike,
) -> pd.DataFrame:
    """
    Explode the feature split into tidy rows.
    """
    payload = _as_payload(audit)
    split = payload.get("feature_split", {}) or {}

    rows: list[dict[str, Any]] = []
    for group_name, group_payload in split.items():
        if not isinstance(group_payload, dict):
            continue

        total = int(group_payload.get("count", 0))
        for bucket in (
            "scaled_by_main_scaler",
            "si_unscaled",
            "other_unscaled_mixed",
        ):
            features = list(
                group_payload.get(bucket, []) or []
            )
            if not features:
                rows.append(
                    {
                        "group": group_name,
                        "bucket": bucket,
                        "feature": None,
                        "count": total,
                        "bucket_size": 0,
                    }
                )
                continue

            for feature in features:
                rows.append(
                    {
                        "group": group_name,
                        "bucket": bucket,
                        "feature": feature,
                        "count": total,
                        "bucket_size": len(features),
                    }
                )
    return pd.DataFrame(rows)


def stage1_stats_frame(
    audit: Stage1AuditLike,
    *,
    section: str = "physics_df_stats",
) -> pd.DataFrame:
    """
    Return a tidy frame for nested variable stats.
    """
    payload = _as_payload(audit)
    src = payload.get(section, {}) or {}

    rows: list[dict[str, Any]] = []
    for name, stats in src.items():
        if not isinstance(stats, dict):
            continue
        for stat_name, value in stats.items():
            if isinstance(value, (int, float)):
                rows.append(
                    {
                        "section": section,
                        "name": str(name),
                        "stat": str(stat_name),
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def summarize_stage1_audit(
    audit: Stage1AuditLike,
) -> dict[str, Any]:
    """
    Build a compact semantic summary for inspection.
    """
    payload = _as_payload(audit)

    split = payload.get("feature_split", {}) or {}
    coord_ranges = nested_get(
        payload,
        "coord_scaler",
        "coord_ranges(chain_rule)",
        default={},
    )
    scaled_cols = list(
        payload.get("scaled_ml_numeric_cols", []) or []
    )

    summary = {
        "brief": {
            "kind": "stage1_audit",
            "city": payload.get("city"),
            "model": payload.get("model"),
            "stage": payload.get("stage", "stage1"),
        },
        "feature_counts": {
            group: int(dict(group_payload).get("count", 0))
            for group, group_payload in split.items()
            if isinstance(group_payload, dict)
        },
        "coord_ranges": dict(coord_ranges),
        "scaled_ml_numeric_cols_count": int(len(scaled_cols)),
        "checks": {
            "coords_normalized": bool(
                nested_get(
                    payload,
                    "provenance",
                    "NORMALIZE_COORDS",
                    default=False,
                )
            ),
            "keep_coords_raw": bool(
                nested_get(
                    payload,
                    "provenance",
                    "KEEP_COORDS_RAW",
                    default=False,
                )
            ),
            "has_coord_ranges": bool(coord_ranges),
            "has_feature_split": bool(split),
            "has_scaler_info": bool(
                nested_get(
                    payload,
                    "scalers",
                    "scaler_info",
                    default={},
                )
            ),
            "has_scaled_numeric_cols": bool(scaled_cols),
            "has_future_features": bool(
                nested_get(
                    split,
                    "future_features",
                    "count",
                    default=0,
                )
            ),
            "has_si_features": any(
                "__si" in str(name)
                for name in scaled_cols
                + list(
                    nested_get(
                        split,
                        "dynamic_features",
                        "si_unscaled",
                        default=[],
                    )
                )
                + list(
                    nested_get(
                        split,
                        "static_features",
                        "si_unscaled",
                        default=[],
                    )
                )
            ),
        },
        "key_variables": {
            "physics": list(
                (payload.get("physics_df_stats") or {}).keys()
            ),
            "targets": list(
                (payload.get("targets_stats") or {}).keys()
            ),
        },
    }
    return summary


def plot_stage1_coord_ranges(
    audit: Stage1AuditLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Stage-1 coord ranges",
    ylabel: str = "range",
    show_grid: bool = True,
    grid_kws: dict[str, Any] | None = None,
    annotate: bool = True,
    annotate_kws: dict[str, Any] | None = None,
    error: str = "ignore",
    **plot_kws: Any,
) -> plt.Axes:
    """Plot chain-rule coordinate ranges."""
    fig, plot_ax, _ = prepare_plot(ax=ax, figsize=(7.0, 4.0))

    frame = stage1_coord_ranges_frame(audit)
    if frame.empty:
        _, plot_ax = empty_plot(
            fig,
            plot_ax,
            title=title,
            message="No coord ranges",
        )
        return plot_ax

    bar_kws = filter_plot_kwargs(
        plot_ax.bar,
        plot_kws,
        error=error,
    )
    bars = plot_ax.bar(
        frame["coord"].astype(str),
        frame["range"],
        **bar_kws,
    )

    _, plot_ax = finalize_plot(
        fig,
        plot_ax,
        title=title,
        ylabel=ylabel,
        show_grid=show_grid,
        grid_kws=grid_kws or {"axis": "y", "alpha": 0.25},
    )

    if annotate:
        text_kws = filter_plot_kwargs(
            plot_ax.text,
            annotate_kws,
            error=error,
        )
        for bar, value in zip(
            bars, frame["range"], strict=False
        ):
            plot_ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{float(value):.4g}",
                ha="center",
                va="bottom",
                **text_kws,
            )
    return plot_ax


def plot_stage1_feature_split(
    audit: Stage1AuditLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Stage-1 feature split",
    xlabel: str = "feature group",
    ylabel: str = "features",
    show_grid: bool = True,
    grid_kws: dict[str, Any] | None = None,
    legend_kws: dict[str, Any] | None = None,
    error: str = "ignore",
    **plot_kws: Any,
) -> plt.Axes:
    """
    Plot feature bucket counts by feature group.
    """
    fig, plot_ax, _ = prepare_plot(ax=ax, figsize=(8.2, 4.4))

    frame = stage1_feature_split_frame(audit)
    if frame.empty:
        _, plot_ax = empty_plot(
            fig,
            plot_ax,
            title=title,
            message="No feature split",
        )
        return plot_ax

    count_frame = (
        frame.groupby(["group", "bucket"])
        .size()
        .rename("n")
        .reset_index()
    )
    pivot = count_frame.pivot(
        index="group",
        columns="bucket",
        values="n",
    ).fillna(0.0)

    plot_call_kws = filter_plot_kwargs(
        pivot.plot,
        plot_kws,
        error=error,
    )
    pivot.plot(
        kind="bar",
        stacked=True,
        ax=plot_ax,
        **plot_call_kws,
    )

    _, plot_ax = finalize_plot(
        fig,
        plot_ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        show_grid=show_grid,
        grid_kws=grid_kws or {"axis": "y", "alpha": 0.25},
        legend=True,
        legend_kws=legend_kws or {"title": "bucket"},
    )
    return plot_ax


def plot_stage1_variable_stats(
    audit: Stage1AuditLike,
    *,
    section: str = "physics_df_stats",
    stat: str = "mean",
    ax: plt.Axes | None = None,
    title: str | None = None,
    error: str = "ignore",
    **plot_kws: Any,
) -> plt.Axes:
    """
    Plot one statistic across variables.
    """
    fig, plot_ax, _ = prepare_plot(ax=ax, figsize=(8.0, 4.2))

    frame = stage1_stats_frame(audit, section=section)
    if frame.empty:
        _, plot_ax = empty_plot(
            fig,
            plot_ax,
            title=title or section,
            message="No numeric variable stats",
        )
        return plot_ax

    sub = frame.loc[frame["stat"].eq(stat)].copy()
    if sub.empty:
        _, plot_ax = empty_plot(
            fig,
            plot_ax,
            title=title or section,
            message=f"Statistic {stat!r} not found",
        )
        return plot_ax

    metrics = dict(
        zip(sub["name"], sub["value"], strict=False)
    )
    plot_metric_bars(
        plot_ax,
        metrics,
        title=title or f"{section}: {stat}",
        sort_by_value=True,
        xlabel=stat,
        error=error,
        **plot_kws,
    )
    return plot_ax


def plot_stage1_target_stats(
    audit: Stage1AuditLike,
    *,
    stat: str = "mean",
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    Plot target summary statistics.
    """
    return plot_stage1_variable_stats(
        audit,
        section="targets_stats",
        stat=stat,
        ax=ax,
        title=title or f"Stage-1 targets: {stat}",
    )


def plot_stage1_boolean_summary(
    audit: Stage1AuditLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Stage-1 audit checks",
    error: str = "ignore",
    **plot_kws: Any,
) -> plt.Axes:
    """Plot semantic pass/fail checks."""
    fig, plot_ax, _ = prepare_plot(ax=ax, figsize=(7.2, 4.0))

    checks = summarize_stage1_audit(audit)["checks"]
    plot_boolean_checks(
        plot_ax,
        checks,
        title=title,
        error=error,
        **plot_kws,
    )
    return plot_ax


def inspect_stage1_audit(
    audit: Stage1AuditLike,
    *,
    output_dir: PathLike | None = None,
    stem: str = "stage1_audit",
    save_figures: bool = True,
) -> dict[str, Any]:
    """
    Inspect a Stage-1 audit and optionally save figures.

    Returns
    -------
    dict
        Bundle containing summary, tabular frames, and
        optionally written figure paths.
    """
    payload = _as_payload(audit)
    summary = summarize_stage1_audit(payload)

    bundle: dict[str, Any] = {
        "summary": summary,
        "frames": {
            "coord_ranges": stage1_coord_ranges_frame(
                payload
            ),
            "feature_split": stage1_feature_split_frame(
                payload
            ),
            "physics_df_stats": stage1_stats_frame(
                payload,
                section="physics_df_stats",
            ),
            "targets_stats": stage1_stats_frame(
                payload,
                section="targets_stats",
            ),
        },
        "figure_paths": {},
    }

    if not (output_dir and save_figures):
        return bundle

    out_dir = as_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plots = {
        f"{stem}_coord_ranges.png": (
            plot_stage1_coord_ranges,
            {},
        ),
        f"{stem}_feature_split.png": (
            plot_stage1_feature_split,
            {},
        ),
        f"{stem}_physics_means.png": (
            plot_stage1_variable_stats,
            {
                "section": "physics_df_stats",
                "stat": "mean",
            },
        ),
        f"{stem}_target_means.png": (
            plot_stage1_target_stats,
            {"stat": "mean"},
        ),
        f"{stem}_checks.png": (
            plot_stage1_boolean_summary,
            {},
        ),
    }

    for name, (func, kwargs) in plots.items():
        fig, ax = plt.subplots(figsize=(8.0, 4.4))
        func(payload, ax=ax, **kwargs)
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        bundle["figure_paths"][name] = str(path)

    return bundle
