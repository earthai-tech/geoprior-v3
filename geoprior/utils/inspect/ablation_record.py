# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Ablation-record generation and inspection helpers.

This module focuses on newline-delimited JSON (JSONL)
ablation logs, typically files such as
``ablation_record.jsonl`` where each line represents one
ablation run or one evaluation snapshot.

Compared with the first starter version of this module,
this revision is aligned with the real ablation records
currently used in GeoPrior. The sample record includes:

- workflow identity fields such as timestamp, city, and
  model,
- physics/config knobs such as pde_mode,
  use_effective_h, kappa_mode, hd_factor, and the
  lambda_* weights,
- compact scalar forecast metrics,
- a nested ``metrics`` block that mirrors top-level
  metrics and carries units,
- per-horizon metric maps such as ``per_horizon_mae`` and
  ``per_horizon_r2``.

The helpers here therefore support two modes at once:

1. faithful handling of the current real schema,
2. tolerant handling of future JSONL drift.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import (
    empty_plot,
    ensure_parent_dir,
    filter_plot_kwargs,
    finalize_plot,
    json_ready,
    plot_boolean_checks,
    prepare_plot,
)

PathLike = str | Path
AblationRecordLike = Sequence[Mapping[str, Any]] | str | Path

__all__ = [
    "ablation_config_frame",
    "ablation_metrics_frame",
    "ablation_per_horizon_frame",
    "ablation_record_flags_frame",
    "ablation_record_runs_frame",
    "default_ablation_record_payload",
    "generate_ablation_record",
    "inspect_ablation_record",
    "load_ablation_record",
    "plot_ablation_boolean_summary",
    "plot_ablation_lambda_weights",
    "plot_ablation_metric_by_variant",
    "plot_ablation_per_horizon_metric",
    "plot_ablation_run_counts",
    "plot_ablation_top_variants",
    "summarize_ablation_record",
]

_TOP_LEVEL_METRICS = [
    "r2",
    "mse",
    "mae",
    "rmse",
    "coverage80",
    "sharpness80",
    "epsilon_prior",
    "epsilon_cons",
    "epsilon_gw",
]

_LAMBDA_KEYS = [
    "lambda_cons",
    "lambda_gw",
    "lambda_prior",
    "lambda_smooth",
    "lambda_mv",
    "lambda_bounds",
    "lambda_q",
]

_CONFIG_KEYS = [
    "timestamp",
    "city",
    "model",
    "pde_mode",
    "use_effective_h",
    "kappa_mode",
    "hd_factor",
    *_LAMBDA_KEYS,
]

_BOOL_FLAG_KEYS = [
    "use_effective_h",
]

_PER_H_KEYS = [
    "per_horizon_mae",
    "per_horizon_r2",
]

_UNITS_KEYS = [
    "subs_unit_to_si",
    "subs_factor_si_to_real",
    "subs_metrics_unit",
    "time_units",
    "seconds_per_time_unit",
]


def _try_float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except Exception:
        return None


def _try_int(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except Exception:
        return None


def _deep_get(
    mapping: Mapping[str, Any] | None,
    *keys: str,
    default: Any = None,
) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _record_label(
    record: Mapping[str, Any],
    index: int,
) -> str:
    for key in (
        "ablation",
        "variant",
        "label",
        "name",
        "experiment",
        "tag",
        "timestamp",
    ):
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text

    parts = []
    city = record.get("city")
    pde_mode = record.get("pde_mode")
    if city is not None:
        parts.append(str(city))
    if pde_mode is not None:
        parts.append(str(pde_mode))
    if parts:
        return " | ".join(parts)

    return f"record_{index}"


def _record_seed(
    record: Mapping[str, Any],
) -> int | None:
    for key in (
        "seed",
        "random_state",
        "trial_seed",
    ):
        value = record.get(key)
        if value is not None:
            return _try_int(value)
    return None


def _metrics_block(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    block = record.get("metrics")
    if isinstance(block, Mapping):
        return dict(block)
    return {}


def _units_block(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    top = record.get("units")
    if isinstance(top, Mapping):
        return dict(top)

    nested = _deep_get(record, "metrics", "units", default={})
    if isinstance(nested, Mapping):
        return dict(nested)

    return {}


def _metric_value(
    record: Mapping[str, Any],
    key: str,
) -> float | None:
    direct = _try_float(record.get(key))
    if direct is not None:
        return direct

    metrics = _metrics_block(record)
    nested = _try_float(metrics.get(key))
    if nested is not None:
        return nested

    return None


def _per_horizon_block(
    record: Mapping[str, Any],
    key: str,
) -> dict[str, float]:
    value = record.get(key)
    if value is None:
        value = _metrics_block(record).get(key)

    if not isinstance(value, Mapping):
        return {}

    out: dict[str, float] = {}
    for h_key, h_val in value.items():
        num = _try_float(h_val)
        if num is None:
            continue
        out[str(h_key)] = float(num)

    def _sort_key(text: str) -> tuple[int, str]:
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            return (0, f"{int(digits):06d}")
        return (1, text)

    return {
        key: out[key] for key in sorted(out, key=_sort_key)
    }


def _normalize_record(
    record: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "record_id": int(index),
        "variant": _record_label(record, index),
        "seed": _record_seed(record),
    }

    for key in _CONFIG_KEYS:
        if key in record:
            row[key] = record.get(key)

    for key in _TOP_LEVEL_METRICS:
        value = _metric_value(record, key)
        if value is not None:
            row[key] = value

    units = _units_block(record)
    for key in _UNITS_KEYS:
        if key in units:
            row[key] = units.get(key)

    return row


def _read_records(path: PathLike) -> list[dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    records: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if not isinstance(obj, Mapping):
                raise ValueError(
                    "Expected a JSON object at line "
                    f"{lineno} in {p!s}, got "
                    f"{type(obj).__name__}."
                )
            records.append(dict(obj))
    return records


def _write_records(
    records: list[dict[str, Any]],
    path: PathLike,
) -> Path:
    p = ensure_parent_dir(path)
    with p.open("w", encoding="utf-8") as stream:
        for rec in json_ready(records):
            json.dump(rec, stream, ensure_ascii=False)
            stream.write("\n")
    return p


def _as_records(
    obj: AblationRecordLike,
) -> list[dict[str, Any]]:
    if isinstance(obj, (str, Path)):
        return _read_records(obj)

    out: list[dict[str, Any]] = []
    for item in obj:
        if isinstance(item, Mapping):
            out.append(dict(item))
    return out


def default_ablation_record_payload(
    *,
    city: str = "nansha",
    model: str = "GeoPriorSubsNet",
) -> list[dict[str, Any]]:
    """
    Build a realistic default ablation JSONL payload.

    The structure mirrors the current real record shape:
    one JSON object per line, with top-level config knobs,
    repeated compact metrics, a nested ``metrics`` block,
    units, and per-horizon metric maps.
    """
    base = {
        "timestamp": "20260228-191355",
        "city": str(city),
        "model": str(model),
        "pde_mode": "both",
        "use_effective_h": True,
        "kappa_mode": "kb",
        "hd_factor": 0.6,
        "lambda_cons": 0.0,
        "lambda_gw": 0.1,
        "lambda_prior": 0.0,
        "lambda_smooth": 0.01,
        "lambda_mv": 0.0,
        "lambda_bounds": 0.05,
        "lambda_q": 0.0005,
        "r2": 0.8797,
        "mse": 3.15e-4,
        "mae": 0.0119,
        "rmse": 0.0178,
        "coverage80": 0.8554,
        "sharpness80": 0.0454,
        "metrics": {
            "r2": 0.8797,
            "mse": 3.15e-4,
            "mae": 0.0119,
            "rmse": 0.0178,
            "coverage80": 0.8554,
            "sharpness80": 0.0454,
            "units": {
                "subs_unit_to_si": 0.001,
                "subs_factor_si_to_real": 1000.0,
                "subs_metrics_unit": "m",
                "time_units": "year",
                "seconds_per_time_unit": 31556952.0,
            },
        },
        "units": {
            "subs_unit_to_si": 0.001,
            "subs_factor_si_to_real": 1000.0,
            "subs_metrics_unit": "m",
            "time_units": "year",
            "seconds_per_time_unit": 31556952.0,
        },
        "epsilon_prior": 8.78,
        "epsilon_cons": 5.52e-3,
        "epsilon_gw": 4.38e-7,
        "per_horizon_mae": {
            "H1": 0.00536,
            "H2": 0.01229,
            "H3": 0.01807,
        },
        "per_horizon_r2": {
            "H1": 0.8924,
            "H2": 0.8812,
            "H3": 0.8720,
        },
    }

    variants = [
        ("baseline", 0.0, 0.0),
        ("gw_heavier", -0.0004, 0.05),
        ("smoother", 0.0003, 0.0),
        ("bounds_stronger", 0.0002, 0.0),
    ]

    rows: list[dict[str, Any]] = []
    for idx, (name, mae_delta, gw_delta) in enumerate(
        variants,
        start=1,
    ):
        rec = copy.deepcopy(base)
        rec["ablation"] = name
        rec["seed"] = idx
        rec["mae"] = float(rec["mae"] + mae_delta)
        rec["rmse"] = float(rec["rmse"] + mae_delta)
        rec["r2"] = float(rec["r2"] - mae_delta * 2.0)
        rec["lambda_gw"] = float(rec["lambda_gw"] + gw_delta)
        rec["metrics"]["mae"] = rec["mae"]
        rec["metrics"]["rmse"] = rec["rmse"]
        rec["metrics"]["r2"] = rec["r2"]
        rows.append(rec)
    return rows


def _deep_update(
    base: dict[str, Any],
    updates: Mapping[str, Any] | None,
) -> dict[str, Any]:
    out = copy.deepcopy(base)
    if not updates:
        return out

    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(
            out.get(key), Mapping
        ):
            out[key] = _deep_update(dict(out[key]), value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def generate_ablation_record(
    output_path: PathLike,
    *,
    overrides: dict[str, Any]
    | list[dict[str, Any]]
    | None = None,
    city: str = "nansha",
    model: str = "GeoPriorSubsNet",
) -> Path:
    """Write a realistic demo ablation JSONL file."""
    records = default_ablation_record_payload(
        city=city,
        model=model,
    )

    if overrides is None:
        return _write_records(records, output_path)

    if isinstance(overrides, Mapping):
        updated = [
            _deep_update(rec, overrides) for rec in records
        ]
        return _write_records(updated, output_path)

    if isinstance(overrides, list):
        updated = [copy.deepcopy(rec) for rec in records]
        n = min(len(updated), len(overrides))
        for idx in range(n):
            item = overrides[idx]
            if isinstance(item, Mapping):
                updated[idx] = _deep_update(
                    updated[idx], item
                )
        return _write_records(updated, output_path)

    raise TypeError(
        "`overrides` must be a dict, "
        "a list of dicts, or None."
    )


def load_ablation_record(
    src: AblationRecordLike,
) -> list[dict[str, Any]]:
    """Load ablation JSONL records into a plain list."""
    return _as_records(src)


def ablation_record_runs_frame(
    src: AblationRecordLike,
) -> pd.DataFrame:
    """Return one tidy row per ablation record."""
    rows = [
        _normalize_record(rec, idx)
        for idx, rec in enumerate(_as_records(src), start=1)
    ]
    return pd.DataFrame(rows)


def ablation_metrics_frame(
    src: AblationRecordLike,
) -> pd.DataFrame:
    """Return long-form scalar metric rows."""
    rows: list[dict[str, Any]] = []
    records = _as_records(src)

    for idx, rec in enumerate(records, start=1):
        label = _record_label(rec, idx)
        seed = _record_seed(rec)
        for key in _TOP_LEVEL_METRICS:
            value = _metric_value(rec, key)
            if value is None:
                continue
            rows.append(
                {
                    "record_id": idx,
                    "variant": label,
                    "seed": seed,
                    "metric": key,
                    "value": value,
                }
            )

    return pd.DataFrame(rows)


def ablation_per_horizon_frame(
    src: AblationRecordLike,
) -> pd.DataFrame:
    """Return long-form per-horizon metric rows."""
    rows: list[dict[str, Any]] = []
    records = _as_records(src)

    for idx, rec in enumerate(records, start=1):
        label = _record_label(rec, idx)
        seed = _record_seed(rec)
        for key in _PER_H_KEYS:
            block = _per_horizon_block(rec, key)
            metric = key.replace("per_horizon_", "")
            for horizon, value in block.items():
                rows.append(
                    {
                        "record_id": idx,
                        "variant": label,
                        "seed": seed,
                        "metric": metric,
                        "horizon": horizon,
                        "value": value,
                    }
                )

    return pd.DataFrame(rows)


def ablation_record_flags_frame(
    src: AblationRecordLike,
) -> pd.DataFrame:
    """Return long-form boolean/config flags."""
    rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(_as_records(src), start=1):
        label = _record_label(rec, idx)
        seed = _record_seed(rec)
        for key in _BOOL_FLAG_KEYS:
            value = rec.get(key)
            if isinstance(value, bool):
                rows.append(
                    {
                        "record_id": idx,
                        "variant": label,
                        "seed": seed,
                        "flag": key,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def ablation_config_frame(
    src: AblationRecordLike,
) -> pd.DataFrame:
    """Return one row per record with config knobs."""
    rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(_as_records(src), start=1):
        row = {
            "record_id": idx,
            "variant": _record_label(rec, idx),
            "seed": _record_seed(rec),
        }
        for key in _CONFIG_KEYS:
            if key in rec:
                row[key] = rec.get(key)
        units = _units_block(rec)
        for key in _UNITS_KEYS:
            if key in units:
                row[key] = units.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_ablation_record(
    src: AblationRecordLike,
) -> dict[str, Any]:
    """Return a semantic summary of ablation JSONL."""
    records = _as_records(src)
    runs = ablation_record_runs_frame(records)
    metrics = ablation_metrics_frame(records)
    per_h = ablation_per_horizon_frame(records)
    ablation_record_flags_frame(records)

    variants = []
    if not runs.empty:
        variants = (
            runs["variant"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        variants = sorted(variants)

    seeds = []
    if not runs.empty and runs["seed"].notna().any():
        seeds = (
            runs["seed"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        seeds = sorted(seeds)

    summary: dict[str, Any] = {
        "record_count": int(len(records)),
        "variant_count": int(len(variants)),
        "seed_count": int(len(seeds)),
        "variants": variants,
        "has_metrics": bool(not metrics.empty),
        "has_per_horizon": bool(not per_h.empty),
        "has_units": bool(
            not runs.empty
            and runs["time_units"].notna().any()
        ),
        "has_lambda_weights": bool(
            not runs.empty
            and any(
                key in runs.columns for key in _LAMBDA_KEYS
            )
        ),
        "best_by_rmse": None,
        "best_by_r2": None,
    }

    if not metrics.empty:
        sub = metrics.loc[metrics["metric"].eq("rmse")]
        if not sub.empty:
            agg = (
                sub.groupby("variant", as_index=False)[
                    "value"
                ]
                .mean()
                .sort_values("value", ascending=True)
            )
            summary["best_by_rmse"] = agg.iloc[0].to_dict()

        sub = metrics.loc[metrics["metric"].eq("r2")]
        if not sub.empty:
            agg = (
                sub.groupby("variant", as_index=False)[
                    "value"
                ]
                .mean()
                .sort_values("value", ascending=False)
            )
            summary["best_by_r2"] = agg.iloc[0].to_dict()

    summary["checks"] = {
        "has_records": bool(records),
        "has_timestamp": bool(
            not runs.empty and runs["timestamp"].notna().any()
        ),
        "has_core_metrics": bool(not metrics.empty),
        "has_per_horizon_metrics": bool(not per_h.empty),
        "has_units_block": bool(summary["has_units"]),
        "has_config_knobs": bool(
            not runs.empty
            and runs[
                [c for c in _CONFIG_KEYS if c in runs.columns]
            ]
            .notna()
            .any()
            .any()
        ),
    }
    return summary


def plot_ablation_run_counts(
    src: AblationRecordLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Ablation runs per variant",
    xlabel: str = "variant",
    ylabel: str = "runs",
    show_grid: bool = True,
    grid_kws: dict[str, Any] | None = None,
    rotate_xticks: float | int | None = 25,
    annotate: bool = False,
    annotate_kws: dict[str, Any] | None = None,
    error: str = "ignore",
    **plot_kws: Any,
) -> plt.Axes:
    fig, plot_ax, _ = prepare_plot(ax=ax, figsize=(8.0, 4.4))

    frame = ablation_record_runs_frame(src)
    if frame.empty:
        _, plot_ax = empty_plot(
            fig,
            plot_ax,
            title=title,
            message="No ablation records",
        )
        return plot_ax

    counts = (
        frame.groupby("variant")
        .size()
        .sort_values(ascending=False)
    )
    x = np.arange(len(counts), dtype=float)
    bar_kws = filter_plot_kwargs(
        plot_ax.bar,
        plot_kws,
        error=error,
    )
    bars = plot_ax.bar(x, counts.to_numpy(), **bar_kws)
    plot_ax.set_xticks(x)
    plot_ax.set_xticklabels(counts.index)

    _, plot_ax = finalize_plot(
        fig,
        plot_ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        show_grid=show_grid,
        grid_kws=grid_kws or {"axis": "y", "alpha": 0.25},
        rotate_xticks=rotate_xticks,
    )

    if annotate:
        text_kws = filter_plot_kwargs(
            plot_ax.text,
            annotate_kws,
            error=error,
        )
        for bar in bars:
            height = bar.get_height()
            plot_ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                **text_kws,
            )

    return plot_ax


def plot_ablation_metric_by_variant(
    src: AblationRecordLike,
    *,
    metric: str = "rmse",
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str = "variant",
    ylabel: str | None = None,
    show_grid: bool = True,
    grid_kws: dict[str, Any] | None = None,
    rotate_xticks: float | int | None = 25,
    annotate: bool = False,
    annotate_kws: dict[str, Any] | None = None,
    error: str = "ignore",
    **plot_kws: Any,
) -> plt.Axes:
    fig, plot_ax, _ = prepare_plot(ax=ax, figsize=(8.0, 4.4))

    frame = ablation_metrics_frame(src)
    sub = frame.loc[frame["metric"].eq(metric)].copy()
    if sub.empty:
        _, plot_ax = empty_plot(
            fig,
            plot_ax,
            title=title or metric,
            message=f"Metric {metric!r} not found",
        )
        return plot_ax

    ascending = metric not in {"r2", "coverage80"}
    agg = (
        sub.groupby("variant", as_index=False)["value"]
        .mean()
        .sort_values("value", ascending=ascending)
    )

    x = np.arange(len(agg), dtype=float)
    bar_kws = filter_plot_kwargs(
        plot_ax.bar,
        plot_kws,
        error=error,
    )
    bars = plot_ax.bar(x, agg["value"].to_numpy(), **bar_kws)
    plot_ax.set_xticks(x)
    plot_ax.set_xticklabels(agg["variant"].tolist())

    _, plot_ax = finalize_plot(
        fig,
        plot_ax,
        title=title or f"Mean {metric} by variant",
        xlabel=xlabel,
        ylabel=ylabel or metric,
        show_grid=show_grid,
        grid_kws=grid_kws or {"axis": "y", "alpha": 0.25},
        rotate_xticks=rotate_xticks,
    )

    if annotate:
        text_kws = filter_plot_kwargs(
            plot_ax.text,
            annotate_kws,
            error=error,
        )
        for bar, value in zip(
            bars, agg["value"], strict=False
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


def plot_ablation_lambda_weights(
    src: AblationRecordLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Lambda weights by variant",
    xlabel: str = "variant",
    ylabel: str = "weight",
    show_grid: bool = True,
    grid_kws: dict[str, Any] | None = None,
    rotate_xticks: float | int | None = 25,
    legend: bool = True,
    legend_kws: dict[str, Any] | None = None,
    error: str = "ignore",
    **plot_kws: Any,
) -> plt.Axes:
    fig, plot_ax, _ = prepare_plot(ax=ax, figsize=(8.2, 4.8))

    frame = ablation_config_frame(src)
    keep = [
        key for key in _LAMBDA_KEYS if key in frame.columns
    ]
    if frame.empty or not keep:
        _, plot_ax = empty_plot(
            fig,
            plot_ax,
            title=title,
            message="No lambda weights available",
        )
        return plot_ax

    agg = (
        frame.groupby("variant", as_index=False)[keep]
        .mean()
        .set_index("variant")
    )
    n_groups = len(agg.index)
    n_series = len(keep)
    x = np.arange(n_groups, dtype=float)
    width = 0.8 / max(1, n_series)
    bar_kws = filter_plot_kwargs(
        plot_ax.bar,
        plot_kws,
        error=error,
    )

    for idx, key in enumerate(keep):
        offset = (idx - (n_series - 1) / 2.0) * width
        plot_ax.bar(
            x + offset,
            agg[key].to_numpy(),
            width=width,
            label=key,
            **bar_kws,
        )

    plot_ax.set_xticks(x)
    plot_ax.set_xticklabels(agg.index.tolist())

    _, plot_ax = finalize_plot(
        fig,
        plot_ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        show_grid=show_grid,
        grid_kws=grid_kws or {"axis": "y", "alpha": 0.25},
        rotate_xticks=rotate_xticks,
        legend=legend,
        legend_kws=legend_kws,
    )
    return plot_ax


def plot_ablation_per_horizon_metric(
    src: AblationRecordLike,
    *,
    metric: str = "mae",
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str = "horizon",
    ylabel: str | None = None,
    show_grid: bool = True,
    grid_kws: dict[str, Any] | None = None,
    legend: bool = True,
    legend_kws: dict[str, Any] | None = None,
    error: str = "ignore",
    **plot_kws: Any,
) -> plt.Axes:
    fig, plot_ax, _ = prepare_plot(ax=ax, figsize=(8.0, 4.6))

    frame = ablation_per_horizon_frame(src)
    sub = frame.loc[frame["metric"].eq(metric)].copy()
    if sub.empty:
        _, plot_ax = empty_plot(
            fig,
            plot_ax,
            title=title or metric,
            message=f"Per-horizon metric {metric!r} not found",
        )
        return plot_ax

    pivot = sub.pivot_table(
        index="horizon",
        columns="variant",
        values="value",
        aggfunc="mean",
    )
    line_kws = filter_plot_kwargs(
        plot_ax.plot,
        plot_kws,
        error=error,
    )
    if "marker" not in line_kws:
        line_kws["marker"] = "o"

    x = np.arange(len(pivot.index), dtype=float)
    for variant in pivot.columns:
        plot_ax.plot(
            x,
            pivot[variant].to_numpy(dtype=float),
            label=str(variant),
            **line_kws,
        )

    plot_ax.set_xticks(x)
    plot_ax.set_xticklabels(pivot.index.tolist())

    _, plot_ax = finalize_plot(
        fig,
        plot_ax,
        title=title or f"Per-horizon {metric}",
        xlabel=xlabel,
        ylabel=ylabel or metric,
        show_grid=show_grid,
        grid_kws=grid_kws or {"axis": "y", "alpha": 0.25},
        legend=legend,
        legend_kws=legend_kws,
    )
    return plot_ax


def plot_ablation_top_variants(
    src: AblationRecordLike,
    *,
    metric: str = "rmse",
    top_n: int = 5,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "variant",
    show_grid: bool = True,
    grid_kws: dict[str, Any] | None = None,
    annotate: bool = False,
    annotate_kws: dict[str, Any] | None = None,
    error: str = "ignore",
    **plot_kws: Any,
) -> plt.Axes:
    fig, plot_ax, _ = prepare_plot(ax=ax, figsize=(8.0, 4.4))

    frame = ablation_metrics_frame(src)
    sub = frame.loc[frame["metric"].eq(metric)].copy()
    if sub.empty:
        _, plot_ax = empty_plot(
            fig,
            plot_ax,
            title=title or metric,
            message=f"Metric {metric!r} not available",
        )
        return plot_ax

    ascending = metric not in {"r2", "coverage80"}
    agg = sub.groupby("variant", as_index=False)[
        "value"
    ].mean()
    agg = agg.sort_values("value", ascending=ascending)
    agg = agg.head(max(1, int(top_n)))

    y = np.arange(len(agg), dtype=float)
    bar_kws = filter_plot_kwargs(
        plot_ax.barh,
        plot_kws,
        error=error,
    )
    bars = plot_ax.barh(y, agg["value"].to_numpy(), **bar_kws)
    plot_ax.set_yticks(y)
    plot_ax.set_yticklabels(agg["variant"].tolist())

    _, plot_ax = finalize_plot(
        fig,
        plot_ax,
        title=title or f"Top variants by {metric}",
        xlabel=xlabel or metric,
        ylabel=ylabel,
        show_grid=show_grid,
        grid_kws=grid_kws or {"axis": "x", "alpha": 0.25},
    )
    if ascending:
        plot_ax.invert_yaxis()

    if annotate:
        text_kws = filter_plot_kwargs(
            plot_ax.text,
            annotate_kws,
            error=error,
        )
        for bar, value in zip(
            bars, agg["value"], strict=False
        ):
            plot_ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2.0,
                f" {float(value):.4g}",
                va="center",
                **text_kws,
            )

    return plot_ax


def plot_ablation_boolean_summary(
    src: AblationRecordLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Ablation record checks",
    **plot_kws: Any,
) -> plt.Axes:
    _, plot_ax, _ = prepare_plot(ax=ax, figsize=(7.6, 4.0))
    checks = summarize_ablation_record(src)["checks"]
    plot_boolean_checks(
        plot_ax,
        checks,
        title=title,
        **plot_kws,
    )
    return plot_ax


def inspect_ablation_record(
    src: AblationRecordLike,
    *,
    output_dir: PathLike | None = None,
    stem: str = "ablation_record",
    save_figures: bool = True,
) -> dict[str, Any]:
    """Inspect ablation JSONL and optionally save figures."""
    records = _as_records(src)
    bundle: dict[str, Any] = {
        "summary": summarize_ablation_record(records),
        "frames": {
            "runs": ablation_record_runs_frame(records),
            "metrics": ablation_metrics_frame(records),
            "per_horizon": ablation_per_horizon_frame(
                records
            ),
            "flags": ablation_record_flags_frame(records),
            "config": ablation_config_frame(records),
        },
        "figure_paths": {},
    }

    if not (output_dir and save_figures):
        return bundle

    out_dir = ensure_parent_dir(Path(output_dir) / "dummy")
    out_dir = out_dir.parent
    plots = {
        f"{stem}_run_counts.png": (
            plot_ablation_run_counts,
            {},
        ),
        f"{stem}_rmse.png": (
            plot_ablation_metric_by_variant,
            {"metric": "rmse"},
        ),
        f"{stem}_r2.png": (
            plot_ablation_metric_by_variant,
            {"metric": "r2"},
        ),
        f"{stem}_lambda_weights.png": (
            plot_ablation_lambda_weights,
            {},
        ),
        f"{stem}_per_h_mae.png": (
            plot_ablation_per_horizon_metric,
            {"metric": "mae"},
        ),
        f"{stem}_checks.png": (
            plot_ablation_boolean_summary,
            {},
        ),
    }

    for name, (func, kwargs) in plots.items():
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        func(records, ax=ax, **kwargs)
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        bundle["figure_paths"][name] = str(path)

    return bundle
