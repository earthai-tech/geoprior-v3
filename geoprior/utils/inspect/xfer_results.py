# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Transfer-results generation and inspection helpers.

This module focuses on the ``xfer_results.json`` artifact
used for transfer-learning and cross-city workflow review.

Unlike most other inspection artifacts in this folder,
``xfer_results.json`` is typically a **JSON list of result
records**, not a single JSON object. Each record describes
one transfer job and usually combines:

- transfer direction and city pair,
- strategy / calibration / rescale choices,
- overall evaluation metrics,
- per-horizon metrics,
- schema-alignment diagnostics,
- warm-start details when relevant,
- exported CSV paths for evaluation and future forecasts.

The helpers here are designed for two common uses:

1. Sphinx-Gallery examples that need a realistic transfer
   results artifact without rerunning the transfer workflow.
2. Real workflow inspection when a user wants to compare
   directions, strategies, or schema quality at a glance.

Notes
-----
This module intentionally does **not** use the shared
``read_json()`` helper from ``inspect.utils`` because that
helper expects a top-level JSON object, while
``xfer_results.json`` is a top-level JSON list.
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
    ensure_parent_dir,
    json_ready,
    plot_boolean_checks,
)

PathLike = str | Path
XferResultsLike = (
    Sequence[Mapping[str, Any]]
    | str
    | Path
)

__all__ = [
    "default_xfer_results_payload",
    "generate_xfer_results",
    "inspect_xfer_results",
    "load_xfer_results",
    "plot_xfer_boolean_summary",
    "plot_xfer_direction_metric",
    "plot_xfer_overall_metrics",
    "plot_xfer_per_horizon_metrics",
    "plot_xfer_schema_counts",
    "summarize_xfer_results",
    "xfer_overall_frame",
    "xfer_per_horizon_frame",
    "xfer_schema_frame",
    "xfer_warm_frame",
]

_OVERALL_KEYS = [
    "coverage80",
    "sharpness80",
    "overall_mae",
    "overall_mse",
    "overall_rmse",
    "overall_r2",
]

_SCHEMA_BOOL_KEYS = [
    "static_aligned",
    "dynamic_reordered",
    "future_reordered",
    "dynamic_order_mismatch",
    "future_order_mismatch",
]

_SCHEMA_COUNT_KEYS = [
    "static_missing_n",
    "static_extra_n",
]


def _try_float(value: Any) -> float | None:
    """Return ``value`` as float when possible."""
    try:
        return float(value)
    except Exception:
        return None


def _read_records(path: PathLike) -> list[dict[str, Any]]:
    """Read a transfer-results JSON list."""
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    if not isinstance(data, list):
        raise ValueError(
            "Expected a JSON list at "
            f"{p!s}, got {type(data).__name__}."
        )

    records: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, Mapping):
            records.append(dict(item))
    return records


def _write_records(
    records: list[dict[str, Any]],
    path: PathLike,
    *,
    indent: int = 2,
) -> Path:
    """Write transfer-results records as JSON."""
    p = ensure_parent_dir(path)
    safe = json_ready(records)
    with p.open("w", encoding="utf-8") as stream:
        json.dump(
            safe,
            stream,
            indent=indent,
            ensure_ascii=False,
        )
        stream.write("\n")
    return p


def _as_records(
    xfer: XferResultsLike,
) -> list[dict[str, Any]]:
    """Return a plain list of transfer records."""
    if isinstance(xfer, (str, Path)):
        return _read_records(xfer)

    records: list[dict[str, Any]] = []
    for item in xfer:
        if isinstance(item, Mapping):
            records.append(dict(item))
    return records


def _deep_update_record(
    base: dict[str, Any],
    updates: dict[str, Any] | None,
) -> dict[str, Any]:
    """Recursively update one record."""
    out = copy.deepcopy(base)
    if not updates:
        return out

    for key, value in updates.items():
        if (
            isinstance(value, dict)
            and isinstance(out.get(key), dict)
        ):
            out[key] = _deep_update_record(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _apply_overrides(
    records: list[dict[str, Any]],
    overrides: (
        dict[str, Any]
        | list[dict[str, Any]]
        | None
    ) = None,
) -> list[dict[str, Any]]:
    """
    Apply optional overrides to transfer records.

    Accepted forms
    --------------
    - dict:
        applied to every record.
    - list[dict]:
        applied record-wise up to the shortest length.
    """
    out = [copy.deepcopy(rec) for rec in records]
    if overrides is None:
        return out

    if isinstance(overrides, dict):
        return [
            _deep_update_record(rec, overrides)
            for rec in out
        ]

    if isinstance(overrides, list):
        n = min(len(out), len(overrides))
        for idx in range(n):
            if isinstance(overrides[idx], dict):
                out[idx] = _deep_update_record(
                    out[idx], overrides[idx]
                )
        return out

    raise TypeError(
        "`overrides` must be a dict, a list of dicts, or None."
    )


def _record_label(record: dict[str, Any]) -> str:
    """Return a compact record label."""
    direction = str(record.get("direction", "")).strip()
    strategy = str(record.get("strategy", "")).strip()
    calibration = str(record.get("calibration", "")).strip()
    rescale = str(record.get("rescale_mode", "")).strip()

    bits = [
        part
        for part in [direction, strategy, calibration, rescale]
        if part
    ]
    return " | ".join(bits) if bits else "record"


def _schema_block(record: dict[str, Any]) -> dict[str, Any]:
    """Return the most relevant schema block."""
    schema = record.get("schema", None)
    if isinstance(schema, dict):
        return schema

    warm = record.get("warm", None)
    if isinstance(warm, dict):
        nested = warm.get("schema", None)
        if isinstance(nested, dict):
            return nested

    return {}


def _warm_block(record: dict[str, Any]) -> dict[str, Any]:
    """Return warm-start details if present."""
    warm = record.get("warm", None)
    return warm if isinstance(warm, dict) else {}


def _feature_count(record: dict[str, Any], key: str) -> int:
    """Return feature-mismatch count from schema."""
    val = _try_float(_schema_block(record).get(key))
    return int(val) if val is not None else 0


def default_xfer_results_payload(
    *,
    src_city: str = "nansha",
    tgt_city: str = "zhongshan",
    model_name: str = "GeoPriorSubsNet",
    split: str = "test",
    calibration: str = "source",
    strategy: str = "warm",
    rescale_mode: str = "strict",
) -> list[dict[str, Any]]:
    """
    Build a realistic default transfer-results payload.

    The payload mirrors the common pattern where one
    ``xfer_results.json`` file stores multiple transfer
    records, often one per direction.
    """
    base = [
        {
            "strategy": str(strategy),
            "rescale_mode": str(rescale_mode),
            "warm": {
                "warm_split": "val",
                "warm_samples": 20000,
                "warm_frac": None,
                "warm_epochs": 3,
                "warm_lr": 1e-4,
                "warm_seed": 42,
                "schema": {
                    "src_city": str(src_city),
                    "tgt_city": str(tgt_city),
                    "static_aligned": True,
                    "dynamic_reordered": False,
                    "future_reordered": False,
                    "dynamic_order_mismatch": False,
                    "future_order_mismatch": False,
                    "static_missing_n": 9,
                    "static_extra_n": 6,
                },
            },
            "model_path": (
                "results/"
                f"{src_city}_{model_name}_stage1/"
                "train_20260222-141331/"
                f"{src_city}_{model_name}_H3_best.keras"
            ),
            "split": str(split),
            "calibration": str(calibration),
            "quantiles": [0.1, 0.5, 0.9],
            "coverage80": 0.8696590273,
            "sharpness80": 51.5739153704,
            "overall_mae": 14.3019965746,
            "overall_mse": 499.0390742467,
            "overall_rmse": 22.3391824883,
            "overall_r2": 0.8237503940,
            "per_horizon_mae": {
                "H1": 9.2531121755,
                "H2": 14.4207786362,
                "H3": 19.2320989120,
            },
            "per_horizon_mse": {
                "H1": 196.9444467984,
                "H2": 465.9660166509,
                "H3": 834.2067592909,
            },
            "per_horizon_rmse": {
                "H1": 14.0336897072,
                "H2": 21.5862460064,
                "H3": 28.8826376789,
            },
            "per_horizon_r2": {
                "H1": 0.8880748376,
                "H2": 0.8260040842,
                "H3": 0.7469995261,
            },
            "csv_eval": (
                "results/xfer/"
                f"{src_city}__{tgt_city}/"
                "20260227-101651/"
                f"{src_city}_to_{tgt_city}_"
                "warm_test_source_strict_eval.csv"
            ),
            "csv_future": (
                "results/xfer/"
                f"{src_city}__{tgt_city}/"
                "20260227-101651/"
                f"{src_city}_to_{tgt_city}_"
                "warm_test_source_strict_future.csv"
            ),
            "model_dir": (
                "results/"
                f"{src_city}_{model_name}_stage1/"
                "train_20260222-141331"
            ),
            "schema": {
                "src_city": str(src_city),
                "tgt_city": str(tgt_city),
                "static_aligned": True,
                "dynamic_reordered": False,
                "future_reordered": False,
                "dynamic_order_mismatch": False,
                "future_order_mismatch": False,
                "static_missing_n": 9,
                "static_extra_n": 6,
            },
            "source_model": "auto",
            "source_load": "auto",
            "hps_mode": "auto",
            "model_name": str(model_name),
            "prefer_artifact": "keras",
            "metrics_source": "eval_csv",
            "subsidence_unit": "mm",
            "metrics_unit": "mm",
            "direction": "A_to_B",
            "source_city": str(src_city),
            "target_city": str(tgt_city),
            "job_index": 1,
            "job_total": 2,
        },
        {
            "strategy": str(strategy),
            "rescale_mode": str(rescale_mode),
            "warm": {
                "warm_split": "val",
                "warm_samples": 20000,
                "warm_frac": None,
                "warm_epochs": 3,
                "warm_lr": 1e-4,
                "warm_seed": 42,
                "schema": {
                    "src_city": str(tgt_city),
                    "tgt_city": str(src_city),
                    "static_aligned": True,
                    "dynamic_reordered": False,
                    "future_reordered": False,
                    "dynamic_order_mismatch": False,
                    "future_order_mismatch": False,
                    "static_missing_n": 6,
                    "static_extra_n": 9,
                },
            },
            "model_path": (
                "results/"
                f"{tgt_city}_{model_name}_stage1/"
                "train_20260218-175001/"
                f"{tgt_city}_{model_name}_H3_best.keras"
            ),
            "split": str(split),
            "calibration": str(calibration),
            "quantiles": [0.1, 0.5, 0.9],
            "coverage80": 0.8231599843,
            "sharpness80": 108.6511001552,
            "overall_mae": 36.4496400584,
            "overall_mse": 3719.0734195425,
            "overall_rmse": 60.9842063123,
            "overall_r2": 0.7610593342,
            "per_horizon_mae": {
                "H1": 12.4219541768,
                "H2": 34.8364833708,
                "H3": 62.0904826275,
            },
            "per_horizon_mse": {
                "H1": 318.3459065124,
                "H2": 2597.6668210567,
                "H3": 8241.2075310584,
            },
            "per_horizon_rmse": {
                "H1": 17.8422506011,
                "H2": 50.9673112991,
                "H3": 90.7810967716,
            },
            "per_horizon_r2": {
                "H1": 0.8925951204,
                "H2": 0.8019549161,
                "H3": 0.6433359875,
            },
            "csv_eval": (
                "results/xfer/"
                f"{src_city}__{tgt_city}/"
                "20260227-101651/"
                f"{tgt_city}_to_{src_city}_"
                "warm_test_source_strict_eval.csv"
            ),
            "csv_future": (
                "results/xfer/"
                f"{src_city}__{tgt_city}/"
                "20260227-101651/"
                f"{tgt_city}_to_{src_city}_"
                "warm_test_source_strict_future.csv"
            ),
            "model_dir": (
                "results/"
                f"{tgt_city}_{model_name}_stage1/"
                "train_20260218-175001"
            ),
            "schema": {
                "src_city": str(tgt_city),
                "tgt_city": str(src_city),
                "static_aligned": True,
                "dynamic_reordered": False,
                "future_reordered": False,
                "dynamic_order_mismatch": False,
                "future_order_mismatch": False,
                "static_missing_n": 6,
                "static_extra_n": 9,
            },
            "source_model": "auto",
            "source_load": "auto",
            "hps_mode": "auto",
            "model_name": str(model_name),
            "prefer_artifact": "keras",
            "metrics_source": "eval_csv",
            "subsidence_unit": "mm",
            "metrics_unit": "mm",
            "direction": "B_to_A",
            "source_city": str(tgt_city),
            "target_city": str(src_city),
            "job_index": 2,
            "job_total": 2,
        },
    ]
    return base


def generate_xfer_results(
    path: PathLike,
    *,
    template: list[dict[str, Any]] | None = None,
    overrides: (
        dict[str, Any]
        | list[dict[str, Any]]
        | None
    ) = None,
) -> Path:
    """
    Generate a reproducible transfer-results artifact.

    Parameters
    ----------
    path : path-like
        Destination JSON path.
    template : list of dict, optional
        Existing transfer records to reuse.
    overrides : dict or list of dict, optional
        Optional updates applied either to all records
        or record-wise.
    """
    records = copy.deepcopy(
        template if template is not None
        else default_xfer_results_payload()
    )
    records = _apply_overrides(records, overrides=overrides)
    return _write_records(records, path)


def load_xfer_results(
    xfer: XferResultsLike,
) -> list[dict[str, Any]]:
    """Load transfer-results records."""
    return _as_records(xfer)


def xfer_overall_frame(
    xfer: XferResultsLike,
) -> pd.DataFrame:
    """
    Return one tidy row per transfer record.

    The frame exposes the most useful comparison
    columns for quick ranking and filtering.
    """
    rows: list[dict[str, Any]] = []
    for record in _as_records(xfer):
        row = {
            "label": _record_label(record),
            "direction": record.get("direction"),
            "source_city": record.get("source_city"),
            "target_city": record.get("target_city"),
            "strategy": record.get("strategy"),
            "rescale_mode": record.get("rescale_mode"),
            "split": record.get("split"),
            "calibration": record.get("calibration"),
            "model_name": record.get("model_name"),
            "metrics_unit": record.get("metrics_unit"),
            "subsidence_unit": record.get("subsidence_unit"),
            "metrics_source": record.get("metrics_source"),
            "prefer_artifact": record.get("prefer_artifact"),
        }
        for key in _OVERALL_KEYS:
            num = _try_float(record.get(key))
            if num is not None:
                row[key] = float(num)
        row["n_quantiles"] = len(record.get("quantiles", []) or [])
        row["job_index"] = _try_float(record.get("job_index"))
        row["job_total"] = _try_float(record.get("job_total"))
        rows.append(row)

    frame = pd.DataFrame(rows)
    return frame.reset_index(drop=True)


def xfer_per_horizon_frame(
    xfer: XferResultsLike,
) -> pd.DataFrame:
    """
    Return a tidy per-horizon metric frame.

    The output is useful for comparing whether
    transfer quality degrades differently across
    directions or strategies as horizon increases.
    """
    rows: list[dict[str, Any]] = []
    for record in _as_records(xfer):
        label = _record_label(record)
        base = {
            "label": label,
            "direction": record.get("direction"),
            "source_city": record.get("source_city"),
            "target_city": record.get("target_city"),
            "strategy": record.get("strategy"),
            "rescale_mode": record.get("rescale_mode"),
            "calibration": record.get("calibration"),
        }

        for metric in ["mae", "mse", "rmse", "r2"]:
            mapping = record.get(f"per_horizon_{metric}", {}) or {}
            if not isinstance(mapping, Mapping):
                continue
            for horizon, value in mapping.items():
                num = _try_float(value)
                if num is None:
                    continue
                rows.append(
                    {
                        **base,
                        "metric": metric,
                        "horizon": str(horizon),
                        "value": float(num),
                    }
                )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["horizon_index"] = (
            frame["horizon"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .astype(float)
        )
        frame = frame.sort_values(
            ["metric", "direction", "horizon_index"]
        )
    return frame.reset_index(drop=True)


def xfer_schema_frame(
    xfer: XferResultsLike,
) -> pd.DataFrame:
    """
    Return schema-alignment diagnostics in tidy form.
    """
    rows: list[dict[str, Any]] = []
    for record in _as_records(xfer):
        schema = _schema_block(record)
        base = {
            "label": _record_label(record),
            "direction": record.get("direction"),
            "source_city": record.get("source_city"),
            "target_city": record.get("target_city"),
            "strategy": record.get("strategy"),
        }

        for key in _SCHEMA_BOOL_KEYS:
            val = schema.get(key)
            if isinstance(val, bool):
                rows.append(
                    {
                        **base,
                        "kind": "bool",
                        "name": key,
                        "value": bool(val),
                    }
                )

        for key in _SCHEMA_COUNT_KEYS:
            num = _try_float(schema.get(key))
            if num is not None:
                rows.append(
                    {
                        **base,
                        "kind": "count",
                        "name": key,
                        "value": float(num),
                    }
                )

    return pd.DataFrame(rows).reset_index(drop=True)


def xfer_warm_frame(
    xfer: XferResultsLike,
) -> pd.DataFrame:
    """Return warm-start settings in tidy form."""
    rows: list[dict[str, Any]] = []
    for record in _as_records(xfer):
        warm = _warm_block(record)
        row = {
            "label": _record_label(record),
            "direction": record.get("direction"),
            "strategy": record.get("strategy"),
            "warm_split": warm.get("warm_split"),
        }
        for key in [
            "warm_samples",
            "warm_frac",
            "warm_epochs",
            "warm_lr",
            "warm_seed",
        ]:
            val = warm.get(key)
            num = _try_float(val)
            row[key] = float(num) if num is not None else val
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def summarize_xfer_results(
    xfer: XferResultsLike,
) -> dict[str, Any]:
    """
    Build a compact transfer-results summary.

    The summary is intentionally workflow-oriented
    rather than exhaustive.
    """
    records = _as_records(xfer)
    overall = xfer_overall_frame(records)
    per_h = xfer_per_horizon_frame(records)

    out: dict[str, Any] = {
        "n_records": len(records),
        "directions": sorted(
            {
                str(rec.get("direction"))
                for rec in records
                if rec.get("direction") is not None
            }
        ),
        "strategies": sorted(
            {
                str(rec.get("strategy"))
                for rec in records
                if rec.get("strategy") is not None
            }
        ),
        "calibrations": sorted(
            {
                str(rec.get("calibration"))
                for rec in records
                if rec.get("calibration") is not None
            }
        ),
        "rescale_modes": sorted(
            {
                str(rec.get("rescale_mode"))
                for rec in records
                if rec.get("rescale_mode") is not None
            }
        ),
    }

    if not overall.empty:
        best_rmse = overall.sort_values("overall_rmse").iloc[0]
        best_r2 = overall.sort_values(
            "overall_r2", ascending=False
        ).iloc[0]
        out["best_overall_rmse"] = {
            "label": best_rmse["label"],
            "value": float(best_rmse["overall_rmse"]),
        }
        out["best_overall_r2"] = {
            "label": best_r2["label"],
            "value": float(best_r2["overall_r2"]),
        }
        out["mean_coverage80"] = float(
            overall["coverage80"].mean()
        )
        out["mean_sharpness80"] = float(
            overall["sharpness80"].mean()
        )

    if not per_h.empty:
        rmse_h = per_h[per_h["metric"] == "rmse"].copy()
        if not rmse_h.empty:
            worst = rmse_h.sort_values("value").iloc[-1]
            out["worst_horizon_rmse"] = {
                "label": worst["label"],
                "horizon": worst["horizon"],
                "value": float(worst["value"]),
            }

    schema = xfer_schema_frame(records)
    if not schema.empty:
        bools = schema[schema["kind"] == "bool"].copy()
        counts = schema[schema["kind"] == "count"].copy()

        if not bools.empty:
            pass_rates = (
                bools.groupby("name")["value"]
                .mean()
                .to_dict()
            )
            out["schema_pass_rates"] = {
                str(k): float(v)
                for k, v in pass_rates.items()
            }

        if not counts.empty:
            count_means = (
                counts.groupby("name")["value"]
                .mean()
                .to_dict()
            )
            out["schema_mean_counts"] = {
                str(k): float(v)
                for k, v in count_means.items()
            }

    return out


def plot_xfer_overall_metrics(
    xfer: XferResultsLike,
    *,
    metrics: list[str] | tuple[str, ...] | None = None,
    figsize: tuple[float, float] = (8.4, 4.8),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot selected overall metrics for each record.
    """
    frame = xfer_overall_frame(xfer)
    if metrics is None:
        metrics = ("overall_rmse", "overall_r2")

    fig, ax = plt.subplots(figsize=figsize)

    if frame.empty:
        ax.set_title("Transfer overall metrics")
        ax.text(
            0.5,
            0.5,
            "No transfer records",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig, ax

    labels = frame["label"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.8 / max(len(metrics), 1)

    for idx, metric in enumerate(metrics):
        if metric not in frame.columns:
            continue
        vals = frame[metric].astype(float).to_numpy()
        offset = (
            idx - (len(metrics) - 1) / 2.0
        ) * width
        ax.bar(x + offset, vals, width=width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Transfer overall metrics")
    ax.set_ylabel("value")
    ax.grid(axis="y", alpha=0.25)
    if len(metrics) > 1:
        ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_xfer_direction_metric(
    xfer: XferResultsLike,
    *,
    metric: str = "overall_rmse",
    figsize: tuple[float, float] = (7.8, 4.2),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot one overall metric by transfer direction.
    """
    frame = xfer_overall_frame(xfer)
    fig, ax = plt.subplots(figsize=figsize)

    if frame.empty or metric not in frame.columns:
        ax.set_title(f"Direction comparison: {metric}")
        ax.text(
            0.5,
            0.5,
            "Metric not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig, ax

    plot = frame.sort_values("direction")
    ax.bar(plot["direction"].astype(str), plot[metric].astype(float))
    ax.set_title(f"Direction comparison: {metric}")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_xfer_per_horizon_metrics(
    xfer: XferResultsLike,
    *,
    metric: str = "rmse",
    figsize: tuple[float, float] = (8.0, 4.8),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot one per-horizon metric as lines over horizon.
    """
    frame = xfer_per_horizon_frame(xfer)
    fig, ax = plt.subplots(figsize=figsize)

    if frame.empty:
        ax.set_title(f"Transfer per-horizon {metric}")
        ax.text(
            0.5,
            0.5,
            "No per-horizon metrics",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig, ax

    sub = frame[frame["metric"] == metric].copy()
    if sub.empty:
        ax.set_title(f"Transfer per-horizon {metric}")
        ax.text(
            0.5,
            0.5,
            "Requested metric not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig, ax

    for label, grp in sub.groupby("label", sort=False):
        g = grp.sort_values("horizon_index")
        ax.plot(
            g["horizon"].astype(str),
            g["value"].astype(float),
            marker="o",
            label=label,
        )

    ax.set_title(f"Transfer per-horizon {metric}")
    ax.set_xlabel("horizon")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_xfer_schema_counts(
    xfer: XferResultsLike,
    *,
    figsize: tuple[float, float] = (8.0, 4.6),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot schema mismatch counts by record.
    """
    schema = xfer_schema_frame(xfer)
    fig, ax = plt.subplots(figsize=figsize)

    if schema.empty:
        ax.set_title("Schema mismatch counts")
        ax.text(
            0.5,
            0.5,
            "No schema diagnostics",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig, ax

    sub = schema[schema["kind"] == "count"].copy()
    if sub.empty:
        ax.set_title("Schema mismatch counts")
        ax.text(
            0.5,
            0.5,
            "No schema count diagnostics",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig, ax

    labels = list(dict.fromkeys(sub["label"].astype(str)))
    x = np.arange(len(labels))
    width = 0.35

    missing = []
    extra = []
    for label in labels:
        grp = sub[sub["label"] == label]
        miss = grp.loc[
            grp["name"] == "static_missing_n", "value"
        ]
        ext = grp.loc[
            grp["name"] == "static_extra_n", "value"
        ]
        missing.append(float(miss.iloc[0]) if not miss.empty else 0.0)
        extra.append(float(ext.iloc[0]) if not ext.empty else 0.0)

    ax.bar(x - width / 2.0, missing, width=width, label="static_missing_n")
    ax.bar(x + width / 2.0, extra, width=width, label="static_extra_n")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Schema mismatch counts")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_xfer_boolean_summary(
    xfer: XferResultsLike,
    *,
    figsize: tuple[float, float] = (8.4, 4.8),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot aggregated schema boolean pass rates.

    This turns per-record schema booleans into one
    compact pass-rate view.
    """
    schema = xfer_schema_frame(xfer)
    fig, ax = plt.subplots(figsize=figsize)

    if schema.empty:
        ax.set_title("Transfer boolean summary")
        ax.text(
            0.5,
            0.5,
            "No schema diagnostics",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig, ax

    sub = schema[schema["kind"] == "bool"].copy()
    if sub.empty:
        ax.set_title("Transfer boolean summary")
        ax.text(
            0.5,
            0.5,
            "No schema boolean checks",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig, ax

    agg = (
        sub.groupby("name")["value"]
        .mean()
        .sort_index()
        .to_dict()
    )
    checks = {
        key: bool(np.isclose(val, 1.0))
        for key, val in agg.items()
    }

    # Reuse the shared boolean plot helper by converting
    # pass rates to strict pass/fail flags.
    plot_boolean_checks(
        ax,
        checks,
        title="Transfer boolean summary",
    )
    return fig, ax


def inspect_xfer_results(
    xfer: XferResultsLike,
) -> dict[str, Any]:
    """
    Build a compact inspection bundle.

    Returns
    -------
    dict
        A dictionary containing:
        - ``summary`` : workflow summary,
        - ``overall`` : overall metrics frame,
        - ``per_horizon`` : per-horizon frame,
        - ``schema`` : schema diagnostics frame,
        - ``warm`` : warm-start frame.
    """
    records = _as_records(xfer)
    return {
        "summary": summarize_xfer_results(records),
        "overall": xfer_overall_frame(records),
        "per_horizon": xfer_per_horizon_frame(records),
        "schema": xfer_schema_frame(records),
        "warm": xfer_warm_frame(records),
    }
