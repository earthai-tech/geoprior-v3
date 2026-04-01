# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Calibration-statistics generation and inspection helpers.

This module focuses on the compact calibration-stats
artifact produced by the forecast calibration workflow.
It is intentionally narrower than ``eval_physics.py``:

- ``eval_physics.py`` inspects the full evaluation payload,
- ``calibration_stats.py`` inspects the saved calibration
  stats object itself.

The module also accepts the richer interpretable eval JSON
because Stage-2 stores the same calibration stats under:

``interval_calibration['factors_per_horizon_from_cal_stats']``.
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
    nested_get,
    plot_boolean_checks,
    plot_metric_bars,
    plot_series_map,
    read_json,
    write_json,
)

PathLike = str | Path
CalibrationStatsLike = (
    ArtifactRecord
    | Mapping[str, Any]
    | str
    | Path
)

__all__ = [
    "calibration_stats_factors_frame",
    "calibration_stats_overall_frame",
    "calibration_stats_per_horizon_frame",
    "default_calibration_stats_payload",
    "generate_calibration_stats",
    "inspect_calibration_stats",
    "load_calibration_stats",
    "plot_calibration_boolean_summary",
    "plot_calibration_factors",
    "plot_calibration_overall_metrics",
    "plot_calibration_per_horizon_coverage",
    "plot_calibration_per_horizon_sharpness",
    "summarize_calibration_stats",
]


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _coerce_horizon_map(
    value: Any,
) -> dict[str, float]:
    """Return a stable string-keyed horizon mapping."""
    if value is None:
        return {}

    if isinstance(value, Mapping):
        out: dict[str, float] = {}
        for key, item in value.items():
            try:
                out[str(int(float(key)))] = float(item)
            except Exception:
                continue
        return out

    if isinstance(value, (list, tuple, np.ndarray)):
        out = {}
        for idx, item in enumerate(value, start=1):
            try:
                out[str(idx)] = float(item)
            except Exception:
                continue
        return out

    return {}


def _extract_payload(
    obj: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Extract calibration stats from either:

    - a direct calibration-stats payload, or
    - an interpretable eval JSON containing the nested
      ``interval_calibration`` block.
    """
    if "interval_calibration" in obj:
        interval_block = obj.get("interval_calibration") or {}
        if not isinstance(interval_block, Mapping):
            return dict(obj)

        nested = (
            interval_block.get(
                "factors_per_horizon_from_cal_stats"
            )
            or {}
        )

        if isinstance(nested, Mapping) and nested:
            payload = dict(nested)
            if (
                "factors" not in payload
                and "factors_per_horizon" in interval_block
            ):
                payload["factors"] = _coerce_horizon_map(
                    interval_block.get("factors_per_horizon")
                )
            return payload

        return dict(interval_block)

    return dict(obj)


def _as_payload(
    stats: CalibrationStatsLike,
) -> dict[str, Any]:
    """Return a plain calibration-stats payload."""
    if isinstance(stats, ArtifactRecord):
        return _extract_payload(stats.payload)

    if isinstance(stats, Mapping):
        return _extract_payload(stats)

    payload = read_json(stats)
    return _extract_payload(payload)


def _per_horizon_rows(
    payload: dict[str, Any],
    *,
    which: str,
) -> list[dict[str, Any]]:
    """Build tidy per-horizon rows."""
    block = nested_get(payload, which, "per_horizon", default={})
    if not isinstance(block, Mapping):
        return []

    rows: list[dict[str, Any]] = []
    for key, item in block.items():
        if not isinstance(item, Mapping):
            continue

        row = {
            "which": str(which),
            "horizon": str(key),
            "coverage": item.get("coverage"),
            "sharpness": item.get("sharpness"),
        }
        rows.append(row)

    try:
        rows.sort(key=lambda r: int(float(r["horizon"])))
    except Exception:
        rows.sort(key=lambda r: r["horizon"])

    return rows


# ------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------

def default_calibration_stats_payload(
    *,
    target: float = 0.80,
    interval: tuple[float, float] = (0.10, 0.90),
    f_max: float = 5.0,
    tol: float = 0.02,
    factors: dict[str, float] | None = None,
    coverage_before: float = 0.865,
    coverage_after: float = 0.867,
    sharpness_before: float = 33.08,
    sharpness_after: float = 33.38,
) -> dict[str, Any]:
    """
    Build a realistic default calibration-stats payload.

    The structure follows the object saved by the
    calibration workflow and later embedded into the
    interpretable eval JSON.
    """
    factors = factors or {
        "1": 1.0,
        "2": 1.0,
        "3": 1.0183744430541992,
    }

    payload = {
        "target": float(target),
        "interval": [float(interval[0]), float(interval[1])],
        "f_max": float(f_max),
        "tol": float(tol),
        "overall_key": "__overall__",
        "factors_source": "fit",
        "factors": {str(k): float(v) for k, v in factors.items()},
        "eval_before": {
            "coverage": float(coverage_before),
            "sharpness": float(sharpness_before),
            "per_horizon": {
                "1": {
                    "coverage": 0.9790543662405667,
                    "sharpness": 23.244874687581664,
                },
                "2": {
                    "coverage": 0.8223728117459829,
                    "sharpness": 27.592224706985775,
                },
                "3": {
                    "coverage": 0.7935725653267621,
                    "sharpness": 48.40746668258111,
                },
            },
        },
        "eval_after": {
            "coverage": float(coverage_after),
            "sharpness": float(sharpness_after),
            "per_horizon": {
                "1": {
                    "coverage": 0.9790543662405667,
                    "sharpness": 23.244874687581664,
                },
                "2": {
                    "coverage": 0.8223728117459829,
                    "sharpness": 27.592224706985775,
                },
                "3": {
                    "coverage": 0.8000410698701166,
                    "sharpness": 49.29692692253824,
                },
            },
        },
    }
    return payload



def generate_calibration_stats(
    path: PathLike,
    *,
    template: CalibrationStatsLike | None = None,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """
    Generate and save a calibration-stats JSON file.

    Parameters
    ----------
    path : str or pathlib.Path
        Output JSON path.
    template : mapping, path, ArtifactRecord, optional
        Optional source payload. If omitted, a realistic
        default payload is used.
    overrides : dict, optional
        Deep overrides applied after template resolution.
    """
    base = (
        _as_payload(template)
        if template is not None
        else default_calibration_stats_payload()
    )
    payload = clone_artifact(base, overrides=overrides)
    return write_json(payload, path)


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load_calibration_stats(
    path: PathLike,
) -> ArtifactRecord:
    """
    Load a calibration-stats artifact.

    Notes
    -----
    This loader is list-safe and nested-block aware. It can
    read either a direct ``calibration_stats.json`` payload or
    an interpretable eval JSON from which the nested block is
    extracted.
    """
    raw = read_json(path)
    payload = _extract_payload(raw)
    p = as_path(path)

    meta = {
        "top_keys": list(payload),
        "n_top_keys": len(payload),
        "source_name": p.name,
        "has_eval_before": "eval_before" in payload,
        "has_eval_after": "eval_after" in payload,
        "has_factors": "factors" in payload,
    }

    return ArtifactRecord(
        path=p,
        kind="calibration_stats",
        payload=payload,
        stage=None,
        city=nested_get(raw, "city"),
        model=nested_get(raw, "model"),
        meta=meta,
    )


# ------------------------------------------------------------------
# Frames
# ------------------------------------------------------------------

def calibration_stats_factors_frame(
    stats: CalibrationStatsLike,
) -> pd.DataFrame:
    """Return per-horizon calibration factors."""
    payload = _as_payload(stats)
    factors = _coerce_horizon_map(payload.get("factors"))

    rows = [
        {
            "horizon": key,
            "factor": value,
        }
        for key, value in factors.items()
    ]

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    try:
        frame = frame.sort_values(
            by="horizon",
            key=lambda s: s.astype(float),
        )
    except Exception:
        frame = frame.sort_values("horizon")

    return frame.reset_index(drop=True)



def calibration_stats_overall_frame(
    stats: CalibrationStatsLike,
) -> pd.DataFrame:
    """Return before/after overall calibration metrics."""
    payload = _as_payload(stats)

    rows = []
    target = payload.get("target")
    for which in ("eval_before", "eval_after"):
        block = payload.get(which) or {}
        if not isinstance(block, Mapping):
            continue

        coverage = block.get("coverage")
        sharpness = block.get("sharpness")
        rows.append(
            {
                "which": which,
                "coverage": coverage,
                "sharpness": sharpness,
                "coverage_error": (
                    abs(float(coverage) - float(target))
                    if coverage is not None and target is not None
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)



def calibration_stats_per_horizon_frame(
    stats: CalibrationStatsLike,
    *,
    which: str = "eval_after",
) -> pd.DataFrame:
    """
    Return per-horizon coverage and sharpness.

    Parameters
    ----------
    which : {'eval_before', 'eval_after'}
        Which calibration stage to extract.
    """
    payload = _as_payload(stats)
    return pd.DataFrame(_per_horizon_rows(payload, which=which))


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

def summarize_calibration_stats(
    stats: CalibrationStatsLike,
) -> dict[str, Any]:
    """Return a compact summary of calibration behavior."""
    payload = _as_payload(stats)

    target = payload.get("target")
    tol = payload.get("tol")
    factors = _coerce_horizon_map(payload.get("factors"))

    before_cov = nested_get(payload, "eval_before", "coverage")
    after_cov = nested_get(payload, "eval_after", "coverage")
    before_sharp = nested_get(payload, "eval_before", "sharpness")
    after_sharp = nested_get(payload, "eval_after", "sharpness")

    before_error = (
        abs(float(before_cov) - float(target))
        if before_cov is not None and target is not None
        else None
    )
    after_error = (
        abs(float(after_cov) - float(target))
        if after_cov is not None and target is not None
        else None
    )

    max_factor = max(factors.values()) if factors else None
    min_factor = min(factors.values()) if factors else None

    summary = {
        "target": target,
        "interval_low": nested_get(payload, "interval", default=[None, None])[0],
        "interval_high": nested_get(payload, "interval", default=[None, None])[1],
        "tol": tol,
        "n_horizons": len(factors),
        "factors_source": payload.get("factors_source"),
        "coverage_before": before_cov,
        "coverage_after": after_cov,
        "sharpness_before": before_sharp,
        "sharpness_after": after_sharp,
        "coverage_error_before": before_error,
        "coverage_error_after": after_error,
        "coverage_error_improved": (
            (after_error <= before_error)
            if after_error is not None and before_error is not None
            else None
        ),
        "target_reached_after": (
            (after_error <= float(tol))
            if after_error is not None and tol is not None
            else None
        ),
        "max_factor": max_factor,
        "min_factor": min_factor,
        "has_eval_before": isinstance(
            payload.get("eval_before"),
            Mapping,
        ),
        "has_eval_after": isinstance(
            payload.get("eval_after"),
            Mapping,
        ),
        "has_factors": bool(factors),
        "skipped": bool(payload.get("skipped", False)),
    }
    return summary


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_calibration_factors(
    ax: plt.Axes,
    stats: CalibrationStatsLike,
    *,
    title: str = "Calibration factors",
) -> plt.Axes:
    """Plot per-horizon widening factors."""
    frame = calibration_stats_factors_frame(stats)
    if frame.empty:
        ax.set_title(title)
        ax.text(
            0.5,
            0.5,
            "No calibration factors",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    ax.bar(frame["horizon"], frame["factor"])
    ax.set_title(title)
    ax.set_xlabel("horizon")
    ax.set_ylabel("factor")
    ax.grid(axis="y", alpha=0.25)
    return ax



def plot_calibration_overall_metrics(
    ax: plt.Axes,
    stats: CalibrationStatsLike,
    *,
    title: str = "Calibration summary",
) -> plt.Axes:
    """Plot overall before/after calibration metrics."""
    frame = calibration_stats_overall_frame(stats)
    if frame.empty:
        ax.set_title(title)
        ax.text(
            0.5,
            0.5,
            "No overall calibration metrics",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    plot_frame = pd.DataFrame(
        {
            "metric": [
                "coverage_before",
                "coverage_after",
                "sharpness_before",
                "sharpness_after",
            ],
            "value": [
                frame.loc[
                    frame["which"] == "eval_before",
                    "coverage",
                ].iloc[0]
                if (frame["which"] == "eval_before").any()
                else np.nan,
                frame.loc[
                    frame["which"] == "eval_after",
                    "coverage",
                ].iloc[0]
                if (frame["which"] == "eval_after").any()
                else np.nan,
                frame.loc[
                    frame["which"] == "eval_before",
                    "sharpness",
                ].iloc[0]
                if (frame["which"] == "eval_before").any()
                else np.nan,
                frame.loc[
                    frame["which"] == "eval_after",
                    "sharpness",
                ].iloc[0]
                if (frame["which"] == "eval_after").any()
                else np.nan,
            ],
        }
    )
    return plot_metric_bars(
        ax,
        plot_frame,
        title=title,
    )



def plot_calibration_per_horizon_coverage(
    ax: plt.Axes,
    stats: CalibrationStatsLike,
    *,
    which: str = "eval_after",
    title: str | None = None,
) -> plt.Axes:
    """Plot per-horizon coverage."""
    frame = calibration_stats_per_horizon_frame(
        stats,
        which=which,
    )
    plot_title = title or f"Coverage by horizon ({which})"

    if frame.empty:
        ax.set_title(plot_title)
        ax.text(
            0.5,
            0.5,
            "No per-horizon coverage",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    data = dict(zip(frame["horizon"], frame["coverage"], strict=False))
    return plot_series_map(
        ax,
        data,
        title=plot_title,
        xlabel="horizon",
        ylabel="coverage",
    )



def plot_calibration_per_horizon_sharpness(
    ax: plt.Axes,
    stats: CalibrationStatsLike,
    *,
    which: str = "eval_after",
    title: str | None = None,
) -> plt.Axes:
    """Plot per-horizon sharpness."""
    frame = calibration_stats_per_horizon_frame(
        stats,
        which=which,
    )
    plot_title = title or f"Sharpness by horizon ({which})"

    if frame.empty:
        ax.set_title(plot_title)
        ax.text(
            0.5,
            0.5,
            "No per-horizon sharpness",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    data = dict(zip(frame["horizon"], frame["sharpness"], strict=False))
    return plot_series_map(
        ax,
        data,
        title=plot_title,
        xlabel="horizon",
        ylabel="sharpness",
    )



def plot_calibration_boolean_summary(
    ax: plt.Axes,
    stats: CalibrationStatsLike,
    *,
    title: str = "Calibration checks",
) -> plt.Axes:
    """Plot compact boolean checks for calibration status."""
    summary = summarize_calibration_stats(stats)
    checks = {
        "has_eval_before": summary.get("has_eval_before"),
        "has_eval_after": summary.get("has_eval_after"),
        "has_factors": summary.get("has_factors"),
        "coverage_error_improved": summary.get(
            "coverage_error_improved"
        ),
        "target_reached_after": summary.get(
            "target_reached_after"
        ),
        "not_skipped": not bool(summary.get("skipped", False)),
    }
    return plot_boolean_checks(
        ax,
        checks,
        title=title,
    )


# ------------------------------------------------------------------
# Inspection bundle
# ------------------------------------------------------------------

def inspect_calibration_stats(
    stats: CalibrationStatsLike,
) -> dict[str, Any]:
    """
    Build a compact inspection bundle.

    Returns
    -------
    dict
        A dictionary containing the raw payload, a compact
        summary, and tidy frames useful for gallery lessons,
        notebooks, or debugging.
    """
    payload = _as_payload(stats)
    summary = summarize_calibration_stats(payload)

    return {
        "payload": payload,
        "summary": summary,
        "overall": calibration_stats_overall_frame(payload),
        "factors": calibration_stats_factors_frame(payload),
        "per_horizon_before": calibration_stats_per_horizon_frame(
            payload,
            which="eval_before",
        ),
        "per_horizon_after": calibration_stats_per_horizon_frame(
            payload,
            which="eval_after",
        ),
    }
