# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Evaluation-diagnostics generation and inspection helpers.

This module focuses on the compact forecast-evaluation
JSON artifact usually written as ``*_eval_diagnostics_*.json``.

Unlike the richer Stage-2 interpretable evaluation payload,
this artifact is intentionally small and centered on:

- per-year evaluation summaries,
- an ``__overall__`` aggregate block,
- per-horizon point metrics,
- interval diagnostics such as coverage/sharpness,
- temporal stability via PSS.

The functions are designed for two common uses:

1. Sphinx-Gallery examples that need a realistic
   diagnostics artifact without rerunning evaluation.
2. Real workflow inspection when a user wants to review
   per-year and per-horizon forecast quality at a glance.
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
    empty_plot,
    filter_plot_kwargs,
    finalize_plot,
    load_artifact,
    plot_boolean_checks,
    plot_metric_bars,
    prepare_plot,
    read_json,
    write_json,
)

PathLike = str | Path
EvalDiagnosticsLike = (
    ArtifactRecord | Mapping[str, Any] | str | Path
)

__all__ = [
    "default_eval_diagnostics_payload",
    "eval_overall_frame",
    "eval_per_horizon_frame",
    "eval_years_frame",
    "generate_eval_diagnostics",
    "inspect_eval_diagnostics",
    "load_eval_diagnostics",
    "plot_eval_boolean_summary",
    "plot_eval_overall_metrics",
    "plot_eval_per_horizon_metrics",
    "plot_eval_year_metric_trend",
    "summarize_eval_diagnostics",
]

_YEAR_METRICS = [
    "overall_mae",
    "overall_mse",
    "overall_rmse",
    "overall_r2",
    "coverage80",
    "sharpness80",
    "pss",
]

_PER_H_METRICS = ["mae", "mse", "rmse", "r2"]


def _as_payload(
    diagnostics: EvalDiagnosticsLike,
) -> dict[str, Any]:
    """Return a plain evaluation-diagnostics payload."""
    if isinstance(diagnostics, ArtifactRecord):
        return dict(diagnostics.payload)

    if isinstance(diagnostics, Mapping):
        return dict(diagnostics)

    payload = read_json(diagnostics)
    return dict(payload)


def _try_float(value: Any) -> float | None:
    """Return ``value`` as float when possible."""
    try:
        return float(value)
    except Exception:
        return None


def _year_keys(payload: dict[str, Any]) -> list[str]:
    """Return sorted year-like keys excluding ``__overall__``."""
    pairs: list[tuple[float, str]] = []
    for key, value in payload.items():
        if key == "__overall__" or not isinstance(
            value, dict
        ):
            continue
        num = _try_float(key)
        if num is None:
            continue
        pairs.append((num, str(key)))
    return [key for _, key in sorted(pairs)]


def _overall_block(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Return the ``__overall__`` block if present."""
    block = payload.get("__overall__", {})
    return block if isinstance(block, dict) else {}


def _per_h_metric_map(
    block: dict[str, Any] | None,
    metric: str,
) -> dict[str, float]:
    """Extract one ``per_horizon_*`` map as float values."""
    src = (block or {}).get(f"per_horizon_{metric}", {}) or {}
    out: dict[str, float] = {}
    for key, value in src.items():
        num = _try_float(value)
        if num is not None:
            out[str(key)] = float(num)
    return out


def _top_scalar_metrics(
    block: dict[str, Any] | None,
    *,
    keys: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Extract selected top-level scalar metrics from a block."""
    want = list(keys or _YEAR_METRICS)
    out: dict[str, float] = {}
    for key in want:
        num = _try_float((block or {}).get(key))
        if num is not None:
            out[str(key)] = float(num)
    return out


def default_eval_diagnostics_payload(
    *,
    years: list[int] | tuple[int, ...] | None = None,
    per_horizon_mae: list[float] | None = None,
    per_horizon_mse: list[float] | None = None,
    per_horizon_rmse: list[float] | None = None,
    per_horizon_r2: list[float] | None = None,
    coverage80: list[float] | None = None,
    sharpness80: list[float] | None = None,
    pss: list[float] | None = None,
) -> dict[str, Any]:
    """
    Build a realistic default eval-diagnostics payload.

    The payload is template-based. It is not meant to
    reproduce the full evaluation pipeline. Instead, it
    creates a stable and inspectable diagnostics artifact
    with the same broad structure as the real compact
    ``eval_diagnostics`` JSON.
    """
    years = list(years or [2020, 2021, 2022])
    n = len(years)

    mae = list(per_horizon_mae or [3.52, 8.31, 15.98])
    mse = list(per_horizon_mse or [24.32, 152.73, 610.36])
    rmse = list(
        per_horizon_rmse
        or [float(np.sqrt(max(v, 0.0))) for v in mse]
    )
    r2 = list(per_horizon_r2 or [0.896, 0.888, 0.874])
    cov = list(coverage80 or [0.979, 0.822, 0.794])
    sharp = list(sharpness80 or [23.24, 27.59, 48.41])
    pss_vals = list(pss or [38.24, 53.74, 70.71])

    # Align lengths conservatively.
    n = min(
        n,
        len(mae),
        len(mse),
        len(rmse),
        len(r2),
        len(cov),
        len(sharp),
        len(pss_vals),
    )
    years = years[:n]
    mae = mae[:n]
    mse = mse[:n]
    rmse = rmse[:n]
    r2 = r2[:n]
    cov = cov[:n]
    sharp = sharp[:n]
    pss_vals = pss_vals[:n]

    payload: dict[str, Any] = {}
    for idx, year in enumerate(years, start=1):
        payload[f"{float(year):.1f}"] = {
            "overall_mae": float(mae[idx - 1]),
            "overall_mse": float(mse[idx - 1]),
            "overall_rmse": float(rmse[idx - 1]),
            "overall_r2": float(r2[idx - 1]),
            "coverage80": float(cov[idx - 1]),
            "sharpness80": float(sharp[idx - 1]),
            "per_horizon_mae": {
                str(idx): float(mae[idx - 1])
            },
            "per_horizon_mse": {
                str(idx): float(mse[idx - 1])
            },
            "per_horizon_rmse": {
                str(idx): float(rmse[idx - 1])
            },
            "per_horizon_r2": {str(idx): float(r2[idx - 1])},
            "pss": float(pss_vals[idx - 1]),
        }

    payload["__overall__"] = {
        "overall_mae": float(np.mean(mae)),
        "overall_mse": float(np.mean(mse)),
        "overall_rmse": float(np.mean(rmse)),
        "overall_r2": float(np.mean(r2)),
        "coverage80": float(np.mean(cov)),
        "sharpness80": float(np.mean(sharp)),
        "per_horizon_mae": {
            str(i): float(v)
            for i, v in enumerate(mae, start=1)
        },
        "per_horizon_mse": {
            str(i): float(v)
            for i, v in enumerate(mse, start=1)
        },
        "per_horizon_rmse": {
            str(i): float(v)
            for i, v in enumerate(rmse, start=1)
        },
        "per_horizon_r2": {
            str(i): float(v)
            for i, v in enumerate(r2, start=1)
        },
        "pss": float(np.mean(pss_vals)),
    }
    return payload


def generate_eval_diagnostics(
    *,
    output_path: PathLike | None = None,
    template: EvalDiagnosticsLike | None = None,
    overrides: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | Path:
    """
    Generate an eval-diagnostics payload or file.

    Parameters
    ----------
    output_path : path-like, optional
        Destination JSON path. If omitted, the payload
        is returned instead of written.
    template : mapping, ArtifactRecord, or path, optional
        Real or synthetic diagnostics template used as
        the generation base.
    overrides : dict, optional
        Nested overrides applied after template/default
        payload creation.
    **kwargs : dict
        Parameters forwarded to
        ``default_eval_diagnostics_payload`` when no
        template is given.
    """
    if template is None:
        payload = default_eval_diagnostics_payload(**kwargs)
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


def load_eval_diagnostics(
    path: PathLike,
) -> ArtifactRecord:
    """
    Load an eval-diagnostics artifact.

    Raises
    ------
    ValueError
        If the artifact does not look like a compact
        evaluation-diagnostics payload.
    """
    record = load_artifact(path, kind="eval_diagnostics")
    payload = record.payload

    if "__overall__" not in payload:
        raise ValueError(
            "The file does not contain the required "
            "'__overall__' diagnostics block."
        )

    if not _year_keys(payload):
        raise ValueError(
            "The file does not contain any year-like "
            "evaluation blocks."
        )

    return record


def eval_years_frame(
    diagnostics: EvalDiagnosticsLike,
) -> pd.DataFrame:
    """
    Return one row per year block.
    """
    payload = _as_payload(diagnostics)

    rows: list[dict[str, Any]] = []
    for key in _year_keys(payload):
        block = payload.get(key, {}) or {}
        year_num = _try_float(key)
        row: dict[str, Any] = {
            "year_key": key,
            "year": year_num,
        }
        row.update(_top_scalar_metrics(block))
        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty and "year" in frame.columns:
        frame = frame.sort_values("year")
    return frame.reset_index(drop=True)


def eval_overall_frame(
    diagnostics: EvalDiagnosticsLike,
) -> pd.DataFrame:
    """
    Return a compact frame for the ``__overall__`` block.
    """
    payload = _as_payload(diagnostics)
    overall = _overall_block(payload)

    row = _top_scalar_metrics(overall)
    row["n_horizons"] = len(_per_h_metric_map(overall, "mae"))
    row["n_year_blocks"] = len(_year_keys(payload))
    return pd.DataFrame([row])


def eval_per_horizon_frame(
    diagnostics: EvalDiagnosticsLike,
) -> pd.DataFrame:
    """
    Return a tidy per-horizon metrics frame.
    """
    payload = _as_payload(diagnostics)
    overall = _overall_block(payload)

    maps = {
        metric: _per_h_metric_map(overall, metric)
        for metric in _PER_H_METRICS
    }

    horizons = sorted(
        {
            int(float(key))
            for mapping in maps.values()
            for key in mapping
        }
    )

    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        key = str(horizon)
        row: dict[str, Any] = {"horizon": int(horizon)}
        for metric in _PER_H_METRICS:
            row[metric] = maps[metric].get(key)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("horizon")
    return frame.reset_index(drop=True)


def summarize_eval_diagnostics(
    diagnostics: EvalDiagnosticsLike,
) -> dict[str, Any]:
    """
    Build a compact semantic summary for inspection.
    """
    payload = _as_payload(diagnostics)
    years = _year_keys(payload)
    overall = _overall_block(payload)

    overall_per_h = {
        metric: _per_h_metric_map(overall, metric)
        for metric in _PER_H_METRICS
    }
    horizon_count = len(overall_per_h["mae"])

    year_frame = eval_years_frame(payload)

    core_metrics_present = all(
        metric in overall for metric in _YEAR_METRICS
    )

    summary_map = {
        "brief": {
            "kind": "eval_diagnostics",
            "n_year_blocks": len(years),
            "year_keys": years,
            "n_horizons": horizon_count,
        },
        "overall": {
            "overall_mae": _try_float(
                overall.get("overall_mae")
            ),
            "overall_rmse": _try_float(
                overall.get("overall_rmse")
            ),
            "overall_r2": _try_float(
                overall.get("overall_r2")
            ),
            "coverage80": _try_float(
                overall.get("coverage80")
            ),
            "sharpness80": _try_float(
                overall.get("sharpness80")
            ),
            "pss": _try_float(overall.get("pss")),
        },
        "checks": {
            "has_overall_block": bool(overall),
            "has_year_blocks": bool(years),
            "overall_has_core_metrics": bool(
                core_metrics_present
            ),
            "overall_has_per_horizon_mae": bool(
                overall_per_h["mae"]
            ),
            "overall_has_per_horizon_rmse": bool(
                overall_per_h["rmse"]
            ),
            "overall_has_per_horizon_r2": bool(
                overall_per_h["r2"]
            ),
            "all_years_have_pss": (
                True
                if year_frame.empty
                else bool(year_frame["pss"].notna().all())
            ),
            "all_years_have_coverage80": (
                True
                if year_frame.empty
                else bool(
                    year_frame["coverage80"].notna().all()
                )
            ),
            "all_years_have_sharpness80": (
                True
                if year_frame.empty
                else bool(
                    year_frame["sharpness80"].notna().all()
                )
            ),
            "horizon_count_matches_year_count": (
                horizon_count == len(years)
                if years and horizon_count
                else False
            ),
        },
    }
    return summary_map


def plot_eval_overall_metrics(
    diagnostics: EvalDiagnosticsLike,
    *,
    keys: list[str] | tuple[str, ...] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    error: str = "ignore",
    **plot_kws: Any,
) -> plt.Axes:
    """Plot selected top-level metrics from ``__overall__``."""
    fig, ax, _ = prepare_plot(
        ax=ax, figsize=(8.2, 4.8) if ax is None else None
    )

    overall = _overall_block(_as_payload(diagnostics))
    metrics = _top_scalar_metrics(
        overall,
        keys=keys
        or [
            "overall_mae",
            "overall_rmse",
            "overall_r2",
            "coverage80",
            "sharpness80",
            "pss",
        ],
    )
    return plot_metric_bars(
        ax,
        metrics,
        title=title or "Overall evaluation metrics",
        sort_by_value=True,
        top_n=None,
        absolute=True,
        error=error,
        **plot_kws,
    )


def plot_eval_year_metric_trend(
    diagnostics: EvalDiagnosticsLike,
    *,
    metric: str = "overall_mae",
    ax: plt.Axes | None = None,
    title: str | None = None,
    show_grid: bool = True,
    grid_kws: dict[str, Any] | None = None,
    error: str = "ignore",
    **plot_kws: Any,
) -> plt.Axes:
    """Plot one metric across year blocks."""
    fig, ax, _ = prepare_plot(
        ax=ax, figsize=(8.0, 4.6) if ax is None else None
    )

    frame = eval_years_frame(diagnostics)
    if frame.empty or metric not in frame.columns:
        _, ax = empty_plot(
            fig,
            ax,
            title=title or f"Year trend: {metric}",
            message="No year metric data",
        )
        return ax

    line_kws = filter_plot_kwargs(
        ax.plot, plot_kws, error=error
    )
    if "marker" not in line_kws:
        line_kws["marker"] = "o"
    ax.plot(frame["year"], frame[metric], **line_kws)
    _, ax = finalize_plot(
        fig,
        ax,
        title=title or f"Year trend: {metric}",
        xlabel="year",
        ylabel=metric,
        show_grid=show_grid,
        grid_kws=grid_kws or {"alpha": 0.25},
    )
    return ax


def plot_eval_per_horizon_metrics(
    diagnostics: EvalDiagnosticsLike,
    *,
    metric: str = "rmse",
    ax: plt.Axes | None = None,
    title: str | None = None,
    show_grid: bool = True,
    grid_kws: dict[str, Any] | None = None,
    error: str = "ignore",
    **plot_kws: Any,
) -> plt.Axes:
    """Plot one per-horizon metric from ``__overall__``."""
    fig, ax, _ = prepare_plot(
        ax=ax, figsize=(8.0, 4.6) if ax is None else None
    )

    frame = eval_per_horizon_frame(diagnostics)
    if frame.empty or metric not in frame.columns:
        _, ax = empty_plot(
            fig,
            ax,
            title=title or f"Per-horizon {metric}",
            message="No per-horizon data",
        )
        return ax

    bar_kws = filter_plot_kwargs(
        ax.bar, plot_kws, error=error
    )
    ax.bar(
        frame["horizon"].astype(str), frame[metric], **bar_kws
    )
    _, ax = finalize_plot(
        fig,
        ax,
        title=title or f"Per-horizon {metric}",
        xlabel="horizon",
        ylabel=metric,
        show_grid=show_grid,
        grid_kws=grid_kws or {"axis": "y", "alpha": 0.25},
    )
    return ax


def plot_eval_boolean_summary(
    diagnostics: EvalDiagnosticsLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Evaluation diagnostics checks",
    error: str = "ignore",
    **plot_kws: Any,
) -> plt.Axes:
    """Plot semantic pass/fail checks."""
    fig, ax, _ = prepare_plot(
        ax=ax, figsize=(8.0, 4.6) if ax is None else None
    )
    checks = summarize_eval_diagnostics(diagnostics)["checks"]
    return plot_boolean_checks(
        ax,
        checks,
        title=title,
        error=error,
        **plot_kws,
    )


def inspect_eval_diagnostics(
    diagnostics: EvalDiagnosticsLike,
    *,
    output_dir: PathLike | None = None,
    stem: str = "eval_diagnostics",
    save_figures: bool = True,
) -> dict[str, Any]:
    """
    Inspect eval diagnostics and optionally save figures.

    Returns
    -------
    dict
        Bundle containing summary, tabular frames, and
        optionally written figure paths.
    """
    payload = _as_payload(diagnostics)
    summary_map = summarize_eval_diagnostics(payload)

    bundle: dict[str, Any] = {
        "summary": summary_map,
        "frames": {
            "years": eval_years_frame(payload),
            "overall": eval_overall_frame(payload),
            "per_horizon": eval_per_horizon_frame(payload),
        },
        "figure_paths": {},
    }

    if not (output_dir and save_figures):
        return bundle

    out_dir = as_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plots = {
        f"{stem}_overall_metrics.png": (
            plot_eval_overall_metrics,
            {},
        ),
        f"{stem}_year_overall_mae.png": (
            plot_eval_year_metric_trend,
            {"metric": "overall_mae"},
        ),
        f"{stem}_per_horizon_rmse.png": (
            plot_eval_per_horizon_metrics,
            {"metric": "rmse"},
        ),
        f"{stem}_checks.png": (
            plot_eval_boolean_summary,
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
