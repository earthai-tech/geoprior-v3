# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Evaluation-physics generation and inspection helpers.

This module focuses on the richer Stage-2 interpretable
physics-evaluation artifact usually written as something like
``geoprior_eval_phys_<stamp>_interpretable.json``.

Unlike the compact ``eval_diagnostics`` artifact, this payload
bridges multiple inspection concerns:

- point and interval forecast metrics,
- physics loss and epsilon diagnostics,
- calibration factors and before/after interval stats,
- optional censor-stratified summaries,
- per-horizon point metrics,
- unit metadata for interpretable reporting.

The functions are designed for two common uses:

1. Sphinx-Gallery examples that need a realistic
   physics-evaluation payload without rerunning Stage-2.
2. Real workflow inspection when a user wants to review
   forecast quality, physics residual diagnostics, interval
   calibration, and reporting units in one place.
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
    clone_artifact,
    deep_update,
    empty_plot,
    filter_plot_kwargs,
    finalize_plot,
    flatten_dict,
    load_artifact,
    metrics_frame,
    plot_boolean_checks,
    plot_metric_bars,
    plot_series_map,
    prepare_plot,
    read_json,
    write_json,
)

PathLike = str | Path
EvalPhysicsLike = (
    ArtifactRecord | Mapping[str, Any] | str | Path
)

__all__ = [
    "default_eval_physics_payload",
    "eval_physics_calibration_frame",
    "eval_physics_calibration_per_horizon_frame",
    "eval_physics_censor_frame",
    "eval_physics_metrics_frame",
    "eval_physics_per_horizon_frame",
    "eval_physics_point_metrics_frame",
    "eval_physics_units_frame",
    "generate_eval_physics",
    "inspect_eval_physics",
    "load_eval_physics",
    "plot_eval_physics_boolean_summary",
    "plot_eval_physics_calibration_factors",
    "plot_eval_physics_epsilons",
    "plot_eval_physics_metrics",
    "plot_eval_physics_per_horizon_metrics",
    "plot_eval_physics_point_metrics",
    "summarize_eval_physics",
]

_METRICS_EVALUATE_KEYS = [
    "subs_pred_mae_q50",
    "subs_pred_mse_q50",
    "subs_pred_rmse_q50",
    "subs_pred_coverage80",
    "subs_pred_sharpness80",
    "gwl_pred_mae_q50",
    "gwl_pred_mse_q50",
    "gwl_pred_rmse_q50",
    "loss",
    "data_loss",
    "physics_loss",
    "physics_loss_scaled",
    "physics_mult",
    "lambda_offset",
    "consolidation_loss",
    "gw_flow_loss",
    "prior_loss",
    "smooth_loss",
    "mv_prior_loss",
    "bounds_loss",
    "epsilon_prior",
    "epsilon_cons",
    "epsilon_gw",
    "epsilon_cons_raw",
    "epsilon_gw_raw",
    "q_reg_loss",
    "q_rms",
    "q_gate",
    "subs_resid_gate",
]

_EPSILON_KEYS = [
    "epsilon_prior",
    "epsilon_cons",
    "epsilon_gw",
    "epsilon_cons_raw",
    "epsilon_gw_raw",
]

_POINT_KEYS = ["mae", "mse", "rmse", "r2"]

_CAL_KEYS = [
    "coverage80_uncalibrated",
    "coverage80_calibrated",
    "sharpness80_uncalibrated",
    "sharpness80_calibrated",
    "coverage80_uncalibrated_phys",
    "coverage80_calibrated_phys",
    "sharpness80_uncalibrated_phys",
    "sharpness80_calibrated_phys",
]


def _as_payload(
    payload: EvalPhysicsLike,
) -> dict[str, Any]:
    """Return a plain eval-physics payload."""
    if isinstance(payload, ArtifactRecord):
        return dict(payload.payload)

    if isinstance(payload, Mapping):
        return dict(payload)

    data = read_json(payload)
    return dict(data)


def _try_float(value: Any) -> float | None:
    """Return ``value`` as float when possible."""
    try:
        return float(value)
    except Exception:
        return None


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


def _horizon_sort_key(label: Any) -> tuple[int, str]:
    """Sort horizon labels such as ``H1`` or ``1``."""
    text = str(label)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return (0, f"{int(digits):06d}")
    return (1, text)


def _per_horizon_map(
    payload: dict[str, Any],
    metric: str,
) -> dict[str, float]:
    """Extract one per-horizon metric mapping."""
    src = (payload.get("per_horizon", {}) or {}).get(
        metric, {}
    )
    out: dict[str, float] = {}
    for key, value in (src or {}).items():
        num = _try_float(value)
        if num is not None:
            out[str(key)] = float(num)
    return {
        key: out[key]
        for key in sorted(out, key=_horizon_sort_key)
    }


def _calibration_nested_block(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Return nested calibration stats if present."""
    block = payload.get("interval_calibration", {}) or {}
    nested = block.get(
        "factors_per_horizon_from_cal_stats", {}
    )
    return nested if isinstance(nested, dict) else {}


def _calibration_per_horizon_rows(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build tidy per-horizon calibration rows."""
    nested = _calibration_nested_block(payload)
    fac = nested.get("factors", {}) or {}
    before = (nested.get("eval_before") or {}).get(
        "per_horizon", {}
    ) or {}
    after = (nested.get("eval_after") or {}).get(
        "per_horizon", {}
    ) or {}

    keys = sorted(
        set(fac) | set(before) | set(after),
        key=_horizon_sort_key,
    )

    rows: list[dict[str, Any]] = []
    for key in keys:
        row = {
            "horizon": str(key),
            "factor": _try_float(fac.get(key)),
            "coverage_before": _try_float(
                (before.get(key) or {}).get("coverage")
            ),
            "coverage_after": _try_float(
                (after.get(key) or {}).get("coverage")
            ),
            "sharpness_before": _try_float(
                (before.get(key) or {}).get("sharpness")
            ),
            "sharpness_after": _try_float(
                (after.get(key) or {}).get("sharpness")
            ),
        }
        rows.append(row)
    return rows


def default_eval_physics_payload(
    *,
    timestamp: str = "20260222-215049",
    city: str = "demo_city",
    model: str = "GeoPriorSubsNet",
    quantiles: list[float] | None = None,
    horizon: int = 3,
    batch_size: int = 32,
    subs_unit: str = "mm",
    time_units: str = "year",
) -> dict[str, Any]:
    """
    Build a realistic default eval-physics payload.

    The payload is template-based. It is not meant to
    reproduce the Stage-2 evaluation path. Instead, it
    creates a stable and inspectable artifact with the
    same broad structure as the interpretable physics
    evaluation JSON.
    """
    q = list(quantiles or [0.1, 0.5, 0.9])

    metrics_evaluate = {
        "subs_pred_mae_q50": 30.37,
        "subs_pred_mse_q50": 4346.39,
        "subs_pred_coverage80": 0.570,
        "subs_pred_sharpness80": 29.94,
        "gwl_pred_mae_q50": 0.239,
        "gwl_pred_mse_q50": 0.0768,
        "loss": 0.1577,
        "total_loss": 0.1577,
        "data_loss": 0.1577,
        "physics_loss": 4.14e-9,
        "physics_mult": 1.0,
        "physics_loss_scaled": 4.14e-9,
        "lambda_offset": 1.0,
        "consolidation_loss": 1.58e-13,
        "gw_flow_loss": 1.56e-14,
        "prior_loss": 2.06e-8,
        "smooth_loss": 0.0,
        "mv_prior_loss": 0.0,
        "bounds_loss": 2.11e-10,
        "epsilon_prior": 3.54e-4,
        "epsilon_cons": 2.11e-6,
        "epsilon_gw": 1.18e-7,
        "epsilon_cons_raw": 0.0125,
        "epsilon_gw_raw": 3.94e-6,
        "q_reg_loss": 0.0,
        "q_rms": 0.0,
        "q_gate": 0.0,
        "subs_resid_gate": 0.0,
        "subs_pred_rmse_q50": 65.93,
        "gwl_pred_rmse_q50": 0.277,
    }

    payload = {
        "timestamp": str(timestamp),
        "city": city,
        "model": model,
        "tf_version": "2.20.0",
        "numpy_version": "2.0.2",
        "quantiles": q,
        "horizon": int(horizon),
        "batch_size": int(batch_size),
        "metrics_evaluate": metrics_evaluate,
        "physics_diagnostics": {
            "epsilon_prior": metrics_evaluate[
                "epsilon_prior"
            ],
            "epsilon_cons": metrics_evaluate["epsilon_cons"],
            "epsilon_gw": metrics_evaluate["epsilon_gw"],
        },
        "interval_calibration": {
            "target": 0.80,
            "factors_per_horizon": [1.0, 1.0, 1.48],
            "factors_per_horizon_from_cal_stats": {
                "target": 0.80,
                "interval": [0.1, 0.9],
                "f_max": 5.0,
                "tol": 0.02,
                "overall_key": "__overall__",
                "factors_source": "fit",
                "factors": {
                    "1": 1.0,
                    "2": 1.0,
                    "3": 1.018,
                },
                "eval_before": {
                    "coverage": 0.865,
                    "sharpness": 33.08,
                    "per_horizon": {
                        "1": {
                            "coverage": 0.979,
                            "sharpness": 23.24,
                        },
                        "2": {
                            "coverage": 0.822,
                            "sharpness": 27.59,
                        },
                        "3": {
                            "coverage": 0.794,
                            "sharpness": 48.41,
                        },
                    },
                },
                "eval_after": {
                    "coverage": 0.867,
                    "sharpness": 33.38,
                    "per_horizon": {
                        "1": {
                            "coverage": 0.979,
                            "sharpness": 23.24,
                        },
                        "2": {
                            "coverage": 0.822,
                            "sharpness": 27.59,
                        },
                        "3": {
                            "coverage": 0.800,
                            "sharpness": 49.30,
                        },
                    },
                },
            },
            "coverage80_uncalibrated": 0.813,
            "coverage80_calibrated": 0.865,
            "sharpness80_uncalibrated": 0.0278,
            "sharpness80_calibrated": 0.0331,
            "coverage80_uncalibrated_phys": 0.813,
            "coverage80_calibrated_phys": 0.865,
            "sharpness80_uncalibrated_phys": 27.82,
            "sharpness80_calibrated_phys": 33.08,
        },
        "censor_stratified": {
            "flag_name": "soil_thickness_censored",
            "threshold": 0.5,
            "mae_censored": 0.0,
            "mae_uncensored": 9.27,
        },
        "point_metrics": {
            "mae": 9.27,
            "mse": 262.47,
            "r2": 0.883,
            "rmse": 16.20,
        },
        "per_horizon": {
            "mae": {
                "H1": 3.52,
                "H2": 8.31,
                "H3": 15.98,
            },
            "r2": {
                "H1": 0.896,
                "H2": 0.888,
                "H3": 0.874,
            },
        },
        "units": {
            "subs_unit_to_si": 0.001,
            "subs_factor_si_to_real": 1000.0,
            "subs_metrics_unit": str(subs_unit),
            "time_units": str(time_units),
            "seconds_per_time_unit": 31556952.0,
            "epsilon_cons_raw_unit": f"{subs_unit}/{time_units}",
            "epsilon_gw_raw_unit": f"1/{time_units}",
        },
    }
    return payload


def generate_eval_physics(
    *,
    output_path: PathLike | None = None,
    template: EvalPhysicsLike | None = None,
    overrides: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | Path:
    """
    Generate an eval-physics payload or file.

    Parameters
    ----------
    output_path : path-like, optional
        Destination JSON path. If omitted, the payload
        is returned instead of written.
    template : mapping, ArtifactRecord, or path, optional
        Real or synthetic eval-physics template used as
        the generation base.
    overrides : dict, optional
        Nested overrides applied after template/default
        payload creation.
    **kwargs : dict
        Parameters forwarded to
        ``default_eval_physics_payload`` when no template
        is given.
    """
    if template is None:
        payload = default_eval_physics_payload(**kwargs)
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


def load_eval_physics(
    path: PathLike,
) -> ArtifactRecord:
    """
    Load an eval-physics artifact.

    Raises
    ------
    ValueError
        If the artifact does not look like an
        eval-physics payload.
    """
    record = load_artifact(path, kind="eval_physics")
    needed = {
        "metrics_evaluate",
        "physics_diagnostics",
    }
    if not needed.issubset(record.payload):
        raise ValueError(
            "The file does not contain the expected "
            "eval-physics sections."
        )
    return record


def eval_physics_metrics_frame(
    payload: EvalPhysicsLike,
) -> pd.DataFrame:
    """Return a tidy frame for ``metrics_evaluate``."""
    data = _as_payload(payload)
    return metrics_frame(
        data.get("metrics_evaluate", {}),
        section="metrics_evaluate",
    )


def eval_physics_point_metrics_frame(
    payload: EvalPhysicsLike,
) -> pd.DataFrame:
    """Return a tidy frame for point metrics."""
    data = _as_payload(payload)
    return metrics_frame(
        data.get("point_metrics", {}),
        section="point_metrics",
    )


def eval_physics_units_frame(
    payload: EvalPhysicsLike,
) -> pd.DataFrame:
    """Return a tidy frame for units metadata."""
    data = _as_payload(payload)
    units = data.get("units", {}) or {}

    rows = []
    for key, value in units.items():
        rows.append(
            {
                "key": str(key),
                "value": value,
                "is_numeric": isinstance(value, (int, float))
                and not isinstance(value, bool),
            }
        )
    return pd.DataFrame(rows)


def eval_physics_censor_frame(
    payload: EvalPhysicsLike,
) -> pd.DataFrame:
    """Return a tidy frame for censor-aware metrics."""
    data = _as_payload(payload)
    censor = data.get("censor_stratified", {}) or {}

    rows = []
    for key, value in censor.items():
        rows.append(
            {
                "key": str(key),
                "value": value,
                "is_numeric": isinstance(value, (int, float))
                and not isinstance(value, bool),
            }
        )
    return pd.DataFrame(rows)


def eval_physics_per_horizon_frame(
    payload: EvalPhysicsLike,
) -> pd.DataFrame:
    """
    Return a tidy frame for exported per-horizon metrics.
    """
    data = _as_payload(payload)
    per_h = data.get("per_horizon", {}) or {}

    rows: list[dict[str, Any]] = []
    for metric, mapping in per_h.items():
        if not isinstance(mapping, dict):
            continue
        for horizon, value in mapping.items():
            num = _try_float(value)
            if num is None:
                continue
            rows.append(
                {
                    "metric": str(metric),
                    "horizon": str(horizon),
                    "value": float(num),
                }
            )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["_sort"] = frame["horizon"].map(
            _horizon_sort_key
        )
        frame = frame.sort_values(["metric", "_sort"])
        frame = frame.drop(columns="_sort")
    return frame.reset_index(drop=True)


def eval_physics_calibration_frame(
    payload: EvalPhysicsLike,
) -> pd.DataFrame:
    """
    Return a tidy frame for top-level calibration scalars.
    """
    data = _as_payload(payload)
    cal = data.get("interval_calibration", {}) or {}

    base = _numeric_subset(cal, keys=_CAL_KEYS)
    base.update(
        _numeric_subset(
            {
                "target": cal.get("target"),
            }
        )
    )

    nested = _calibration_nested_block(data)
    for key in ["target", "f_max", "tol"]:
        num = _try_float(nested.get(key))
        if num is not None:
            base[f"cal_stats.{key}"] = float(num)

    before = (
        (nested.get("eval_before") or {}) if nested else {}
    )
    after = (nested.get("eval_after") or {}) if nested else {}
    for key in ["coverage", "sharpness"]:
        num_b = _try_float(before.get(key))
        num_a = _try_float(after.get(key))
        if num_b is not None:
            base[f"eval_before.{key}"] = float(num_b)
        if num_a is not None:
            base[f"eval_after.{key}"] = float(num_a)

    return metrics_frame(base, section="interval_calibration")


def eval_physics_calibration_per_horizon_frame(
    payload: EvalPhysicsLike,
) -> pd.DataFrame:
    """
    Return a tidy per-horizon calibration frame.
    """
    data = _as_payload(payload)
    rows = _calibration_per_horizon_rows(data)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["_sort"] = frame["horizon"].map(
            _horizon_sort_key
        )
        frame = frame.sort_values("_sort")
        frame = frame.drop(columns="_sort")
    return frame.reset_index(drop=True)


def summarize_eval_physics(
    payload: EvalPhysicsLike,
) -> dict[str, Any]:
    """
    Build a compact semantic summary for inspection.
    """
    data = _as_payload(payload)
    eval_metrics = data.get("metrics_evaluate", {}) or {}
    phys = data.get("physics_diagnostics", {}) or {}
    cal = data.get("interval_calibration", {}) or {}
    point = data.get("point_metrics", {}) or {}
    units = data.get("units", {}) or {}
    nested = _calibration_nested_block(data)

    before = (
        (nested.get("eval_before") or {}) if nested else {}
    )
    after = (nested.get("eval_after") or {}) if nested else {}

    factor_rows = eval_physics_calibration_per_horizon_frame(
        data
    )
    if factor_rows.empty:
        factor_max = None
        factor_min = None
    else:
        factor_max = float(factor_rows["factor"].max())
        factor_min = float(factor_rows["factor"].min())

    summary_map = {
        "brief": {
            "kind": "eval_physics",
            "timestamp": data.get("timestamp"),
            "city": data.get("city"),
            "model": data.get("model"),
            "horizon": data.get("horizon"),
            "batch_size": data.get("batch_size"),
            "quantiles": list(
                data.get("quantiles", []) or []
            ),
        },
        "core_metrics": {
            "subs_mae_q50": _try_float(
                eval_metrics.get("subs_pred_mae_q50")
            ),
            "subs_rmse_q50": _try_float(
                eval_metrics.get("subs_pred_rmse_q50")
            ),
            "gwl_mae_q50": _try_float(
                eval_metrics.get("gwl_pred_mae_q50")
            ),
            "gwl_rmse_q50": _try_float(
                eval_metrics.get("gwl_pred_rmse_q50")
            ),
            "loss": _try_float(eval_metrics.get("loss")),
            "data_loss": _try_float(
                eval_metrics.get("data_loss")
            ),
            "physics_loss": _try_float(
                eval_metrics.get("physics_loss")
            ),
            "physics_loss_scaled": _try_float(
                eval_metrics.get("physics_loss_scaled")
            ),
            "point_mae": _try_float(point.get("mae")),
            "point_rmse": _try_float(point.get("rmse")),
            "point_r2": _try_float(point.get("r2")),
        },
        "physics": {
            "epsilon_prior": _try_float(
                phys.get("epsilon_prior")
            ),
            "epsilon_cons": _try_float(
                phys.get("epsilon_cons")
            ),
            "epsilon_gw": _try_float(phys.get("epsilon_gw")),
            "epsilon_cons_raw": _try_float(
                eval_metrics.get("epsilon_cons_raw")
            ),
            "epsilon_gw_raw": _try_float(
                eval_metrics.get("epsilon_gw_raw")
            ),
            "lambda_offset": _try_float(
                eval_metrics.get("lambda_offset")
            ),
            "physics_mult": _try_float(
                eval_metrics.get("physics_mult")
            ),
        },
        "calibration": {
            "target": _try_float(cal.get("target")),
            "coverage80_uncalibrated": _try_float(
                cal.get("coverage80_uncalibrated")
            ),
            "coverage80_calibrated": _try_float(
                cal.get("coverage80_calibrated")
            ),
            "sharpness80_uncalibrated": _try_float(
                cal.get("sharpness80_uncalibrated")
            ),
            "sharpness80_calibrated": _try_float(
                cal.get("sharpness80_calibrated")
            ),
            "coverage_before": _try_float(
                before.get("coverage")
            ),
            "coverage_after": _try_float(
                after.get("coverage")
            ),
            "sharpness_before": _try_float(
                before.get("sharpness")
            ),
            "sharpness_after": _try_float(
                after.get("sharpness")
            ),
            "factor_min": factor_min,
            "factor_max": factor_max,
        },
        "units": {
            "subs_metrics_unit": units.get(
                "subs_metrics_unit"
            ),
            "time_units": units.get("time_units"),
            "epsilon_cons_raw_unit": units.get(
                "epsilon_cons_raw_unit"
            ),
            "epsilon_gw_raw_unit": units.get(
                "epsilon_gw_raw_unit"
            ),
        },
        "checks": {
            "has_metrics_evaluate": bool(eval_metrics),
            "has_physics_diagnostics": bool(phys),
            "has_interval_calibration": bool(cal),
            "has_point_metrics": bool(point),
            "has_units": bool(units),
            "has_per_horizon": bool(
                data.get("per_horizon", {})
            ),
            "has_quantiles": bool(data.get("quantiles", [])),
            "physics_loss_nonnegative": (
                (
                    _try_float(
                        eval_metrics.get("physics_loss")
                    )
                    or 0.0
                )
                >= 0.0
            ),
            "epsilons_present": all(
                key in phys
                for key in [
                    "epsilon_prior",
                    "epsilon_cons",
                    "epsilon_gw",
                ]
            ),
            "calibration_target_in_01": (
                _try_float(cal.get("target")) is not None
                and 0.0 <= float(cal.get("target")) <= 1.0
            ),
            "coverage_improves_or_matches": (
                _try_float(cal.get("coverage80_calibrated"))
                is not None
                and _try_float(
                    cal.get("coverage80_uncalibrated")
                )
                is not None
                and float(cal.get("coverage80_calibrated"))
                >= float(cal.get("coverage80_uncalibrated"))
            ),
            "reported_unit_present": bool(
                units.get("subs_metrics_unit")
            ),
        },
    }
    return summary_map


def plot_eval_physics_metrics(
    payload: EvalPhysicsLike,
    *,
    keys: list[str] | tuple[str, ...] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    error: str = "ignore",
    **plot_kws: Any,
) -> plt.Axes:
    """Plot selected ``metrics_evaluate`` values."""
    fig, ax, _ = prepare_plot(
        ax=ax, figsize=(8.2, 4.8) if ax is None else None
    )

    data = _as_payload(payload)
    metrics = _numeric_subset(
        data.get("metrics_evaluate", {}),
        keys=keys or _METRICS_EVALUATE_KEYS,
    )
    return plot_metric_bars(
        ax,
        metrics,
        title=title or "Eval physics: metrics_evaluate",
        sort_by_value=True,
        top_n=14,
        absolute=True,
        error=error,
        **plot_kws,
    )


def plot_eval_physics_epsilons(
    payload: EvalPhysicsLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Eval physics: epsilon diagnostics",
    error: str = "ignore",
    **plot_kws: Any,
) -> plt.Axes:
    """Plot epsilon-related diagnostics."""
    fig, ax, _ = prepare_plot(
        ax=ax, figsize=(8.0, 4.6) if ax is None else None
    )

    data = _as_payload(payload)
    metrics = {
        **_numeric_subset(
            data.get("physics_diagnostics", {}),
            keys=[
                "epsilon_prior",
                "epsilon_cons",
                "epsilon_gw",
            ],
        ),
        **_numeric_subset(
            data.get("metrics_evaluate", {}),
            keys=["epsilon_cons_raw", "epsilon_gw_raw"],
        ),
    }
    return plot_metric_bars(
        ax,
        metrics,
        title=title,
        sort_by_value=True,
        top_n=None,
        absolute=True,
        error=error,
        **plot_kws,
    )


def plot_eval_physics_calibration_factors(
    payload: EvalPhysicsLike,
    *,
    source: str = "top",
    ax: plt.Axes | None = None,
    title: str | None = None,
    error: str = "ignore",
    **plot_kws: Any,
) -> plt.Axes:
    """
    Plot per-horizon calibration factors.

    Parameters
    ----------
    source : {'top', 'nested'}, default='top'
        ``'top'`` uses ``factors_per_horizon``.
        ``'nested'`` uses ``factors_per_horizon_from_cal_stats``.
    """
    fig, ax, _ = prepare_plot(
        ax=ax, figsize=(7.8, 4.6) if ax is None else None
    )

    data = _as_payload(payload)
    cal = data.get("interval_calibration", {}) or {}

    if str(source).strip().lower() == "nested":
        nested = _calibration_nested_block(data)
        src = nested.get("factors", {}) if nested else {}
        series = {
            str(k): float(v)
            for k, v in (src or {}).items()
            if _try_float(v) is not None
        }
    else:
        vals = cal.get("factors_per_horizon", []) or []
        series = {
            f"H{i}": float(v)
            for i, v in enumerate(vals, start=1)
            if _try_float(v) is not None
        }

    return plot_series_map(
        ax,
        series,
        title=title or "Calibration factors by horizon",
        xlabel="horizon",
        ylabel="factor",
        error=error,
        **plot_kws,
    )


def plot_eval_physics_point_metrics(
    payload: EvalPhysicsLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Eval physics: point metrics",
    error: str = "ignore",
    **plot_kws: Any,
) -> plt.Axes:
    """Plot point-metric summary."""
    fig, ax, _ = prepare_plot(
        ax=ax, figsize=(7.8, 4.6) if ax is None else None
    )

    data = _as_payload(payload)
    metrics = _numeric_subset(
        data.get("point_metrics", {}),
        keys=_POINT_KEYS,
    )
    return plot_metric_bars(
        ax,
        metrics,
        title=title,
        sort_by_value=True,
        top_n=None,
        absolute=True,
        error=error,
        **plot_kws,
    )


def plot_eval_physics_per_horizon_metrics(
    payload: EvalPhysicsLike,
    *,
    metric: str = "mae",
    ax: plt.Axes | None = None,
    title: str | None = None,
    error: str = "ignore",
    **plot_kws: Any,
) -> plt.Axes:
    """Plot one exported per-horizon metric map."""
    fig, ax, _ = prepare_plot(
        ax=ax, figsize=(7.8, 4.6) if ax is None else None
    )

    data = _as_payload(payload)
    series = _per_horizon_map(data, metric)
    return plot_series_map(
        ax,
        series,
        title=title or f"Per-horizon {metric}",
        xlabel="horizon",
        ylabel=metric,
        error=error,
        **plot_kws,
    )


def plot_eval_physics_boolean_summary(
    payload: EvalPhysicsLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Eval physics checks",
    error: str = "ignore",
    **plot_kws: Any,
) -> plt.Axes:
    """Plot semantic pass/fail checks."""
    fig, ax, _ = prepare_plot(
        ax=ax, figsize=(8.0, 4.6) if ax is None else None
    )

    checks = summarize_eval_physics(payload)["checks"]
    return plot_boolean_checks(
        ax,
        checks,
        title=title,
        error=error,
        **plot_kws,
    )


def inspect_eval_physics(
    payload: EvalPhysicsLike,
    *,
    output_dir: PathLike | None = None,
    stem: str = "eval_physics",
    save_figures: bool = True,
) -> dict[str, Any]:
    """
    Inspect an eval-physics artifact and optionally save figures.

    Returns
    -------
    dict
        Bundle containing summary, tabular frames, and
        optionally written figure paths.
    """
    data = _as_payload(payload)
    summary_map = summarize_eval_physics(data)

    bundle: dict[str, Any] = {
        "summary": summary_map,
        "frames": {
            "metrics_evaluate": eval_physics_metrics_frame(
                data
            ),
            "point_metrics": eval_physics_point_metrics_frame(
                data
            ),
            "units": eval_physics_units_frame(data),
            "censor_stratified": eval_physics_censor_frame(
                data
            ),
            "per_horizon": eval_physics_per_horizon_frame(
                data
            ),
            "interval_calibration": (
                eval_physics_calibration_frame(data)
            ),
            "interval_calibration_per_horizon": (
                eval_physics_calibration_per_horizon_frame(
                    data
                )
            ),
        },
        "figures": {},
    }

    if not save_figures or output_dir is None:
        return bundle

    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    fig_specs = {
        f"{stem}_metrics.png": plot_eval_physics_metrics,
        f"{stem}_epsilons.png": plot_eval_physics_epsilons,
        f"{stem}_cal_factors.png": (
            plot_eval_physics_calibration_factors
        ),
        f"{stem}_point_metrics.png": (
            plot_eval_physics_point_metrics
        ),
        f"{stem}_per_h_mae.png": (
            lambda p, ax=None: (
                plot_eval_physics_per_horizon_metrics(
                    p,
                    metric="mae",
                    ax=ax,
                )
            )
        ),
        f"{stem}_checks.png": plot_eval_physics_boolean_summary,
    }

    for name, fn in fig_specs.items():
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        fn(data, ax=ax)
        fig.tight_layout()
        path = outdir / name
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        bundle["figures"][name] = str(path)

    return bundle
