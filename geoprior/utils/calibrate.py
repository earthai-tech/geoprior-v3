# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# https://lkouadio.com
# Copyright (c) 2026-present
# Author: LKouadio <etanoyau@gmail.com>

"""
Forecast utilities.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    Any,
    Literal,
)

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..core.checks import check_empty
from ..core.io import SaveFile
from ..decorators import isdf
from .generic_utils import normalize_model_inputs, vlog


def _infer_quantile_map(
    df: pd.DataFrame,
    *,
    target_name: str,
    column_map: Mapping[str, Any] | None = None,
) -> dict[float, str]:
    column_map = dict(column_map or {})
    q_spec = column_map.get("quantiles", None)
    q_map: dict[float, str] = {}

    if q_spec is not None:
        if isinstance(q_spec, Mapping):
            for q, col in q_spec.items():
                qf = float(q)
                if col not in df.columns:
                    raise KeyError(
                        f"Quantile col '{col}' (q={qf}) missing."
                    )
                q_map[qf] = col
        else:
            for col in q_spec:
                if col not in df.columns:
                    raise KeyError(
                        f"Quantile col '{col}' missing."
                    )
                # user provided list => infer q from suffix _qXX
                import re

                m = re.match(
                    rf"^{target_name}_q(\d+)$", str(col)
                )
                if not m:
                    raise ValueError(
                        f"Cannot infer q from '{col}'. "
                        f"Expected '{target_name}_qXX'."
                    )
                q_map[int(m.group(1)) / 100.0] = col
        return q_map

    # auto-detect
    import re

    pat = re.compile(rf"^{target_name}_q(\d+)$")
    for col in df.columns:
        m = pat.match(str(col))
        if m:
            q_map[int(m.group(1)) / 100.0] = str(col)

    return q_map


def _infer_actual_col(
    df: pd.DataFrame,
    *,
    target_name: str,
    column_map: Mapping[str, Any] | None = None,
) -> str | None:
    column_map = dict(column_map or {})
    spec = column_map.get("actual", f"{target_name}_actual")
    if spec is None:
        return None
    if isinstance(spec, str):
        return spec if spec in df.columns else None
    cols = list(spec)
    if len(cols) != 1:
        raise ValueError("Only 1 actual col supported.")
    return cols[0] if cols[0] in df.columns else None


def _pick_median_q(
    q_map: Mapping[float, str],
    *,
    median_q: float = 0.5,
) -> float:
    if not q_map:
        raise ValueError("No quantiles detected.")
    return min(q_map.keys(), key=lambda q: abs(q - median_q))


def _pick_interval_qs(
    q_map: Mapping[float, str],
    *,
    interval: tuple[float, float],
) -> tuple[float, float]:
    lo_t, hi_t = interval
    qs = sorted(q_map.keys())
    q_lo = min(qs, key=lambda q: abs(q - lo_t))
    q_hi = min(qs, key=lambda q: abs(q - hi_t))
    if q_lo >= q_hi:
        raise ValueError(
            f"Bad interval selection: q_lo={q_lo}, q_hi={q_hi}."
        )
    return q_lo, q_hi


def _coverage_and_width(
    y_true: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
) -> tuple[float, float]:
    m = (
        np.isfinite(y_true)
        & np.isfinite(y_lo)
        & np.isfinite(y_hi)
    )
    if not np.any(m):
        return float("nan"), float("nan")
    yt = y_true[m]
    lo = y_lo[m]
    hi = y_hi[m]
    cov = float(np.mean((yt >= lo) & (yt <= hi)))
    wid = float(np.mean(hi - lo))
    return cov, wid


def _fit_factor_bisect(
    y_true: np.ndarray,
    q_lo: np.ndarray,
    q_md: np.ndarray,
    q_hi: np.ndarray,
    *,
    target: float = 0.8,
    tol: float = 1e-3,
    f_max: float = 5.0,
    max_iter: int = 32,
) -> float:
    y_true = np.asarray(y_true, float)
    q_lo = np.asarray(q_lo, float)
    q_md = np.asarray(q_md, float)
    q_hi = np.asarray(q_hi, float)

    def _cov(f: float) -> float:
        lo = q_md - f * (q_md - q_lo)
        hi = q_md + f * (q_hi - q_md)
        cov, _ = _coverage_and_width(y_true, lo, hi)
        return cov

    c1 = _cov(1.0)
    if np.isnan(c1):
        return 1.0
    if c1 >= target - tol:
        return 1.0

    cmax = _cov(f_max)
    if np.isnan(cmax):
        return 1.0
    if cmax < target - tol:
        return float(f_max)

    lo, hi = 1.0, float(f_max)
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        cm = _cov(mid)
        if cm >= target:
            hi = mid
        else:
            lo = mid
        if (hi - lo) <= 1e-6:
            break

    return float(hi)


def fit_interval_factors_df(
    df_eval: pd.DataFrame,
    *,
    target_name: str = "subsidence",
    column_map: Mapping[str, Any] | None = None,
    step_col: str = "forecast_step",
    interval: tuple[float, float] = (0.1, 0.9),
    target_coverage: float = 0.8,
    median_q: float = 0.5,
    tol: float = 1e-3,
    f_max: float = 5.0,
    max_iter: int = 32,
    verbose: int = 1,
    logger: logging.Logger | None = None,
) -> dict[int, float]:
    q_map = _infer_quantile_map(
        df_eval,
        target_name=target_name,
        column_map=column_map,
    )
    act_col = _infer_actual_col(
        df_eval,
        target_name=target_name,
        column_map=column_map,
    )
    if act_col is None:
        raise ValueError(
            "Cannot fit calibration factors: missing actual col."
        )

    q_md = _pick_median_q(q_map, median_q=median_q)
    q_lo, q_hi = _pick_interval_qs(q_map, interval=interval)

    if step_col not in df_eval.columns:
        groups = [(1, df_eval)]
    else:
        groups = df_eval.groupby(step_col)

    factors: dict[int, float] = {}
    for h, g in groups:
        h_int = int(h)
        yt = g[act_col].to_numpy()
        lo = g[q_map[q_lo]].to_numpy()
        md = g[q_map[q_md]].to_numpy()
        hi = g[q_map[q_hi]].to_numpy()

        f = _fit_factor_bisect(
            yt,
            lo,
            md,
            hi,
            target=target_coverage,
            tol=tol,
            f_max=f_max,
            max_iter=max_iter,
        )
        factors[h_int] = float(f)

        vlog(
            f"[calib] fitted factor H{h_int}={f:.6g}",
            verbose=verbose,
            level=2,
            logger=logger,
        )

    return factors


def _enforce_monotonic_q(
    q_vals: np.ndarray,
    *,
    mode: str = "cummax",
) -> np.ndarray:
    if mode in (None, "", "none"):
        return q_vals
    if mode == "sort":
        return np.sort(q_vals, axis=1)
    if mode == "cummax":
        return np.maximum.accumulate(q_vals, axis=1)
    raise ValueError(
        "enforce_monotonic must be one of "
        "{'none','cummax','sort'}."
    )


def apply_interval_factors_df(
    df: pd.DataFrame,
    factors: Mapping[int, float] | float,
    *,
    target_name: str = "subsidence",
    column_map: Mapping[str, Any] | None = None,
    step_col: str = "forecast_step",
    median_q: float = 0.5,
    keep_original: bool = False,
    factor_col: str = "calibration_factor",
    calibrated_col: str = "is_calibrated",
    enforce_monotonic: str = "cummax",
    verbose: int = 1,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    df2 = df.copy()
    q_map = _infer_quantile_map(
        df2, target_name=target_name, column_map=column_map
    )
    if not q_map:
        raise ValueError("No quantile columns to calibrate.")

    q_md = _pick_median_q(q_map, median_q=median_q)
    qs = sorted(q_map.keys())
    cols = [q_map[q] for q in qs]

    if keep_original:
        for c in cols:
            df2[f"{c}_raw"] = df2[c]

    # Resolve factors
    if isinstance(factors, int | float):
        f_map = None
        f_scalar = float(factors)
    else:
        f_map = {int(k): float(v) for k, v in factors.items()}
        f_scalar = None

    if step_col not in df2.columns:
        df2[step_col] = 1

    # Apply per horizon
    med_col = q_map[q_md]
    for h, g_idx in df2.groupby(step_col).groups.items():
        h_int = int(h)
        f_h = (
            float(f_map.get(h_int, 1.0))
            if f_map is not None
            else float(f_scalar)
        )

        idx = np.asarray(list(g_idx))
        med = df2.loc[idx, med_col].to_numpy(
            dtype=float,
            copy=True,
        )

        # Pandas/NumPy may return a read-only array here depending on
        # version, block layout, and copy-on-write behavior. We mutate
        # q_vals in place below, so force an explicit writable copy.
        q_vals = df2.loc[idx, cols].to_numpy(
            dtype=float,
            copy=True,
        )

        # scale each quantile around median
        for j, q in enumerate(qs):
            if q < q_md:
                q_vals[:, j] = med - f_h * (
                    med - q_vals[:, j]
                )
            elif q > q_md:
                q_vals[:, j] = med + f_h * (
                    q_vals[:, j] - med
                )
            else:
                q_vals[:, j] = med

        q_vals = _enforce_monotonic_q(
            q_vals, mode=enforce_monotonic
        )

        df2.loc[idx, cols] = q_vals
        df2.loc[idx, factor_col] = f_h

    df2[calibrated_col] = True
    return df2


def _looks_calibrated(
    df_eval: pd.DataFrame,
    *,
    target_name: str,
    column_map: Mapping[str, Any] | None,
    step_col: str,
    interval: tuple[float, float],
    target: float,
    tol: float,
    calibrated_col: str,
    verbose: int,
    logger: logging.Logger | None,
) -> bool:
    if calibrated_col in df_eval.columns:
        v = df_eval[calibrated_col]
        if bool(np.all(v.astype(bool).to_numpy())):
            vlog(
                "[calib] detected calibrated_col=True.",
                verbose=verbose,
                level=2,
                logger=logger,
            )
            return True

    act = _infer_actual_col(
        df_eval,
        target_name=target_name,
        column_map=column_map,
    )
    if act is None:
        return False

    q_map = _infer_quantile_map(
        df_eval,
        target_name=target_name,
        column_map=column_map,
    )
    if not q_map:
        return False

    q_lo, q_hi = _pick_interval_qs(q_map, interval=interval)
    lo_col = q_map[q_lo]
    hi_col = q_map[q_hi]

    yt = df_eval[act].to_numpy(dtype=float)
    lo = df_eval[lo_col].to_numpy(dtype=float)
    hi = df_eval[hi_col].to_numpy(dtype=float)
    cov, _ = _coverage_and_width(yt, lo, hi)

    if np.isnan(cov):
        return False

    ok = abs(cov - target) <= tol
    vlog(
        f"[calib] auto-check coverage={cov:.4f} "
        f"target={target:.4f} tol={tol:.4f} -> {ok}",
        verbose=verbose,
        level=2,
        logger=logger,
    )
    return bool(ok)


def _normalize_factors(
    factors: float | Mapping[Any, Any],
) -> float | dict[int, float]:
    if isinstance(factors, int | float):
        return float(factors)

    out: dict[int, float] = {}
    for k, v in dict(factors).items():
        out[int(k)] = float(v)
    return out


def calibrate_quantile_forecasts(
    *,
    df_eval: pd.DataFrame | None = None,
    df_future: pd.DataFrame | None = None,
    target_name: str = "subsidence",
    column_map: Mapping[str, Any] | None = None,
    step_col: str = "forecast_step",
    interval: tuple[float, float] = (0.1, 0.9),
    target_coverage: float = 0.8,
    median_q: float = 0.5,
    use: str | bool = "auto",
    tol: float = 0.02,
    f_max: float = 5.0,
    max_iter: int = 32,
    keep_original: bool = False,
    enforce_monotonic: str = "cummax",
    overall_key: str | None = "__overall__",
    calibrated_col: str = "is_calibrated",
    factor_col: str = "calibration_factor",
    factors: float
    | Mapping[int, float]
    | Mapping[str, float]
    | None = None,
    save_eval: str | Path | None = None,
    save_future: str | Path | None = None,
    save_stats: str | Path | None = None,
    verbose: int = 1,
    logger: logging.Logger | None = None,
) -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    dict[str, Any],
]:
    use_s: Any = use
    if isinstance(use, str):
        use_s = use.strip().lower()

    def _save_df(dfX: pd.DataFrame | None, p: Any) -> None:
        if not p or dfX is None:
            return
        pp = Path(p)
        pp.parent.mkdir(parents=True, exist_ok=True)
        dfX.to_csv(pp, index=False)

    def _save_stats(statsX: dict[str, Any]) -> None:
        if not save_stats:
            return
        pp = Path(save_stats)
        pp.parent.mkdir(parents=True, exist_ok=True)
        with open(pp, "w", encoding="utf-8") as f:
            json.dump(statsX, f, indent=2)

    def _jsonable_factors(fx: Any) -> Any:
        if isinstance(fx, int | float):
            return float(fx)
        return {
            str(int(k)): float(v) for k, v in dict(fx).items()
        }

    stats: dict[str, Any] = {
        "target": float(target_coverage),
        "interval": [float(interval[0]), float(interval[1])],
        "f_max": float(f_max),
        "tol": float(tol),
        "overall_key": overall_key,
    }

    # ----------------------------
    # 0) Hard disable
    # ----------------------------
    if use is False or use_s == "false":
        stats["skipped"] = True
        stats["reason"] = "use_false"
        _save_df(df_eval, save_eval)
        _save_df(df_future, save_future)
        _save_stats(stats)
        return df_eval, df_future, stats

    # ----------------------------
    # 1) Auto skip if calibrated
    #    (safe default, even if
    #     user passes factors)
    #    -> to force, set use=True
    # ----------------------------
    if use_s == "auto" and df_eval is not None:
        if _looks_calibrated(
            df_eval,
            target_name=target_name,
            column_map=column_map,
            step_col=step_col,
            interval=interval,
            target=target_coverage,
            tol=tol,
            calibrated_col=calibrated_col,
            verbose=verbose,
            logger=logger,
        ):
            stats["skipped"] = True
            stats["reason"] = "already_calibrated"
            _save_df(df_eval, save_eval)
            _save_df(df_future, save_future)
            _save_stats(stats)
            return df_eval, df_future, stats

    # ----------------------------
    # 2) Choose factors:
    #    - user factors win
    #    - else fit from eval
    # ----------------------------
    f_apply: Any = None

    if factors is not None:
        f_apply = _normalize_factors(factors)
        stats["factors_source"] = "user"
        stats["factors"] = _jsonable_factors(f_apply)

    elif df_eval is not None:
        f_apply = fit_interval_factors_df(
            df_eval,
            target_name=target_name,
            column_map=column_map,
            step_col=step_col,
            interval=interval,
            target_coverage=target_coverage,
            median_q=median_q,
            tol=tol * 0.1,
            f_max=f_max,
            max_iter=max_iter,
            verbose=verbose,
            logger=logger,
        )
        stats["factors_source"] = "fit"
        stats["factors"] = _jsonable_factors(f_apply)

    else:
        stats["skipped"] = True
        stats["reason"] = "no_eval_no_factors"
        _save_df(df_eval, save_eval)
        _save_df(df_future, save_future)
        _save_stats(stats)
        return df_eval, df_future, stats

    # If still None (shouldn't happen), skip safely.
    if f_apply is None:
        stats["skipped"] = True
        stats["reason"] = "no_factors"
        _save_df(df_eval, save_eval)
        _save_df(df_future, save_future)
        _save_stats(stats)
        return df_eval, df_future, stats

    # ----------------------------
    # 3) Apply
    # ----------------------------
    df_eval_cal = None
    df_future_cal = None

    if df_eval is not None:
        df_eval_cal = apply_interval_factors_df(
            df_eval,
            f_apply,
            target_name=target_name,
            column_map=column_map,
            step_col=step_col,
            median_q=median_q,
            keep_original=keep_original,
            factor_col=factor_col,
            calibrated_col=calibrated_col,
            enforce_monotonic=enforce_monotonic,
            verbose=verbose,
            logger=logger,
        )

    if df_future is not None:
        df_future_cal = apply_interval_factors_df(
            df_future,
            f_apply,
            target_name=target_name,
            column_map=column_map,
            step_col=step_col,
            median_q=median_q,
            keep_original=keep_original,
            factor_col=factor_col,
            calibrated_col=calibrated_col,
            enforce_monotonic=enforce_monotonic,
            verbose=verbose,
            logger=logger,
        )

    # Optional: compute before/after stats (eval only)
    if df_eval is not None:
        act = _infer_actual_col(
            df_eval,
            target_name=target_name,
            column_map=column_map,
        )
        q_map = _infer_quantile_map(
            df_eval,
            target_name=target_name,
            column_map=column_map,
        )
        if act is not None and q_map:
            q_lo, q_hi = _pick_interval_qs(
                q_map, interval=interval
            )
            lo_col = q_map[q_lo]
            hi_col = q_map[q_hi]

            def _summ(dfX: pd.DataFrame) -> dict[str, Any]:
                out: dict[str, Any] = {}
                yt = dfX[act].to_numpy(float)
                lo = dfX[lo_col].to_numpy(float)
                hi = dfX[hi_col].to_numpy(float)
                cov, wid = _coverage_and_width(yt, lo, hi)
                out["coverage"] = cov
                out["sharpness"] = wid

                if step_col in dfX.columns:
                    per: dict[str, Any] = {}
                    for h, g in dfX.groupby(step_col):
                        yt_h = g[act].to_numpy(float)
                        lo_h = g[lo_col].to_numpy(float)
                        hi_h = g[hi_col].to_numpy(float)
                        c_h, w_h = _coverage_and_width(
                            yt_h, lo_h, hi_h
                        )
                        per[str(int(h))] = {
                            "coverage": c_h,
                            "sharpness": w_h,
                        }
                    out["per_horizon"] = per
                return out

            stats["eval_before"] = _summ(df_eval)
            stats["eval_after"] = _summ(df_eval_cal)

    # Save outputs
    _save_df(df_eval_cal, save_eval)
    _save_df(df_future_cal, save_future)
    _save_stats(stats)

    return df_eval_cal, df_future_cal, stats


@check_empty(["y_val", "forecast_val"])
def calibrate_forecasts_in(
    y_val: pd.Series,
    forecasts_val: pd.DataFrame,
    *forecasts_to_calibrate: pd.DataFrame
    | Mapping[str, pd.DataFrame],
    quantiles: tuple[int, ...] = (
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
    ),
    prefix: str = "subsidence",
    verbose: int | None = None,
    _logger=None,
):
    vlog(
        "Starting forecast calibration...",
        verbose,
        level=3,
        logger=_logger,
    )
    calibrators = {}
    # Train a calibration model for each quantile
    for q in quantiles:
        col = f"{prefix}_q{q}"
        if col not in forecasts_val.columns:
            vlog(
                f"Skip quantile {q}: '{col}' missing",  # noqa
                verbose,
                level=2,
                logger=_logger,
            )
            continue
        y_bin = (y_val < forecasts_val[col]).astype(int)
        iso = IsotonicRegression(
            y_min=0, y_max=1, out_of_bounds="clip"
        )
        calibrators[col] = iso.fit(forecasts_val[col], y_bin)
    # Normalize input forecasts into dict format
    inputs = normalize_model_inputs(*forecasts_to_calibrate)
    results = {}
    vlog(
        f"Applying calibration to {len(inputs)} model(s)...",
        verbose,
        level=3,
        logger=_logger,
    )
    # Apply calibrators to each DataFrame
    for name, df in inputs.items():
        df_cal = df.copy()
        for q in quantiles:
            col = f"{prefix}_q{q}"
            if col in calibrators:
                out_col = f"{col}_cal_prob"
                df_cal[out_col] = calibrators[col].predict(
                    df[col]
                )
        results[name] = df_cal
    return results, calibrators


calibrate_forecasts_in.__doc__ = r"""
Train and apply isotonic calibration models on forecast data.

Parameters
----------
y_val : pandas.Series
    Observed target values for calibration.
forecasts_val : pandas.DataFrame
    Validation forecasts with columns named
    `<prefix>_q{quantile}` for each quantile.
*forecasts_to_calibrate : DataFrame or dict
    One or more DataFrames (or a mapping) of new forecasts
    to calibrate using trained models.
quantiles : tuple of int, optional
    List of quantile levels (e.g., 10, 50, 90) to calibrate.
prefix : str, default 'subsidence'
    Column name prefix for quantile forecasts.
verbose : int, optional
    Verbosity level passed to vlog for logging.
_logger : Logger or callable, optional
    Sink for log messages (print or logging.Logger).

Returns
-------
results : dict
    Mapping of model names to calibrated DataFrames,
    each containing new `<prefix>_q{q}_cal_prob`
    probability columns.
calibrators : dict
    Fitted IsotonicRegression models per quantile.

Notes
-----
This function fits an isotonic regression for each
quantile on validation data and then predicts
calibrated probabilities for new forecasts. Use
`normalize_model_inputs` to handle varied input formats.

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from geoprior.utils.calibrate import calibrate_forecasts
>>> # Create dummy validation series
>>> idx = pd.date_range('2021-01-01', periods=50)
>>> y_val = pd.Series(
...     np.random.randn(50), index=idx
... )
>>> # Simulate validation forecasts
>>> val_df = pd.DataFrame({
...     'subsidence_q10': y_val - 0.2,
...     'subsidence_q90': y_val + 0.2
... }, index=idx)
>>> # Simulate two new forecast sets
>>> newA = val_df + 0.1
>>> newB = val_df - 0.1
>>> # Calibrate forecasts
>>> results, models = calibrate_forecasts(
...     y_val,
...     val_df,
...     newA,
...     newB,
...     quantiles=(10, 90),
...     verbose=2
... )
>>> # Access calibrated probabilities
>>> results['ModelA']['subsidence_q10_cal_prob']
>>> # Inspect fitted calibrator for q90
>>> models['subsidence_q90'].threshold_
"""


@SaveFile
@check_empty(["df"])
@isdf
def calibrate_probability_forecast(
    df: pd.DataFrame,
    prob_col: str,
    actual_col: str,
    method: str = "isotonic",
    out_col: str | None = None,
    clip: bool = True,
    savefile: str | None = None,
) -> pd.DataFrame:
    df = df.copy()
    if out_col is None:
        out_col = f"{prob_col}_calib"
    y_pred = df[prob_col].values
    y_true = df[actual_col].values

    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_pred, y_true)
        y_cal = iso.transform(y_pred)
    elif method == "logistic":
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(y_pred.reshape(-1, 1), y_true)
        y_cal = lr.predict_proba(y_pred.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unknown method '{method}'")

    if clip:
        y_cal = np.clip(y_cal, 0, 1)

    df[out_col] = y_cal
    return df


@SaveFile
@check_empty(["df"])
@isdf
def calibrate_forecasts(
    df: pd.DataFrame,
    quantiles: Sequence[float],
    q_prefix: str,
    actual_col: str,
    method: Literal["isotonic", "logistic"] = "isotonic",
    out_prefix: str = "calib",
    grid_mode: Literal["unit", "range"] = "unit",
    grid_size: int = 1001,
    group_by: str | None = None,
    savefile: str
    | None = None,  # will handle by decorator so once the function
    # return dataframe, decorator take it and explort to csv format instead .
) -> pd.DataFrame:
    """
    Calibrate quantile forecasts by fitting a monotonic classifier
    at each quantile level, then inverting the CDF at the nominal q.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing raw quantile forecasts and actuals.
    quantiles : sequence of float
        Nominal quantile levels, e.g. [0.1,0.5,0.9].
    q_prefix : str
        Prefix for quantile columns (e.g. 'subsidence' for
        columns subsidence_q10, subsidence_q50, …).
    actual_col : str
        Column name of the true continuous outcome.
    method : {'isotonic','logistic'}, default 'isotonic'
        Calibration method to fit P(actual <= threshold).
    out_prefix : str, default 'calib'
        Prefix for the calibrated columns (e.g. 'calib_subsidence_q10').
    grid_mode : {'unit','range'}, default 'unit'
        Which domain to build the inversion grid over:
          - 'unit' → np.linspace(0,1,grid_size)
          - 'range' → np.linspace(min(raw), max(raw), grid_size)
    grid_size : int, default 1001
        Number of points in the inversion grid.
    group_by : str, optional
        If provided, calibrate separately per `df[group_by]`
        (e.g. 'forecast_step').

    Returns
    -------
    pd.DataFrame
        A copy of `df` with new columns
        `{out_prefix}_{q_prefix}_qXX` appended.
    """

    def _calibrate_block(block: pd.DataFrame) -> pd.DataFrame:
        out = block.copy()
        for q in quantiles:
            raw_col = f"{q_prefix}_q{int(q * 100)}"
            cal_col = f"{out_prefix}_{raw_col}"

            # Build grid
            if grid_mode == "unit":
                grid = np.linspace(0.0, 1.0, grid_size)
            else:  # 'range'
                vals = out[raw_col].to_numpy()
                grid = np.linspace(
                    vals.min(), vals.max(), grid_size
                )

            # Binary target: actual ≤ raw_pred
            y_thr = (
                (out[actual_col] <= out[raw_col])
                .astype(int)
                .values
            )
            x_thr = out[raw_col].values

            # Fit calibrator
            if method == "isotonic":
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(x_thr, y_thr)
                cdf_vals = iso.predict(grid)
            else:  # 'logistic'
                lr = LogisticRegression(solver="lbfgs")
                lr.fit(x_thr.reshape(-1, 1), y_thr)
                cdf_vals = lr.predict_proba(
                    grid.reshape(-1, 1)
                )[:, 1]

            # Invert at q
            idx = np.searchsorted(cdf_vals, q, side="left")
            idx = np.clip(idx, 0, grid_size - 1)
            out[cal_col] = grid[idx]

        return out

    # Apply either globally or per-group
    if group_by and group_by in df.columns:
        pieces = []
        for _, grp in df.groupby(group_by, sort=False):
            pieces.append(_calibrate_block(grp))
        df_out = pd.concat(pieces).sort_index()
    else:
        df_out = _calibrate_block(df)

    return df_out


fit_interval_factors_df.__doc__ = r"""
Fit empirical interval-width correction factors from an evaluation
DataFrame.

This function learns one multiplicative factor per forecast horizon
so that an existing predictive interval reaches a desired empirical
coverage on evaluation data. It is designed for the common case where
a model already produces quantile forecasts, but the resulting
intervals are systematically too narrow or too wide.

For each group of rows, the function identifies

- one lower quantile,
- one median-like quantile, and
- one upper quantile,

then searches for the smallest factor :math:`f \ge 1` such that the
rescaled interval reaches the requested target coverage.

If :math:`q_{lo}`, :math:`q_{md}`, and :math:`q_{hi}` denote the
selected lower, median, and upper forecasts, the calibrated interval
used during fitting is

.. math::

   \tilde q_{lo} = q_{md} - f \, (q_{md} - q_{lo})

.. math::

   \tilde q_{hi} = q_{md} + f \, (q_{hi} - q_{md})

and the fitted factor is the smallest value whose empirical coverage
is at least the target, up to the requested tolerance.

This is a lightweight post-hoc calibration strategy that is especially
useful for multi-horizon quantile forecasts, where sharpness and
coverage must be balanced carefully in downstream risk analysis
:cite:p:`Limetal2021,Kouadio2025XTFT`.

Parameters
----------
df_eval : pandas.DataFrame
    Evaluation table containing the observed target and the forecast
    quantile columns used for calibration.
target_name : str, default="subsidence"
    Base name of the forecasted variable. When ``column_map`` is not
    provided, quantile columns are auto-detected using names such as
    ``"{target_name}_q10"``, ``"{target_name}_q50"``, and
    ``"{target_name}_q90"``.
column_map : mapping or None, default=None
    Optional explicit column mapping. This may contain at least

    - ``"quantiles"``: a mapping from quantile level to column name,
      or a list of quantile column names,
    - ``"actual"``: the observed target column.

    This is useful when the DataFrame does not follow the default
    naming convention.
step_col : str, default="forecast_step"
    Column used to fit separate factors per horizon or lead time.
    If this column is missing, the full DataFrame is treated as a
    single group and a single factor is returned with key ``1``.
interval : tuple of float, default=(0.1, 0.9)
    Nominal lower and upper quantiles defining the interval to
    calibrate. The function uses the closest available quantiles in
    the DataFrame.
target_coverage : float, default=0.8
    Desired empirical coverage of the calibrated interval.
median_q : float, default=0.5
    Target quantile used as the center of the interval expansion.
    The nearest available quantile in the DataFrame is used.
tol : float, default=1e-3
    Numerical tolerance used during the bisection search.
f_max : float, default=5.0
    Upper bound for the searched interval factor. If the target
    coverage cannot be reached before this bound, ``f_max`` is
    returned for that group.
max_iter : int, default=32
    Maximum number of bisection iterations used to fit each factor.
verbose : int, default=1
    Verbosity level forwarded to the internal logging helper.
logger : logging.Logger or None, default=None
    Optional logger used for progress messages.

Returns
-------
dict[int, float]
    Dictionary mapping each forecast horizon to its fitted interval
    factor.

Raises
------
ValueError
    If no actual column can be resolved, or if no valid quantile
    interval can be formed from the available columns.
KeyError
    If a user-specified column in ``column_map`` is missing.

Notes
-----
This function does not change the forecasts directly. It only learns
the correction factors. To apply the fitted factors to evaluation or
future forecasts, use :func:`apply_interval_factors_df`.

Because the method calibrates interval width around a median-like
center, it is most appropriate when miscalibration is primarily a
sharpness problem rather than a severe bias in the central forecast.

Examples
--------
>>> import pandas as pd
>>> from geoprior.utils.calibrate import fit_interval_factors_df
>>> df = pd.DataFrame(
...     {
...         "forecast_step": [1, 1, 1, 2, 2, 2],
...         "subsidence_actual": [0.4, 0.7, 1.0, 0.5, 0.8, 1.2],
...         "subsidence_q10": [0.3, 0.5, 0.8, 0.4, 0.6, 0.9],
...         "subsidence_q50": [0.4, 0.7, 1.0, 0.5, 0.8, 1.1],
...         "subsidence_q90": [0.5, 0.9, 1.1, 0.6, 1.0, 1.3],
...     }
... )
>>> factors = fit_interval_factors_df(
...     df,
...     target_name="subsidence",
...     interval=(0.1, 0.9),
...     target_coverage=0.8,
... )
>>> isinstance(factors, dict)
True

See Also
--------
apply_interval_factors_df :
    Apply learned interval factors to forecast quantiles.
calibrate_quantile_forecasts :
    High-level wrapper that can fit and apply factors in one call.

References
----------
:cite:t:`Limetal2021` discuss multi-horizon quantile forecasting in a
setting where calibration and interpretability both matter.

For subsidence-oriented uncertainty-aware forecasting and practical
post-hoc probabilistic refinement in this project context, see
:cite:t:`Kouadio2025XTFT`.
"""


apply_interval_factors_df.__doc__ = r"""
Apply interval-width calibration factors to quantile forecasts stored
in a DataFrame.

This function rescales forecast quantiles around a median-like central
forecast using either

- a single scalar factor applied to every row, or
- a mapping of ``forecast_step -> factor``.

The main goal is to widen or shrink predictive intervals after the
factors have been estimated on evaluation data with
:func:`fit_interval_factors_df`.

If :math:`q_{md}` is the selected median-like forecast and
:math:`f_h` is the factor for horizon :math:`h`, then for a lower
quantile :math:`q < q_{md}` the calibrated value is

.. math::

   \tilde q = q_{md} - f_h \, (q_{md} - q)

and for an upper quantile :math:`q > q_{md}` it is

.. math::

   \tilde q = q_{md} + f_h \, (q - q_{md})

while the median itself is left unchanged.

Parameters
----------
df : pandas.DataFrame
    DataFrame containing forecast quantile columns.
factors : mapping or float
    Calibration factor specification. This may be

    - a scalar applied to every row, or
    - a mapping from forecast horizon to factor.

target_name : str, default="subsidence"
    Base variable name used to auto-detect quantile columns when
    ``column_map`` is not supplied.
column_map : mapping or None, default=None
    Optional explicit mapping describing the quantile columns.
step_col : str, default="forecast_step"
    Horizon column used when ``factors`` is a mapping. If this column
    is absent, it is created and filled with ``1``.
median_q : float, default=0.5
    Quantile used as the center of the recalibration. The closest
    available forecast quantile is used.
keep_original : bool, default=False
    If True, each raw quantile column is preserved in an additional
    ``"{col}_raw"`` column before calibration is applied.
factor_col : str, default="calibration_factor"
    Name of the column storing the factor applied to each row.
calibrated_col : str, default="is_calibrated"
    Name of the Boolean marker column added to the returned DataFrame.
enforce_monotonic : {"cummax", "sort", "none"}, default="cummax"
    Strategy used after recalibration to keep quantiles ordered.

    - ``"cummax"`` applies a cumulative maximum across quantiles,
    - ``"sort"`` sorts the calibrated quantiles row-wise,
    - ``"none"`` leaves the result unchanged.

verbose : int, default=1
    Verbosity level forwarded to the internal logger helper.
logger : logging.Logger or None, default=None
    Optional logger used for progress messages.

Returns
-------
pandas.DataFrame
    A calibrated copy of ``df`` containing updated quantiles and the
    metadata columns specified by ``factor_col`` and
    ``calibrated_col``.

Raises
------
ValueError
    If no quantile columns can be resolved from the DataFrame or if
    an invalid ``enforce_monotonic`` mode is requested.

Notes
-----
Monotonicity enforcement is important because quantile-wise
post-processing can otherwise produce crossing quantiles. Keeping
quantiles ordered is especially important when the calibrated outputs
are used to derive intervals, exceedance probabilities, or downstream
risk metrics :cite:p:`Limetal2021,Kouadio2025XTFT`.

This function only modifies the quantile forecasts. It does not
recompute summary coverage statistics. For fit-and-apply workflows
with optional evaluation summaries, use
:func:`calibrate_quantile_forecasts`.

Examples
--------
>>> import pandas as pd
>>> from geoprior.utils.calibrate import apply_interval_factors_df
>>> df = pd.DataFrame(
...     {
...         "forecast_step": [1, 1, 2, 2],
...         "subsidence_q10": [0.3, 0.5, 0.4, 0.6],
...         "subsidence_q50": [0.4, 0.7, 0.5, 0.8],
...         "subsidence_q90": [0.5, 0.9, 0.6, 1.0],
...     }
... )
>>> out = apply_interval_factors_df(
...     df,
...     factors={1: 1.2, 2: 1.1},
...     target_name="subsidence",
... )
>>> "calibration_factor" in out.columns
True

See Also
--------
fit_interval_factors_df :
    Fit per-horizon interval scaling factors.
calibrate_quantile_forecasts :
    High-level wrapper for fitting, applying, and summarizing
    interval calibration.
"""


calibrate_quantile_forecasts.__doc__ = r"""
Fit and apply post-hoc interval calibration for quantile forecasts.

This is the high-level DataFrame-oriented entry point for interval
recalibration in ``geoprior.utils.calibrate``. It can

1. detect whether evaluation forecasts already appear calibrated,
2. fit interval-width correction factors from evaluation data,
3. apply those factors to evaluation and/or future forecasts,
4. compute before/after summary diagnostics on the evaluation set,
5. optionally save the outputs to disk.

The function is designed for workflows where quantile forecasts are
already available in tabular form and calibration should be handled
without retraining the forecasting model.

Conceptually, the function widens or narrows a predictive interval
around a median-like forecast so that the empirical interval coverage
better matches the requested target. This is a practical post-hoc
strategy for uncertainty refinement in multi-horizon forecasting
pipelines :cite:p:`Limetal2021,Kouadio2025XTFT`.

Parameters
----------
df_eval : pandas.DataFrame or None, default=None
    Evaluation forecasts used to fit calibration factors and to
    compute before/after diagnostics. This table should contain the
    observed target column in addition to the quantile forecasts.
df_future : pandas.DataFrame or None, default=None
    Future or inference forecasts to which the fitted factors should
    be applied. This table does not need observed targets.
target_name : str, default="subsidence"
    Base name used to infer forecast and observation columns when
    ``column_map`` is not explicitly supplied.
column_map : mapping or None, default=None
    Optional mapping describing the observed column and the quantile
    columns. This is helpful when the table does not follow the
    default naming conventions.
step_col : str, default="forecast_step"
    Column used to fit and apply separate factors per forecast
    horizon.
interval : tuple of float, default=(0.1, 0.9)
    Lower and upper quantiles defining the interval to calibrate.
    The nearest available quantiles are used.
target_coverage : float, default=0.8
    Desired empirical coverage after calibration.
median_q : float, default=0.5
    Central quantile used as the expansion anchor.
use : {"auto", True, False}, default="auto"
    Control flag for whether calibration is performed.

    - ``False`` disables calibration and returns inputs unchanged.
    - ``"auto"`` skips calibration when evaluation forecasts already
      look calibrated.
    - ``True`` forces calibration even if the automatic check would
      skip it.

tol : float, default=0.02
    Tolerance used by the automatic already-calibrated check.
f_max : float, default=5.0
    Maximum factor allowed during fitting.
max_iter : int, default=32
    Maximum number of bisection iterations used when fitting factors.
keep_original : bool, default=False
    If True, raw quantiles are copied into ``*_raw`` columns before
    calibration is applied.
enforce_monotonic : {"cummax", "sort", "none"}, default="cummax"
    Strategy used to prevent quantile crossing after recalibration.
overall_key : str or None, default="__overall__"
    Reserved label stored in the returned statistics dictionary for
    overall summary reporting.
calibrated_col : str, default="is_calibrated"
    Column name added to calibrated outputs as a Boolean marker.
factor_col : str, default="calibration_factor"
    Column name used to store the factor applied to each row.
factors : float or mapping or None, default=None
    Optional user-specified calibration factors. If provided, these
    take precedence over factors fitted from ``df_eval``.
save_eval : str or path-like or None, default=None
    Optional CSV path for saving the calibrated evaluation table.
save_future : str or path-like or None, default=None
    Optional CSV path for saving the calibrated future table.
save_stats : str or path-like or None, default=None
    Optional JSON path for saving the calibration summary.
verbose : int, default=1
    Verbosity level forwarded to logging helpers.
logger : logging.Logger or None, default=None
    Optional logger used for progress messages.

Returns
-------
df_eval_cal : pandas.DataFrame or None
    Calibrated evaluation DataFrame, or None when no evaluation table
    was provided.
df_future_cal : pandas.DataFrame or None
    Calibrated future DataFrame, or None when no future table was
    provided.
stats : dict[str, Any]
    Dictionary describing the calibration workflow. Depending on the
    path taken, it may contain

    - the target interval and target coverage,
    - the fitted or user-specified factors,
    - skip reasons,
    - evaluation summaries before and after calibration.

Notes
-----
In ``use="auto"`` mode, the function first checks for an explicit
``calibrated_col`` and then falls back to a simple empirical
coverage-based decision. This makes the wrapper conservative in
repeated workflows, where the same tables may pass through the
calibration stage more than once.

The returned ``stats`` dictionary is designed to be JSON-friendly and
therefore suitable for audit trails, experiment manifests, or gallery
artifacts.

Examples
--------
>>> import pandas as pd
>>> from geoprior.utils.calibrate import (
...     calibrate_quantile_forecasts,
... )
>>> df_eval = pd.DataFrame(
...     {
...         "forecast_step": [1, 1, 2, 2],
...         "subsidence_actual": [0.4, 0.7, 0.5, 0.9],
...         "subsidence_q10": [0.3, 0.5, 0.4, 0.6],
...         "subsidence_q50": [0.4, 0.7, 0.5, 0.8],
...         "subsidence_q90": [0.5, 0.9, 0.6, 1.0],
...     }
... )
>>> df_eval_cal, df_future_cal, stats = (
...     calibrate_quantile_forecasts(
...         df_eval=df_eval,
...         target_name="subsidence",
...         target_coverage=0.8,
...     )
... )
>>> isinstance(stats, dict)
True

See Also
--------
fit_interval_factors_df :
    Fit per-horizon interval-width correction factors.
apply_interval_factors_df :
    Apply a scalar or per-horizon factor map to quantile forecasts.

References
----------
For the broader role of calibrated probabilistic multi-horizon
forecasting, see :cite:t:`Limetal2021`.

For uncertainty-rich forecasting in the present project ecosystem, see
:cite:t:`Kouadio2025XTFT`.
"""


calibrate_probability_forecast.__doc__ = r"""
Calibrate binary probability forecasts with isotonic regression or
logistic scaling.

This function post-processes a column of raw forecast probabilities so
that they better agree with observed binary outcomes. It returns a
copy of the input DataFrame with one additional calibrated
probability column.

Two calibration modes are supported:

- ``"isotonic"``
    Non-parametric monotone calibration using isotonic regression.
- ``"logistic"``
    Parametric calibration using logistic regression on the raw
    probabilities.

Parameters
----------
df : pandas.DataFrame
    DataFrame containing a column of raw forecast probabilities and a
    column of observed binary outcomes.
prob_col : str
    Name of the column containing the raw forecast probabilities.
    Values are usually expected in the interval ``[0, 1]``.
actual_col : str
    Name of the column containing the realized binary outcomes,
    encoded as ``0`` and ``1``.
method : {"isotonic", "logistic"}, default="isotonic"
    Calibration method to apply.
out_col : str or None, default=None
    Name of the calibrated probability column. If None,
    ``f"{prob_col}_calib"`` is used.
clip : bool, default=True
    If True, calibrated outputs are clipped to ``[0, 1]`` before
    being written to the result.
savefile : str or None, default=None
    Optional output path handled by the :func:`SaveFile` decorator.

Returns
-------
pandas.DataFrame
    Copy of ``df`` with an additional calibrated probability column.

Raises
------
ValueError
    If ``method`` is not one of ``"isotonic"`` or ``"logistic"``.

Notes
-----
This function is appropriate for event-probability forecasts, not for
continuous quantile forecasts. For quantile-table recalibration, use
:func:`calibrate_forecasts` or
:func:`calibrate_quantile_forecasts` depending on the workflow.

The isotonic option is often preferred when a flexible monotone
mapping is desired, whereas the logistic option is useful when a
simple parametric correction is sufficient.

Examples
--------
>>> import pandas as pd
>>> from geoprior.utils.calibrate import (
...     calibrate_probability_forecast,
... )
>>> df = pd.DataFrame(
...     {
...         "p_raw": [0.10, 0.30, 0.60, 0.85],
...         "y": [0, 0, 1, 1],
...     }
... )
>>> out = calibrate_probability_forecast(
...     df,
...     prob_col="p_raw",
...     actual_col="y",
...     method="isotonic",
... )
>>> "p_raw_calib" in out.columns
True

See Also
--------
calibrate_forecasts :
    Calibrate continuous quantile forecasts by inverting calibrated
    CDF estimates.
calibrate_quantile_forecasts :
    Fit-and-apply interval calibration for tabular quantile forecasts.
"""


calibrate_forecasts.__doc__ = r"""
Calibrate continuous quantile forecasts by fitting a calibrated CDF
surrogate and inverting it at the nominal quantile levels.

This function works on a DataFrame containing continuous quantile
forecasts and the corresponding observed target. For each requested
quantile level :math:`q`, it builds a binary supervision signal of
the form

.. math::

   y_q = \mathbb{1}\{y \le \hat q\}

where :math:`y` is the observed outcome and :math:`\hat q` is the raw
forecast quantile at level :math:`q`.

A monotone classifier is then fitted to approximate the conditional
CDF at that threshold. Finally, the calibrated forecast is obtained by
inverting the fitted CDF on a grid and taking the first threshold
whose calibrated CDF reaches the nominal quantile level.

This provides a practical post-hoc recalibration mechanism for
continuous quantile forecasts in multi-horizon settings
:cite:p:`Limetal2021,Kouadio2025XTFT`.

Parameters
----------
df : pandas.DataFrame
    DataFrame containing the raw forecast quantile columns and the
    observed target column.
quantiles : sequence of float
    Nominal quantile levels to recalibrate, expressed on the unit
    interval, for example ``[0.1, 0.5, 0.9]``.
q_prefix : str
    Base prefix for the raw quantile columns. For example, if
    ``q_prefix="subsidence"``, the function expects columns such as
    ``"subsidence_q10"``, ``"subsidence_q50"``, and
    ``"subsidence_q90"``.
actual_col : str
    Name of the observed continuous target column.
method : {"isotonic", "logistic"}, default="isotonic"
    Calibration method used to estimate the monotone CDF surrogate at
    each quantile level.
out_prefix : str, default="calib"
    Prefix used to build the new calibrated quantile column names.
    The output columns follow the pattern
    ``"{out_prefix}_{q_prefix}_qXX"``.
grid_mode : {"unit", "range"}, default="unit"
    Domain used to construct the inversion grid.

    - ``"unit"`` builds ``np.linspace(0, 1, grid_size)``.
      This is mainly appropriate when the forecast support is already
      normalized to the unit interval.
    - ``"range"`` builds a grid from the observed range of each raw
      quantile column and is typically the safer choice for forecasts
      on the original target scale.

grid_size : int, default=1001
    Number of points used in the inversion grid.
group_by : str or None, default=None
    Optional grouping column. When provided, calibration is performed
    separately within each group, for example by forecast horizon.
savefile : str or None, default=None
    Optional output path handled by the :func:`SaveFile` decorator.

Returns
-------
pandas.DataFrame
    Copy of ``df`` with calibrated quantile columns appended.

Notes
-----
This function differs from :func:`calibrate_quantile_forecasts`.
Here, each quantile is recalibrated through a classifier-plus-
inversion approach. By contrast,
:func:`calibrate_quantile_forecasts` applies multiplicative interval
width factors around a median-like forecast.

The two approaches answer different needs:

- use :func:`calibrate_forecasts` when you want to recalibrate the
  quantile levels themselves through a threshold-CDF view;
- use :func:`calibrate_quantile_forecasts` when your main issue is
  interval under- or over-dispersion and you want a simpler,
  auditable width correction.

When ``group_by`` is used, the function preserves the original row
index order after concatenating the calibrated groups.

Examples
--------
>>> import pandas as pd
>>> from geoprior.utils.calibrate import calibrate_forecasts
>>> df = pd.DataFrame(
...     {
...         "forecast_step": [1, 1, 2, 2],
...         "subsidence_actual": [0.4, 0.8, 0.5, 1.0],
...         "subsidence_q10": [0.3, 0.6, 0.4, 0.8],
...         "subsidence_q50": [0.4, 0.8, 0.5, 1.0],
...         "subsidence_q90": [0.5, 1.0, 0.6, 1.2],
...     }
... )
>>> out = calibrate_forecasts(
...     df,
...     quantiles=[0.1, 0.5, 0.9],
...     q_prefix="subsidence",
...     actual_col="subsidence_actual",
...     method="isotonic",
...     grid_mode="range",
...     group_by="forecast_step",
... )
>>> "calib_subsidence_q50" in out.columns
True

See Also
--------
calibrate_probability_forecast :
    Calibrate event probabilities rather than continuous quantiles.
calibrate_quantile_forecasts :
    Wrapper for interval-width calibration on tabular forecasts.
fit_interval_factors_df :
    Learn empirical per-horizon interval scaling factors.

References
----------
For interpretable multi-horizon quantile forecasting, see
:cite:t:`Limetal2021`.

For uncertainty-rich forecasting in the GeoPrior ecosystem, see
:cite:t:`Kouadio2025XTFT`.
"""
