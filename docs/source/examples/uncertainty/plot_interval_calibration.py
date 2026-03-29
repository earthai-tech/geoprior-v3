"""
Interval calibration with ``calibrate_quantile_forecasts``
==========================================================

This lesson introduces one of the most important uncertainty utilities
in GeoPrior:
:func:`geoprior.utils.calibrate.calibrate_quantile_forecasts`.

Why this page matters
---------------------
A forecast can be visually impressive and still be poorly calibrated.

For quantile forecasts, a central question is:

**Do the predictive intervals contain the truth as often as they claim?**

For example, if we use the q10-q90 interval as an 80% interval, then
we expect the observed value to fall inside that band about 80% of the
time. If it falls inside much less often, the intervals are too narrow
and the forecast is overconfident.

That is the problem this utility addresses.

The calibration workflow in GeoPrior is built around three closely
connected helpers:

- ``fit_interval_factors_df``:
  fit a widening factor per horizon step so the interval coverage
  approaches a target;
- ``apply_interval_factors_df``:
  widen or shrink the quantile columns around the median;
- ``calibrate_quantile_forecasts``:
  the high-level wrapper that fits or accepts factors, applies them
  to evaluation and future forecasts, and returns summary statistics.

The lower-level factor fitter groups by forecast step, estimates a
factor per horizon, and targets a requested coverage level. The
application helper then rescales lower and upper quantiles around the
median and can enforce monotonic quantile ordering. The wrapper adds
automatic skip logic and before/after summaries. 

What this lesson teaches
------------------------
We will:

1. build a synthetic evaluation and future forecast table,
2. deliberately make the raw intervals too narrow,
3. measure coverage and sharpness before calibration,
4. fit horizon-wise interval factors,
5. apply them manually,
6. repeat the same workflow with the high-level wrapper,
7. build explanatory plots showing why the calibration is useful.

This lesson is intentionally synthetic so it remains fully executable
during the documentation build.
"""

# %%
# Imports
# -------
# We use the real uncertainty helpers from GeoPrior.

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from geoprior.utils.calibrate import (
    apply_interval_factors_df,
    calibrate_quantile_forecasts,
    fit_interval_factors_df,
)

# %%
# Step 1 - Build synthetic forecast tables
# ----------------------------------------
# We create:
#
# - ``df_eval``:
#   evaluation rows with actual values available;
# - ``df_future``:
#   future rows with predictions only.
#
# Each row represents one spatial sample at one forecast step.
#
# We deliberately make the q10-q90 intervals too narrow so that the
# initial empirical coverage is below the nominal 80% target.

rng = np.random.default_rng(7)

nx = 9
ny = 6
steps = [1, 2, 3]
future_years = {1: 2024, 2: 2025, 3: 2026}

xv = np.linspace(0.0, 12_000.0, nx)
yv = np.linspace(0.0, 8_000.0, ny)
X, Y = np.meshgrid(xv, yv)

x_flat = X.ravel()
y_flat = Y.ravel()
n_sites = x_flat.size

xn = (x_flat - x_flat.min()) / (x_flat.max() - x_flat.min())
yn = (y_flat - y_flat.min()) / (y_flat.max() - y_flat.min())

hotspot = np.exp(
    -(
        ((xn - 0.72) ** 2) / 0.020
        + ((yn - 0.34) ** 2) / 0.030
    )
)
ridge = 0.50 * np.exp(
    -(
        ((xn - 0.26) ** 2) / 0.030
        + ((yn - 0.74) ** 2) / 0.050
    )
)
gradient = 0.44 * xn + 0.20 * (1.0 - yn)

eval_rows: list[dict[str, float | int]] = []
future_rows: list[dict[str, float | int]] = []

for step in steps:
    scale = {1: 1.00, 2: 1.18, 3: 1.40}[step]

    q50 = (
        2.2
        + 1.4 * gradient
        + 2.0 * hotspot
        + 0.9 * ridge
    ) * scale

    # Deliberately too narrow intervals
    width_raw = (
        0.12
        + 0.04 * xn
        + 0.02 * hotspot
        + 0.02 * step
    )

    q10 = q50 - width_raw
    q90 = q50 + width_raw

    # Truth is noisier than the interval expects -> undercoverage
    actual = q50 + rng.normal(0.0, 0.22 + 0.05 * step, size=n_sites)

    for i in range(n_sites):
        eval_rows.append(
            {
                "sample_idx": i,
                "forecast_step": step,
                "coord_t": future_years[step],
                "coord_x": float(x_flat[i]),
                "coord_y": float(y_flat[i]),
                "subsidence_q10": float(q10[i]),
                "subsidence_q50": float(q50[i]),
                "subsidence_q90": float(q90[i]),
                "subsidence_actual": float(actual[i]),
            }
        )

    # Future forecast uses the same raw uncertainty pattern but no actuals
    q50_future = q50 * 1.06
    q10_future = q50_future - width_raw
    q90_future = q50_future + width_raw

    for i in range(n_sites):
        future_rows.append(
            {
                "sample_idx": i,
                "forecast_step": step,
                "coord_t": future_years[step] + 1,
                "coord_x": float(x_flat[i]),
                "coord_y": float(y_flat[i]),
                "subsidence_q10": float(q10_future[i]),
                "subsidence_q50": float(q50_future[i]),
                "subsidence_q90": float(q90_future[i]),
            }
        )

df_eval = pd.DataFrame(eval_rows)
df_future = pd.DataFrame(future_rows)

print("df_eval shape:", df_eval.shape)
print("df_future shape:", df_future.shape)
print("")
print(df_eval.head(8).to_string(index=False))

# %%
# Step 2 - Compute empirical coverage and sharpness before calibration
# --------------------------------------------------------------------
# Calibration is about matching observed coverage to nominal coverage.
#
# For the q10-q90 interval we treat the nominal target as 80%.
#
# We compute:
#
# - empirical coverage:
#   fraction of actual values inside [q10, q90],
# - sharpness:
#   average interval width q90 - q10.
#
# These two numbers should always be read together.
#
# - Wider intervals often improve coverage.
# - Narrower intervals often reduce coverage.
#
# Good uncertainty quantification balances both.

def coverage_and_width(
    df: pd.DataFrame,
    *,
    lo_col: str = "subsidence_q10",
    hi_col: str = "subsidence_q90",
    actual_col: str = "subsidence_actual",
) -> tuple[float, float]:
    yt = df[actual_col].to_numpy(float)
    lo = df[lo_col].to_numpy(float)
    hi = df[hi_col].to_numpy(float)
    covered = (yt >= lo) & (yt <= hi)
    coverage = float(np.mean(covered))
    width = float(np.mean(hi - lo))
    return coverage, width


before_rows = []
for step, g in df_eval.groupby("forecast_step", sort=True):
    cov, wid = coverage_and_width(g)
    before_rows.append(
        {
            "forecast_step": int(step),
            "coverage_before": cov,
            "sharpness_before": wid,
        }
    )

df_before = pd.DataFrame(before_rows)

print("")
print("Before calibration")
print(df_before.to_string(index=False))

# %%
# Interpretation
# --------------
# In this synthetic example, the empirical coverage should be below the
# intended 0.80 target for at least some horizon steps.
#
# That is the uncertainty problem we want to fix: the forecast intervals
# are too sharp relative to the observed residual spread.

# %%
# Step 3 - Fit interval factors with the low-level helper
# -------------------------------------------------------
# ``fit_interval_factors_df`` estimates one factor per horizon step.
#
# Conceptually, each factor rescales the interval around the median:
#
# - lower quantiles move farther below q50,
# - upper quantiles move farther above q50.
#
# The function groups by ``forecast_step`` and uses a bisection search
# so the empirical coverage approaches the requested target. 

factors = fit_interval_factors_df(
    df_eval,
    target_name="subsidence",
    step_col="forecast_step",
    interval=(0.1, 0.9),
    target_coverage=0.8,
    median_q=0.5,
    verbose=0,
)

print("")
print("Fitted factors by horizon")
print(factors)

# %%
# How to read the factors
# -----------------------
# The factor is interpreted around the median:
#
# - factor > 1:
#   widen the interval,
# - factor = 1:
#   leave the interval unchanged,
# - factor < 1:
#   shrink the interval.
#
# In undercovered forecasts, we usually expect factors larger than 1.

# %%
# Step 4 - Apply interval factors manually
# ----------------------------------------
# ``apply_interval_factors_df`` is the second low-level helper.
#
# It rescales all quantiles around the median and can enforce monotonic
# ordering of the quantiles using strategies such as ``cummax`` or
# ``sort``. It also stores the applied factor and marks the forecast as
# calibrated. 

df_eval_manual = apply_interval_factors_df(
    df_eval,
    factors,
    target_name="subsidence",
    step_col="forecast_step",
    keep_original=True,
    factor_col="calibration_factor",
    calibrated_col="is_calibrated",
    enforce_monotonic="cummax",
    verbose=0,
)

df_future_manual = apply_interval_factors_df(
    df_future,
    factors,
    target_name="subsidence",
    step_col="forecast_step",
    keep_original=True,
    factor_col="calibration_factor",
    calibrated_col="is_calibrated",
    enforce_monotonic="cummax",
    verbose=0,
)

after_rows = []
for step, g in df_eval_manual.groupby("forecast_step", sort=True):
    cov, wid = coverage_and_width(g)
    after_rows.append(
        {
            "forecast_step": int(step),
            "coverage_after": cov,
            "sharpness_after": wid,
        }
    )

df_after = pd.DataFrame(after_rows)

df_compare = df_before.merge(df_after, on="forecast_step")
print("")
print("Before/after manual calibration")
print(df_compare.to_string(index=False))

# %%
# What changed
# ------------
# After calibration:
#
# - coverage should move closer to the target 0.80,
# - sharpness will usually increase because the interval has widened.
#
# This is the central trade-off in interval calibration.

# %%
# Step 5 - Run the high-level wrapper
# -----------------------------------
# ``calibrate_quantile_forecasts`` packages the full workflow:
#
# - optional auto-skip if already calibrated,
# - use user-supplied factors or fit from ``df_eval``,
# - apply to both ``df_eval`` and ``df_future``,
# - return a stats dictionary with before/after summaries.
#
# This is the function that should be taught first in the uncertainty
# gallery because it naturally explains the two helper functions as
# part of one coherent pipeline. 

df_eval_cal, df_future_cal, stats = calibrate_quantile_forecasts(
    df_eval=df_eval,
    df_future=df_future,
    target_name="subsidence",
    step_col="forecast_step",
    interval=(0.1, 0.9),
    target_coverage=0.8,
    median_q=0.5,
    use="auto",
    tol=0.02,
    keep_original=True,
    enforce_monotonic="cummax",
    verbose=0,
)

print("")
print("Calibration stats")
print(stats)

# %%
# Why this wrapper is useful
# --------------------------
# The wrapper adds several practical behaviors that matter in real
# workflows:
#
# - it can skip calibration if the evaluation intervals already look
#   calibrated;
# - it can use fixed user factors instead of refitting;
# - it can save calibrated eval/future tables and JSON stats;
# - it summarizes coverage and sharpness before and after calibration. 

# %%
# Step 6 - Build compact explanatory plots
# ----------------------------------------
# The calibration helpers themselves return DataFrames and dictionaries,
# not figures. But lesson pages become much easier to understand when we
# turn the returned information into small diagnostic plots.
#
# We build two plots:
#
# - coverage before vs after by horizon,
# - sharpness before vs after by horizon.
#
# This makes the calibration trade-off visible immediately.

stats_before = stats["eval_before"]["per_horizon"]
stats_after = stats["eval_after"]["per_horizon"]

plot_rows = []
for h in sorted(stats_before, key=lambda x: int(x)):
    plot_rows.append(
        {
            "forecast_step": int(h),
            "coverage_before": stats_before[h]["coverage"],
            "coverage_after": stats_after[h]["coverage"],
            "sharpness_before": stats_before[h]["sharpness"],
            "sharpness_after": stats_after[h]["sharpness"],
        }
    )

df_plot = pd.DataFrame(plot_rows)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

axes[0].plot(
    df_plot["forecast_step"],
    df_plot["coverage_before"],
    marker="o",
    label="Before",
)
axes[0].plot(
    df_plot["forecast_step"],
    df_plot["coverage_after"],
    marker="s",
    label="After",
)
axes[0].axhline(0.80, linestyle="--", linewidth=1.5, label="Target 0.80")
axes[0].set_xlabel("Forecast step")
axes[0].set_ylabel("Empirical coverage")
axes[0].set_title("Coverage moves toward the target")
axes[0].grid(True, linestyle=":", alpha=0.6)
axes[0].legend()

axes[1].plot(
    df_plot["forecast_step"],
    df_plot["sharpness_before"],
    marker="o",
    label="Before",
)
axes[1].plot(
    df_plot["forecast_step"],
    df_plot["sharpness_after"],
    marker="s",
    label="After",
)
axes[1].set_xlabel("Forecast step")
axes[1].set_ylabel("Mean interval width")
axes[1].set_title("Sharper vs wider intervals")
axes[1].grid(True, linestyle=":", alpha=0.6)
axes[1].legend()

plt.tight_layout()
plt.show()

# %%
# How to read these plots
# -----------------------
# Left panel
# ~~~~~~~~~~
# The goal is not necessarily to hit the target exactly at every step,
# but to move empirical coverage toward the desired level.
#
# Right panel
# ~~~~~~~~~~~
# Better coverage usually comes at the price of wider intervals.
#
# That is why uncertainty pages should always talk about both:
#
# - **coverage**:
#   are we honest enough?
# - **sharpness**:
#   are we still informative?
#
# Calibration is useful when it improves honesty without exploding the
# interval width unnecessarily.

# %%
# Step 7 - Inspect the calibrated tables directly
# -----------------------------------------------
# The resulting forecast tables now carry explicit calibration metadata.

print("")
print("Calibrated evaluation columns")
print(df_eval_cal.columns.tolist())

print("")
print("Calibrated future head")
print(df_future_cal.head(8).to_string(index=False))

# %%
# Notice two practical columns:
#
# - ``calibration_factor``:
#   the factor actually used for that horizon;
# - ``is_calibrated``:
#   a flag that helps downstream logic detect that calibration was
#   already applied.
#
# This is also why the wrapper can auto-skip re-calibration later. 

# %%
# Step 8 - Why this page should come first in uncertainty
# -------------------------------------------------------
# This page teaches the foundational uncertainty idea in the most
# concrete possible way:
#
# - forecasts can be overconfident,
# - interval coverage can be measured directly,
# - intervals can be recalibrated per horizon,
# - future forecasts can inherit the same fitted factors.
#
# That makes this the best first page in the uncertainty gallery.
#
# Later pages can then build naturally on top of it:
#
# - reliability diagrams,
# - coverage-versus-sharpness trade-offs,
# - probability calibration,
# - exceedance-oriented uncertainty analysis.