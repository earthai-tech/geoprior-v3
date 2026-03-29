"""
Coverage versus sharpness in probabilistic forecasts
====================================================

This lesson teaches one of the most important ideas in forecast
uncertainty:

**a good interval forecast must be both honest and useful.**

In practice, that means balancing two competing goals:

- **coverage**:
  does the interval contain the truth often enough?
- **sharpness**:
  is the interval still narrow enough to be informative?

Why this page matters
---------------------
After interval calibration, the next question is no longer only
"can we widen the intervals?" but:

**how should we judge whether the resulting uncertainty is actually
better?**

That is the role of a coverage-versus-sharpness analysis.

In GeoPrior, this lesson naturally combines three pieces:

- :func:`geoprior.utils.forecast_utils.evaluate_forecast`,
  which computes coverage and sharpness from an evaluation forecast
  table in quantile mode;
- :func:`geoprior.utils.calibrate.calibrate_quantile_forecasts`,
  which can recalibrate under-covered intervals;
- :func:`geoprior.utils.forecast_utils.plot_reliability_diagram`,
  which visualizes empirical coverage against nominal probability
  for interval forecasts.

The evaluator reports ``coverage80`` and ``sharpness80`` for a chosen
interval such as q10-q90, and the reliability helper draws the perfect
calibration line against empirical coverage from the forecast
intervals.

What this lesson teaches
------------------------
We will:

1. create one common synthetic truth field,
2. build several probabilistic forecast variants:
   overconfident, balanced, conservative,
3. recalibrate the overconfident forecast,
4. compare all models in coverage-sharpness space,
5. inspect how that trade-off changes by horizon,
6. finish with a reliability diagram.

This page is intentionally synthetic so it remains fully executable
during the documentation build.
"""

# %%
# Imports
# -------
# We use the real uncertainty and forecast-evaluation helpers.

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from geoprior.utils.calibrate import calibrate_quantile_forecasts
from geoprior.utils.forecast_utils import (
    evaluate_forecast,
    plot_reliability_diagram,
)

# %%
# Step 1 - Build one shared synthetic truth field
# -----------------------------------------------
# We mimic a compact spatial forecasting problem.
#
# The important design choice here is that *all* forecast variants
# will share the same true outcomes. This makes their uncertainty
# comparison fair.

rng = np.random.default_rng(21)

nx = 9
ny = 6
steps = [1, 2, 3]
years = {1: 2024, 2: 2025, 3: 2026}

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
        + ((yn - 0.34) ** 2) / 0.032
    )
)
ridge = 0.52 * np.exp(
    -(
        ((xn - 0.28) ** 2) / 0.028
        + ((yn - 0.72) ** 2) / 0.050
    )
)
gradient = 0.44 * xn + 0.22 * (1.0 - yn)

# Quantile levels available in the forecast tables.
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Standard-normal style z scores for symmetric quantiles.
z_map = {
    0.1: -1.2816,
    0.2: -0.8416,
    0.3: -0.5244,
    0.4: -0.2533,
    0.5: 0.0,
    0.6: 0.2533,
    0.7: 0.5244,
    0.8: 0.8416,
    0.9: 1.2816,
}

# Shared truth-generating process by horizon step.
truth_mean_by_step: dict[int, np.ndarray] = {}
truth_actual_by_step: dict[int, np.ndarray] = {}
base_sigma_by_step: dict[int, np.ndarray] = {}

for step in steps:
    step_scale = {1: 1.00, 2: 1.18, 3: 1.42}[step]

    mu = (
        2.2
        + 1.45 * gradient
        + 2.05 * hotspot
        + 0.95 * ridge
    ) * step_scale

    # "Natural" uncertainty scale before model over/under-confidence.
    sigma = (
        0.22
        + 0.07 * xn
        + 0.04 * hotspot
        + 0.03 * step
    )

    # Shared actual values that all models will face.
    actual = mu + rng.normal(0.0, sigma, size=n_sites)

    truth_mean_by_step[step] = mu
    truth_actual_by_step[step] = actual
    base_sigma_by_step[step] = sigma

print("Number of spatial samples:", n_sites)
print("Forecast steps:", steps)
print("Quantiles:", quantiles)

# %%
# Step 2 - Create forecast tables with different uncertainty behavior
# -------------------------------------------------------------------
# We now build three forecast styles:
#
# - **overconfident**:
#   intervals too narrow;
# - **balanced**:
#   intervals reasonably matched to the data;
# - **conservative**:
#   intervals too wide.
#
# All three share the same median forecast so the comparison focuses
# on uncertainty behavior rather than mean-forecast accuracy.

def make_quantile_df(
    *,
    width_scale: float,
    median_bias: float = 0.0,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for step in steps:
        mu = truth_mean_by_step[step] + median_bias
        actual = truth_actual_by_step[step]
        sigma = base_sigma_by_step[step] * width_scale

        for i in range(n_sites):
            row = {
                "sample_idx": i,
                "forecast_step": step,
                "coord_t": years[step],
                "coord_x": float(x_flat[i]),
                "coord_y": float(y_flat[i]),
                "subsidence_actual": float(actual[i]),
            }

            for q in quantiles:
                col = f"subsidence_q{int(q * 100):02d}"
                row[col] = float(mu[i] + z_map[q] * sigma[i])

            rows.append(row)

    return pd.DataFrame(rows)


df_over = make_quantile_df(width_scale=0.55)
df_bal = make_quantile_df(width_scale=1.00)
df_cons = make_quantile_df(width_scale=1.70)

print("Overconfident rows:", len(df_over))
print("Balanced rows:", len(df_bal))
print("Conservative rows:", len(df_cons))

print("")
print(df_over.head(6).to_string(index=False))

# %%
# Step 3 - Create a future forecast table for calibration transfer
# ----------------------------------------------------------------
# The calibration wrapper expects both an evaluation table and, when
# available, a future table. The future table does not contain actuals.
#
# We only need one future table here because we are going to calibrate
# the overconfident forecast and compare the calibrated result to the
# other variants.

def make_future_df(
    *,
    width_scale: float,
    median_growth: float = 1.06,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for step in steps:
        mu = truth_mean_by_step[step] * median_growth
        sigma = base_sigma_by_step[step] * width_scale

        for i in range(n_sites):
            row = {
                "sample_idx": i,
                "forecast_step": step,
                "coord_t": years[step] + 1,
                "coord_x": float(x_flat[i]),
                "coord_y": float(y_flat[i]),
            }

            for q in quantiles:
                col = f"subsidence_q{int(q * 100):02d}"
                row[col] = float(mu[i] + z_map[q] * sigma[i])

            rows.append(row)

    return pd.DataFrame(rows)


df_future_over = make_future_df(width_scale=0.55)

# %%
# Step 4 - Calibrate the overconfident forecast
# ---------------------------------------------
# We now produce a fourth model:
#
# - **calibrated**:
#   obtained by applying the real GeoPrior interval-calibration
#   workflow to the overconfident forecast.
#
# The calibration wrapper fits horizon-wise widening factors on
# ``df_eval`` and applies them to both evaluation and future tables. 

df_over_cal, df_future_over_cal, stats_cal = calibrate_quantile_forecasts(
    df_eval=df_over,
    df_future=df_future_over,
    target_name="subsidence",
    step_col="forecast_step",
    interval=(0.1, 0.9),
    target_coverage=0.8,
    median_q=0.5,
    use="auto",
    keep_original=True,
    enforce_monotonic="cummax",
    verbose=0,
)

print("Calibration stats")
print(stats_cal)

# %%
# Step 5 - Evaluate all models with the real evaluator
# ----------------------------------------------------
# ``evaluate_forecast`` computes deterministic accuracy plus interval
# quality. In quantile mode it reports:
#
# - ``coverage80`` for the chosen interval,
# - ``sharpness80`` as mean interval width,
# - and optional per-horizon deterministic summaries. 
model_tables = {
    "Overconfident": df_over,
    "Balanced": df_bal,
    "Conservative": df_cons,
    "Calibrated": df_over_cal,
}

overall_rows = []
for name, df_model in model_tables.items():
    metrics = evaluate_forecast(
        df_model,
        target_name="subsidence",
        per_horizon=True,
        quantile_interval=(0.1, 0.9),
        verbose=0,
    )
    overall = metrics["__overall__"]

    overall_rows.append(
        {
            "model": name,
            "overall_mae": overall["overall_mae"],
            "overall_rmse": overall["overall_rmse"],
            "overall_r2": overall["overall_r2"],
            "coverage80": overall["coverage80"],
            "sharpness80": overall["sharpness80"],
        }
    )

df_overall = pd.DataFrame(overall_rows)
print("")
print("Overall uncertainty summary")
print(df_overall.to_string(index=False))

# %%
# Step 6 - Compute coverage and sharpness by horizon
# --------------------------------------------------
# ``evaluate_forecast`` gives overall interval quality directly, but
# for a teaching page it is helpful to also build a small per-step
# table by hand.
#
# This lets us see whether uncertainty problems become worse at longer
# horizons.

def coverage_and_sharpness_by_step(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for step, g in df.groupby("forecast_step", sort=True):
        actual = g["subsidence_actual"].to_numpy(float)
        lo = g["subsidence_q10"].to_numpy(float)
        hi = g["subsidence_q90"].to_numpy(float)

        coverage = float(np.mean((actual >= lo) & (actual <= hi)))
        sharpness = float(np.mean(hi - lo))

        rows.append(
            {
                "forecast_step": int(step),
                "coverage80": coverage,
                "sharpness80": sharpness,
            }
        )

    return pd.DataFrame(rows)


step_tables = {
    name: coverage_and_sharpness_by_step(df_model)
    for name, df_model in model_tables.items()
}

for name, df_step in step_tables.items():
    print("")
    print(name)
    print(df_step.to_string(index=False))

# %%
# Step 7 - Plot the coverage-versus-sharpness trade-off
# -----------------------------------------------------
# This is the main figure of the lesson.
#
# Read this scatter plot as follows:
#
# - moving **up** improves empirical coverage;
# - moving **right** widens the interval;
# - the "best" point depends on the target use case, but typically
#   we want enough coverage without becoming excessively wide.

fig, ax = plt.subplots(figsize=(7.2, 5.5))

for _, row in df_overall.iterrows():
    ax.scatter(
        row["sharpness80"],
        row["coverage80"],
        s=90,
        label=row["model"],
    )
    ax.annotate(
        row["model"],
        (row["sharpness80"], row["coverage80"]),
        xytext=(6, 6),
        textcoords="offset points",
    )

ax.axhline(0.80, linestyle="--", linewidth=1.5)
ax.set_xlabel("Sharpness80 (mean interval width)")
ax.set_ylabel("Coverage80 (empirical coverage)")
ax.set_title("Coverage versus sharpness")
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

# %%
# How to read this trade-off
# --------------------------
# Typical interpretation:
#
# - **Overconfident**:
#   sharp but under-covered;
# - **Conservative**:
#   well-covered but very wide;
# - **Balanced**:
#   closer to a useful compromise;
# - **Calibrated**:
#   should move upward from the overconfident point, usually with some
#   rightward movement because intervals widen.

# %%
# Step 8 - Plot horizon-wise coverage and sharpness
# -------------------------------------------------
# A forecast can look acceptable overall and still behave badly at the
# longer horizons. So we now inspect the trade-off step by step.

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

for name, df_step in step_tables.items():
    axes[0].plot(
        df_step["forecast_step"],
        df_step["coverage80"],
        marker="o",
        label=name,
    )
axes[0].axhline(0.80, linestyle="--", linewidth=1.2)
axes[0].set_xlabel("Forecast step")
axes[0].set_ylabel("Coverage80")
axes[0].set_title("Coverage by horizon")
axes[0].grid(True, linestyle=":", alpha=0.6)
axes[0].legend()

for name, df_step in step_tables.items():
    axes[1].plot(
        df_step["forecast_step"],
        df_step["sharpness80"],
        marker="o",
        label=name,
    )
axes[1].set_xlabel("Forecast step")
axes[1].set_ylabel("Sharpness80")
axes[1].set_title("Sharpness by horizon")
axes[1].grid(True, linestyle=":", alpha=0.6)
axes[1].legend()

plt.tight_layout()
plt.show()

# %%
# Why this horizon view matters
# -----------------------------
# This panel often reveals the real uncertainty story.
#
# In many forecasting systems:
#
# - later steps need wider intervals,
# - undercoverage becomes more severe with horizon,
# - recalibration helps most at the longer steps.
#
# That is exactly why the first uncertainty lesson fit interval factors
# per horizon instead of using one global factor for the whole forecast.

# %%
# Step 9 - Add a reliability diagram
# ----------------------------------
# The reliability diagram belongs naturally in this section because it
# compares nominal interval probability to empirical coverage. The
# helper can wrap simple DataFrames and draw the diagonal "perfect
# calibration" reference line. 
#
# We compare the overconfident, balanced, conservative, and calibrated
# forecasts on the same observed outcomes.

y_true_series = pd.Series(df_over["subsidence_actual"].to_numpy())

rel_models = {
    "Overconfident": {
        "forecasts": df_over[
            [f"subsidence_q{int(q * 100):02d}" for q in quantiles]
        ].reset_index(drop=True),
        "marker": "o",
    },
    "Balanced": {
        "forecasts": df_bal[
            [f"subsidence_q{int(q * 100):02d}" for q in quantiles]
        ].reset_index(drop=True),
        "marker": "s",
    },
    "Conservative": {
        "forecasts": df_cons[
            [f"subsidence_q{int(q * 100):02d}" for q in quantiles]
        ].reset_index(drop=True),
        "marker": "^",
    },
    "Calibrated": {
        "forecasts": df_over_cal[
            [f"subsidence_q{int(q * 100):02d}" for q in quantiles]
        ].reset_index(drop=True),
        "marker": "D",
    },
}

plot_reliability_diagram(
    models_data=rel_models,
    y_true=y_true_series.reset_index(drop=True),
    prefix="subsidence",
    figsize=(7, 7),
    title="Reliability diagram across forecast variants",
    plot_style="default",
    verbose=0,
)

# %%
# How to read the reliability diagram
# -----------------------------------
# The diagonal represents perfect calibration.
#
# A model below the diagonal is typically overconfident for those
# nominal probabilities, while a model above it is conservative.
#
# This is why the reliability diagram and the coverage-sharpness
# scatter complement each other:
#
# - the scatter summarizes one chosen interval,
# - the reliability diagram shows calibration over a range of nominal
#   interval probabilities.

# %%
# Step 10 - Final practical takeaway
# ----------------------------------
# A good uncertainty analysis should never report coverage alone.
#
# Why?
#
# Because coverage can always be improved by making intervals wider.
# That is not enough.
#
# The right question is:
#
# - how much coverage did we gain,
# - how much sharpness did we lose,
# - and does the new forecast sit closer to a useful operating point?
#
# That is why this page belongs early in the uncertainty section:
# it teaches the core trade-off that the later calibration and
# exceedance pages build on.