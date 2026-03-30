"""
From raw model outputs to forecast tables with ``format_and_forecast``
======================================================================

This lesson teaches one of the most important utilities in the
forecasting workflow:
:func:`geoprior.utils.forecast_utils.format_and_forecast`.

Why this page matters
---------------------
Most plotting pages in the forecasting gallery start from already
formatted forecast tables such as:

- ``df_eval``:
  evaluation rows with predictions and actuals,
- ``df_future``:
  future rows with predictions only.

But a real model does not produce those tables directly.

Instead, the model produces raw arrays such as:

- ``y_pred["subs_pred"]`` with shape ``(B, H, Q, O)`` in quantile mode,
- ``y_true[...]`` with shape ``(B, H, O)``,
- optional ``coords`` with shape ``(B, H, 3)``.

The job of ``format_and_forecast`` is to convert those raw outputs into
clean, long-format DataFrames that downstream tools can inspect, plot,
and evaluate.

What the real function does
---------------------------
The utility takes raw prediction dictionaries, optional truth arrays,
optional coordinates, and temporal configuration, then returns:

- ``df_eval``:
  predictions + actuals for the evaluation side,
- ``df_future``:
  predictions for the future horizon.

It supports quantile mode, point mode, optional spatial coordinates,
evaluation export control, cumulative or absolute-cumulative
transformations, optional calibration, and optional calls to
``evaluate_forecast``.

The resulting DataFrames carry a standard structure centered on
``sample_idx``, ``forecast_step``, ``coord_t``, optional
``coord_x``/``coord_y``, prediction columns such as
``subsidence_q10`` / ``subsidence_q50`` / ``subsidence_q90`` or
``subsidence_pred``, and an actual column for the evaluation side.
That is exactly why the later forecasting pages can read these outputs
directly. 

What this lesson teaches
------------------------
We will:

1. build synthetic spatial forecast data that mimics a real city grid,
2. create raw ``y_pred`` and ``y_true`` arrays in the same shape a
   probabilistic model would return,
3. call the real ``format_and_forecast`` utility,
4. inspect the resulting evaluation and future forecast tables,
5. show how ``eval_export`` changes what is written on the evaluation side,
6. show how ``value_mode`` changes rate forecasts into cumulative and
   absolute cumulative forecasts,
7. finish by sending the resulting tables into a real plotting helper.

The example is fully synthetic so the page is executable during the
documentation build.
"""

# %%
# Imports
# -------
# We use the real utility that the forecasting workflow relies on.

import numpy as np
import pandas as pd

from geoprior.plot import plot_eval_future
from geoprior.utils.forecast_utils import (
    evaluate_forecast,
    format_and_forecast,
)

# %%
# Step 1 - Build a compact synthetic spatial layout
# -------------------------------------------------
# We mimic a small city-like spatial grid.
#
# Each spatial location becomes one "sample" from the perspective of
# the forecast formatting utility. The model will later return H-step
# forecasts for every one of these samples.

rng = np.random.default_rng(42)

nx = 7
ny = 5
xv = np.linspace(0.0, 12_000.0, nx)
yv = np.linspace(0.0, 8_000.0, ny)

X, Y = np.meshgrid(xv, yv)
x_flat = X.ravel()
y_flat = Y.ravel()

B = x_flat.size          # number of spatial samples
H = 3                    # forecast horizon
Q = 3                    # q10, q50, q90
O = 1                    # one subsidence target

quantiles = [0.1, 0.5, 0.9]
future_years = np.array([2023, 2024, 2025], dtype=int)
train_end_year = 2022

print("Number of samples (B):", B)
print("Forecast horizon (H):", H)
print("Quantiles (Q):", quantiles)

# %%
# Step 2 - Build synthetic raw coordinates in model layout
# --------------------------------------------------------
# ``format_and_forecast`` expects optional coordinates shaped like
# ``(B, H, 3)`` with columns typically interpreted as:
#
# - t
# - x
# - y
#
# Importantly, the function uses x/y for spatial coordinates, while
# the time column is overwritten by the temporal configuration when
# we provide ``train_end_time`` and ``future_time_grid``. So the time
# values inside ``coords`` are not the main source of truth here. 

coords = np.zeros((B, H, 3), dtype=float)

for i in range(B):
    for h in range(H):
        coords[i, h, 0] = h + 1         # placeholder t-like slot
        coords[i, h, 1] = x_flat[i]     # x
        coords[i, h, 2] = y_flat[i]     # y

print("coords shape:", coords.shape)

# %%
# Step 3 - Build synthetic raw model outputs
# ------------------------------------------
# We now create arrays that mimic a probabilistic subsidence model.
#
# Target contract:
#
# - ``y_pred["subs_pred"]`` in quantile mode:
#   shape ``(B, H, Q, O)``
# - ``y_true["subsidence"]``:
#   shape ``(B, H, O)``
#
# The synthetic field is designed so that:
#
# - one main hotspot exists,
# - a secondary ridge is present,
# - the response strengthens through time,
# - the predictive interval widens with horizon.

xn = (x_flat - x_flat.min()) / (x_flat.max() - x_flat.min())
yn = (y_flat - y_flat.min()) / (y_flat.max() - y_flat.min())

hotspot = np.exp(
    -(
        ((xn - 0.72) ** 2) / 0.022
        + ((yn - 0.34) ** 2) / 0.030
    )
)
ridge = 0.55 * np.exp(
    -(
        ((xn - 0.26) ** 2) / 0.030
        + ((yn - 0.70) ** 2) / 0.060
    )
)
gradient = 0.48 * xn + 0.22 * (1.0 - yn)

# raw true values: (B, H, O)
y_true_subs = np.zeros((B, H, O), dtype=float)

# raw predicted quantiles: (B, H, Q, O)
y_pred_subs = np.zeros((B, H, Q, O), dtype=float)

for h in range(H):
    year_scale = [1.00, 1.18, 1.42][h]

    median = (
        1.9
        + 1.4 * gradient
        + 2.0 * hotspot
        + 0.95 * ridge
    ) * year_scale

    width = (
        0.28
        + 0.07 * xn
        + 0.05 * hotspot
        + 0.05 * (h + 1)
    )

    q10 = median - width
    q50 = median
    q90 = median + width

    # truth stays close to q50 but is not identical
    actual = median + rng.normal(0.0, 0.14, size=B)

    y_true_subs[:, h, 0] = actual
    y_pred_subs[:, h, 0, 0] = q10
    y_pred_subs[:, h, 1, 0] = q50
    y_pred_subs[:, h, 2, 0] = q90

y_pred = {"subs_pred": y_pred_subs}
y_true = {"subsidence": y_true_subs}

print("y_pred['subs_pred'] shape:", y_pred["subs_pred"].shape)
print("y_true['subsidence'] shape:", y_true["subsidence"].shape)

# %%
# Step 4 - Call the real ``format_and_forecast`` utility
# ------------------------------------------------------
# This is the main lesson step.
#
# Important settings:
#
# - ``quantiles=[0.1, 0.5, 0.9]``:
#   quantile mode, so the function emits q10/q50/q90 columns;
# - ``train_end_time=2022``:
#   anchors the evaluation-side time reconstruction;
# - ``future_time_grid=[2023, 2024, 2025]``:
#   explicit future horizon labels;
# - ``eval_export="all"``:
#   export all evaluation horizons rather than only the final one;
# - ``value_mode="rate"``:
#   treat each step as an incremental forecast, not yet cumulative.
#
# In the real implementation, ``eval_forecast_step`` defaults to the
# last horizon if not provided, and ``eval_export="all"`` promotes the
# multi-horizon evaluation DataFrame when it is available. 

df_eval, df_future = format_and_forecast(
    y_pred=y_pred,
    y_true=y_true,
    coords=coords,
    quantiles=quantiles,
    target_name="subsidence",
    train_end_time=train_end_year,
    future_time_grid=future_years,
    eval_export="all",
    value_mode="rate",
    city_name="SyntheticCity",
    model_name="GeoPrior-demo",
    dataset_name="synthetic-spatial-grid",
    verbose=0,
)

print("df_eval shape:", df_eval.shape)
print("df_future shape:", df_future.shape)

print("")
print("df_eval head")
print(df_eval.head(10).to_string(index=False))

print("")
print("df_future head")
print(df_future.head(10).to_string(index=False))

# %%
# Step 5 - Understand the two returned DataFrames
# -----------------------------------------------
# ``df_eval`` and ``df_future`` are the central products of this
# function.
#
# On the evaluation side we expect:
#
# - ``sample_idx``
# - ``forecast_step``
# - ``coord_t``
# - ``coord_x``, ``coord_y``
# - quantile columns
# - ``subsidence_actual``
#
# On the future side we expect the same prediction structure, but
# no actual column because those years are beyond the observed data
# window. The implementation explicitly drops the actual column from
# the future DataFrame. 

print("")
print("df_eval columns")
print(df_eval.columns.tolist())

print("")
print("df_future columns")
print(df_future.columns.tolist())

# %%
# A compact year / horizon summary helps explain the table layout.

eval_summary = (
    df_eval.groupby(["coord_t", "forecast_step"])
    .agg(
        n_rows=("sample_idx", "size"),
        mean_q50=("subsidence_q50", "mean"),
        mean_actual=("subsidence_actual", "mean"),
    )
    .reset_index()
)

future_summary = (
    df_future.groupby(["coord_t", "forecast_step"])
    .agg(
        n_rows=("sample_idx", "size"),
        mean_q10=("subsidence_q10", "mean"),
        mean_q50=("subsidence_q50", "mean"),
        mean_q90=("subsidence_q90", "mean"),
    )
    .reset_index()
)

print("")
print("Evaluation summary")
print(eval_summary.to_string(index=False))

print("")
print("Future summary")
print(future_summary.to_string(index=False))

# %%
# What these summaries mean
# -------------------------
# The evaluation DataFrame here is multi-horizon because we used
# ``eval_export="all"``.
#
# That means the evaluation side now contains one row per:
#
# - sample
# - forecast step
#
# rather than only the final evaluation step.
#
# This is one of the most important things to understand about the
# utility: the returned table can represent either a single-step
# evaluation view or the full multi-horizon evaluation view depending
# on the export control. 

# %%
# Step 6 - Compare ``eval_export="all"`` versus ``eval_export="last"``
# --------------------------------------------------------------------
# The easiest way to understand ``eval_export`` is to run the function
# twice and compare the outputs.

df_eval_last, df_future_last = format_and_forecast(
    y_pred=y_pred,
    y_true=y_true,
    coords=coords,
    quantiles=quantiles,
    target_name="subsidence",
    train_end_time=train_end_year,
    future_time_grid=future_years,
    eval_export="last",
    value_mode="rate",
    verbose=0,
)

print("Rows with eval_export='all': ", len(df_eval))
print("Rows with eval_export='last':", len(df_eval_last))

print("")
print("Single-step evaluation head")
print(df_eval_last.head(8).to_string(index=False))

# %%
# Interpretation
# --------------
# Use:
#
# - ``eval_export="all"`` when you want full multi-horizon evaluation
#   analysis or horizon-wise metrics;
# - ``eval_export="last"`` when you want the compact, single-step
#   evaluation table that mirrors the original backward-compatible
#   behavior. 

# %%
# Step 7 - Demonstrate cumulative and absolute cumulative modes
# -------------------------------------------------------------
# Another important lesson in this utility is that it can reinterpret
# the horizon values as:
#
# - rates,
# - relative cumulative values,
# - absolute cumulative values.
#
# In the real implementation:
#
# - ``"cumulative"`` applies a cumulative sum by sample over
#   ``forecast_step``,
# - ``"absolute_cumulative"`` does the same and then adds a baseline
#   value or mapping. 

baseline_map = {
    int(i): 12.0 + 0.5 * np.sin(i / 4.0)
    for i in range(B)
}

df_eval_abs, df_future_abs = format_and_forecast(
    y_pred=y_pred,
    y_true=y_true,
    coords=coords,
    quantiles=quantiles,
    target_name="subsidence",
    train_end_time=train_end_year,
    future_time_grid=future_years,
    eval_export="all",
    value_mode="absolute_cumulative",
    absolute_baseline=baseline_map,
    verbose=0,
)

cum_summary = (
    df_future_abs.groupby("coord_t")
    .agg(
        mean_q50=("subsidence_q50", "mean"),
        mean_q90=("subsidence_q90", "mean"),
    )
    .reset_index()
)

print("")
print("Absolute cumulative future summary")
print(cum_summary.to_string(index=False))

# %%
# Why this matters
# ----------------
# The same raw model output can support different downstream stories.
#
# - In **rate** mode, each step is interpreted as a per-step increment.
# - In **cumulative** mode, the horizon becomes an accumulated path.
# - In **absolute cumulative** mode, the path is shifted to an absolute
#   reference level.
#
# That distinction is crucial in geohazard forecasting because users may
# want either:
#
# - incremental annual change,
# - relative cumulative change since the start of the horizon,
# - or absolute cumulative state referenced to the end of training.

# %%
# Step 8 - Run automatic forecast evaluation
# ------------------------------------------
# ``format_and_forecast`` can optionally trigger the evaluation helper,
# but for a lesson page it is often clearer to call
# ``evaluate_forecast`` explicitly afterward.
#
# The evaluator is designed around the same DataFrame contract and
# computes deterministic metrics plus coverage and sharpness in quantile
# mode. It can also compute per-horizon metrics. 

metrics = evaluate_forecast(
    df_eval,
    target_name="subsidence",
    per_horizon=True,
    quantile_interval=(0.1, 0.9),
    verbose=0,
)

print("")
print("Evaluation metrics")
print(metrics)

# %%
# How to read these metrics
# -------------------------
# The main things to inspect are:
#
# - ``overall_mae`` / ``overall_mse`` / ``overall_r2``:
#   central forecast accuracy,
# - ``coverage80``:
#   how often the actual value falls inside the q10-q90 interval,
# - ``sharpness80``:
#   average interval width,
# - ``per_horizon_*``:
#   whether accuracy degrades across the horizon.
#
# This is why ``format_and_forecast`` is such a key bridge utility:
# once the tables exist, both plotting and evaluation become simple.

# %%
# Step 9 - Send the formatted tables into a real plotting function
# ----------------------------------------------------------------
# To close the lesson, we connect this utility page to the rest of the
# forecasting gallery.
#
# The tables returned here can be passed directly into
# ``plot_eval_future`` for a holdout-versus-future comparison.

plot_eval_future(
    df_eval=df_eval,
    df_future=df_future,
    target_name="subsidence",
    quantiles=quantiles,
    spatial_cols=("coord_x", "coord_y"),
    time_col="coord_t",
    eval_years=[2020, 2021, 2022],
    future_years=[2023, 2024, 2025],
    eval_view_quantiles=[0.5],
    future_view_quantiles=[0.1, 0.5, 0.9],
    cbar="uniform",
    axis_off=False,
    show_grid=True,
    grid_props={"linestyle": ":", "alpha": 0.45},
    figsize_eval=(10.0, 5.0),
    figsize_future=(10.5, 8.0),
    show=True,
    verbose=0,
)

# %%
# Final lesson takeaway
# ---------------------
# ``format_and_forecast`` is the utility that turns raw model arrays
# into the canonical long-format forecast tables used across the
# forecasting workflow.
#
# Once you understand this function, the rest of the forecasting gallery
# becomes much easier to read because you know exactly where the tables
# come from and why they have the columns they do.