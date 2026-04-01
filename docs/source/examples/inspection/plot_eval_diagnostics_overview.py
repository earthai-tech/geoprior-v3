# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Inspect compact evaluation diagnostics before trusting forecast quality
===========================================================================

This lesson explains how to inspect the compact
``eval_diagnostics.json`` artifact produced by the evaluation workflow.

This file is intentionally smaller than the richer interpretable
Stage-2 evaluation payload. That is exactly why it is useful: it gives
users a fast way to review forecast quality at three complementary
levels:

- the overall block ``__overall__``,
- the per-year evaluation blocks,
- the per-horizon point-metric maps.

The goal of this page is therefore not only to call helper functions.
It is to teach how to read the artifact step by step, understand what
its small set of metrics is saying, and decide whether the forecast
looks strong enough for reporting, calibration, or further diagnosis.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd

from geoprior.utils.inspect import (
    eval_overall_frame,
    eval_per_horizon_frame,
    eval_years_frame,
    generate_eval_diagnostics,
    inspect_eval_diagnostics,
    load_eval_diagnostics,
    plot_eval_boolean_summary,
    plot_eval_overall_metrics,
    plot_eval_per_horizon_metrics,
    plot_eval_year_metric_trend,
    summarize_eval_diagnostics,
)

pd.set_option("display.max_columns", 24)
pd.set_option("display.width", 108)
pd.set_option("display.float_format", lambda v: f"{v:0.6f}")

EVAL_DIAG_PALETTE = {
    "overall": ["#1D3557", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"],
    "mae_trend": "#5A189A",
    "pss_trend": "#C1121F",
    "rmse_h": ["#277DA1", "#577590", "#F9844A"],
    "r2_h": ["#43AA8B", "#90BE6D", "#4D908E"],
    "checks": "#6D597A",
}


# %%
# Why this artifact matters
# -------------------------
#
# The compact evaluation-diagnostics artifact is one of the easiest
# files to inspect after running forecast evaluation.
#
# It is smaller than the interpretable physics-evaluation JSON, but it
# still preserves the information most users need for a first-pass
# decision:
#
# 1. how forecast quality behaves year by year,
# 2. how the overall aggregated forecast behaves,
# 3. whether later horizons degrade sharply,
# 4. whether interval diagnostics stay believable,
# 5. whether the prediction-stability score (PSS) stays acceptable.
#
# In other words, this artifact is a quick decision file. It helps the
# user answer a practical question before spending more time downstream:
#
# *Does this forecast already look good enough to trust, or does it need
# calibration, model comparison, or deeper physics inspection first?*


# %%
# Create a realistic demo diagnostics artifact
# --------------------------------------------
#
# For documentation, we usually want a stable evaluation artifact that
# behaves like a real saved file but does not require re-running a full
# model evaluation.
#
# In this example we intentionally encode a common forecasting pattern:
#
# - early horizons perform well,
# - later horizons degrade,
# - coverage moves from slightly conservative toward slightly under the
#   nominal 80% level,
# - and PSS becomes worse as the forecast reaches further into the
#   future.
#
# That makes the lesson easier to interpret because the metrics tell a
# coherent story instead of looking randomly generated.

out_dir = Path(tempfile.mkdtemp(prefix="gp_eval_diag_"))
diag_path = out_dir / "nansha_eval_diagnostics.json"

generate_eval_diagnostics(
    output_path=diag_path,
    years=[2023, 2024, 2025],
    per_horizon_mae=[3.84, 8.96, 17.42],
    per_horizon_mse=[26.10, 166.40, 742.25],
    per_horizon_rmse=[5.109795, 12.899612, 27.244266],
    per_horizon_r2=[0.912, 0.887, 0.841],
    coverage80=[0.888, 0.821, 0.768],
    sharpness80=[21.84, 28.46, 47.30],
    pss=[34.20, 52.75, 78.10],
)

print("Written evaluation-diagnostics file")
print(f" - {diag_path}")
# %%
# Load the artifact with the real reader
# --------------------------------------
#
# Even in a gallery lesson, it is useful to follow the same loading
# path a real user would take with a saved ``*_eval_diagnostics``
# JSON under ``results/...``.

diag_record = load_eval_diagnostics(diag_path)

print("\nArtifact header")
pprint(
    {
        "kind": diag_record.kind,
        "stage": diag_record.stage,
        "city": diag_record.city,
        "model": diag_record.model,
        "path": str(diag_record.path),
    }
)

# %%
# Start with the compact semantic summary
# ---------------------------------------
#
# A good inspection habit is to begin with the semantic summary.
# It answers the first structural questions before you look at any
# plots:
#
# - does the artifact contain the required ``__overall__`` block?
# - are year blocks present?
# - are the per-horizon maps complete?
# - does each year expose interval and stability diagnostics?
#
# When those checks fail, it is usually not worth over-interpreting
# the numbers yet. Structural completeness comes first.

summary = summarize_eval_diagnostics(diag_record)

print("\nCompact summary")
print(json.dumps(summary, indent=2))

# %%
# Read the three core table views
# -------------------------------
#
# The compact diagnostics file becomes much easier to read when we
# separate it into three tidy views:
#
# 1. one row per evaluated year,
# 2. one-row overall aggregate,
# 3. one row per horizon from the aggregated block.
#
# These three levels answer different questions:
#
# - the year table shows *temporal drift*,
# - the overall table shows the *headline quality*,
# - the per-horizon table shows *forecast-range degradation*.

years_frame = eval_years_frame(diag_record)
overall_frame = eval_overall_frame(diag_record)
per_h_frame = eval_per_horizon_frame(diag_record)

print("\nPer-year diagnostics")
print(years_frame)

print("\nOverall diagnostics")
print(overall_frame)

print("\nPer-horizon diagnostics")
print(per_h_frame)

# %%
# How to interpret the per-year view
# ----------------------------------
#
# The year table is useful when evaluation is reported separately for
# each forecast year. A robust reading order is:
#
# 1. ``overall_mae`` or ``overall_rmse`` for absolute error,
# 2. ``overall_r2`` for explained variance,
# 3. ``coverage80`` and ``sharpness80`` for interval usefulness,
# 4. ``pss`` for temporal stability.
#
# In this demo, the later years look harder:
#
# - MAE and RMSE rise,
# - R2 declines,
# - coverage drifts downward,
# - sharpness widens,
# - and PSS increases.
#
# That combination is a common forecasting warning sign: the model is
# still producing forecasts, but confidence quality and stability are
# deteriorating with forecast range.

year_view = years_frame.copy()
if not year_view.empty:
    first_mae = float(year_view["overall_mae"].iloc[0])
    first_pss = float(year_view["pss"].iloc[0])
    year_view["mae_vs_first"] = (
        year_view["overall_mae"] - first_mae
    )
    year_view["pss_vs_first"] = year_view["pss"] - first_pss
    year_view["coverage80_error"] = (
        year_view["coverage80"] - 0.80
    ).abs()

print("\nPer-year interpretation table")
print(year_view)

# %%
# How to interpret the overall block
# ----------------------------------
#
# The overall block is the headline view many users look at first.
# That is useful, but it can also be misleading if read alone.
#
# A good interpretation pattern is:
#
# - lower ``overall_mae`` and ``overall_rmse`` are better,
# - higher ``overall_r2`` is better,
# - ``coverage80`` should usually sit reasonably close to 0.80,
# - lower ``sharpness80`` means narrower intervals,
# - lower ``pss`` means more temporally stable predictions.
#
# Notice the tension here: a model can have decent RMSE yet still
# show suspicious interval behavior or poor temporal stability.
# That is exactly why this compact artifact includes all of these
# metrics together.

overall_view = overall_frame.copy()
if not overall_view.empty:
    overall_view["coverage80_error"] = (
        overall_view["coverage80"] - 0.80
    ).abs()
    overall_view["interval_comment"] = [
        (
            "close to nominal 80%"
            if abs(float(overall_view.loc[0, "coverage80"]) - 0.80)
            <= 0.03
            else "coverage drift worth reviewing"
        )
    ]

print("\nOverall interpretation table")
print(overall_view)

# %%
# How to interpret the per-horizon block
# --------------------------------------
#
# The per-horizon table is often the most actionable part of the
# whole artifact because it shows whether the forecast degrades in a
# smooth and believable way.
#
# A common healthy pattern is:
#
# - MAE/RMSE gradually increase,
# - R2 gradually decreases,
# - but no horizon collapses abruptly.
#
# A common warning pattern is a very sharp jump at the last horizon.
# That often suggests the model is extrapolating beyond what the
# training information can support well.

per_h_view = per_h_frame.copy()
if not per_h_view.empty:
    per_h_view["rmse_step_change"] = per_h_view["rmse"].diff()
    per_h_view["r2_step_change"] = per_h_view["r2"].diff()
    per_h_view["rmse_growth_ratio"] = (
        per_h_view["rmse"] / float(per_h_view["rmse"].iloc[0])
    )

print("\nPer-horizon interpretation table")
print(per_h_view)

# %%
# Use the all-in-one inspector when you want the main bundle together
# -------------------------------------------------------------------
#
# ``inspect_eval_diagnostics(...)`` is convenient when you want the
# semantic summary, tidy frames, and saved figure paths in one call.
# That makes it useful for gallery generation, report pipelines, or
# future CLI helpers that inspect folders of evaluation artifacts.

bundle = inspect_eval_diagnostics(diag_record)

print("\nInspector bundle keys")
print(sorted(bundle))

# %%
# Plot the main compact views
# ---------------------------
#
# A compact first-pass review usually benefits from four plots:
#
# 1. the overall aggregated metrics,
# 2. the year trend for absolute error,
# 3. the year trend for prediction stability,
# 4. the per-horizon RMSE curve.
#
# These four views work well together because they combine one
# headline panel with three structural trend panels.

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12.4, 8.8),
    constrained_layout=True,
)

plot_eval_overall_metrics(
    diag_record,
    ax=axes[0, 0],
    title="Overall compact evaluation metrics",
    color=EVAL_DIAG_PALETTE["overall"],
    edgecolor="#1F2937",
    linewidth=0.9,
    alpha=0.92,
)
plot_eval_year_metric_trend(
    diag_record,
    metric="overall_mae",
    ax=axes[0, 1],
    title="Year trend: overall MAE",
    color=EVAL_DIAG_PALETTE["mae_trend"],
    marker="o",
    linewidth=2.4,
    markersize=7,
)
plot_eval_year_metric_trend(
    diag_record,
    metric="pss",
    ax=axes[1, 0],
    title="Year trend: PSS",
    color=EVAL_DIAG_PALETTE["pss_trend"],
    marker="D",
    linewidth=2.4,
    markersize=6.5,
)
plot_eval_per_horizon_metrics(
    diag_record,
    metric="rmse",
    ax=axes[1, 1],
    title="Per-horizon RMSE",
    color=EVAL_DIAG_PALETTE["rmse_h"],
    edgecolor="#1F2937",
    linewidth=0.9,
    alpha=0.92,
)

# %%
# How to read these plots
# -----------------------
#
# These plots are meant to be read together, not in isolation.
#
# - The overall bar plot tells us the headline quality at a glance.
# - The MAE year trend reveals whether forecast error is worsening as
#   the target year moves away from the observed window.
# - The PSS trend reveals whether the prediction path is becoming less
#   stable over time.
# - The per-horizon RMSE bar plot shows whether deterioration happens
#   gradually or whether one horizon becomes disproportionately hard.
#
# In this demo, the answer is fairly coherent: later years and later
# horizons are clearly harder, and the degradation is not just an
# interval issue. The point forecast itself is also weakening.

# %%
# Inspect another horizon-specific view
# -------------------------------------
#
# RMSE is not the whole story. The same horizon may have rising error
# but still preserve some explained variance. Looking at R2 gives a
# second perspective on horizon degradation.

fig, ax = plt.subplots(
    figsize=(8.0, 4.6),
    constrained_layout=True,
)
plot_eval_per_horizon_metrics(
    diag_record,
    metric="r2",
    ax=ax,
    title="Per-horizon R²",
    color=EVAL_DIAG_PALETTE["r2_h"],
    edgecolor="#1F2937",
    linewidth=0.9,
    alpha=0.92,
)

# %%
# Plot the structural checks separately
# -------------------------------------
#
# The boolean summary converts the most important structural checks
# into a fast pass/fail view. This is especially useful when many
# evaluation files are being compared and the user wants to avoid
# manually checking whether every block is present.

fig, ax = plt.subplots(
    figsize=(8.2, 4.6),
    constrained_layout=True,
)
plot_eval_boolean_summary(
    diag_record,
    ax=ax,
    title="Evaluation diagnostics decision checks",
    color=EVAL_DIAG_PALETTE["checks"],
    edgecolor="#1F2937",
    linewidth=0.8,
    alpha=0.9,
)

# %%
# Save a full inspection bundle
# -----------------------------
#
# When you want more than live figures, the bundle helper can also
# save the core inspection plots to disk. This is useful for gallery
# generation and later report automation.

bundle_dir = out_dir / "inspection_bundle"
saved = inspect_eval_diagnostics(
    diag_record,
    output_dir=bundle_dir,
    stem="lesson_eval_diag",
    save_figures=True,
)

print("\nSaved inspection figures")
for name, path in saved["figure_paths"].items():
    print(f" - {name}: {path}")

# %%
# A practical reading rule
# ------------------------
#
# For this artifact family, a compact decision rule can be:
#
# - the artifact contains the overall block,
# - year blocks are present,
# - per-horizon maps are complete,
# - overall coverage80 is not too far from the nominal 0.80 level,
# - later horizons degrade, but not in a catastrophic jump,
# - and PSS stays within a range you consider operationally usable.
#
# The exact thresholds depend on your project. But even without fixed
# thresholds, this artifact already helps you separate three common
# cases:
#
# 1. forecast looks healthy,
# 2. forecast is usable but should be calibrated or compared,
# 3. forecast quality is weak enough that retraining or redesign is
#    probably needed.

checks = summary["checks"]
coverage_ok = abs(float(summary["overall"]["coverage80"] or 0.0) - 0.80) <= 0.05
horizon_jump_ok = True
if not per_h_view.empty and per_h_view["rmse_step_change"].notna().any():
    max_jump = float(per_h_view["rmse_step_change"].dropna().max())
    horizon_jump_ok = max_jump < 20.0

pss_ok = float(summary["overall"]["pss"] or 0.0) < 80.0

ready = all(
    bool(checks.get(name, False))
    for name in [
        "has_overall_block",
        "has_year_blocks",
        "overall_has_core_metrics",
        "overall_has_per_horizon_mae",
        "overall_has_per_horizon_rmse",
        "overall_has_per_horizon_r2",
    ]
)
ready = ready and coverage_ok and horizon_jump_ok and pss_ok

print("\nDecision note")
if ready:
    print(
        "This demo evaluation artifact looks coherent enough for "
        "reporting or for use as the input to a deeper comparison "
        "and calibration review."
    )
else:
    print(
        "This evaluation artifact suggests that the forecast should "
        "be reviewed more carefully before being trusted downstream."
    )
