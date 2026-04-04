# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Inspect ablation records before choosing a configuration
========================================================

This lesson explains how to inspect the JSONL artifact
``ablation_record.jsonl``.

Why this file matters
---------------------
Ablation work is not only about finding the *best* score. It is about
understanding *why* one configuration behaves differently from another.
In practice, an ablation record helps answer questions such as:

- Which variant really reduced RMSE or MAE?
- Was the apparent gain only at short horizons?
- Did a physics-weight change improve fit while harming stability?
- Are two variants actually comparable in their lambda weights?
- Is a strong score still believable when epsilon diagnostics degrade?

This page is therefore written as a **reading lesson**, not only as an
API demo. We will use the ablation inspector to move from raw JSONL
records to interpretable tables, comparison plots, and a final decision
rule.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd

from geoprior.utils.inspect import (
    ablation_config_frame,
    ablation_metrics_frame,
    ablation_per_horizon_frame,
    ablation_record_flags_frame,
    ablation_record_runs_frame,
    generate_ablation_record,
    inspect_ablation_record,
    load_ablation_record,
    plot_ablation_boolean_summary,
    plot_ablation_lambda_weights,
    plot_ablation_metric_by_variant,
    plot_ablation_per_horizon_metric,
    plot_ablation_run_counts,
    plot_ablation_top_variants,
    summarize_ablation_record,
)

pd.set_option("display.max_columns", 32)
pd.set_option("display.width", 110)
pd.set_option("display.float_format", lambda v: f"{v:0.6f}")

ABLATION_PALETTE = {
    "counts": ["#4C78A8", "#72B7B2", "#54A24B", "#F58518"],
    "rmse": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"],
    "lambdas": ["#355070", "#6D597A", "#B56576", "#E56B6F", "#EAAC8B", "#84A59D", "#52796F"],
    "horizon_lines": ["#3A86FF", "#8338EC", "#FF006E", "#FB5607"],
    "top": ["#118AB2", "#06D6A0", "#FFD166", "#EF476F"],
    "checks": "#5C677D",
}


# %%
# JSONL is a little different from the other artifacts
# ----------------------------------------------------
#
# Most inspection files in this gallery are single JSON objects.
# ``ablation_record.jsonl`` is different: it is a newline-delimited log,
# so each line is one ablation record.
#
# That difference matters conceptually:
#
# 1. we are usually comparing *variants*, not reading one run in
#    isolation,
# 2. there may be multiple seeds or repeated evaluations,
# 3. one metric alone is rarely enough to choose a variant.
#
# In other words, the goal is to compare *patterns* across records.


# %%
# Create a realistic demo ablation file
# -------------------------------------
#
# For a gallery lesson, we want a stable and readable set of ablation
# rows without rerunning the full experiment pipeline.
#
# The helper already creates a realistic family of records, but here we
# push the variants a little further apart so the lesson becomes easier
# to interpret:
#
# - ``baseline`` stays as the reference,
# - ``gw_heavier`` gets a larger groundwater weight,
# - ``smoother`` makes the smoothness term more visible,
# - ``bounds_stronger`` increases the bounds penalty.
#
# This is a good teaching setup because it gives us meaningful
# differences in both scalar and horizon-wise behavior.

workdir = Path(tempfile.mkdtemp(prefix="gp_ablation_"))
out_dir = workdir
ablation_path = out_dir / "nansha_ablation_record.jsonl"

generate_ablation_record(
    ablation_path,
    overrides=[
        {
            "ablation": "baseline",
            "seed": 11,
            "lambda_gw": 0.10,
            "lambda_smooth": 0.010,
            "lambda_bounds": 0.05,
            "mae": 0.01190,
            "rmse": 0.01776,
            "r2": 0.87970,
            "epsilon_prior": 8.78,
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
        },
        {
            "ablation": "gw_heavier",
            "seed": 12,
            "lambda_gw": 0.18,
            "lambda_smooth": 0.010,
            "lambda_bounds": 0.05,
            "mae": 0.01135,
            "rmse": 0.01710,
            "r2": 0.88450,
            "epsilon_prior": 9.35,
            "per_horizon_mae": {
                "H1": 0.00510,
                "H2": 0.01166,
                "H3": 0.01729,
            },
            "per_horizon_r2": {
                "H1": 0.8960,
                "H2": 0.8865,
                "H3": 0.8758,
            },
        },
        {
            "ablation": "smoother",
            "seed": 13,
            "lambda_gw": 0.10,
            "lambda_smooth": 0.050,
            "lambda_bounds": 0.05,
            "mae": 0.01225,
            "rmse": 0.01802,
            "r2": 0.87720,
            "epsilon_prior": 7.95,
            "per_horizon_mae": {
                "H1": 0.00562,
                "H2": 0.01256,
                "H3": 0.01857,
            },
            "per_horizon_r2": {
                "H1": 0.8910,
                "H2": 0.8790,
                "H3": 0.8680,
            },
        },
        {
            "ablation": "bounds_stronger",
            "seed": 14,
            "lambda_gw": 0.10,
            "lambda_smooth": 0.010,
            "lambda_bounds": 0.12,
            "mae": 0.01170,
            "rmse": 0.01742,
            "r2": 0.88210,
            "epsilon_prior": 8.12,
            "per_horizon_mae": {
                "H1": 0.00518,
                "H2": 0.01202,
                "H3": 0.01796,
            },
            "per_horizon_r2": {
                "H1": 0.8948,
                "H2": 0.8832,
                "H3": 0.8731,
            },
        },
    ],
)

print("Written ablation file")
print(f" - {ablation_path}")

# %%
# Look at the raw JSONL form first
# --------------------------------
#
# Before using the inspection helpers, it is useful to remember what
# this artifact really looks like on disk. Each line is its own JSON
# record. This is one reason why the ablation inspector uses tables so
# heavily: raw JSONL becomes hard to compare once the file grows.

print("\nFirst two raw lines")
with ablation_path.open("r", encoding="utf-8") as stream:
    for idx, line in enumerate(stream, start=1):
        print(line.strip())
        if idx >= 2:
            break

# %%
# Load the artifact through the real reader
# -----------------------------------------
#
# The reader returns a plain list of normalized records. That keeps
# the JSONL structure faithful while giving us a stable starting point
# for tables and plots.

records = load_ablation_record(ablation_path)

print("\nHow many records were loaded?")
print(len(records))

print("\nFirst normalized record")
pprint(records[0])

# %%
# Start with the semantic summary
# -------------------------------
#
# A useful first question is not "Which plot should I draw?" but
# rather:
#
# *Does this file look complete enough to support a fair comparison?*
#
# The semantic summary answers exactly that. It tells us whether the
# file contains records, core metrics, horizon-wise metrics, units,
# and key configuration knobs.

summary = summarize_ablation_record(records)

print("\nCompact summary")
print(json.dumps(summary, indent=2))

# %%
# Build the tidy comparison tables
# --------------------------------
#
# JSONL is best inspected after converting it into tidy tables.
# We will use four different views:
#
# 1. a run-level table,
# 2. a long-form scalar-metrics table,
# 3. a per-horizon table,
# 4. a compact config/weights table.
#
# Each table answers a different question, so it is worth keeping
# them conceptually separate.

runs = ablation_record_runs_frame(records)
metrics = ablation_metrics_frame(records)
per_h = ablation_per_horizon_frame(records)
config = ablation_config_frame(records)
flags = ablation_record_flags_frame(records)

print("\nRun-level view")
print(runs)

print("\nScalar metric rows")
print(metrics.head(18))

print("\nPer-horizon rows")
print(per_h)

print("\nConfiguration / weight view")
print(config)

print("\nBoolean flags")
print(flags)

# %%
# Read the run-level table as a comparison checklist
# --------------------------------------------------
#
# The run-level table is the fastest way to check whether records are
# *comparable* at all.
#
# Things worth checking here:
#
# - Are all variants from the same city and model?
# - Did the PDE mode stay fixed, or are we accidentally mixing two
#   kinds of experiments?
# - Are units consistent across rows?
# - Did only one configuration knob change, or did several knobs move
#   together?
#
# A common ablation mistake is to compare variants that changed more
# than one meaningful thing. The table makes that easier to detect.

compare_cols = [
    col
    for col in [
        "variant",
        "city",
        "model",
        "pde_mode",
        "use_effective_h",
        "kappa_mode",
        "hd_factor",
        "time_units",
    ]
    if col in runs.columns
]
print("\nComparison checklist view")
print(runs.loc[:, compare_cols])

# %%
# Aggregate scalar metrics by variant
# -----------------------------------
#
# The long-form metric table becomes much easier to interpret once we
# aggregate by variant. For a deep inspection lesson, this is where we
# start asking ranking questions.
#
# Here we compute a compact mean metric view. In a real experiment,
# this step becomes even more useful when you have repeated seeds.

metric_pivot = (
    metrics.pivot_table(
        index="variant",
        columns="metric",
        values="value",
        aggfunc="mean",
    )
    .reset_index()
    .sort_values("rmse")
)

print("\nMean scalar metrics by variant")
print(metric_pivot)

# %%
# How to interpret scalar ranking
# -------------------------------
#
# A careful ablation reading usually follows this order:
#
# 1. lower ``rmse`` or ``mae`` is good,
# 2. higher ``r2`` is good,
# 3. coverage and sharpness should still look reasonable together,
# 4. epsilon diagnostics should not quietly become much worse.
#
# That last point matters a lot. A variant can improve fit while also
# pushing the physics consistency in a less trustworthy direction.
#
# In this demo, ``gw_heavier`` looks strongest on fit metrics, but we
# should still compare its epsilon level against the others before we
# celebrate it.

epsilon_view = metric_pivot.loc[
    :, [
        col
        for col in [
            "variant",
            "rmse",
            "r2",
            "coverage80",
            "sharpness80",
            "epsilon_prior",
            "epsilon_cons",
            "epsilon_gw",
        ]
        if col in metric_pivot.columns
    ]
]

print("\nFit metrics together with epsilon diagnostics")
print(epsilon_view)

# %%
# Inspect the lambda weights directly
# -----------------------------------
#
# One of the easiest ways to misread an ablation file is to focus only
# on the outcome metrics and forget the actual weights that produced
# them.
#
# The lambda view is therefore essential. It tells us whether the
# variants are changing:
#
# - the groundwater residual weight,
# - the smoothness weight,
# - the bounds penalty,
# - or a more complex combination.
#
# Good ablation reading connects *weight changes* to *metric changes*.

lambda_cols = ["variant"] + [
    col
    for col in [
        "lambda_cons",
        "lambda_gw",
        "lambda_prior",
        "lambda_smooth",
        "lambda_mv",
        "lambda_bounds",
        "lambda_q",
    ]
    if col in config.columns
]

print("\nLambda-weight comparison")
print(config.loc[:, lambda_cols])

# %%
# Inspect the horizon-wise behavior
# ---------------------------------
#
# Scalar averages can hide a very important phenomenon:
# a variant may help early horizons but degrade later ones.
#
# This is why the per-horizon table matters. We can read it as a
# *degradation curve*.
#
# In many forecasting tasks, a good variant should:
#
# - keep short-horizon error low,
# - avoid a sharp long-horizon blow-up,
# - and preserve a sensible ranking across horizons.

per_h_pivot = (
    per_h.pivot_table(
        index=["variant", "horizon"],
        columns="metric",
        values="value",
        aggfunc="mean",
    )
    .reset_index()
)

print("\nPer-horizon comparison table")
print(per_h_pivot)

# %%
# Plot the main comparison views
# ------------------------------
#
# A compact ablation review is usually easier when we look at four
# complementary views:
#
# 1. how many runs belong to each variant,
# 2. which variants rank best on RMSE,
# 3. how the lambda weights differ,
# 4. how MAE behaves across horizons.
#
# Notice that each plot answers a different decision question. That is
# much better than drawing many redundant score charts.

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13.0, 9.0),
    constrained_layout=True,
)

axes[1, 0].set_prop_cycle(color=ABLATION_PALETTE["lambdas"])
axes[1, 1].set_prop_cycle(color=ABLATION_PALETTE["horizon_lines"])

plot_ablation_run_counts(
    records,
    ax=axes[0, 0],
    title="Runs available per variant",
    color=ABLATION_PALETTE["counts"],
    edgecolor="#1F2937",
    linewidth=0.9,
    alpha=0.92,
)
plot_ablation_metric_by_variant(
    records,
    metric="rmse",
    ax=axes[0, 1],
    title="Mean RMSE by variant",
    color=ABLATION_PALETTE["rmse"],
    edgecolor="#1F2937",
    linewidth=0.9,
    alpha=0.92,
)
plot_ablation_lambda_weights(
    records,
    ax=axes[1, 0],
    title="Lambda weights behind each variant",
    edgecolor="white",
    linewidth=0.8,
    alpha=0.92,
    legend_kws={"ncol": 2, "frameon": False, "fontsize": 9},
)
plot_ablation_per_horizon_metric(
    records,
    metric="mae",
    ax=axes[1, 1],
    title="Per-horizon MAE by variant",
    linewidth=2.2,
    markersize=7,
    marker="o",
    alpha=0.95,
    legend_kws={"frameon": False, "fontsize": 9},
)

# %%
# How to read these plots
# -----------------------
#
# Here is a practical interpretation order:
#
# **Run count plot**
#   Check whether comparison is balanced. If one variant has many more
#   runs or seeds, its mean score may be more stable than the others.
#
# **RMSE plot**
#   This is the quick performance ranking. It is often the first chart
#   a reader notices, but it should never be read in isolation.
#
# **Lambda plot**
#   This explains *what actually changed*. Without this chart, a good
#   score might look mysterious or misleading.
#
# **Per-horizon MAE plot**
#   This shows where the gain happens. A variant that only improves
#   H1 but worsens H3 may not be the best operational choice.


# %%
# Plot the best-ranked variants explicitly
# ----------------------------------------
#
# The top-variants plot is useful when the file becomes longer and you
# want a quick shortlist. In a real workflow, this is often the chart
# that helps decide which configurations deserve a rerun, deeper
# diagnosis, or reporting.

fig, ax = plt.subplots(
    figsize=(8.6, 4.6),
    constrained_layout=True,
)
plot_ablation_top_variants(
    records,
    metric="rmse",
    top_n=4,
    ax=ax,
    title="Best variants by mean RMSE",
    color=ABLATION_PALETTE["top"],
    edgecolor="#1F2937",
    linewidth=0.9,
    alpha=0.94,
)

# %%
# Plot the structural checks separately
# -------------------------------------
#
# The boolean summary is a small but useful guardrail. It tells us
# whether the file carries the minimum information needed for a real
# comparison: records, metrics, horizon metrics, units, and config
# knobs.

fig, ax = plt.subplots(
    figsize=(8.0, 4.2),
    constrained_layout=True,
)
plot_ablation_boolean_summary(
    records,
    ax=ax,
    title="Ablation artifact structural checks",
    color=ABLATION_PALETTE["checks"],
    edgecolor="#1F2937",
    linewidth=0.8,
    alpha=0.9,
)

# %%
# Save the full inspection bundle
# -------------------------------
#
# The all-in-one inspector is useful when you want to keep the
# semantic summary, tidy frames, and saved figures together. This is a
# convenient pattern for reports, gallery generation, or later CLI
# helpers that may inspect a whole experiment folder.

bundle_dir = out_dir / "inspection_bundle"
bundle = inspect_ablation_record(
    records,
    output_dir=bundle_dir,
    stem="lesson_ablation",
    save_figures=True,
)

print("\nSaved inspection figures")
for name, path in bundle["figure_paths"].items():
    print(f" - {name}: {path}")

# %%
# A practical decision rule
# -------------------------
#
# A simple reading rule for ablation records can be:
#
# - choose variants with strong scalar fit metrics,
# - reject variants whose horizon-wise behavior degrades badly,
# - reject variants whose epsilon diagnostics become suspicious,
# - and always confirm the configuration knobs that changed.
#
# In this demo, the most plausible candidate is the one that combines
# a good RMSE ranking with stable horizon behavior and no obvious
# structural warning.

best = summary.get("best_by_rmse") or {}
best_variant = best.get("variant")

print("\nDecision note")
if best_variant:
    print(
        "A good next candidate for deeper review is: "
        f"{best_variant!r}. But the final choice should still "
        "consider horizon-wise behavior and epsilon diagnostics, "
        "not only the scalar RMSE ranking."
    )
else:
    print(
        "No clear best variant could be identified from the "
        "available ablation records."
    )
