# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Inspect a Stage-1 audit before Stage-2
========================================

This lesson teaches how to *read* a Stage-1 audit artifact,
not only how to call plotting helpers.

Why this file deserves attention
--------------------------------
The Stage-1 audit is one of the earliest workflow artifacts that
can prevent downstream confusion. Before training starts, it helps
answer questions such as:

- Were coordinates normalized in the way we expected?
- Did Stage-1 keep enough coordinate information for the later
  chain-rule calculations used by physics-aware components?
- Which feature groups are scaled by the main scaler, already in
  SI-ready form, or still mixed / unscaled?
- Do the physics-side variables and the learning targets look
  numerically plausible?

The point of this page is therefore pedagogical:
we inspect the file section by section, explain what each view means,
and end with a simple decision rule for whether the preprocessing
handshake looks healthy enough to move on to the Stage-2 handshake.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd

from geoprior.utils.inspect import (
    generate_stage1_audit,
    inspect_stage1_audit,
    load_stage1_audit,
    plot_stage1_boolean_summary,
    plot_stage1_coord_ranges,
    plot_stage1_feature_split,
    plot_stage1_target_stats,
    plot_stage1_variable_stats,
    stage1_coord_ranges_frame,
    stage1_feature_split_frame,
    stage1_stats_frame,
    summarize_stage1_audit,
)

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 100)

AUDIT_COLORS = {
    "primary": "#15803d",
    "secondary": "#0f766e",
    "accent": "#d97706",
    "violet": "#7c3aed",
    "ink": "#0f172a",
    "muted": "#64748b",
    "grid": "#cbd5e1",
    "face": "#f8fafc",
    "pass": "#16a34a",
    "fail": "#dc2626",
}


def _style_axes(ax: plt.Axes, *, facecolor: str = AUDIT_COLORS["face"]) -> None:
    ax.set_facecolor(facecolor)
    ax.set_axisbelow(True)
    ax.grid(True, color=AUDIT_COLORS["grid"], alpha=0.45, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#94a3b8")
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=AUDIT_COLORS["ink"], labelsize=9)
    if ax.get_title():
        ax.set_title(ax.get_title(), fontsize=11, fontweight="bold", color=AUDIT_COLORS["ink"], pad=12)
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontsize=9.5, color=AUDIT_COLORS["muted"])
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontsize=9.5, color=AUDIT_COLORS["muted"])


def _style_bars(ax: plt.Axes, colors: list[str], *, edgecolor: str = AUDIT_COLORS["ink"], linewidth: float = 1.1, alpha: float = 0.95) -> None:
    for idx, patch in enumerate(ax.patches):
        patch.set_facecolor(colors[idx % len(colors)])
        patch.set_edgecolor(edgecolor)
        patch.set_linewidth(linewidth)
        patch.set_alpha(alpha)


def _style_stacked_bars(ax: plt.Axes, colors: list[str]) -> None:
    for idx, container in enumerate(ax.containers):
        color = colors[idx % len(colors)]
        for patch in container:
            patch.set_facecolor(color)
            patch.set_edgecolor(AUDIT_COLORS["ink"])
            patch.set_linewidth(0.9)
            patch.set_alpha(0.95)
    if ax.get_legend() is not None:
        ax.get_legend().get_frame().set_facecolor("white")
        ax.get_legend().get_frame().set_edgecolor("#cbd5e1")


def _style_boolean_bars(ax: plt.Axes) -> None:
    for patch in ax.patches:
        value = patch.get_width() if patch.get_width() else patch.get_height()
        color = AUDIT_COLORS["pass"] if value >= 0.5 else AUDIT_COLORS["fail"]
        patch.set_facecolor(color)
        patch.set_edgecolor(AUDIT_COLORS["ink"])
        patch.set_linewidth(1.1)
        patch.set_alpha(0.95)


# %%
# What this lesson will teach
# ---------------------------
#
# We will follow the same order a careful user would follow with a
# real artifact from ``results/...``:
#
# 1. open the audit and confirm its identity,
# 2. read the semantic summary first,
# 3. inspect the coordinate ranges,
# 4. inspect the feature split,
# 5. inspect the numeric variable summaries,
# 6. use the boolean checks as a fast decision panel,
# 7. decide whether the Stage-1 output is credible enough for the
#    next workflow step.
#
# This ordering matters. If the summary already shows missing
# sections or broken coordinate metadata, there is little value in
# overinterpreting later plots.


# %%
# Create a realistic demo audit
# -----------------------------
#
# For documentation, we do not want to rerun the whole preprocessing
# pipeline every time the gallery builds. The generation helper gives
# us a stable and realistic audit payload with the same broad shape as
# the real Stage-1 audit artifact.
#
# In a real project, you would replace this temporary path with your
# own Stage-1 audit JSON file.

out_dir = Path(tempfile.mkdtemp(prefix="gp_stage1_audit_"))
audit_path = out_dir / "nansha_stage1_scaling_audit.json"

generate_stage1_audit(
    output_path=audit_path,
    city="nansha",
    model="GeoPriorSubsNet",
    normalize_coords=True,
    keep_coords_raw=True,
    shift_raw_coords=True,
    coords_in_degrees=False,
    coord_mode="projected_meters",
    coord_epsg_used=32649,
    n_sequences=1998,
    forecast_horizon=3,
    time_steps=5,
)

print("Written audit file")
print(f" - {audit_path}")
# %%
# Load the artifact with the real reader
# --------------------------------------
#
# Even in a lesson, it is worth loading the file through the real
# helper. That keeps the example close to how users will inspect
# their own artifacts later.

audit_record = load_stage1_audit(audit_path)

print("\nArtifact header")
pprint(
    {
        "kind": audit_record.kind,
        "stage": audit_record.stage,
        "city": audit_record.city,
        "model": audit_record.model,
        "path": str(audit_record.path),
    }
)

# %%
# Start with the semantic summary
# -------------------------------
#
# The summary is the best first stop because it compresses the
# nested audit into a decision-oriented overview:
#
# - what artifact family we are looking at,
# - how many features each group contains,
# - whether coordinate ranges are present,
# - whether the feature split exists,
# - and whether the scaler metadata appears complete.
#
# A good habit is to read this before looking at the plots.
# If the summary already says a core section is missing, the later
# visualizations should be interpreted cautiously.

summary = summarize_stage1_audit(audit_record)

print("\nCompact summary")
print(json.dumps(summary, indent=2))

# %%
# Read the key summary signals aloud
# ----------------------------------
#
# Rather than printing the whole summary and moving on, let us turn
# it into a short interpretation. This is closer to how a user
# would actually reason about the file.

checks = summary["checks"]
feature_counts = summary["feature_counts"]
coord_ranges = summary["coord_ranges"]

print("\nInterpretation of the summary")
print(
    "- Dynamic / static / future feature counts:",
    feature_counts,
)
print(
    "- Coordinate ranges available for chain-rule scaling:",
    coord_ranges,
)
print(
    "- Structural checks:",
    {k: checks[k] for k in sorted(checks)},
)

# %%
# Inspect the coordinate-range frame
# ----------------------------------
#
# For physics-guided models, coordinate ranges are more than a
# cosmetic detail. They determine how normalized coordinates are
# interpreted when derivatives are converted back into physical
# units.
#
# When this frame looks suspicious, the user should stop and verify
# the preprocessing assumptions before trusting later physics or
# calibration diagnostics.

coord_frame = stage1_coord_ranges_frame(audit_record)

print("\nCoordinate ranges")
print(coord_frame)

if not coord_frame.empty:
    print("\nWhy these values matter")
    print(
        "- A positive time range means normalized time can be "
        "mapped back to the original time units."
    )
    print(
        "- Positive x/y ranges indicate that spatial derivatives "
        "can also be rescaled consistently."
    )
    print(
        "- Extremely tiny ranges would be a warning sign because "
        "they may amplify derivative noise."
    )

# %%
# Inspect the feature split frame
# -------------------------------
#
# The Stage-1 audit separates features into semantic buckets:
#
# - ``scaled_by_main_scaler``
# - ``si_unscaled``
# - ``other_unscaled_mixed``
#
# This separation is especially useful when a user wants to verify
# whether physically meaningful channels were left in SI-ready form
# while ordinary machine-learning inputs were scaled more
# conventionally.

split_frame = stage1_feature_split_frame(audit_record)

print("\nFeature split overview")
print(
    split_frame.loc[
        :, ["group", "bucket", "feature", "bucket_size"]
    ].head(24)
)

bucket_counts = (
    split_frame.groupby(["group", "bucket"])
    .size()
    .rename("n_features")
    .reset_index()
)

print("\nFeature counts by group and bucket")
print(bucket_counts)

# %%
# Interpret the feature split
# ---------------------------
#
# A good feature split does not have to be perfectly balanced.
# What matters is whether it is *intentional*.
#
# In many GeoPrior workflows, users often want:
#
# - some ML-style covariates scaled by a main scaler,
# - some physically meaningful variables already expressed in
#   SI-ready form,
# - and only a limited set of mixed unscaled columns.
#
# Here we produce a short analysis instead of only showing the raw
# table.

split_pivot = (
    bucket_counts.pivot(
        index="group",
        columns="bucket",
        values="n_features",
    )
    .fillna(0)
    .astype(int)
)

print("\nInterpretation of the feature split")
for group_name in split_pivot.index:
    row = split_pivot.loc[group_name]
    print(f"- {group_name}:")
    print(
        f"  scaled={int(row.get('scaled_by_main_scaler', 0))}, "
        f"si_unscaled={int(row.get('si_unscaled', 0))}, "
        f"other_mixed={int(row.get('other_unscaled_mixed', 0))}"
    )

# %%
# Inspect the numeric summaries
# -----------------------------
#
# The audit also stores compact variable statistics. These are not
# meant to replace full exploratory data analysis, but they are very
# useful for quick sanity checks.
#
# We look separately at:
#
# - physics-side variables, and
# - learning targets.

physics_stats = stage1_stats_frame(
    audit_record,
    section="physics_df_stats",
)
target_stats = stage1_stats_frame(
    audit_record,
    section="targets_stats",
)

print("\nPhysics variable statistics")
print(physics_stats)

print("\nTarget statistics")
print(target_stats)

# %%
# Read the variable statistics like an inspector
# ----------------------------------------------
#
# For a first-pass lesson, a very practical way to read these
# tables is:
#
# - inspect the means to understand the rough scale,
# - inspect the standard deviations to see whether some variables
#   are nearly constant,
# - inspect the min / max values to detect obvious outliers or
#   sign mistakes.
#
# Below we extract a small mean table because that is often what a
# user needs first.

physics_means = (
    physics_stats.loc[physics_stats["stat"].eq("mean")]
    .loc[:, ["name", "value"]]
    .rename(columns={"value": "mean_value"})
)
target_means = (
    target_stats.loc[target_stats["stat"].eq("mean")]
    .loc[:, ["name", "value"]]
    .rename(columns={"value": "mean_value"})
)

print("\nPhysics means")
print(physics_means)

print("\nTarget means")
print(target_means)

# %%
# Plot 1: coordinate ranges
# -------------------------
#
# This plot answers a narrow but important question:
#
# *Do the stored chain-rule coordinate ranges look positive,
# nondegenerate, and compatible with the expected geometry of the
# dataset?*
#
# What to look for:
#
# - all ranges should usually be positive,
# - time should not collapse to zero,
# - spatial ranges should not be implausibly tiny when the dataset
#   is supposed to cover a real city or basin.

fig, ax = plt.subplots(
    figsize=(7.2, 4.4),
    constrained_layout=True,
)
plot_stage1_coord_ranges(
    audit_record,
    ax=ax,
    title="Stage-1 coordinate ranges used by the chain rule",
)
_style_axes(ax, facecolor="#f8fafc")
_style_bars(ax, [AUDIT_COLORS["primary"], AUDIT_COLORS["secondary"], AUDIT_COLORS["accent"]])

# %%
# Plot 2: feature split by group
# ------------------------------
#
# This stacked view is one of the quickest ways to understand the
# composition of the model inputs.
#
# What to look for:
#
# - dynamic, static, and future groups should all be represented
#   when you expect them,
# - the bucket proportions should look intentional,
# - a completely empty future group is often a warning sign for
#   forecasting setups that expect known drivers.

fig, ax = plt.subplots(
    figsize=(7.4, 4.6),
    constrained_layout=True,
)
plot_stage1_feature_split(
    audit_record,
    ax=ax,
    title="How Stage-1 grouped and scaled the input features",
)
_style_axes(ax, facecolor="#fbfcff")
_style_stacked_bars(ax, [AUDIT_COLORS["primary"], AUDIT_COLORS["accent"], AUDIT_COLORS["violet"]])

# %%
# Plot 3: physics-variable means
# ------------------------------
#
# This plot does not tell us whether the variables are "correct" in
# an absolute scientific sense. It tells us whether their scales and
# signs are *plausible enough* to deserve trust for the next stage.
#
# What to look for:
#
# - variables on wildly different scales may deserve another look,
# - unexpected sign patterns may suggest a convention mismatch,
# - a mean near zero is not automatically bad, but a near-zero
#   standard deviation for an important variable can signal a nearly
#   constant channel.

fig, ax = plt.subplots(
    figsize=(8.2, 4.8),
    constrained_layout=True,
)
plot_stage1_variable_stats(
    audit_record,
    section="physics_df_stats",
    stat="mean",
    ax=ax,
    title="Physics-side variables: mean values",
)
_style_axes(ax, facecolor="#fffdf8")
_style_bars(ax, [AUDIT_COLORS["secondary"], AUDIT_COLORS["primary"], AUDIT_COLORS["accent"], AUDIT_COLORS["violet"]])

# %%
# Plot 4: target means
# --------------------
#
# Targets deserve separate attention because they shape the learning
# problem directly.
#
# What to look for:
#
# - are the target means in a plausible range,
# - does the sign convention look consistent,
# - does the target scale roughly match what you expect from the
#   dataset and units?

fig, ax = plt.subplots(
    figsize=(7.6, 4.4),
    constrained_layout=True,
)
plot_stage1_target_stats(
    audit_record,
    stat="mean",
    ax=ax,
    title="Learning targets: mean values",
)
_style_axes(ax, facecolor="#f8fafc")
_style_bars(ax, [AUDIT_COLORS["accent"], AUDIT_COLORS["primary"]])

# %%
# Plot 5: boolean decision panel
# ------------------------------
#
# The boolean summary is not a replacement for detailed reading.
# It is a compact decision panel that answers:
#
# *Are the core structural sections present, and do the broad
# expectations appear to hold?*
#
# It is often the last plot to check before deciding whether you
# are comfortable moving to Stage-2.

fig, ax = plt.subplots(
    figsize=(8.2, 4.6),
    constrained_layout=True,
)
plot_stage1_boolean_summary(
    audit_record,
    ax=ax,
    title="Stage-1 audit decision checks",
)
_style_axes(ax, facecolor="#f8fafc")
_style_boolean_bars(ax)

# %%
# Use the all-in-one inspector
# ----------------------------
#
# When you want more than live figures, ``inspect_stage1_audit`` is
# useful because it returns the summary, tidy frames, and optionally
# writes the main figures to disk.
#
# This pattern is convenient for gallery generation, CLI tools, and
# later reporting workflows.

bundle_dir = out_dir / "inspection_bundle"
bundle = inspect_stage1_audit(
    audit_record,
    output_dir=bundle_dir,
    stem="lesson_stage1",
    save_figures=True,
)

print("\nInspector bundle keys")
print(sorted(bundle))

print("\nSaved inspection figures")
for name, path in bundle["figure_paths"].items():
    print(f" - {name}: {path}")

# %%
# Final reading rule
# ------------------
#
# A simple lesson-level decision rule is:
#
# - coordinates are normalized when expected,
# - coordinate ranges are present,
# - the feature split exists,
# - scaler metadata exists,
# - future features are present for the forecasting setup.
#
# This rule is intentionally simple. It is not a substitute for
# domain knowledge, but it is a good first-pass gate before moving
# toward the Stage-2 handshake.

must_pass = [
    "coords_normalized",
    "has_coord_ranges",
    "has_feature_split",
    "has_scaler_info",
    "has_future_features",
]
ready = all(bool(checks.get(name, False)) for name in must_pass)

print("\nDecision note")
if ready:
    print(
        "This audit looks structurally credible enough to continue "
        "toward a Stage-2 handshake review."
    )
else:
    print(
        "This audit should be reviewed more carefully before moving "
        "to Stage-2."
    )
