# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Inspect a Stage-1 manifest before downstream stages
=====================================================

This lesson teaches how to *read* a Stage-1 ``manifest.json``
artifact as a workflow handshake.

Why this file matters
---------------------
Compared with the Stage-1 audit, the manifest is broader. It answers
questions such as:

- Which dataset and split settings were actually resolved?
- Which columns and feature groups will downstream stages rely on?
- Were holdout counts and sequence counts exported?
- Which CSV, joblib, and NPZ artifacts were written?
- Does the Stage-1 bundle look complete enough for Stage-2?

The goal of this page is not only to produce plots. It is to guide the
reader through the file in the order an inspector would normally follow:
identity, compact configuration, feature groups, holdout counts,
artifact inventory, shape summaries, and final decision checks.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd

from geoprior.utils.inspect import (
    generate_manifest,
    inspect_manifest,
    load_manifest,
    manifest_artifacts_frame,
    manifest_config_frame,
    manifest_feature_groups_frame,
    manifest_holdout_frame,
    manifest_identity_frame,
    manifest_paths_frame,
    manifest_shapes_frame,
    manifest_versions_frame,
    plot_manifest_artifact_inventory,
    plot_manifest_boolean_summary,
    plot_manifest_coord_ranges,
    plot_manifest_feature_group_sizes,
    plot_manifest_holdout_counts,
    summarize_manifest,
)

pd.set_option("display.max_columns", 24)
pd.set_option("display.width", 110)


MANIFEST_PALETTE = {
    "inventory": "#4C6EF5",
    "features": "#0F766E",
    "holdout": "#C2410C",
    "coords": "#7C3AED",
    "pass": "#15803D",
    "fail": "#B91C1C",
    "edge": "#243B53",
    "panel": "#FBFCFE",
}


def _style_bar_panel(
    ax: plt.Axes,
    *,
    color: str,
    edge: str = MANIFEST_PALETTE["edge"],
) -> None:
    """Polish a bar-based gallery panel."""
    for patch in ax.patches:
        patch.set_facecolor(color)
        patch.set_edgecolor(edge)
        patch.set_linewidth(1.25)
        patch.set_alpha(0.94)
    ax.set_facecolor(MANIFEST_PALETTE["panel"])
    ax.tick_params(labelsize=9)
    ax.title.set_fontweight("bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#CBD5E1")


def _style_boolean_panel(ax: plt.Axes) -> None:
    """Apply a distinct pass/fail palette to boolean panels."""
    for patch in ax.patches:
        width = patch.get_width()
        height = patch.get_height()
        score = width if width != 0 else height
        color = (
            MANIFEST_PALETTE["pass"]
            if score >= 0.5
            else MANIFEST_PALETTE["fail"]
        )
        patch.set_facecolor(color)
        patch.set_edgecolor(MANIFEST_PALETTE["edge"])
        patch.set_linewidth(1.15)
        patch.set_alpha(0.94)
    ax.set_facecolor("#FCFCFD")
    ax.tick_params(labelsize=9)
    ax.title.set_fontweight("bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#CBD5E1")


# %%
# What this lesson will teach
# ---------------------------
#
# We will inspect the manifest in a deliberate workflow order:
#
# 1. confirm what artifact we are opening,
# 2. read the compact summary,
# 3. inspect the run identity and compact config,
# 4. inspect feature groups and holdout counts,
# 5. inspect artifact paths and runtime versions,
# 6. inspect the saved tensor-shape summary,
# 7. use the boolean checks as a final decision panel.
#
# This order matters because the manifest is a workflow contract.
# Before caring about fine details, we want to know whether the file is
# structurally complete enough to support downstream steps.


# %%
# Create a realistic demo manifest
# --------------------------------
#
# For a gallery lesson, a stable synthetic-but-realistic manifest is
# usually better than re-running the whole preprocessing pipeline.
# The helper below creates such a file while preserving the broad
# structure of a real Stage-1 manifest.
#
# In real work, you would simply point these inspection helpers to the
# manifest written under your own ``results/...`` directory.

out_dir = Path(tempfile.mkdtemp(prefix="gp_manifest_"))
manifest_path = out_dir / "manifest.json"

generate_manifest(
    manifest_path,
    overrides={
        "city": "nansha",
        "model": "GeoPriorSubsNet",
        "timestamp": "2026-03-28 13:41:18",
        "config": {
            "TIME_STEPS": 5,
            "FORECAST_HORIZON_YEARS": 3,
            "MODE": "tft_like",
            "TRAIN_END_YEAR": 2022,
            "FORECAST_START_YEAR": 2023,
        },
    },
)

print("Written manifest file")
print(f" - {manifest_path}")

# %%
# Load the manifest with the real reader
# --------------------------------------
#
# The inspection path should look as close as possible to real usage,
# so we load the file through ``load_manifest(...)`` rather than
# treating it as a plain JSON blob.

manifest_record = load_manifest(manifest_path)

print("\nArtifact header")
pprint(
    {
        "kind": manifest_record.kind,
        "stage": manifest_record.stage,
        "city": manifest_record.city,
        "model": manifest_record.model,
        "path": str(manifest_record.path),
    }
)

# %%
# Start with the compact summary
# ------------------------------
#
# The summary is the best first stop because it compresses the broad
# manifest into a quick decision-oriented overview.
#
# It answers questions such as:
#
# - how many feature groups were recorded,
# - whether scaling kwargs and holdout metadata exist,
# - how many artifact paths and top-level paths were saved,
# - and whether core sections such as ``config`` and ``versions``
#   are present.

summary = summarize_manifest(manifest_record)

print("\nCompact summary")
print(json.dumps(summary, indent=2))

# %%
# Interpret the summary before digging deeper
# -------------------------------------------
#
# A lesson should not stop at printing the summary. We also want to
# interpret it.
#
# In a real workflow, this is the stage where a user asks whether the
# manifest seems complete enough to deserve deeper reading.

print("\nInterpretation of the summary")
print(
    "- Feature-group sizes:",
    {
        "static": summary.get("static_feature_count"),
        "dynamic": summary.get("dynamic_feature_count"),
        "future": summary.get("future_feature_count"),
        "group_id_cols": summary.get("group_id_cols_count"),
    },
)
print(
    "- Inventory counts:",
    {
        "artifact_paths": summary.get("artifact_path_count"),
        "paths": summary.get("path_count"),
        "versions": summary.get("version_count"),
    },
)
print(
    "- Split counts:",
    {
        "train_groups": summary.get("train_groups"),
        "val_groups": summary.get("val_groups"),
        "test_groups": summary.get("test_groups"),
        "train_seq": summary.get("train_seq"),
        "val_seq": summary.get("val_seq"),
        "test_seq": summary.get("test_seq"),
    },
)

# %%
# Inspect the identity and compact config
# ---------------------------------------
#
# The identity frame is the first place to check whether we are
# looking at the file we think we are looking at.
#
# The compact config frame then answers a different question:
# what high-level settings actually governed Stage-1?

identity = manifest_identity_frame(manifest_record)
config_frame = manifest_config_frame(manifest_record)

print("\nIdentity view")
print(identity)

print("\nCompact config view")
print(config_frame)

# %%
# Why the compact config matters
# ------------------------------
#
# These few values often explain later workflow behavior. For
# example:
#
# - ``TIME_STEPS`` and ``FORECAST_HORIZON_YEARS`` shape the sequence
#   layout,
# - ``MODE`` helps explain how downstream tensors are organized,
# - ``TRAIN_END_YEAR`` and ``FORECAST_START_YEAR`` provide temporal
#   context for the split.
#
# A mismatch here can explain many later surprises.

# %%
# Inspect feature groups and holdout counts
# -----------------------------------------
#
# After identity and config, the next practical question is whether
# the data organization looks plausible.
#
# The feature-group frame tells us what kinds of information were
# exported. The holdout frame tells us whether the split sizes look
# coherent enough for later stages.

feature_groups = manifest_feature_groups_frame(manifest_record)
holdout = manifest_holdout_frame(manifest_record)

print("\nFeature groups")
print(feature_groups.loc[:, ["group", "count", "values"]])

print("\nHoldout and split counts")
print(holdout)

# %%
# Interpret the feature groups and holdout counts
# -----------------------------------------------
#
# A strong inspection habit is to convert these tables into a short
# narrative rather than treating them as passive output.

print("\nInterpretation of feature groups")
for _, row in feature_groups.iterrows():
    print(f"- {row['group']}: {row['count']} entries")

holdout_map = dict(zip(holdout["key"], holdout["value"], strict=False))
print("\nInterpretation of holdout structure")
print(
    "- Group counts suggest how many spatial / logical groups are "
    "present in each split."
)
print(
    "- Sequence counts indicate how many sequence samples were "
    "actually exported for training and validation."
)
print(
    "- A very small validation or test split would deserve a closer "
    "look before trusting later metrics."
)
print(
    "- Current values:",
    {
        key: holdout_map.get(key)
        for key in [
            "train_groups",
            "val_groups",
            "test_groups",
            "train_seq",
            "val_seq",
            "test_seq",
        ]
    },
)

# %%
# Inspect artifact, path, version, and shape inventories
# ------------------------------------------------------
#
# This is the part that makes the manifest especially valuable as a
# downstream handshake.
#
# It tells us:
#
# - what Stage-1 actually wrote,
# - where those outputs live,
# - what runtime versions were recorded,
# - and what the exported tensors look like structurally.

artifacts = manifest_artifacts_frame(manifest_record)
paths = manifest_paths_frame(manifest_record)
versions = manifest_versions_frame(manifest_record)
shapes = manifest_shapes_frame(manifest_record)

print("\nArtifact inventory (first rows)")
print(artifacts.head(18))

print("\nTop-level paths")
print(paths)

print("\nRuntime versions")
print(versions)

print("\nTensor shape summary")
print(shapes.head(12))

# %%
# How to read the inventory sections
# ----------------------------------
#
# These sections answer slightly different questions:
#
# - the artifact inventory says *what* was exported,
# - the path table says *where* it was exported,
# - the version table says *with which software context*,
# - the shape table says *what tensor layouts downstream code should
#   expect*.
#
# When something later fails in Stage-2, the shape table is often one
# of the most useful places to revisit.

# %%
# Use the all-in-one inspector
# ----------------------------
#
# ``inspect_manifest(...)`` is convenient when a workflow or CLI
# helper wants the normalized payload, summary, and main tidy frames
# in one call.

bundle = inspect_manifest(manifest_record)

print("\nInspector bundle keys")
print(sorted(bundle))

# %%
# Plot 1: artifact inventory
# --------------------------
#
# This plot is a compact answer to the question:
#
# *Did Stage-1 actually export the bundle we expect?*
#
# What to look for:
#
# - at least some artifact paths should exist,
# - the path and version counts should not be empty,
# - numpy / shape blocks should exist when later stages depend on
#   them.

fig, ax = plt.subplots(
    figsize=(7.6, 4.5),
    constrained_layout=True,
)
plot_manifest_artifact_inventory(
    ax,
    manifest_record,
    title="Manifest inventory counts",
)
_style_bar_panel(ax, color=MANIFEST_PALETTE["inventory"])

# %%
# Plot 2: feature-group sizes
# ---------------------------
#
# This plot makes the feature composition readable at a glance.
#
# What to look for:
#
# - static, dynamic, and future groups should all be present when
#   your workflow expects them,
# - group-id columns should not silently disappear,
# - a missing future group may explain later forecasting issues.

fig, ax = plt.subplots(
    figsize=(7.0, 4.4),
    constrained_layout=True,
)
plot_manifest_feature_group_sizes(
    ax,
    manifest_record,
    title="Stage-1 feature groups",
)
_style_bar_panel(ax, color=MANIFEST_PALETTE["features"])

# %%
# Plot 3: holdout and sequence counts
# -----------------------------------
#
# This plot helps answer whether the split sizes look balanced enough
# for the next workflow step.
#
# What to look for:
#
# - train counts should usually dominate,
# - validation and test counts should not be accidentally tiny,
# - sequence counts should not collapse to zero.

fig, ax = plt.subplots(
    figsize=(8.0, 4.8),
    constrained_layout=True,
)
plot_manifest_holdout_counts(
    ax,
    manifest_record,
    title="Holdout and sequence counts",
)
_style_bar_panel(ax, color=MANIFEST_PALETTE["holdout"])

# %%
# Plot 4: coordinate ranges from scaling kwargs
# ---------------------------------------------
#
# Even though the manifest is broader than the Stage-1 audit, it
# still carries important scaling context. These ranges remind the
# user that the manifest also preserves downstream conventions, not
# only file paths.
#
# What to look for:
#
# - ranges should be positive,
# - values should be plausible for the dataset extent,
# - the presence of this block confirms that scaling metadata was
#   exported into the Stage-1 handshake.

fig, ax = plt.subplots(
    figsize=(7.2, 4.4),
    constrained_layout=True,
)
plot_manifest_coord_ranges(
    ax,
    manifest_record,
    title="Scaling coordinate ranges",
)
_style_bar_panel(ax, color=MANIFEST_PALETTE["coords"])

# %%
# Plot 5: structural decision checks
# ----------------------------------
#
# This boolean summary works like a final gate.
#
# It does not replace the detailed reading above, but it provides a
# fast answer to the question:
#
# *Does this manifest contain the core sections a downstream stage
# would expect?*

fig, ax = plt.subplots(
    figsize=(8.2, 4.6),
    constrained_layout=True,
)
plot_manifest_boolean_summary(
    ax,
    manifest_record,
    title="Stage-1 manifest decision checks",
)
_style_boolean_panel(ax)

# %%
# Final reading rule
# ------------------
#
# A simple lesson-level decision rule for a Stage-1 manifest is:
#
# - schema and stage are present,
# - config / artifacts / paths / versions are present,
# - holdout and scaling kwargs exist,
# - at least one real artifact path was recorded.
#
# When these checks pass, the manifest usually looks healthy enough
# to trust as a Stage-1 handshake for later steps.

must_pass = [
    "has_config",
    "has_artifacts",
    "has_paths",
    "has_versions",
    "has_holdout",
    "has_scaling_kwargs",
]
ready = all(bool(summary.get(name, False)) for name in must_pass)
ready = ready and int(summary.get("artifact_path_count", 0) or 0) > 0

print("\nDecision note")
if ready:
    print(
        "This manifest looks structurally complete enough to support "
        "downstream workflow stages."
    )
else:
    print(
        "This manifest deserves more attention before you rely on it "
        "downstream."
    )
