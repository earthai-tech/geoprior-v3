# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Inspect physics-payload metadata before opening the full payload
==================================================================

This lesson explains how to inspect a physics-payload metadata
sidecar, usually a file such as ``physics_payload.npz.meta.json``.

This artifact is small, but it carries some of the most important
semantic information attached to an exported physics payload:

- which model and split produced the payload,
- which PDE modes were active,
- which groundwater convention was used,
- which closure / tau-prior definition was assumed,
- which units should be used when interpreting the arrays,
- and which compact epsilon-style payload metrics were recorded.

The goal of this page is therefore not only to call helper
functions. It is to teach how to inspect the metadata first,
so a user can decide whether the heavier NPZ payload is worth
opening and whether its conventions match the intended analysis.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd

from geoprior.utils.inspect import (
    generate_physics_payload_meta,
    inspect_physics_payload_meta,
    load_physics_payload_meta,
    physics_payload_meta_closure_frame,
    physics_payload_meta_identity_frame,
    physics_payload_meta_metrics_frame,
    physics_payload_meta_units_frame,
    plot_physics_payload_meta_boolean_summary,
    plot_physics_payload_meta_core_scalars,
    plot_physics_payload_meta_payload_metrics,
    summarize_physics_payload_meta,
)

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 112)
pd.set_option("display.max_colwidth", 88)


PAYLOAD_META_PALETTE = {
    "core": "#0F766E",
    "metrics": "#7C3AED",
    "pass": "#15803D",
    "fail": "#B91C1C",
    "edge": "#334155",
    "panel": "#FCFCFE",
}


def _style_bar_panel(
    ax: plt.Axes,
    *,
    color: str,
    edge: str = PAYLOAD_META_PALETTE["edge"],
) -> None:
    """Polish a bar-based gallery panel."""
    for patch in ax.patches:
        patch.set_facecolor(color)
        patch.set_edgecolor(edge)
        patch.set_linewidth(1.25)
        patch.set_alpha(0.94)
    ax.set_facecolor(PAYLOAD_META_PALETTE["panel"])
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
            PAYLOAD_META_PALETTE["pass"]
            if score >= 0.5
            else PAYLOAD_META_PALETTE["fail"]
        )
        patch.set_facecolor(color)
        patch.set_edgecolor(PAYLOAD_META_PALETTE["edge"])
        patch.set_linewidth(1.15)
        patch.set_alpha(0.94)
    ax.set_facecolor(PAYLOAD_META_PALETTE["panel"])
    ax.tick_params(labelsize=9)
    ax.title.set_fontweight("bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#CBD5E1")


# %%
# Why this artifact matters
# -------------------------
#
# The full physics payload is often an NPZ archive that stores heavy
# arrays. Opening it directly can be unnecessary if the exported run
# already uses conventions or closures that do not match the question
# we want to study.
#
# The metadata sidecar is therefore the ideal first checkpoint.
# It lets us answer practical questions such as:
#
# 1. Was this payload produced by the model and split I expected?
# 2. Were both ``consolidation`` and ``gw_flow`` active, or only one?
# 3. Are groundwater values interpreted as depth below ground or
#    something else?
# 4. Which tau-prior / closure formula was used to define the saved
#    residual context?
# 5. Are the saved units compatible with the way I plan to report
#    results or compare payloads across runs?
#
# In other words, this artifact is a semantic filter. It helps the
# user validate meaning before reading arrays.


# %%
# Create a realistic demo metadata sidecar
# ----------------------------------------
#
# For documentation pages, we want a stable artifact that behaves like
# a real metadata sidecar without rebuilding a full physics payload.
# The generation helper was designed exactly for that use case.
#
# Here we create a metadata file with realistic identity fields,
# unit conventions, closure definitions, and compact payload metrics.

out_dir = Path(tempfile.mkdtemp(prefix="gp_payload_meta_"))
meta_path = out_dir / "physics_payload_demo.npz.meta.json"

generate_physics_payload_meta(
    meta_path,
    city="nansha",
    model_name="GeoPriorSubsNet",
    split="TestSet",
    created_utc="2026-03-30T10:18:55Z",
    pde_modes_active=["consolidation", "gw_flow"],
    time_units="year",
    time_coord_units="year",
    gwl_kind="depth_bgs",
    gwl_sign="down_positive",
    use_head_proxy=True,
    kappa_mode="kb",
    use_effective_h=True,
    hd_factor=0.6,
    lambda_offset=1.0,
    kappa_value=0.04639,
    tau_prior_definition="tau_closure_from_learned_fields",
    tau_prior_human_name="tau_closure",
    tau_prior_source="model.evaluate_physics() closure head",
    tau_closure_formula="Hd^2 * Ss / (pi^2 * kappa_b * K)",
    payload_metrics={
        "eps_prior_rms": 3.57e-4,
        "eps_cons_scaled_rms": 2.61e-6,
        "closure_consistency_rms": None,
        "r2_logtau": 1.0,
        "kappa_used_for_closure": None,
    },
)

print("Written physics-payload metadata file")
print(f" - {meta_path}")

# %%
# Load the artifact with the real reader
# --------------------------------------
#
# Even in a lesson, it is worth using the same loading entry point
# a real workflow would use. That keeps the page close to a user's
# actual inspection path.

meta_record = load_physics_payload_meta(meta_path)

print("\nArtifact header")
pprint(
    {
        "kind": meta_record.kind,
        "stage": meta_record.stage,
        "city": meta_record.city,
        "model": meta_record.model,
        "path": str(meta_record.path),
    }
)

# %%
# Start with the compact semantic summary
# ---------------------------------------
#
# Before opening every table, start with the compact summary.
#
# This summary is useful because it reorganizes the raw metadata
# into a few inspection layers:
#
# - a short identity block,
# - conventions for groundwater and time,
# - closure / tau-prior semantics,
# - payload-level metrics,
# - and boolean checks for structural completeness.
#
# When you inspect many payload exports, this summary is often the
# fastest way to reject a mismatched sidecar before reading more.

summary = summarize_physics_payload_meta(meta_record)

print("\nCompact summary")
print(json.dumps(summary, indent=2))

# %%
# Read the identity frame first
# -----------------------------
#
# The identity frame answers the first practical question:
#
# *What exactly does this metadata sidecar describe?*
#
# This is where we verify model name, city, split, PDE modes,
# time units, and groundwater conventions.
#
# In practice, this frame is often the quickest way to catch
# mistakes such as:
#
# - opening a payload from the wrong split,
# - mixing runs built with different PDE modes,
# - or confusing groundwater depth conventions.

identity = physics_payload_meta_identity_frame(meta_record)

print("\nIdentity and convention view")
print(identity)

# %%
# Read the closure frame carefully
# --------------------------------
#
# The closure frame is one of the most important parts of this
# artifact family.
#
# It tells us how the payload relates its physical fields to the
# tau prior and whether effective thickness was used. This matters
# because two payloads can look similar numerically while encoding
# different closure assumptions.
#
# When this frame changes across experiments, the comparison is not
# purely a metric comparison anymore. It becomes a modeling-choice
# comparison.

closure = physics_payload_meta_closure_frame(meta_record)

print("\nClosure / tau-prior view")
print(closure)

# %%
# Inspect the units before opening arrays
# ---------------------------------------
#
# The units block is the safeguard that prevents misreading the NPZ
# payload. A user should not interpret fields such as ``K``, ``Ss``,
# ``tau``, or residual values before checking the declared units.
#
# A healthy reading pattern is:
#
# 1. verify that the expected physical quantities are present,
# 2. check that ``tau`` and ``tau_prior`` share compatible units,
# 3. confirm that the residual values are expressed in the units
#    you expect for later reporting.

units = physics_payload_meta_units_frame(meta_record)

print("\nUnits block")
print(units)

# %%
# Read the compact payload metrics
# --------------------------------
#
# The payload metrics are not a replacement for the full evaluation
# artifact. They are a compact checkpoint attached directly to the
# exported payload.
#
# In this lesson, the most informative metrics are usually:
#
# - ``eps_prior_rms``
# - ``eps_cons_scaled_rms``
# - ``closure_consistency_rms``
# - ``r2_logtau``
# - ``kappa_used_for_closure``
#
# Together they help answer a small but important question:
#
# *Does this payload still carry a physically interpretable closure
# context, or does it already look inconsistent before I read the
# heavier arrays?*

metrics = physics_payload_meta_metrics_frame(meta_record)

print("\nPayload metrics")
print(metrics)

# %%
# Use the all-in-one inspector when you want the key tables together
# ------------------------------------------------------------------
#
# ``inspect_physics_payload_meta(...)`` is useful when you want a
# compact bundle with:
#
# - the semantic summary,
# - the identity frame,
# - the closure frame,
# - the units frame,
# - and the metrics frame.
#
# This is especially useful for batch inspection tools that need to
# scan many sidecars quickly before deciding which payload archives
# deserve a deeper review.

bundle = inspect_physics_payload_meta(meta_record)

print("\nInspector bundle keys")
print(sorted(bundle))

# %%
# Plot the metadata scalars and payload metrics
# ---------------------------------------------
#
# The first figure answers two different questions.
#
# The *core scalars* plot shows top-level numeric metadata such as
# ``hd_factor``, ``Hd_factor``, ``lambda_offset``, and ``kappa_value``.
#
# This helps the user check whether the saved payload corresponds to
# the physical closure settings they intended.
#
# The *payload metrics* plot then summarizes the small diagnostic
# numbers attached directly to the payload. This is a quick way to
# see whether epsilon-like values are tiny, moderate, or obviously
# suspicious.

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.4, 4.8),
    constrained_layout=True,
)

plot_physics_payload_meta_core_scalars(
    meta_record,
    ax=axes[0],
    title="Physics-payload meta: core scalars",
)
_style_bar_panel(axes[0], color=PAYLOAD_META_PALETTE["core"])
plot_physics_payload_meta_payload_metrics(
    meta_record,
    ax=axes[1],
    title="Physics-payload meta: payload metrics",
)
_style_bar_panel(axes[1], color=PAYLOAD_META_PALETTE["metrics"], edge="#4C1D95")

# %%
# How should you read these two plots?
# ------------------------------------
#
# A good reading strategy is:
#
# - use the left plot to confirm that the saved metadata still
#   reflects the closure regime you intended,
# - use the right plot to judge whether the payload-level epsilon
#   values remain small enough to support a later deeper read.
#
# Not every project will use the same thresholds, but some common
# warning signs are:
#
# - missing or null critical scalars,
# - unexpectedly different ``hd_factor`` or ``kappa_value`` between
#   runs you thought were comparable,
# - very large epsilon-style metrics compared with your usual runs,
# - or missing payload metrics when you expected the exporter to
#   attach them.

# %%
# Plot the structural decision checks separately
# ----------------------------------------------
#
# The boolean summary is useful because it turns the metadata into a
# compact pass/fail-style review:
#
# - identity present,
# - PDE modes present,
# - units present,
# - closure definition present,
# - payload metrics present,
# - key physical scalars present.
#
# This is the fastest plot for deciding whether a sidecar is ready
# for serious downstream inspection.

fig, ax = plt.subplots(
    figsize=(8.6, 4.8),
    constrained_layout=True,
)
plot_physics_payload_meta_boolean_summary(
    meta_record,
    ax=ax,
    title="Physics-payload meta: decision checks",
)
_style_boolean_panel(ax)

# %%
# A practical reading rule
# ------------------------
#
# For this artifact family, a useful first-pass decision rule can be:
#
# - model, city, and split are present,
# - PDE modes are recorded,
# - units are present,
# - tau-prior definition and closure formula are present,
# - payload metrics are present,
# - and the key scalar fields such as ``kappa_value`` and
#   ``lambda_offset`` are not missing.
#
# If these checks pass, the metadata is usually healthy enough for
# the user to open the heavier NPZ payload with confidence.

checks = summary["checks"]
must_pass = [
    "has_model_name",
    "has_city",
    "has_split",
    "has_pde_modes",
    "has_units",
    "has_payload_metrics",
    "has_tau_prior_definition",
    "has_tau_closure_formula",
    "time_units_present",
    "time_coord_units_present",
    "kappa_value_present",
    "lambda_offset_present",
    "hd_factor_present",
]
ready = all(bool(checks.get(name, False)) for name in must_pass)

print("\nDecision note")
if ready:
    print(
        "This metadata sidecar looks structurally ready for a "
        "deeper read of the full physics payload archive."
    )
else:
    print(
        "This metadata sidecar needs attention before you rely "
        "on the heavier payload it describes."
    )
