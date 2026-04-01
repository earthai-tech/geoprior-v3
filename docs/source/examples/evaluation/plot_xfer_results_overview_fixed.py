# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Inspect transfer-learning results before trusting cross-city conclusions
========================================================================

This lesson explains how to inspect the ``xfer_results.json`` artifact.

Unlike most artifacts in the inspection gallery, transfer results are
usually stored as a **JSON list of records**, not as one single mapping.
Each record describes one transfer job and combines:

- the transfer direction,
- the strategy and calibration choices,
- overall evaluation metrics,
- per-horizon behavior,
- schema mismatch diagnostics,
- warm-start details,
- exported CSV locations.

That makes this artifact especially useful when you want to answer
workflow questions such as:

- Which transfer direction behaves better?
- Does performance collapse as the forecast horizon grows?
- Are poor metrics possibly explained by schema mismatch?
- Did the warm-start settings look sensible?
- Is a cross-city result good enough to compare, report, or extend?

The goal of this page is not only to call plotting helpers. It is to
teach how to read transfer results as a **comparison artifact** and turn
that reading into an informed workflow decision.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geoprior.utils.inspect import (
    generate_xfer_results,
    inspect_xfer_results,
    load_xfer_results,
    plot_xfer_boolean_summary,
    summarize_xfer_results,
    xfer_overall_frame,
    xfer_per_horizon_frame,
    xfer_schema_frame,
    xfer_warm_frame,
)

pd.set_option("display.max_columns", 24)
pd.set_option("display.width", 108)
pd.set_option("display.float_format", lambda v: f"{v:0.6f}")

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


# %%
# Why this artifact matters
# -------------------------
#
# Transfer results are not only about raw accuracy. In a cross-city or
# cross-domain workflow, a record can look weak for several different
# reasons:
#
# 1. the transfer direction may be intrinsically harder,
# 2. horizon-wise error may grow too quickly,
# 3. coverage may be acceptable but intervals may be too wide,
# 4. schema mismatch may make the transfer unfair,
# 5. warm-start settings may be too small or too aggressive.
#
# Because this file keeps those pieces together, it becomes one of the
# best places to inspect a transfer experiment before drawing strong
# scientific or operational conclusions.


# %%
# Create a realistic demo artifact
# --------------------------------
#
# For documentation lessons, we usually want a stable artifact that is
# rich enough to compare directions without rerunning the full transfer
# workflow.
#
# Here we generate a small but realistic transfer-results file and make
# the two directions slightly different on purpose, so the lesson can
# teach how to interpret comparisons instead of only printing tables.
#
# One direction is deliberately healthier: lower overall RMSE, flatter
# horizon growth, and fewer schema mismatches. That gives us a clear
# teaching example for how to compare records responsibly.

with tempfile.TemporaryDirectory(prefix="gp_xfer_results_") as tmp:
    out_dir = Path(tmp)
    xfer_path = out_dir / "xfer_results.json"

    generate_xfer_results(
        xfer_path,
        overrides=[
            {
                "strategy": "warm",
                "calibration": "source",
                "rescale_mode": "strict",
                "coverage80": 0.872,
                "sharpness80": 49.80,
                "overall_mae": 13.90,
                "overall_rmse": 21.75,
                "overall_r2": 0.832,
                "per_horizon_rmse": {"H1": 13.90, "H2": 20.90, "H3": 27.80},
                "schema": {"static_missing_n": 7, "static_extra_n": 5},
            },
            {
                "strategy": "warm",
                "calibration": "source",
                "rescale_mode": "strict",
                "coverage80": 0.821,
                "sharpness80": 109.20,
                "overall_mae": 37.10,
                "overall_rmse": 61.20,
                "overall_r2": 0.748,
                "per_horizon_rmse": {"H1": 18.40, "H2": 51.60, "H3": 91.50},
                "schema": {"static_missing_n": 10, "static_extra_n": 12},
            },
        ],
    )

    print("Written transfer-results file")
    print(f" - {xfer_path}")

    # %%
    # Load the artifact with the real reader
    # --------------------------------------
    #
    # This artifact family is slightly special: the top-level payload is
    # a list of records rather than a single JSON object. That is already
    # something worth teaching because users often assume every artifact
    # in the workflow has the same shape.

    records = load_xfer_results(xfer_path)

    print("\nTransfer artifact basics")
    pprint(
        {
            "type": type(records).__name__,
            "n_records": len(records),
            "first_record_keys": sorted(records[0])[:12],
        }
    )

    print("\nFirst record preview")
    print(json.dumps(records[0], indent=2)[:1200] + "\n...")

    # %%
    # Start with the workflow summary
    # -------------------------------
    summary = summarize_xfer_results(records)

    print("\nCompact summary")
    print(json.dumps(summary, indent=2))

    # %%
    # Read the main comparison table
    # ------------------------------
    overall = xfer_overall_frame(records)

    print("\nOverall comparison table")
    print(
        overall[
            [
                "label",
                "direction",
                "strategy",
                "calibration",
                "rescale_mode",
                "coverage80",
                "sharpness80",
                "overall_mae",
                "overall_rmse",
                "overall_r2",
            ]
        ]
    )

    ranked = overall.sort_values("overall_rmse").reset_index(drop=True)
    best = ranked.iloc[0]
    worst = ranked.iloc[-1]

    print("\nInitial reading")
    print(
        f"Best overall record by RMSE: {best['label']} "
        f"(RMSE={best['overall_rmse']:.3f}, R²={best['overall_r2']:.3f})."
    )
    print(
        f"Weakest overall record by RMSE: {worst['label']} "
        f"(RMSE={worst['overall_rmse']:.3f}, R²={worst['overall_r2']:.3f})."
    )

    # %%
    # Interpret overall metrics carefully
    # -----------------------------------
    coverage_gap = float(best["coverage80"] - worst["coverage80"])
    sharpness_gap = float(worst["sharpness80"] - best["sharpness80"])

    print("\nCoverage / sharpness reading")
    print(
        f"Coverage difference between strongest and weakest record: "
        f"{coverage_gap:.3f}."
    )
    print(
        f"Sharpness penalty of the weaker record relative to the stronger one: "
        f"{sharpness_gap:.3f}."
    )

    # %%
    # Inspect per-horizon degradation
    # -------------------------------
    per_h = xfer_per_horizon_frame(records)

    print("\nPer-horizon comparison (RMSE only)")
    print(
        per_h.loc[
            per_h["metric"] == "rmse",
            ["label", "direction", "horizon", "value"],
        ]
    )

    rmse_h = per_h[per_h["metric"] == "rmse"].copy()
    horizon_spread = (
        rmse_h.groupby("label")["value"].agg(["min", "max"])
        .assign(range=lambda d: d["max"] - d["min"])
        .sort_values("range", ascending=False)
    )

    print("\nHorizon spread in RMSE")
    print(horizon_spread)

    print(
        "\nInterpretation:\n"
        "A large RMSE range across H1→H3 usually means the transfer is\n"
        "less reliable for longer horizons, even if the overall metric\n"
        "still looks acceptable."
    )

    # %%
    # Inspect schema mismatch diagnostics
    # -----------------------------------
    schema = xfer_schema_frame(records)

    print("\nSchema diagnostics")
    print(schema)

    schema_counts = schema[schema["kind"] == "count"].copy()
    if not schema_counts.empty:
        schema_pivot = schema_counts.pivot(
            index="label",
            columns="name",
            values="value",
        ).fillna(0.0)
        print("\nSchema mismatch count table")
        print(schema_pivot)

    # %%
    # Inspect warm-start details
    # --------------------------
    warm = xfer_warm_frame(records)

    print("\nWarm-start settings")
    print(warm)

    # %%
    # Use the all-in-one inspector when you want all core views together
    # ------------------------------------------------------------------
    bundle = inspect_xfer_results(records)

    print("\nInspector bundle keys")
    print(sorted(bundle))

    # %%
    # Plot the main comparison views
    # ------------------------------
    #
    # The previous version of this lesson created a 2×2 grid and then
    # called helper functions that internally opened *their own* figures.
    # That left the prepared axes empty on the lesson canvas.
    #
    # Here we instead plot directly on the lesson axes using the tidy
    # frames we already prepared. This keeps every panel populated and
    # lets us control colors, labels, and teaching emphasis clearly.

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.2), constrained_layout=True)

    label_order = list(overall["label"].astype(str))
    label_to_color = {label: PALETTE[idx % len(PALETTE)] for idx, label in enumerate(label_order)}

    # Panel 1: overall RMSE and R² on twin axes so both stay readable.
    ax = axes[0, 0]
    x = np.arange(len(overall))
    rmse_vals = overall["overall_rmse"].astype(float).to_numpy()
    r2_vals = overall["overall_r2"].astype(float).to_numpy()
    colors = [label_to_color[label] for label in label_order]

    bars = ax.bar(x, rmse_vals, color=colors, alpha=0.92, label="overall_rmse")
    ax.set_title("Transfer overall metrics")
    ax.set_ylabel("overall_rmse")
    ax.set_xticks(x)
    ax.set_xticklabels(label_order, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)

    ax_r2 = ax.twinx()
    ax_r2.plot(x, r2_vals, color="#222222", marker="o", linewidth=2.0, label="overall_r2")
    ax_r2.set_ylabel("overall_r2")
    ax_r2.set_ylim(max(0.0, r2_vals.min() - 0.10), 1.0)

    ax.legend([bars, ax_r2.lines[0]], ["overall_rmse", "overall_r2"], loc="upper right", fontsize="small")

    # Panel 2: direct direction comparison.
    ax = axes[0, 1]
    direction_plot = overall.sort_values("direction").reset_index(drop=True)
    ax.bar(
        direction_plot["direction"].astype(str),
        direction_plot["overall_rmse"].astype(float),
        color=[label_to_color[label] for label in direction_plot["label"]],
        alpha=0.92,
    )
    ax.set_title("Direction comparison: overall RMSE")
    ax.set_ylabel("overall_rmse")
    ax.grid(axis="y", alpha=0.25)

    # Panel 3: per-horizon degradation.
    ax = axes[1, 0]
    for label, grp in rmse_h.groupby("label", sort=False):
        g = grp.sort_values("horizon_index")
        ax.plot(
            g["horizon"].astype(str),
            g["value"].astype(float),
            marker="o",
            linewidth=2.2,
            color=label_to_color.get(label, PALETTE[0]),
            label=label,
        )
    ax.set_title("Transfer per-horizon RMSE")
    ax.set_xlabel("horizon")
    ax.set_ylabel("rmse")
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small")

    # Panel 4: schema mismatch counts.
    ax = axes[1, 1]
    schema_pivot = schema_counts.pivot(index="label", columns="name", values="value").fillna(0.0)
    schema_pivot = schema_pivot.reindex(label_order)
    x = np.arange(len(schema_pivot.index))
    width = 0.34
    missing = schema_pivot.get("static_missing_n", pd.Series(0.0, index=schema_pivot.index))
    extra = schema_pivot.get("static_extra_n", pd.Series(0.0, index=schema_pivot.index))
    ax.bar(x - width / 2.0, missing.to_numpy(dtype=float), width=width, color="#4c78a8", label="static_missing_n")
    ax.bar(x + width / 2.0, extra.to_numpy(dtype=float), width=width, color="#f58518", label="static_extra_n")
    ax.set_xticks(x)
    ax.set_xticklabels(schema_pivot.index, rotation=25, ha="right")
    ax.set_title("Schema mismatch counts")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize="small")

    # %%
    # How to read these plots
    # -----------------------
    #
    # A practical interpretation pattern is:
    #
    # - If one direction is better on overall RMSE *and* keeps a flatter
    #   per-horizon error curve, it is usually the stronger transfer.
    # - If two directions look similar in metrics but one has far fewer
    #   schema mismatches, that direction is usually easier to trust.
    # - If coverage is decent only because sharpness is extremely large,
    #   the intervals may be too conservative to be useful.
    # - If H3 error grows sharply while H1 looks fine, the model may be
    #   acceptable for short-horizon adaptation but not for longer-horizon
    #   forecasting.
    #
    # In this demo, the story is coherent: the lower-RMSE direction also
    # shows a flatter horizon curve and fewer schema mismatches. That is a
    # much stronger basis for choosing a preferred direction than RMSE
    # alone.

    # %%
    # Plot the boolean summary separately
    # -----------------------------------
    #
    # The boolean summary aggregates the schema pass/fail checks into one
    # compact view. It is not a performance plot; it is a structural
    # plausibility check for the transfer setup.
    #
    # We keep this view separate because the helper already knows how to
    # collapse boolean schema checks into a compact pass/fail plot.

    plot_xfer_boolean_summary(records)

    # %%
    # A simple decision rule
    # ----------------------
    winner = ranked.iloc[0]
    loser = ranked.iloc[-1]

    winner_schema = schema_counts[schema_counts["label"] == winner["label"]]["value"].sum()
    loser_schema = schema_counts[schema_counts["label"] == loser["label"]]["value"].sum()

    winner_h3 = rmse_h[(rmse_h["label"] == winner["label"]) & (rmse_h["horizon"] == "H3")]["value"]
    winner_h3 = float(winner_h3.iloc[0]) if not winner_h3.empty else None

    print("\nDecision note")
    if (
        float(winner["overall_rmse"]) < float(loser["overall_rmse"])
        and float(winner["sharpness80"]) <= float(loser["sharpness80"])
        and float(winner_schema) <= float(loser_schema)
    ):
        print(
            "The stronger direction in this demo looks structurally and "
            "numerically preferable for follow-up transfer analysis."
        )
    else:
        print(
            "The transfer comparison still looks ambiguous and deserves "
            "closer schema or calibration review before reporting."
        )

    if winner_h3 is not None:
        print(
            f"The preferred direction still reaches H3 RMSE={winner_h3:.3f}; "
            "decide whether that is acceptable for your scientific or "
            "operational horizon."
        )
