"""
Summarize hotspot point clouds into tidy group tables
=====================================================

This example teaches you how to use GeoPrior's
``summarize-hotspots`` utility.

Unlike the plotting scripts, this command is a table builder.
It starts from a hotspot *point cloud* CSV and converts it into
a tidy summary grouped by city, year, and hotspot kind.

Why this matters
----------------
A hotspot map or point cloud is useful for visual inspection,
but it is not yet the compact artifact that downstream analysis
usually needs.

This builder converts hotspot points into a grouped table with:

- hotspot counts,
- min/mean/max hotspot subsidence values,
- min/mean/max hotspot anomaly values,
- optional baseline summaries,
- optional threshold summaries.

That makes it a strong lesson page for the
``tables_and_summaries`` section.
"""

# %%
# Imports
# -------
# We call the real production entrypoint from the project code.
# Then we read the generated CSV back in and build one compact
# teaching preview.

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geoprior._scripts.summarize_hotspots import (
    summarize_hotspots,
    summarize_hotspots_main,
)

# %%
# Build a compact synthetic hotspot point cloud
# ---------------------------------------------
# The real script expects a hotspot CSV produced by the spatial
# forecast workflow, with fields such as:
#
# - city
# - year
# - kind
# - value
# - metric_value
#
# and optionally:
#
# - baseline_value
# - threshold
#
# For the lesson, we create two cities, three years, and two
# hotspot kinds. That is enough to show how the builder groups the
# point cloud into one summary row per (city, year, kind).

rng = np.random.default_rng(11)

rows: list[dict[str, object]] = []

for city in ["Nansha", "Zhongshan"]:
    city_shift = 0.0 if city == "Nansha" else 6.0

    for year in [2025, 2027, 2030]:
        year_shift = {2025: 0.0, 2027: 3.5, 2030: 8.0}[year]

        for kind in ["q50", "q90"]:
            kind_shift = 0.0 if kind == "q50" else 5.0

            n_hot = 12 if kind == "q50" else 8
            if city == "Zhongshan":
                n_hot += 3

            baseline_mean = 36.0 + city_shift
            threshold = 11.0 + 0.7 * year_shift + 0.4 * kind_shift

            for i in range(n_hot):
                coord_x = 100 + rng.normal(0.0, 18.0)
                coord_y = 200 + rng.normal(0.0, 14.0)

                metric_value = max(
                    0.1,
                    threshold
                    + rng.normal(2.8 + 0.2 * year_shift, 1.3),
                )

                baseline_value = max(
                    0.1,
                    baseline_mean + rng.normal(0.0, 2.2),
                )

                value = baseline_value + metric_value

                rows.append(
                    {
                        "city": city,
                        "panel": "future_hotspots",
                        "kind": kind,
                        "year": year,
                        "coord_x": float(coord_x),
                        "coord_y": float(coord_y),
                        "value": float(value),
                        "hotspot_mode": "delta_abs",
                        "hotspot_quantile": 0.90,
                        "metric_value": float(metric_value),
                        "baseline_value": float(baseline_value),
                        "threshold": float(threshold),
                    }
                )

hotspots_df = pd.DataFrame(rows)

print("Hotspot point-cloud preview")
print(hotspots_df.head(10).to_string(index=False))

# %%
# Write the synthetic hotspot CSV
# -------------------------------
# The production command consumes a point-cloud CSV, so we follow
# the same workflow here.

tmp_dir = Path(
    tempfile.mkdtemp(prefix="gp_sg_hotspots_summary_")
)
hotspot_csv = tmp_dir / "fig6_hotspot_points.synthetic.csv"

hotspots_df.to_csv(hotspot_csv, index=False)

print("")
print(f"Input hotspot CSV written to: {hotspot_csv}")

# %%
# Run the real summarizer
# -----------------------
# We ask the production command to build the grouped summary CSV.

out_csv = tmp_dir / "fig6_hotspot_summary.csv"

summarize_hotspots_main(
    [
        "--hotspot-csv",
        str(hotspot_csv),
        "--out",
        str(out_csv),
        "--quiet",
        "false",
    ],
    prog="summarize-hotspots",
)

# %%
# Read the generated summary table
# --------------------------------
# The output has one row per (city, year, kind) group.

summary = pd.read_csv(out_csv)

print("")
print("Written file")
print(" -", out_csv.name)

print("")
print("Grouped summary table")
print(summary.to_string(index=False))

# %%
# Compare with the direct in-memory API
# -------------------------------------
# The script also exposes the core grouping function directly.
# This is useful for tests and notebook workflows.

summary_api = summarize_hotspots(hotspots_df)

print("")
print("Direct API result matches CSV output:")
print(summary_api.equals(summary))

# %%
# Build one compact visual preview
# --------------------------------
# This preview is not part of the production builder itself.
# It is a teaching aid for the gallery page.
#
# Left:
#   hotspot counts by city-year-kind.
#
# Right:
#   ranked anomaly means.

summary["group"] = (
    summary["city"].astype(str)
    + " | "
    + summary["year"].astype(str)
    + " | "
    + summary["kind"].astype(str)
)

ranked = summary.sort_values(
    ["metric_mean", "n_hotspots"],
    ascending=[False, False],
).reset_index(drop=True)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.0, 4.2),
    constrained_layout=True,
)

# Hotspot counts
ax = axes[0]
ax.bar(summary["group"], summary["n_hotspots"])
ax.set_title("Hotspot counts by group")
ax.set_xlabel("City | Year | Kind")
ax.set_ylabel("n_hotspots")
ax.tick_params(axis="x", rotation=75)

# Ranked anomaly means
ax = axes[1]
ax.bar(ranked["group"], ranked["metric_mean"])
ax.set_title("Ranked anomaly means")
ax.set_xlabel("City | Year | Kind")
ax.set_ylabel("metric_mean [mm/yr]")
ax.tick_params(axis="x", rotation=75)

# %%
# Learn how to read the summary table
# -----------------------------------
# Each row corresponds to one:
#
# - city
# - year
# - kind
#
# combination.
#
# The core reading order is:
#
# 1. read ``n_hotspots`` to see how large the hotspot cloud is;
# 2. read ``value_*`` to understand hotspot subsidence levels;
# 3. read ``metric_*`` to understand hotspot anomaly intensity;
# 4. use ``baseline_*`` and ``threshold_*`` when the source CSV
#    provides those optional columns.
#
# In other words:
#
# - the point-cloud CSV is the geometric/point-level artifact,
# - the summary CSV is the compact grouped artifact.

# %%
# What the key columns mean
# -------------------------
# ``value_*``
#     summary of the hotspot subsidence values themselves.
#
# ``metric_*``
#     summary of the anomaly or delta measure used to define or
#     score hotspot points.
#
# ``baseline_*``
#     optional summary of the baseline values attached to the
#     hotspot points.
#
# ``threshold_*``
#     optional summary of the group threshold values.
#
# The threshold is often constant within a group, but the script
# still keeps both min and max for robustness.

# %%
# Why this builder is useful in practice
# --------------------------------------
# This builder is a bridge between hotspot maps and later
# reporting.
#
# A useful workflow is:
#
# 1. generate hotspot points from a spatial forecast workflow,
# 2. summarize them by city, year, and kind,
# 3. compare counts and anomaly magnitudes across groups,
# 4. only then build narrative tables or paper figures.
#
# This keeps:
#
# - hotspot extraction,
# - group-level tabulation,
# - and final visualization
#
# clearly separated.

# %%
# Why this page belongs after compute_hotspots.py
# -----------------------------------------------
# The previous lesson built a compact city-year hotspot table from
# forecast CSVs directly.
#
# This lesson starts later in the chain:
#
# - it assumes hotspot *points* already exist,
# - and it compresses those points into grouped summary rows.
#
# So the two scripts are related, but they summarize different
# intermediate artifacts.

# %%
# Command-line version
# --------------------
# The same lesson can be reproduced from the CLI.
#
# Legacy dispatcher:
#
# .. code-block:: bash
#
#    python -m scripts summarize-hotspots \
#      --hotspot-csv results/figs/fig6_hotspot_points.csv \
#      --out fig6_hotspot_summary.csv
#
# Quiet mode:
#
# .. code-block:: bash
#
#    python -m scripts summarize-hotspots \
#      --hotspot-csv results/figs/fig6_hotspot_points.csv \
#      --out fig6_hotspot_summary.csv \
#      --quiet true
#
# Modern CLI:
#
# .. code-block:: bash
#
#    geoprior build hotspots-summary \
#      --hotspot-csv results/figs/fig6_hotspot_points.csv \
#      --out fig6_hotspot_summary.csv
#
# The gallery page teaches the builder.
# The command line reproduces it in a workflow.