"""
Build ablation tables from sensitivity records
==============================================

This example teaches you how to use GeoPrior's
``build-ablation-table`` utility.

Unlike the figure-generation scripts, this command does not start
from a plotting function. It starts from raw or semi-processed
ablation records and turns them into tidy, reusable analysis
tables.

Why this matters
----------------
Ablation experiments are only useful if their outputs can be
compared, filtered, exported, and reused downstream.

This builder helps turn a folder of
``ablation_record*.jsonl`` files into:

- one tidy ablation table,
- one optional best-per-city table,
- optional grouped S6 lambda grids,
- optional grouped S7 toggle summaries.

That makes it a strong first lesson for the
``tables_and_summaries`` section.
"""

# %%
# Imports
# -------
# We call the real production entrypoint from the project code.
# Then we read the generated tables back in and build one compact
# visual preview for the lesson page.

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geoprior._scripts.build_ablation_table import (
    build_ablation_table_main,
)

# %%
# Build a compact synthetic ablation study
# ----------------------------------------
# The real script expects ablation records such as
# ``ablation_record*.jsonl``. Each record is a dictionary with
# run metadata and metrics.
#
# For the lesson, we create a small synthetic study with:
#
# - two cities,
# - two physics buckets,
# - a lambda_prior × lambda_cons grid,
# - overall metrics,
# - and per-horizon metrics.
#
# We also include the toggle-style fields used by the script's
# S7 grouped summary logic:
#
# - lambda_smooth
# - lambda_bounds
# - lambda_mv
# - lambda_q
# - use_effective_h
# - kappa_mode
# - hd_factor

lambda_prior_vals = [0.0, 0.1, 0.3, 1.0]
lambda_cons_vals = [0.0, 0.1, 0.3, 1.0]

rows: list[dict[str, object]] = []
counter = 0

for city in ["Nansha", "Zhongshan"]:
    city_shift = 0.0 if city == "Nansha" else 0.55

    for pde_mode in ["both", "none"]:
        phys_penalty = 0.0 if pde_mode == "both" else 1.25

        for lp in lambda_prior_vals:
            for lc in lambda_cons_vals:
                counter += 1

                log_lp = np.log10(lp + 0.12)
                log_lc = np.log10(lc + 0.12)

                mae = (
                    4.8
                    + city_shift
                    + phys_penalty
                    + 1.8 * (log_lp + 0.22) ** 2
                    + 1.1 * (log_lc + 0.18) ** 2
                )

                rmse = mae * 1.18
                mse = rmse**2

                epsilon_prior = (
                    0.26
                    + 0.04 * city_shift
                    + 0.08 * (log_lp + 0.22) ** 2
                    + 0.03 * (log_lc + 0.10) ** 2
                    + 0.08 * (pde_mode == "none")
                )

                epsilon_cons = (
                    0.22
                    + 0.04 * city_shift
                    + 0.03 * (log_lp + 0.08) ** 2
                    + 0.09 * (log_lc + 0.25) ** 2
                    + 0.08 * (pde_mode == "none")
                )

                coverage80 = np.clip(
                    0.95
                    - 0.08 * epsilon_prior
                    - 0.06 * epsilon_cons,
                    0.65,
                    0.98,
                )

                sharpness80 = (
                    15.0
                    + 3.5 * (log_lp + 0.18) ** 2
                    + 5.5 * (log_lc + 0.25) ** 2
                    + 1.0 * (pde_mode == "none")
                )

                r2 = np.clip(
                    0.95
                    - 0.050 * mae
                    - 0.004 * sharpness80,
                    0.10,
                    0.96,
                )

                use_effective_h = bool(
                    city == "Zhongshan" or lp >= 0.3
                )
                hd_factor = 0.6 if use_effective_h else 1.0
                kappa_mode = "bar" if city == "Nansha" else "kb"

                row = {
                    "timestamp": (
                        f"2026-03-28T12:{counter:02d}:00"
                    ),
                    "city": city,
                    "model": "GeoPriorSubsNet",
                    "pde_mode": pde_mode,
                    "use_effective_h": use_effective_h,
                    "kappa_mode": kappa_mode,
                    "hd_factor": float(hd_factor),
                    "lambda_prior": float(lp),
                    "lambda_cons": float(lc),
                    "lambda_gw": (
                        0.20 if pde_mode == "both" else 0.0
                    ),
                    "lambda_smooth": 0.40 if lp >= 0.3 else 0.0,
                    "lambda_mv": 0.08 if lp >= 0.1 else 0.0,
                    "lambda_bounds": 0.05 if lc >= 0.1 else 0.0,
                    "lambda_q": 0.03 if lc >= 0.3 else 0.0,
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "mse": float(mse),
                    "r2": float(r2),
                    "coverage80": float(coverage80),
                    "sharpness80": float(sharpness80),
                    "epsilon_prior": float(epsilon_prior),
                    "epsilon_cons": float(epsilon_cons),
                    "units": {"subs_metrics_unit": "mm"},
                    "per_horizon_mae": {
                        "H1": float(mae * 0.86),
                        "H2": float(mae),
                        "H3": float(mae * 1.15),
                    },
                    "per_horizon_r2": {
                        "H1": float(min(0.99, r2 + 0.04)),
                        "H2": float(r2),
                        "H3": float(max(0.0, r2 - 0.05)),
                    },
                }
                rows.append(row)

print(f"Number of synthetic records: {len(rows)}")
print("")
print("Example record")
print(json.dumps(rows[0], indent=2))

# %%
# Write the synthetic JSONL file
# ------------------------------
# The production command reads ablation records from files, so we
# follow that same workflow here.

tmp_dir = Path(
    tempfile.mkdtemp(prefix="gp_sg_ablation_table_")
)
jsonl_path = tmp_dir / "ablation_record.synthetic.jsonl"

with jsonl_path.open("w", encoding="utf-8") as f:
    for rec in rows:
        f.write(json.dumps(rec) + "\n")

print("")
print(f"Input JSONL written to: {jsonl_path}")

# %%
# Run the real ablation-table builder
# -----------------------------------
# We ask the script to produce:
#
# - the main tidy table,
# - a best-per-city table,
# - grouped S6 lambda grids,
# - and an S7 grouped toggle summary.
#
# We keep the output in CSV/JSON/TXT form for the lesson page.

out_stem = "ablation_table_gallery"

build_ablation_table_main(
    [
        "--input",
        str(jsonl_path),
        "--out-dir",
        str(tmp_dir),
        "--out",
        out_stem,
        "--formats",
        "csv,json,txt",
        "--sort-by",
        "mae",
        "--ascending",
        "auto",
        "--metric-unit",
        "mm",
        "--keep-per-horizon",
        "true",
        "--best-per-city",
        "--group-cols",
        "s6,s7",
        "--s6-metrics",
        "mae,coverage80,sharpness80",
        "--s7-metrics",
        "mae,coverage80,sharpness80",
        "--s7-agg",
        "mean",
    ],
    prog="build-ablation-table",
)

# %%
# Inspect the produced files
# --------------------------
# The command writes multiple outputs. For the lesson, we list the
# main ones so the user sees the artifact family clearly.

written = sorted(tmp_dir.glob("ablation_table_gallery*"))

print("")
print("Written files")
for p in written:
    print(" -", p.name)

# %%
# Read the main table and the grouped outputs
# -------------------------------------------
# The main table is the tidy run-level export.
# S6 is the lambda grid summary.
# S7 is the grouped toggle summary.
# The best-per-city table is a very useful quick summary.

main_csv = tmp_dir / "ablation_table_gallery.csv"
best_csv = (
    tmp_dir / "ablation_table_gallery__best_per_city.csv"
)
s6_csv = (
    tmp_dir
    / "ablation_table_gallery__S6__mae__Nansha__both.csv"
)
s7_csv = tmp_dir / "ablation_table_gallery__S7.csv"

tab = pd.read_csv(main_csv)
best = pd.read_csv(best_csv)
s6 = pd.read_csv(s6_csv)
s7 = pd.read_csv(s7_csv)

print("")
print("Main tidy table")
print(tab.head(8).to_string(index=False))

print("")
print("Best per city")
print(best.to_string(index=False))

print("")
print("S7 grouped summary")
print(s7.head(12).to_string(index=False))

# %%
# Build one compact visual preview from the generated tables
# ----------------------------------------------------------
# This is not part of the production builder itself. It is a
# teaching aid for the gallery page.
#
# Left:
#   an S6 heatmap for MAE in Nansha with physics on.
#
# Right:
#   the best-per-city MAE summary.

# Rebuild the S6 matrix from the flat CSV.
lcons = pd.to_numeric(
    s6.iloc[:, 0], errors="coerce"
).to_numpy()
lpris = np.array([float(c) for c in s6.columns[1:]])
heat = s6.iloc[:, 1:].to_numpy(dtype=float)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(9.0, 3.8),
    constrained_layout=True,
)

# S6 heatmap
ax = axes[0]
im = ax.imshow(
    heat,
    origin="lower",
    aspect="auto",
)
ax.set_title("S6 preview: MAE grid\nNansha, physics=both")
ax.set_xlabel(r"$\lambda_{\mathrm{prior}}$")
ax.set_ylabel(r"$\lambda_{\mathrm{cons}}$")
ax.set_xticks(np.arange(len(lpris)))
ax.set_yticks(np.arange(len(lcons)))
ax.set_xticklabels([f"{x:g}" for x in lpris])
ax.set_yticklabels([f"{x:g}" for x in lcons])

# Mark the best cell.
i_best, j_best = np.unravel_index(
    np.nanargmin(heat),
    heat.shape,
)
ax.scatter(
    [j_best],
    [i_best],
    marker="o",
    s=55,
    facecolors="none",
    edgecolors="white",
    linewidths=1.6,
)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("MAE [mm]")

# Best-per-city bar chart
ax = axes[1]
ax.bar(best["city"], best["mae"])
ax.set_title("Best-per-city summary")
ax.set_ylabel("MAE [mm]")
ax.set_xlabel("City")

for i, v in enumerate(best["mae"].to_numpy(float)):
    ax.text(
        i,
        v + 0.06,
        f"{v:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# %%
# Learn how to read the main table
# --------------------------------
# The main tidy table is the run-level table.
#
# A useful reading order is:
#
# 1. identify the configuration columns
#    such as city, pde_mode, lambda weights, and identifiability
#    settings;
#
# 2. read the main metrics
#    such as MAE, RMSE, R2, coverage80, and sharpness80;
#
# 3. use the per-horizon columns only when you need to inspect
#    how performance drifts across forecast steps.
#
# In other words:
#
# - the main table is the tidy archive,
# - the grouped tables are the paper-style summaries.

# %%
# Learn how to read the S6 table
# ------------------------------
# The S6 output is the lambda-grid summary.
#
# For each city and each physics bucket, the script pivots one
# metric onto:
#
# - rows   = lambda_cons
# - cols   = lambda_prior
#
# This is extremely useful because it converts a long run table
# into a sensitivity surface that can later be:
#
# - plotted as a heatmap,
# - exported to TeX,
# - or inspected directly in CSV form.
#
# In the preview heatmap above, the white ring marks the best MAE
# cell in the Nansha + physics-on grid.

# %%
# Learn how to read the S7 table
# ------------------------------
# The S7 output is the toggle summary.
#
# Instead of summarizing individual runs, it groups runs by
# higher-level settings such as:
#
# - city,
# - pde bucket,
# - use_effective_h,
# - kappa_mode,
# - smooth_on,
# - bounds_on,
# - mv_on,
# - q_on.
#
# Then it computes group means and group sizes.
#
# This is useful when the reader is not asking:
#
# "Which exact lambda pair won?"
#
# but instead:
#
# "Do runs with bounds or smoothness tend to behave better on
# average?"

# %%
# Why this builder is useful in practice
# --------------------------------------
# This script is a strong bridge between raw experiment logs and
# later analysis pages.
#
# It helps with three different kinds of work:
#
# - quick filtering and ranking of runs,
# - paper-style grouped summaries,
# - and downstream figure generation such as S6 sensitivity maps.
#
# That is why it belongs naturally in the
# ``tables_and_summaries`` gallery rather than in
# ``figure_generation``.

# %%
# Practical takeaway
# ------------------
# A good workflow for ablations is:
#
# 1. run the sensitivity or ablation jobs,
# 2. build the tidy ablation table,
# 3. inspect best-per-city rows,
# 4. inspect S6 lambda grids,
# 5. inspect S7 grouped toggle summaries,
# 6. only then move on to paper-ready plots.
#
# This order helps separate:
#
# - data collection,
# - tabulation,
# - and visualization.

# %%
# Command-line version
# --------------------
# The same lesson can be reproduced from the CLI.
#
# Legacy dispatcher:
#
# .. code-block:: bash
#
#    python -m scripts build-ablation-table \
#      --root results \
#      --out table_ablations \
#      --formats csv,json,txt \
#      --best-per-city \
#      --group-cols s6,s7
#
# Paper-friendly TeX export:
#
# .. code-block:: bash
#
#    python -m scripts build-ablation-table \
#      --root results \
#      --for-paper \
#      --err-metric rmse \
#      --keep-r2 \
#      --formats csv,tex \
#      --out table_ablations_paper
#
# Modern CLI:
#
# .. code-block:: bash
#
#    geoprior build ablation-table \
#      --root results \
#      --out table_ablations \
#      --group-cols s6,s7
#
# The gallery page teaches the builder.
# The command line reproduces it in a workflow.