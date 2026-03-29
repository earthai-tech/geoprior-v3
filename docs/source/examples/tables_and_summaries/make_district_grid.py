"""
Create district grid layers from forecast points
================================================

This example teaches you how to use GeoPrior's
``make-district-grid`` utility.

Unlike the plotting scripts, this command builds a reusable spatial
layer. It turns forecast point locations into a regular grid of
named zones, optionally clipped to a city boundary.

Why this matters
----------------
Later workflows often need a simple district-like support layer for:

- aggregating hotspot statistics,
- assigning samples to zones,
- tagging clusters with zone IDs,
- and producing map-ready spatial summaries.

This builder creates exactly that layer from the forecast point cloud.
"""

# %%
# Imports
# -------
# We call the real production entrypoint from the project code.
# Then we read the generated grid back in and build one compact
# teaching preview.

from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from geoprior._scripts.make_district_grid import (
    make_district_grid_main,
)
from geoprior._scripts import utils as script_utils

# %%
# Build compact synthetic point clouds
# ------------------------------------
# The production script only needs:
#
# - sample_idx
# - coord_x
# - coord_y
#
# in both eval and future CSVs.
#
# For the lesson, we create two cities with slightly different
# spatial footprints and one future cloud that extends a bit beyond
# the eval cloud.

rng = np.random.default_rng(13)


def _city_points(
    *,
    city: str,
    center_x: float,
    center_y: float,
    scale_x: float,
    scale_y: float,
    drift_x: float,
    drift_y: float,
    n: int = 140,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ang = rng.uniform(0.0, 2.0 * np.pi, size=n)
    rad = np.sqrt(rng.uniform(0.0, 1.0, size=n))

    xe = center_x + scale_x * rad * np.cos(ang)
    ye = center_y + scale_y * rad * np.sin(ang)

    xf = (
        center_x
        + drift_x
        + 1.08 * scale_x * rad * np.cos(ang)
        + rng.normal(0.0, 2.5, size=n)
    )
    yf = (
        center_y
        + drift_y
        + 1.08 * scale_y * rad * np.sin(ang)
        + rng.normal(0.0, 2.0, size=n)
    )

    eval_df = pd.DataFrame(
        {
            "city": city,
            "sample_idx": np.arange(n, dtype=int),
            "coord_x": xe.astype(float),
            "coord_y": ye.astype(float),
        }
    )

    future_df = pd.DataFrame(
        {
            "city": city,
            "sample_idx": np.arange(n, dtype=int),
            "coord_x": xf.astype(float),
            "coord_y": yf.astype(float),
        }
    )

    return eval_df, future_df


ns_eval_df, ns_future_df = _city_points(
    city="Nansha",
    center_x=1200.0,
    center_y=780.0,
    scale_x=120.0,
    scale_y=80.0,
    drift_x=18.0,
    drift_y=8.0,
)

zh_eval_df, zh_future_df = _city_points(
    city="Zhongshan",
    center_x=1720.0,
    center_y=1080.0,
    scale_x=160.0,
    scale_y=95.0,
    drift_x=26.0,
    drift_y=12.0,
)

print("Nansha eval preview")
print(ns_eval_df.head(6).to_string(index=False))

print("")
print("Zhongshan future preview")
print(zh_future_df.head(6).to_string(index=False))

# %%
# Write the lesson inputs
# -----------------------
# We follow the real command's file-based workflow with explicit
# per-city eval/future CSVs.

tmp_dir = Path(
    tempfile.mkdtemp(prefix="gp_sg_district_grid_")
)

ns_eval_csv = tmp_dir / "nansha_eval.csv"
ns_future_csv = tmp_dir / "nansha_future.csv"
zh_eval_csv = tmp_dir / "zhongshan_eval.csv"
zh_future_csv = tmp_dir / "zhongshan_future.csv"

ns_eval_df.to_csv(ns_eval_csv, index=False)
ns_future_df.to_csv(ns_future_csv, index=False)
zh_eval_df.to_csv(zh_eval_csv, index=False)
zh_future_df.to_csv(zh_future_csv, index=False)

print("")
print("Input files")
for p in [
    ns_eval_csv,
    ns_future_csv,
    zh_eval_csv,
    zh_future_csv,
]:
    print(" -", p.name)

# %%
# Build a simple boundary file for clipping
# -----------------------------------------
# The production script accepts an optional boundary GeoJSON/SHP.
# To make the lesson concrete, we provide one here so the page can
# show boundary clipping and sample assignment together.
#
# A single dissolved geometry is enough because the script unions
# the boundary internally.

boundary_poly = Polygon(
    [
        (1070.0, 700.0),
        (1355.0, 710.0),
        (1385.0, 835.0),
        (1260.0, 900.0),
        (1100.0, 870.0),
        (1055.0, 760.0),
    ]
)

boundary_gdf = gpd.GeoDataFrame(
    {"name": ["lesson_boundary"]},
    geometry=[boundary_poly],
    crs=None,
)

boundary_geojson = tmp_dir / "lesson_boundary.geojson"
boundary_gdf.to_file(boundary_geojson, driver="GeoJSON")

print("")
print(f"Boundary file: {boundary_geojson.name}")

# %%
# Run the real district-grid builder
# ----------------------------------
# We ask the production command to:
#
# - use only Nansha for this compact example,
# - build a 6 x 5 grid,
# - clip it to the boundary,
# - write GeoJSON,
# - and also emit sample-to-zone assignments.

out_stem = "district_grid_gallery"

make_district_grid_main(
    [
        "--cities", "ns",
        "--ns-eval",
        str(ns_eval_csv),
        "--ns-future",
        str(ns_future_csv),
        "--nx",
        "6",
        "--ny",
        "5",
        "--pad",
        "0.03",
        "--boundary",
        str(boundary_geojson),
        "--clip-boundary",
        "--min-area-frac",
        "0.03",
        "--format",
        "geojson",
        "--assign-samples",
        "--out",
        str(tmp_dir / out_stem),
    ],
    prog="make-district-grid",
)

# %%
# Inspect the produced artifacts
# ------------------------------
# In grid mode the command writes:
#
# - one city-specific grid file,
# - and, when requested, one city-specific assignment CSV.

written = sorted(tmp_dir.glob("district_grid_gallery*"))
assignments = sorted(tmp_dir.glob("district_assignments_*.csv"))

print("")
print("Written files")
for p in written + assignments:
    print(" -", p.name)

# %%
# Read the generated outputs
# --------------------------
# We read both the grid layer and the assignment table back in.

grid_geojson = tmp_dir / "district_grid_gallery_nansha.geojson"
assign_csv = script_utils.resolve_out_out(
    "district_assignments_nansha.csv"
)

grid_gdf = gpd.read_file(grid_geojson)
assign_df = pd.read_csv(assign_csv)

print("")
print("Grid preview")
print(
    grid_gdf[
        [
            "city",
            "zone_id",
            "zone_label",
            "row",
            "col",
            "xmin",
            "xmax",
            "ymin",
            "ymax",
        ]
    ].head(10).to_string(index=False)
)

print("")
print("Assignments preview")
print(assign_df.head(10).to_string(index=False))

# %%
# Build one compact visual preview
# --------------------------------
# This preview is not part of the production builder itself.
# It is a teaching aid for the gallery page.
#
# It overlays:
#
# - eval and future points,
# - the boundary,
# - the clipped district grid.

fig, ax = plt.subplots(
    figsize=(6.8, 5.2),
    constrained_layout=True,
)

ax.scatter(
    ns_eval_df["coord_x"].to_numpy(float),
    ns_eval_df["coord_y"].to_numpy(float),
    s=10,
    alpha=0.55,
    label="Eval points",
)
ax.scatter(
    ns_future_df["coord_x"].to_numpy(float),
    ns_future_df["coord_y"].to_numpy(float),
    s=10,
    alpha=0.55,
    label="Future points",
)

boundary_gdf.boundary.plot(ax=ax)
grid_gdf.boundary.plot(ax=ax)

for _, row in grid_gdf.iterrows():
    rp = row.geometry.representative_point()
    ax.text(
        rp.x,
        rp.y,
        str(row["zone_id"]),
        ha="center",
        va="center",
        fontsize=7,
    )

ax.set_title("District grid preview: Nansha")
ax.set_xlabel("coord_x")
ax.set_ylabel("coord_y")
ax.legend(fontsize=8)

# %%
# Learn how to read the grid artifact
# -----------------------------------
# Each grid row represents one zone polygon with:
#
# - ``zone_id``:
#   a compact ID such as ``Z0305``,
# - ``row``:
#   north-to-south row index starting at 0,
# - ``col``:
#   west-to-east column index starting at 0,
# - bounding coordinates,
# - and geometry.
#
# The geometry is the full cell unless boundary clipping is enabled.

# %%
# Learn how the zone IDs are formed
# ---------------------------------
# The script numbers rows from the north (top) downward and columns
# from the west (left) eastward.
#
# So:
#
# - ``Z0101`` is the north-west corner,
# - ``Z0201`` is one row farther south,
# - ``Z0102`` is one column farther east.
#
# This is useful because the IDs remain stable and readable even
# after clipping.

# %%
# Why clipping matters
# --------------------
# Without a boundary, the builder creates a full rectangular grid
# over the padded point extent.
#
# With ``--clip-boundary``, each cell is intersected with the
# boundary polygon, and tiny slivers can be dropped using
# ``--min-area-frac``.
#
# That makes the final grid much more suitable for map overlays or
# zone-based reporting.

# %%
# Why the assignment CSV is useful
# --------------------------------
# With ``--assign-samples``, the script also writes a compact table
# mapping each sample point to:
#
# - ``zone_id``
# - ``zone_row``
# - ``zone_col``
#
# and, when clipping was used, a ``zone_present`` flag telling you
# whether that assigned zone survived clipping.
#
# This is especially handy for later scripts that tag clusters or
# aggregate point-level outputs by zone.

# %%
# Why this page belongs after make_boundary.py
# --------------------------------------------
# The natural workflow is:
#
# 1. build a boundary,
# 2. build a district grid over the city footprint,
# 3. optionally assign samples to grid zones,
# 4. reuse the zones in exposure or cluster-tagging steps.
#
# So this page is best seen as the next support-layer lesson after
# the boundary page.

# %%
# Command-line version
# --------------------
# The same lesson can be reproduced from the CLI.
#
# Legacy dispatcher:
#
# .. code-block:: bash
#
#    python -m scripts make-district-grid \
#      --ns-eval results/nansha_eval.csv \
#      --ns-future results/nansha_future.csv \
#      --nx 12 \
#      --ny 12 \
#      --boundary boundary_nansha.geojson \
#      --clip-boundary \
#      --assign-samples \
#      --format both \
#      --out district_grid
#
# Modern CLI:
#
# .. code-block:: bash
#
#    geoprior build district-grid \
#      --ns-eval results/nansha_eval.csv \
#      --ns-future results/nansha_future.csv \
#      --nx 12 \
#      --ny 12 \
#      --boundary boundary_nansha.geojson \
#      --clip-boundary \
#      --assign-samples \
#      --format both \
#      --out district_grid
#
# Source-folder discovery:
#
# .. code-block:: bash
#
#    geoprior build district-grid \
#      --ns-src results/nansha_run \
#      --zh-src results/zhongshan_run \
#      --split auto \
#      --nx 12 \
#      --ny 12 \
#      --pad 0.02 \
#      --format geojson \
#      --out district_grid
#
# The gallery page teaches the builder.
# The command line reproduces it in a workflow.