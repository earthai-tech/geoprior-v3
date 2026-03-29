"""
Stage-1 data checks with group masks and holdout splitting
==========================================================

This lesson introduces one of the most important diagnostic stages in
the GeoPrior workflow: **checking whether the data are actually valid
for training and forecasting before any model fitting begins**.

Why this page matters
---------------------
A forecasting pipeline can fail for reasons that have nothing to do
with the model itself.

Typical Stage-1 problems include:

- some spatial groups are missing required years,
- some groups are usable for forecasting but not for training,
- random holdout splits accidentally create unrealistic leakage,
- spatial-block splits remove more groups than expected.

That is why GeoPrior exposes explicit Stage-1 data-check helpers.

What the real utilities do
--------------------------
The holdout utilities in ``holdout_utils.py`` provide a compact but
important workflow:

- :func:`compute_group_masks`
  builds two validity masks:
  groups valid for training must contain all years from the last
  ``time_steps + horizon`` window, while groups valid for forecasting
  must contain all years from the last ``time_steps`` window. 

- :func:`filter_df_by_groups`
  keeps only rows whose group identifiers belong to a chosen valid
  group set. 

- :func:`split_groups_holdout`
  splits unique groups into train/validation/test sets using either
  ``"random"`` or ``"spatial_block"`` strategy, and enforces that the
  resulting groups are disjoint. Spatial-block mode requires
  ``x_col``, ``y_col``, and a positive ``block_size``. 

What this lesson teaches
------------------------
We will:

1. create a synthetic spatial-temporal dataset,
2. deliberately remove some years from selected groups,
3. compute Stage-1 validity masks,
4. inspect which groups survive for training and forecasting,
5. filter the raw DataFrame accordingly,
6. compare random and spatial-block holdout splits,
7. build compact diagnostic plots to explain what happened.

This page is synthetic so it remains fully executable during the
documentation build.
"""

# %%
# Imports
# -------
# We use the real Stage-1 utilities from GeoPrior.

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from geoprior.utils.holdout_utils import (
    compute_group_masks,
    filter_df_by_groups,
    split_groups_holdout,
)

# %%
# Step 1 - Build a compact synthetic spatial-temporal dataset
# -----------------------------------------------------------
# Each spatial location acts as one group. We observe annual values
# from 2018 to 2024.
#
# The dataset includes:
#
# - spatial coordinates ``coord_x`` and ``coord_y``,
# - time column ``year``,
# - a synthetic subsidence-like target,
# - a few extra columns so the table looks closer to a real Stage-1
#   preprocessing product.

rng = np.random.default_rng(14)

nx = 8
ny = 6
years = list(range(2018, 2025))

xv = np.linspace(0.0, 12_000.0, nx)
yv = np.linspace(0.0, 8_000.0, ny)
X, Y = np.meshgrid(xv, yv)

x_flat = X.ravel()
y_flat = Y.ravel()
n_groups = x_flat.size

xn = (x_flat - x_flat.min()) / (x_flat.max() - x_flat.min())
yn = (y_flat - y_flat.min()) / (y_flat.max() - y_flat.min())

hotspot = np.exp(
    -(
        ((xn - 0.72) ** 2) / 0.020
        + ((yn - 0.34) ** 2) / 0.032
    )
)
gradient = 0.42 * xn + 0.22 * (1.0 - yn)

rows: list[dict[str, float | int]] = []

for gid in range(n_groups):
    base = 2.1 + 1.4 * gradient[gid] + 1.8 * hotspot[gid]

    for year in years:
        # simple annual trend
        value = (
            base
            + 0.35 * (year - years[0])
            + rng.normal(0.0, 0.10)
        )

        rows.append(
            {
                "group_id": gid,
                "coord_x": float(x_flat[gid]),
                "coord_y": float(y_flat[gid]),
                "year": int(year),
                "subsidence": float(value),
                "rainfall": float(900 + 80 * rng.normal()),
            }
        )

df = pd.DataFrame(rows)

print("Initial dataset shape:", df.shape)
print("")
print(df.head(10).to_string(index=False))

# %%
# Step 2 - Introduce realistic Stage-1 problems
# ---------------------------------------------
# To make the diagnostic lesson meaningful, we remove years from some
# groups.
#
# Design:
#
# - some groups lose one recent year and therefore fail the training
#   requirement,
# - some groups still retain enough history for forecasting,
# - some groups lose more than one year and fail both checks.
#
# This mimics what often happens in real geospatial monitoring data.

drop_rows = []

# groups that will fail training but still pass forecast validity
train_only_fail = [3, 12, 21]
for gid in train_only_fail:
    drop_rows.append((gid, 2024))

# groups that fail both training and forecast validity
all_fail = [8, 17, 29]
for gid in all_fail:
    drop_rows.extend([(gid, 2023), (gid, 2024)])

mask_drop = pd.Series(False, index=df.index)
for gid, yr in drop_rows:
    mask_drop |= (
        (df["group_id"] == gid)
        & (df["year"] == yr)
    )

df = df.loc[~mask_drop].copy().reset_index(drop=True)

print("Dataset shape after removing some years:", df.shape)

# %%
# Step 3 - Compute Stage-1 validity masks
# ---------------------------------------
# We now run the real mask builder.
#
# We choose:
#
# - ``train_end_year=2024``
# - ``time_steps=3``
# - ``horizon=2``
#
# That means:
#
# - training validity requires the last ``T + H = 5`` years,
# - forecast validity requires the last ``T = 3`` years.
#
# This matches the helper's contract directly. 

masks = compute_group_masks(
    df,
    group_cols=["group_id", "coord_x", "coord_y"],
    time_col="year",
    train_end_year=2024,
    time_steps=3,
    horizon=2,
)

print("")
print("Required train years:", masks.required_train_years)
print("Required forecast years:", masks.required_forecast_years)

print("")
print("Valid for training:", len(masks.valid_for_train))
print("Valid for forecasting:", len(masks.valid_for_forecast))
print("Keep for processing:", len(masks.keep_for_processing))

print("")
print("Training-valid groups")
print(masks.valid_for_train.head(8).to_string(index=False))

print("")
print("Forecast-valid groups")
print(masks.valid_for_forecast.head(8).to_string(index=False))

# %%
# How to interpret these masks
# ----------------------------
# ``valid_for_train`` is stricter because it needs enough history to
# support both the training window and the forecast horizon.
#
# ``valid_for_forecast`` is looser because it only needs the lookback
# window.
#
# This is an important Stage-1 distinction:
#
# - some groups are still usable for forecasting artifacts,
# - but not reliable enough for supervised training.

# %%
# Step 4 - Filter the raw DataFrame with the computed masks
# ---------------------------------------------------------
# The next real helper is ``filter_df_by_groups``.
#
# It keeps only rows whose group identifiers belong to the selected
# valid-group DataFrame. 

df_train_valid = filter_df_by_groups(
    df,
    group_cols=["group_id", "coord_x", "coord_y"],
    groups=masks.valid_for_train,
)

df_forecast_valid = filter_df_by_groups(
    df,
    group_cols=["group_id", "coord_x", "coord_y"],
    groups=masks.valid_for_forecast,
)

print("")
print("Rows kept for training:", len(df_train_valid))
print("Rows kept for forecasting:", len(df_forecast_valid))

# %%
# Step 5 - Build a compact group-level status table
# -------------------------------------------------
# A diagnostics page becomes easier to read when the filtering logic is
# summarized into one human-readable table.

all_groups = (
    df[["group_id", "coord_x", "coord_y"]]
    .drop_duplicates()
    .sort_values("group_id")
    .reset_index(drop=True)
)

train_ids = set(masks.valid_for_train["group_id"].tolist())
forecast_ids = set(masks.valid_for_forecast["group_id"].tolist())

status_rows = []
for _, row in all_groups.iterrows():
    gid = int(row["group_id"])
    status_rows.append(
        {
            "group_id": gid,
            "coord_x": float(row["coord_x"]),
            "coord_y": float(row["coord_y"]),
            "train_valid": gid in train_ids,
            "forecast_valid": gid in forecast_ids,
            "status": (
                "train+forecast"
                if gid in train_ids
                else (
                    "forecast_only"
                    if gid in forecast_ids
                    else "invalid"
                )
            ),
        }
    )

status_df = pd.DataFrame(status_rows)

print("")
print("Group status summary")
print(status_df.head(12).to_string(index=False))

# %%
# Step 6 - Plot the Stage-1 validity map
# --------------------------------------
# This is the first important visual diagnostic:
#
# - groups valid for both training and forecasting,
# - groups valid only for forecasting,
# - groups invalid for both.
#
# The point of this plot is not beauty. It is to make the Stage-1
# filtering decision visually obvious.

fig, ax = plt.subplots(figsize=(7.4, 5.4))

for label, marker in [
    ("train+forecast", "o"),
    ("forecast_only", "s"),
    ("invalid", "X"),
]:
    g = status_df[status_df["status"] == label]
    ax.scatter(
        g["coord_x"],
        g["coord_y"],
        marker=marker,
        s=80,
        label=label,
    )

ax.set_xlabel("coord_x")
ax.set_ylabel("coord_y")
ax.set_title("Stage-1 validity of spatial groups")
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend()

plt.tight_layout()
plt.show()

# %%
# How to read this plot
# ---------------------
# This figure answers a simple but essential diagnostic question:
#
# - where are the usable groups?
#
# If the invalid groups cluster spatially, that can indicate a serious
# sampling bias. If they are sparse and scattered, the Stage-1 loss may
# be less damaging.

# %%
# Step 7 - Compare random and spatial-block holdout splitting
# -----------------------------------------------------------
# Once the valid groups are known, the next Stage-1 task is to split
# them into train/validation/test groups.
#
# The real helper supports two strategies:
#
# - ``"random"``
# - ``"spatial_block"``
#
# Spatial-block mode requires coordinate columns and a positive block
# size. It groups nearby spatial points into the same holdout block. 

valid_groups = masks.valid_for_train.copy()

split_random = split_groups_holdout(
    valid_groups,
    seed=42,
    val_frac=0.2,
    test_frac=0.2,
    strategy="random",
)

split_block = split_groups_holdout(
    valid_groups,
    seed=42,
    val_frac=0.2,
    test_frac=0.2,
    strategy="spatial_block",
    x_col="coord_x",
    y_col="coord_y",
    block_size=4000.0,
)

print("")
print("Random split sizes")
print(
    len(split_random.train_groups),
    len(split_random.val_groups),
    len(split_random.test_groups),
)

print("")
print("Spatial-block split sizes")
print(
    len(split_block.train_groups),
    len(split_block.val_groups),
    len(split_block.test_groups),
)

# %%
# Step 8 - Plot the random split
# ------------------------------
# This plot shows how a purely random group split distributes the
# training, validation, and test groups over space.

fig, ax = plt.subplots(figsize=(7.2, 5.2))

for label, gdf, marker in [
    ("train", split_random.train_groups, "o"),
    ("val", split_random.val_groups, "s"),
    ("test", split_random.test_groups, "^"),
]:
    ax.scatter(
        gdf["coord_x"],
        gdf["coord_y"],
        marker=marker,
        s=80,
        label=label,
    )

ax.set_xlabel("coord_x")
ax.set_ylabel("coord_y")
ax.set_title("Random holdout split")
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend()

plt.tight_layout()
plt.show()

# %%
# Step 9 - Plot the spatial-block split
# -------------------------------------
# This second plot shows how spatial blocks create a more spatially
# coherent split.
#
# This is often a stronger diagnostic for geospatial generalization
# because nearby points are less likely to leak information across
# train and test.

fig, ax = plt.subplots(figsize=(7.2, 5.2))

for label, gdf, marker in [
    ("train", split_block.train_groups, "o"),
    ("val", split_block.val_groups, "s"),
    ("test", split_block.test_groups, "^"),
]:
    ax.scatter(
        gdf["coord_x"],
        gdf["coord_y"],
        marker=marker,
        s=80,
        label=label,
    )

ax.set_xlabel("coord_x")
ax.set_ylabel("coord_y")
ax.set_title("Spatial-block holdout split")
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend()

plt.tight_layout()
plt.show()

# %%
# How to read the two split plots
# -------------------------------
# Random split
# ~~~~~~~~~~~~
# Random splitting is simple and often efficient, but nearby points can
# easily land in different subsets. In spatial problems, that can make
# validation look better than true out-of-area generalization.
#
# Spatial-block split
# ~~~~~~~~~~~~~~~~~~~
# Spatial-block splitting is harsher but often more realistic for
# geospatial forecasting. It is therefore a stronger diagnostic of
# spatial generalization.
#
# This is why both views are useful:
#
# - random split is a convenient baseline,
# - spatial-block split is often the more demanding test.

# %%
# Step 10 - Summarize the split composition numerically
# -----------------------------------------------------
# A compact table makes the comparison easier to document.

def summarize_split(name: str, split_obj) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy": name,
                "subset": "train",
                "n_groups": len(split_obj.train_groups),
            },
            {
                "strategy": name,
                "subset": "val",
                "n_groups": len(split_obj.val_groups),
            },
            {
                "strategy": name,
                "subset": "test",
                "n_groups": len(split_obj.test_groups),
            },
        ]
    )

split_summary = pd.concat(
    [
        summarize_split("random", split_random),
        summarize_split("spatial_block", split_block),
    ],
    ignore_index=True,
)

print("")
print("Split summary")
print(split_summary.to_string(index=False))

# %%
# Step 11 - Practical interpretation guide
# ----------------------------------------
# This Stage-1 page should be read as a diagnostic checklist.
#
# A user should leave this page able to answer:
#
# - which groups are actually valid for training?
# - which groups remain usable only for forecasting?
# - how many groups are lost because of missing years?
# - how different is a random split from a spatial-block split?
#
# That is why this page belongs first in diagnostics:
# it explains the data-quality and split-validity checks that happen
# before training curves or tuning summaries mean anything at all.

# %%
# Final takeaway
# --------------
# Stage-1 diagnostics are not an optional extra.
#
# They decide whether the downstream workflow is being run on a valid
# set of groups and whether the holdout strategy is spatially credible.
#
# In GeoPrior, that logic is made explicit through:
#
# - ``compute_group_masks``,
# - ``filter_df_by_groups``,
# - ``split_groups_holdout``.
#
# Understanding these checks is the right first step for the
# diagnostics gallery.