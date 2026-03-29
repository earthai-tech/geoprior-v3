"""
Flatten (t, x, y) coordinates from dataset batches
==================================================

This example teaches you how to use GeoPrior's
``gather_coords_flat`` helper.

Unlike the plotting scripts in ``figure_generation/``, this function
is a small inspection utility. It extracts coordinate arrays from
batched dataset-like objects and returns them in one flat, reusable
form.

Why this matters
----------------
Several later inspection tools want coordinates in a very simple
shape:

- one flat ``t`` array,
- one flat ``x`` array,
- one flat ``y`` array.

But training datasets are usually batched and may package inputs in
different ways. This helper normalizes those cases into one compact
dictionary.
"""

# %%
# Imports
# -------
# We call the real helper from the package and feed it small
# dataset-like iterables.

from __future__ import annotations

import numpy as np

from geoprior.models import gather_coords_flat


# %%
# Build a compact dataset in dict-input form
# ------------------------------------------
# The real helper accepts batches where:
#
# - the batch itself is the inputs, or
# - the batch is ``(inputs, targets)``.
#
# When the inputs are a dict, the helper reads ``inputs[coord_key]``.
#
# Here we create two batches with coordinates stored as:
#
# - ``(B, T, 3)``
#
# where the last axis is ordered as:
#
# - t
# - x
# - y

coords_b1 = np.array(
    [
        [
            [2021.0, 120.0, 35.0],
            [2022.0, 120.0, 35.0],
            [2023.0, 120.0, 35.0],
        ],
        [
            [2021.0, 121.5, 35.8],
            [2022.0, 121.5, 35.8],
            [2023.0, 121.5, 35.8],
        ],
    ],
    dtype=float,
)

coords_b2 = np.array(
    [
        [
            [2021.0, 123.0, 36.1],
            [2022.0, 123.0, 36.1],
            [2023.0, 123.0, 36.1],
        ],
        [
            [2021.0, 125.0, 36.7],
            [2022.0, 125.0, 36.7],
            [2023.0, 125.0, 36.7],
        ],
    ],
    dtype=float,
)

dataset_dict = [
    ({"coords": coords_b1, "other": np.ones((2, 3, 1))}, None),
    ({"coords": coords_b2, "other": np.ones((2, 3, 1))}, None),
]

flat_dict = gather_coords_flat(dataset_dict)

print("Flattened coords from dict-input dataset")
for k, v in flat_dict.items():
    print(k, "->", v)

# %%
# Build a compact dataset in direct-input form
# --------------------------------------------
# The helper also accepts cases where the batch directly *is* the
# coordinate tensor, without a dict wrapper.
#
# Here we use a simpler shape:
#
# - ``(B, 3)``
#
# which is also supported.

dataset_direct = [
    np.array(
        [
            [2024.0, 130.0, 40.0],
            [2024.0, 131.0, 40.5],
            [2024.0, 132.0, 41.0],
        ],
        dtype=float,
    ),
    np.array(
        [
            [2025.0, 130.0, 40.0],
            [2025.0, 131.0, 40.5],
            [2025.0, 132.0, 41.0],
        ],
        dtype=float,
    ),
]

flat_direct = gather_coords_flat(dataset_direct)

print("")
print("Flattened coords from direct-input dataset")
for k, v in flat_direct.items():
    print(k, "->", v)

# %%
# Use a non-default coord_key
# ---------------------------
# When the inputs dict stores coordinates under another field name,
# the helper can be pointed to it with ``coord_key=...``.

dataset_alt_key = [
    ({"txy": coords_b1}, None),
    ({"txy": coords_b2}, None),
]

flat_alt = gather_coords_flat(
    dataset_alt_key,
    coord_key="txy",
)

print("")
print("Flattened coords with coord_key='txy'")
for k, v in flat_alt.items():
    print(k, "->", v)

# %%
# Limit the number of processed batches
# -------------------------------------
# ``max_batches`` is useful when you only want a quick preview rather
# than a full pass over a large dataset.

flat_limited = gather_coords_flat(
    dataset_dict,
    max_batches=1,
)

print("")
print("Flattened coords with max_batches=1")
for k, v in flat_limited.items():
    print(k, "->", v)

# %%
# Learn how to read the output
# ----------------------------
# The returned object is intentionally small:
#
# - ``out["t"]``
# - ``out["x"]``
# - ``out["y"]``
#
# All three arrays are one-dimensional and have the same length.
# That makes them easy to:
#
# - pass into plotting helpers,
# - join with flattened payload values,
# - or inspect directly.

# %%
# Why flattening is useful
# ------------------------
# Later plotting helpers usually want coordinates in a simple
# point-wise form.
#
# But datasets often store them as:
#
# - batched samples,
# - batched horizons,
# - or nested `(inputs, targets)` tuples.
#
# This helper removes that packaging so downstream utilities do not
# have to care about batch structure.

# %%
# What kinds of input packaging are supported
# -------------------------------------------
# The helper supports three common patterns:
#
# 1. ``batch`` is ``(inputs, targets)``
# 2. ``inputs`` is a dict containing a coordinate key
# 3. ``inputs`` is the coordinate tensor directly
#
# It also accepts sequence-style inputs, where the first item is
# treated as the coordinate tensor.
#
# In all cases, the last axis must be exactly three values:
#
# - t
# - x
# - y

# %%
# Why this page belongs in model_inspection
# -----------------------------------------
# This helper does not produce a paper figure and it does not export a
# reusable table artifact.
#
# Its job is to normalize coordinate data for later inspection
# utilities.
#
# So it belongs naturally with the other model-inspection helpers.

# %%
# A natural next lesson
# ---------------------
# The clean next page after this is ``plot_physics_values_in.py``,
# because that helper can either:
#
# - consume an explicit ``coords={"x": ..., "y": ...}`` dict,
# - or derive those coordinates from a dataset using this helper.