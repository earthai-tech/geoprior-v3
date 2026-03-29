"""
Automatically save the standard GeoPrior history diagnostics
============================================================

This example teaches you how to use GeoPrior's
``autoplot_geoprior_history`` helper.

Unlike the publication-oriented scripts in ``figure_generation/``,
this function is a convenience wrapper for recurring training-log
inspection. It automatically writes the two standard GeoPrior history
figures to an output directory.

Why this matters
----------------
When you inspect many runs, you often want the same two plots every
time:

- epsilon diagnostics,
- physics loss terms.

This helper creates both in one call, with stable filenames.
"""

# %%
# Imports
# -------
# We call the real helper from the package and point it at a compact
# synthetic history dictionary.

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from geoprior.models  import autoplot_geoprior_history


# %%
# Build a compact synthetic GeoPrior history
# ------------------------------------------
# The helper delegates to:
#
# - ``plot_epsilons_in(...)``
# - ``plot_physics_losses_in(...)``
#
# so the history only needs to contain the keys those two helpers know
# how to plot.

epochs = np.arange(1, 13, dtype=int)

history = {
    "loss": (
        np.array(
            [2.20, 1.80, 1.48, 1.23, 1.03, 0.89,
             0.79, 0.72, 0.67, 0.63, 0.60, 0.58]
        )
    ).tolist(),
    "physics_loss": (
        np.array(
            [0.82, 0.66, 0.52, 0.40, 0.31, 0.25,
             0.20, 0.17, 0.15, 0.135, 0.126, 0.120]
        )
    ).tolist(),
    "physics_loss_scaled": (
        np.array(
            [0.39, 0.33, 0.27, 0.22, 0.18, 0.15,
             0.13, 0.118, 0.108, 0.101, 0.096, 0.092]
        )
    ).tolist(),
    "consolidation_loss": (
        np.array(
            [0.29, 0.23, 0.18, 0.14, 0.11, 0.088,
             0.070, 0.057, 0.047, 0.039, 0.034, 0.030]
        )
    ).tolist(),
    "gw_flow_loss": (
        np.array(
            [0.22, 0.18, 0.14, 0.11, 0.087, 0.068,
             0.053, 0.042, 0.033, 0.027, 0.023, 0.020]
        )
    ).tolist(),
    "prior_loss": (
        np.array(
            [0.15, 0.12, 0.095, 0.074, 0.058, 0.044,
             0.034, 0.026, 0.020, 0.016, 0.013, 0.011]
        )
    ).tolist(),
    "smooth_loss": (
        np.array(
            [0.10, 0.082, 0.067, 0.054, 0.043, 0.034,
             0.027, 0.022, 0.018, 0.015, 0.013, 0.011]
        )
    ).tolist(),
    "mv_prior_loss": (
        np.array(
            [0.055, 0.046, 0.038, 0.032, 0.027, 0.023,
             0.019, 0.016, 0.014, 0.0125, 0.0115, 0.0105]
        )
    ).tolist(),
    "bounds_loss": (
        np.array(
            [0.084, 0.068, 0.054, 0.043, 0.034, 0.027,
             0.022, 0.018, 0.015, 0.013, 0.0115, 0.0105]
        )
    ).tolist(),
    "q_reg_loss": (
        np.array(
            [0.039, 0.033, 0.028, 0.024, 0.020, 0.017,
             0.014, 0.012, 0.011, 0.010, 0.009, 0.008]
        )
    ).tolist(),
    "q_rms": (
        np.array(
            [0.27, 0.24, 0.21, 0.18, 0.16, 0.145,
             0.130, 0.118, 0.108, 0.100, 0.094, 0.089]
        )
    ).tolist(),
    "q_gate": (
        np.array(
            [1.0, 1.0, 0.96, 0.90, 0.82, 0.73,
             0.61, 0.48, 0.34, 0.18, 0.05, 0.00]
        )
    ).tolist(),
    "subs_resid_gate": (
        np.array(
            [1.0, 1.0, 1.0, 0.96, 0.90, 0.83,
             0.71, 0.56, 0.39, 0.21, 0.08, 0.00]
        )
    ).tolist(),
    "epsilon_prior": (
        np.array(
            [0.22, 0.17, 0.125, 0.092, 0.067, 0.048,
             0.032, 0.019, 0.010, 0.005, 0.002, 0.001]
        )
    ).tolist(),
    "epsilon_cons": (
        np.array(
            [0.17, 0.14, 0.105, 0.080, 0.060, 0.043,
             0.028, 0.016, 0.009, 0.004, 0.002, 0.001]
        )
    ).tolist(),
    "epsilon_gw": (
        np.array(
            [0.12, 0.10, 0.080, 0.061, 0.046, 0.032,
             0.020, 0.010, 0.004, 0.000, -0.002, 0.001]
        )
    ).tolist(),
    "epsilon_cons_raw": (
        np.array(
            [0.29, 0.23, 0.18, 0.14, 0.11, 0.083,
             0.058, 0.038, 0.021, 0.011, 0.006, 0.003]
        )
    ).tolist(),
}

print("History keys used by the auto-export helper")
for k in history:
    print(" -", k)

# %%
# Auto-save the two standard history figures
# ------------------------------------------
# The helper will create:
#
# - ``<prefix>_epsilons.png``
# - ``<prefix>_physics_terms.png``
#
# inside the requested output directory.

tmp_dir = Path(
    tempfile.mkdtemp(prefix="gp_sg_autoplot_history_")
)
outdir = tmp_dir / "history_figures"

autoplot_geoprior_history(
    history,
    outdir=str(outdir),
    prefix="run_demo",
    style="default",
)

# %%
# Inspect the produced files
# --------------------------
# The helper is intentionally tiny, so the main artifact to inspect is
# the pair of written image files.

written = sorted(outdir.glob("*.png"))

print("")
print("Written files")
for p in written:
    print(" -", p.name)

# %%
# Show the stable filename pattern
# --------------------------------
# The filenames are predictable, which is useful when you run the same
# inspection step for many experiments.

expected = [
    outdir / "run_demo_epsilons.png",
    outdir / "run_demo_physics_terms.png",
]

print("")
print("Expected outputs")
for p in expected:
    print(" -", p)

# %%
# Learn how to use this helper in practice
# ----------------------------------------
# This helper is best used when you already know that every training
# run should emit the same standard inspection plots.
#
# A common workflow is:
#
# 1. train or reload a model,
# 2. collect its history,
# 3. call ``autoplot_geoprior_history(...)``,
# 4. inspect the two saved PNG files.
#
# That is faster and more consistent than calling the two individual
# wrappers manually for every run.

# %%
# Why this page belongs in model_inspection
# -----------------------------------------
# This helper does not create a publication figure and it does not
# build a reusable analysis table.
#
# Its only purpose is to automate recurring inspection output for a
# training history.
#
# So it belongs naturally with the other model-inspection helpers.

# %%
# A natural next lesson
# ---------------------
# After this page, the clean next choices are:
#
# - ``gather_coords_flat.py``
# - ``plot_physics_values_in.py``
#
# because they move from history inspection into coordinate- and
# field-level inspection.