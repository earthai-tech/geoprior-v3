# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Learn how to read forecast sharpness with ``plot_mean_interval_width``
======================================================================

This lesson explains how to use
``geoprior.plot.evaluation.plot_mean_interval_width``
when you want to answer a practical uncertainty question:

**How wide are my prediction intervals, and are they becoming
so wide that they stop being useful?**

Why this function matters
-------------------------
Coverage tells you whether the truth falls inside the interval.
Width tells you how much uncertainty the interval is claiming.

That makes mean interval width one of the simplest sharpness
checks in forecast evaluation.

It is tempting to think that narrower is always better, but that
is not correct.

- extremely narrow intervals may look impressive but miss the truth,
- extremely wide intervals may cover almost everything while being
  too vague for decisions,
- and two models can have similar coverage while offering very
  different practical usefulness because their widths differ a lot.

That is exactly why this helper is valuable.
It gives two complementary views:

- a **histogram** of individual interval widths,
- and a **summary bar** for the mean width.

This page is written as a teaching guide, not only as an API demo.
We will build realistic interval arrays, inspect the full width
distribution, compare narrower and wider regimes, look at
multi-output behavior, and finish with a checklist for applying the
function to your own forecast tables.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geoprior.plot.evaluation import plot_mean_interval_width

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 110)
pd.set_option(
    "display.float_format",
    lambda v: f"{v:0.4f}",
)


# %%
# What this function really expects
# ---------------------------------
#
# ``plot_mean_interval_width`` works directly with two aligned arrays:
#
# - ``y_lower``
# - ``y_upper``
#
# The arrays must have the same shape, and the implementation accepts
# only:
#
# - ``(N,)`` for one output,
# - ``(N, O)`` for multiple outputs.
#
# This helper does not need ``y_true`` because it is not checking
# coverage or error. It is checking **sharpness** only.
#
# It offers two viewing modes:
#
# - ``kind='widths_histogram'``
# - ``kind='summary_bar'``
#
# The histogram answers:
#
# *How are the individual interval widths distributed?*
#
# The summary bar answers:
#
# *What is the mean interval width overall?*
#
# For multi-output data, one important rule matters:
#
# - ``output_idx`` is required for the histogram,
# - while the summary bar can show one overall mean or one bar per
#   output when ``metric_kws={'multioutput': 'raw_values'}`` is used.
#
# This is helpful because sharpness questions are often different at
# different outputs. A subsidence interval and a groundwater interval
# rarely live on the same scale.


# %%
# Build a realistic one-output interval example
# ---------------------------------------------
#
# We begin with a single output and 80 forecast cases.
# The intervals will be narrower in the early part of the sample axis
# and wider later on. This creates a realistic width distribution
# instead of one flat repeated value.
#
# The helper computes width simply as:
#
# ``y_upper - y_lower``
#
# so the most important thing for a lesson example is to make the width
# pattern visible and interpretable.

rng = np.random.default_rng(2026)

n_samples = 80
center = 20.0 + 1.6 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
base_width = np.linspace(0.9, 3.1, n_samples)
noise = rng.normal(0.0, 0.12, n_samples)
width = np.clip(base_width + noise, 0.4, None)

y_lower = center - 0.5 * width
y_upper = center + 0.5 * width

preview = pd.DataFrame(
    {
        "center": center,
        "y_lower": y_lower,
        "y_upper": y_upper,
        "interval_width": y_upper - y_lower,
    }
)

print("One-output preview")
print(preview.head(10))

print("\nWidth summary")
print(preview["interval_width"].describe())


# %%
# Start with the width histogram
# ------------------------------
#
# The histogram is the best first plot when you want to understand the
# *distribution* of sharpness rather than only its average.
#
# This matters because two forecasting systems can have the same mean
# width while behaving differently:
#
# - one may keep widths tightly concentrated,
# - another may mix very narrow and very wide intervals,
# - and a third may have a long right tail only at difficult cases.
#
# In this example the right tail is expected because later cases were
# deliberately given wider intervals.

plot_mean_interval_width(
    y_lower=y_lower,
    y_upper=y_upper,
    kind="widths_histogram",
    figsize=(9.8, 5.4),
    title="Distribution of interval widths",
    hist_bins=14,
    hist_color="#7C3AED",
    hist_edgecolor="#312E81",
    show_score=True,
)


# %%
# How to read the histogram correctly
# -----------------------------------
#
# A good reading order is:
#
# 1. check where the bulk of the widths sits,
# 2. inspect whether there is a long tail of much wider intervals,
# 3. then compare that picture with the mean width.
#
# In this demo, the histogram is not symmetric.
# That is a useful warning sign for real projects because it means the
# model is not expressing uncertainty uniformly across cases.
#
# That is not automatically bad, but it tells you uncertainty is being
# concentrated in some parts of the forecast set.


# %%
# The summary bar gives the compact sharpness number
# --------------------------------------------------
#
# Once you understand the distribution, the summary bar gives the
# compact value you would compare in a table or across competing runs.
#
# This is the simplest sharpness statistic for quick reporting.
#
# A smaller mean interval width indicates sharper forecasts, but that
# only becomes a positive result when coverage or calibration remains
# acceptable as well.

plot_mean_interval_width(
    y_lower=y_lower,
    y_upper=y_upper,
    kind="summary_bar",
    figsize=(6.8, 5.0),
    title="Mean interval width",
    bar_color="#8B5CF6",
    score_annotation_format="{:.3f}",
)


# %%
# Why width must never be read alone
# ----------------------------------
#
# A low width score may come from genuinely sharp forecasts, but it may
# also come from intervals that are too narrow and therefore unreliable.
#
# A high width score may indicate proper caution, but it may also signal
# overly diffuse uncertainty that is not operationally useful.
#
# So the best habit is:
#
# - use ``plot_mean_interval_width`` to study sharpness,
# - then read it together with coverage, WIS, or calibration plots.
#
# In practice, width is one side of the reliability–sharpness trade-off.


# %%
# Compare a narrower regime and a wider regime
# --------------------------------------------
#
# A helpful teaching trick is to split the same forecast set into two
# regimes. Here we compare the first half and the second half of the
# sample axis.
#
# This answers a very practical question:
#
# *Is the forecast becoming less sharp in one part of the evaluation
# set?*
#
# In many real workflows that split could represent:
#
# - early vs late horizons,
# - dry vs wet seasons,
# - urban core vs peripheral cells,
# - or one city vs another.

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.2, 4.9),
    constrained_layout=True,
)

plot_mean_interval_width(
    y_lower=y_lower[:40],
    y_upper=y_upper[:40],
    kind="summary_bar",
    ax=axes[0],
    title="Earlier regime: narrower intervals",
    bar_color="#0EA5E9",
    score_annotation_format="{:.3f}",
)

plot_mean_interval_width(
    y_lower=y_lower[40:],
    y_upper=y_upper[40:],
    kind="summary_bar",
    ax=axes[1],
    title="Later regime: wider intervals",
    bar_color="#F97316",
    score_annotation_format="{:.3f}",
)

plt.show()


# %%
# What this regime comparison teaches
# -----------------------------------
#
# The second half should show a visibly larger mean width.
# That is exactly what we encoded in the synthetic data.
#
# The lesson is important for real evaluations:
#
# a single average width can hide regime drift.
#
# If your intervals widen sharply only at late horizons, difficult
# geologies, or a single study area, you need to know that before you
# report one global sharpness number.


# %%
# Multi-output interval widths are often the most realistic case
# --------------------------------------------------------------
#
# Many GeoPrior workflows evaluate more than one target or more than one
# forecast family. The helper accepts 2D arrays of shape ``(N, O)`` for
# that situation.
#
# Here we create two outputs:
#
# - Output 0: relatively sharp intervals,
# - Output 1: visibly wider intervals.
#
# This is useful because the summary bar can now show one bar per output
# when we request raw values from the metric.

center_2 = 7.0 + 0.9 * np.cos(np.linspace(0, 3 * np.pi, n_samples))
width_1 = np.clip(0.8 + rng.normal(0.0, 0.08, n_samples), 0.35, None)
width_2 = np.clip(2.0 + 0.45 * np.sin(np.linspace(0, 2 * np.pi, n_samples))
                  + rng.normal(0.0, 0.10, n_samples), 0.8, None)

y_lower_2d = np.column_stack(
    [
        center - 0.5 * width_1,
        center_2 - 0.5 * width_2,
    ]
)
y_upper_2d = np.column_stack(
    [
        center + 0.5 * width_1,
        center_2 + 0.5 * width_2,
    ]
)

multi_preview = pd.DataFrame(
    {
        "output0_width": y_upper_2d[:, 0] - y_lower_2d[:, 0],
        "output1_width": y_upper_2d[:, 1] - y_lower_2d[:, 1],
    }
)

print("\nTwo-output width preview")
print(multi_preview.head(10))


# %%
# Use one bar per output in the summary view
# ------------------------------------------
#
# This is one of the most useful multi-output patterns.
# It produces one mean-width bar per output instead of collapsing them
# into one average.
#
# That is usually the better teaching choice because interval width is
# scale-sensitive. Different outputs can behave very differently.

plot_mean_interval_width(
    y_lower=y_lower_2d,
    y_upper=y_upper_2d,
    kind="summary_bar",
    metric_kws={"multioutput": "raw_values"},
    figsize=(7.6, 5.0),
    title="Mean interval width per output",
    bar_color=["#14B8A6", "#EC4899"],
    score_annotation_format="{:.3f}",
)


# %%
# For the histogram, choose the output explicitly
# -----------------------------------------------
#
# Histogram mode works on one selected output at a time when the data is
# multi-output. That is why ``output_idx`` is required.
#
# This makes the reading cleaner: the histogram answers the distribution
# question for one output without mixing scales.

plot_mean_interval_width(
    y_lower=y_lower_2d,
    y_upper=y_upper_2d,
    kind="widths_histogram",
    output_idx=1,
    hist_bins=12,
    figsize=(9.6, 5.2),
    title="Width distribution for output 1",
    hist_color="#F59E0B",
    hist_edgecolor="#7C2D12",
)


# %%
# A practical reading rule for your own data
# ------------------------------------------
#
# When you use this helper on real forecast outputs, a good workflow is:
#
# 1. start with the histogram,
# 2. check whether widths are tightly concentrated or widely spread,
# 3. then read the mean width from the summary bar,
# 4. compare outputs or regimes separately if needed,
# 5. and finally interpret width together with reliability metrics.
#
# In real forecast evaluation, ``plot_mean_interval_width`` is rarely a
# final decision plot by itself. It is a sharpness plot that becomes most
# useful when paired with coverage, calibration, and weighted interval
# score.


# %%
# How to adapt this lesson to your own forecast results
# -----------------------------------------------------
#
# In your own project, the key step is simply to provide aligned lower
# and upper bounds.
#
# Typical sources are:
#
# - ``q10`` and ``q90`` columns for an 80% interval,
# - ``q05`` and ``q95`` for a 90% interval,
# - or calibrated lower/upper bounds written by a Stage-2 workflow.
#
# A practical adaptation pattern is:
#
# ``y_lower = df['subsidence_q10'].to_numpy()``
# ``y_upper = df['subsidence_q90'].to_numpy()``
#
# Then:
#
# - use ``kind='widths_histogram'`` to inspect the full spread,
# - use ``kind='summary_bar'`` to report the mean width,
# - use ``metric_kws={'multioutput': 'raw_values'}`` when each column is
#   a separate output,
# - and use ``output_idx=...`` when you want one histogram for one
#   output.
#
# If the result looks extremely narrow, do not celebrate too early.
# Check coverage next.
#
# If the result looks extremely wide, do not reject it too quickly.
# Check whether calibration or long-horizon uncertainty genuinely needed
# that width.
