"""
Spatial forecasts: how to read observed maps, fitted maps,
and future forecast maps together
=================================

This example teaches you how to read the GeoPrior spatial
forecast figure.

The goal of this figure is not only to show a future forecast.
It is to place that forecast in context.

A good spatial forecast page should let the reader answer three
questions:

1. What was observed in the reference year?
2. How closely does the model reproduce that same year?
3. How does the spatial pattern evolve into the forecast years?

That is exactly what the plotting backend in
``plot_spatial_forecasts.py`` is designed to do.

What the figure layout means
----------------------------
The plotting function builds a panel grid with:

- one row per city,
- a first column for the observed map in ``year_val``,
- a second column for the predicted median map in that same
  year,
- and the remaining columns for the future forecast years.

So the reader can move from left to right:

**observation -> fitted reconstruction -> future evolution**

Why this matters
----------------
A future map on its own can be visually impressive but
scientifically incomplete.

If we do not compare the forecast against the observed spatial
pattern in a reference year, we do not know whether the model
has learned the right geography of the process.

This is why the spatial-forecast figure is useful: it joins
calibration-year interpretation with future-year interpretation
in one page.

In the real script, the backend can accept one city or two
cities, can work in cumulative or non-cumulative mode, and can
add hotspot contours. The command-line interface also supports
artifact discovery from ``--ns-src`` and ``--zh-src``, manual
CSV overrides, forecast-year selection, clipping, colormap
choice, and hotspot export. 
"""

# %%
# Imports
# -------
# We use the real figure backend from the project plotting
# module. This gallery page is therefore a teaching wrapper
# around the real plotting function, not a disconnected demo.

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from geoprior._scripts.plot_spatial_forecasts import (
    plot_fig6_spatial_forecasts,
)

# %%
# Step 1 - Build a compact synthetic spatial domain
# -------------------------------------------------
# We create a small regular grid for two synthetic cities.
#
# The intention is pedagogical:
#
# - City A will show one dominant basin-like pattern,
# - City B will show a shifted and slightly different pattern,
# - future years will gradually intensify the main hotspots.
#
# This makes it easy for the reader to see what the figure is
# doing.

nx = 24
ny = 18

xv = np.linspace(0.0, 12_000.0, nx)
yv = np.linspace(0.0, 8_000.0, ny)

X, Y = np.meshgrid(xv, yv)

x = X.ravel()
y = Y.ravel()

xn = (X - X.min()) / (X.max() - X.min())
yn = (Y - Y.min()) / (Y.max() - Y.min())

# %%
# Step 2 - Create one synthetic spatial pattern
# ---------------------------------------------
# We build a helper that creates a smooth subsidence surface with
# a localized depression and a broad regional gradient.
#
# The pattern is deliberately easy to interpret:
#
# - one compact hotspot,
# - one broad background trend,
# - and a future intensification factor.

def make_pattern(
    xn: np.ndarray,
    yn: np.ndarray,
    *,
    cx: float,
    cy: float,
    amp: float,
    width_x: float,
    width_y: float,
    slope_x: float,
    slope_y: float,
    offset: float,
) -> np.ndarray:
    bowl = np.exp(
        -(
            ((xn - cx) ** 2) / width_x
            + ((yn - cy) ** 2) / width_y
        )
    )
    z = (
        offset
        + amp * bowl
        + slope_x * xn
        + slope_y * yn
    )
    return z


# %%
# Step 3 - Build the reference-year observations
# ----------------------------------------------
# The plotting backend expects, for the calibration / observed
# frame, a table with at least:
#
# - sample_idx,
# - forecast_step,
# - coord_t,
# - coord_x,
# - coord_y,
# - subsidence_actual,
# - subsidence_q50.
#
# For the future frame, the required fields are the same except
# that ``subsidence_actual`` is not needed. That is how the
# project loader prepares the tables before plotting. :contentReference[oaicite:6]{index=6}

year_val = 2022
years_fore = [2025, 2027, 2030]

base_a = make_pattern(
    xn,
    yn,
    cx=0.62,
    cy=0.38,
    amp=38.0,
    width_x=0.020,
    width_y=0.035,
    slope_x=8.0,
    slope_y=4.0,
    offset=6.0,
)

base_b = make_pattern(
    xn,
    yn,
    cx=0.35,
    cy=0.62,
    amp=28.0,
    width_x=0.030,
    width_y=0.028,
    slope_x=5.0,
    slope_y=7.0,
    offset=5.5,
)

# Add a small structured difference between actual and fitted
# q50 so the "predicted reference year" panel is realistic.
fit_a = base_a * 0.96 + 1.4 * np.sin(2.0 * np.pi * xn)
fit_b = base_b * 0.95 + 1.0 * np.cos(2.0 * np.pi * yn)

# %%
# Step 4 - Package the calibration-year tables
# --------------------------------------------
# The backend will extract the selected ``year_val`` and then use
# ``subsidence_actual`` for the observed panel and
# ``subsidence_q50`` for the matched prediction panel.

def make_calib_df(
    actual: np.ndarray,
    fitted: np.ndarray,
    *,
    year: int,
) -> pd.DataFrame:
    n = actual.size
    return pd.DataFrame(
        {
            "sample_idx": np.arange(n, dtype=int),
            "forecast_step": np.zeros(n, dtype=int),
            "coord_t": np.full(n, year, dtype=int),
            "coord_x": x,
            "coord_y": y,
            "subsidence_actual": actual.ravel(),
            "subsidence_q50": fitted.ravel(),
            "subsidence_unit": ["mm"] * n,
        }
    )


calib_a = make_calib_df(base_a.ravel(), fit_a.ravel(), year=year_val)
calib_b = make_calib_df(base_b.ravel(), fit_b.ravel(), year=year_val)

# %%
# Step 5 - Package the future forecast tables
# -------------------------------------------
# Now we create future median forecasts. We keep the shape of the
# hotspot but let it intensify through time. This lets the page
# teach one of the main uses of the figure: reading how the
# geography of risk evolves from one forecast year to the next.

def make_future_df(
    fitted_ref: np.ndarray,
    *,
    year_list: list[int],
    growths: list[float],
    wave_scale: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    n = fitted_ref.size

    for i, (yy, g) in enumerate(zip(year_list, growths)):
        wave = wave_scale * np.sin(
            2.0 * np.pi * (xn + 0.2 * i)
        )
        vals = fitted_ref.reshape(X.shape) * g + wave

        rows.append(
            pd.DataFrame(
                {
                    "sample_idx": np.arange(n, dtype=int),
                    "forecast_step": np.full(
                        n, i + 1, dtype=int
                    ),
                    "coord_t": np.full(
                        n, yy, dtype=int
                    ),
                    "coord_x": x,
                    "coord_y": y,
                    "subsidence_q50": vals.ravel(),
                    "subsidence_unit": ["mm"] * n,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


future_a = make_future_df(
    fit_a,
    year_list=years_fore,
    growths=[1.10, 1.22, 1.35],
    wave_scale=0.8,
)

future_b = make_future_df(
    fit_b,
    year_list=years_fore,
    growths=[1.06, 1.15, 1.28],
    wave_scale=0.6,
)

# %%
# Step 6 - Build the city objects expected by the backend
# -------------------------------------------------------
# The plotting function expects a list of city dictionaries. Each
# city contains at least a name, a color, a calibration DataFrame,
# and a future DataFrame. That is the same structure assembled by
# the real CLI helper ``_resolve_city(...)`` before plotting.


cities = [
    {
        "name": "Nansha",
        "color": "#2a6f97",
        "calib_df": calib_a,
        "future_df": future_a,
    },
    {
        "name": "Zhongshan",
        "color": "#c26d3d",
        "calib_df": calib_b,
        "future_df": future_b,
    },
]

# %%
# Step 7 - Render the spatial forecast figure
# -------------------------------------------
# The plotting backend creates:
#
# - column 1: observed map in ``year_val``,
# - column 2: predicted q50 map in ``year_val``,
# - remaining columns: q50 future maps for the requested years.
#
# If cumulative mode is enabled, the backend switches to
# cumulative quantities. It also computes a common color scale
# across all cities and panels, which is important because visual
# comparison is only fair when the color range is shared.

tmp_dir = Path(
    tempfile.mkdtemp(prefix="gp_sg_spatial_")
)
out_base = str(
    tmp_dir / "spatial_forecasts_gallery"
)

plot_fig6_spatial_forecasts(
    cities=cities,
    year_val=year_val,
    years_fore=years_fore,
    cumulative=True,
    subsidence_kind="cumulative",
    grid_res=180,
    clip=98.0,
    cmap_name="viridis",
    hotspot_mode="delta",
    hotspot_q=0.90,
    out=out_base,
    out_hotspots=None,
    dpi=160,
    font=9,
    show_legend=True,
    show_title=True,
    show_panel_titles=True,
    title=(
        "Synthetic spatial forecasts: observed, fitted, "
        "and future median maps"
    ),
)

print("Spatial-forecast figure written to:")
print(f"  {out_base}")

# %%
# Step 8 - Measure how the hotspot grows through time
# ---------------------------------------------------
# Let us compute a simple diagnostic outside the figure itself.
#
# We define "hotspot" as the top 10% of the median forecast in
# each future year. This is not required by the plotting backend,
# but it helps the reader see how to interpret the panels in a
# more quantitative way.

def hotspot_share(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for yy, g in df.groupby("coord_t", sort=True):
        vals = g["subsidence_q50"].to_numpy()
        thr = np.quantile(vals, 0.90)
        frac = float(np.mean(vals >= thr))
        mean_hot = float(np.mean(vals[vals >= thr]))

        rows.append(
            {
                "year": int(yy),
                "hotspot_threshold": float(thr),
                "hotspot_fraction": frac,
                "mean_hotspot_q50": mean_hot,
            }
        )

    return pd.DataFrame(rows)


print("")
print("Nansha hotspot summary:")
print(hotspot_share(future_a).to_string(index=False))

print("")
print("Zhongshan hotspot summary:")
print(hotspot_share(future_b).to_string(index=False))

# %%
# Step 9 - How to read the figure
# -------------------------------
# Here is the teaching logic I would want a user to follow when
# reading the spatial-forecast page.
#
# First column: observed map
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# This tells you what the spatial pattern actually looked like in
# the chosen reference year. Before you trust any forecast, you
# should understand this panel first. Where are the strongest
# zones? Is the pattern compact, diffuse, elongated, coastal, or
# multi-centered?
#
# Second column: predicted median for the same year
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is the fitted reconstruction. It is not yet the future.
# It is the model's spatial summary for the same year as the
# observation panel. The question here is:
#
# "Did the model learn the right geography?"
#
# If the hotspot moves to the wrong place, becomes too diffuse,
# or misses the major gradients, then the future panels should be
# read more cautiously.
#
# Remaining columns: future years
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Once the fitted reference-year panel is credible, the future
# columns become much more meaningful. Now we can ask:
#
# - Does the hotspot intensify?
# - Does it expand or contract?
# - Does a new secondary hotspot appear?
# - Does the regional gradient remain stable?
#
# In this synthetic lesson, the dominant hotspot remains in a
# similar location but strengthens over time. That is the kind
# of visual story a decision-maker or scientific reader can
# understand quickly.
#
# Why shared color scaling matters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The backend uses a global color range across all panels.
# Without that, a weak map and a strong map could look equally
# intense simply because each panel rescaled itself. Shared
# scaling makes spatial comparison honest. 
#
# What hotspot contours add
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# The script can also add hotspot contours using modes such as
# ``none``, ``absolute``, or ``delta`` together with a hotspot
# quantile threshold. That is useful when you want the reader to
# see the highest-risk zones immediately, without relying only on
# color perception. 

# %%
# Step 10 - Practical takeaway
# ----------------------------
# The strongest use of this figure is not merely "show me the
# future map."
#
# Its real strength is:
#
# - it grounds the forecast in a known observed year,
# - it tests whether the learned spatial structure is believable,
# - and it lets the user compare spatial evolution across
#   multiple forecast years in one visual frame.
#
# In other words, it teaches both forecast *content* and forecast
# *credibility*.

# %%
# Command-line version
# --------------------
# The same figure can be produced from the command line.
#
# The project CLI supports two ways to provide data:
#
# 1. artifact discovery from result folders such as
#    ``--ns-src`` and ``--zh-src``,
# 2. or direct CSV overrides using
#    ``--ns-calib`` / ``--ns-future`` and
#    ``--zh-calib`` / ``--zh-future``.
#
# It also supports:
#
# - ``--split`` with ``auto``, ``val``, or ``test``,
# - ``--year-val`` for the reference year,
# - ``--years-forecast`` for the future columns,
# - ``--cumulative`` and ``--subsidence-kind``,
# - ``--grid-res``, ``--clip``, and ``--cmap``,
# - hotspot controls such as ``--hotspot``,
#   ``--hotspot-quantile``, and ``--hotspot-out``.
#   
#
# A typical run from discovered artifacts looks like this:
#
# .. code-block:: bash
#
#    python -m scripts plot-spatial-forecasts \
#      --ns-src results/nansha_run \
#      --zh-src results/zhongshan_run \
#      --split auto \
#      --year-val 2022 \
#      --years-forecast 2025 2027 2030 \
#      --cumulative \
#      --subsidence-kind cumulative \
#      --grid-res 300 \
#      --clip 98 \
#      --cmap viridis \
#      --hotspot delta \
#      --hotspot-quantile 0.90 \
#      --show-title true \
#      --show-panel-titles true \
#      --out fig6-spatial-forecasts
#
# And if you want to bypass artifact discovery completely:
#
# .. code-block:: bash
#
#    python -m scripts plot-spatial-forecasts \
#      --ns-calib results/ns_calib.csv \
#      --ns-future results/ns_future.csv \
#      --zh-calib results/zh_calib.csv \
#      --zh-future results/zh_future.csv \
#      --year-val 2022 \
#      --years-forecast 2025 2027 2030 \
#      --hotspot delta \
#      --out fig6-spatial-forecasts
#
# The gallery page teaches the figure.
# The command line reproduces it in a workflow.