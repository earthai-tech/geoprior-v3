# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/
# geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------
# Artifact export environment
# ---------------------------------------------------------
FIG_DIR_ENV = "GEOPRIOR_FIG_DIR"
OUT_DIR_ENV = "GEOPRIOR_OUT_DIR"
ROOT_ENV = "GEOPRIOR_ARTIFACT_ROOT"

# ---------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------


def _as_path(value: str | os.PathLike[str]) -> Path:
    """Return an expanded absolute path."""
    return Path(value).expanduser().resolve()


def _find_project_root(start: Path | None = None) -> Path:
    """Return the nearest project root."""

    here = (start or Path.cwd()).resolve()

    for root in (here, *here.parents):
        if (root / "pyproject.toml").exists():
            return root
        if (root / ".git").exists():
            return root

    return here


def scripts_root(*, create: bool = True) -> Path:
    """Return the user-facing artifact root."""

    root_env = os.environ.get(ROOT_ENV)
    if root_env:
        root = _as_path(root_env)
    else:
        root = _find_project_root() / "scripts"

    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def fig_dir(*, create: bool = True) -> Path:
    """Return the figure export directory."""

    value = os.environ.get(FIG_DIR_ENV)
    path = (
        _as_path(value)
        if value
        else scripts_root(create=create) / "figs"
    )

    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def out_dir(*, create: bool = True) -> Path:
    """Return the tabular export directory."""

    value = os.environ.get(OUT_DIR_ENV)
    path = (
        _as_path(value)
        if value
        else scripts_root(create=create) / "out"
    )

    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _has_explicit_parent(path: Path) -> bool:
    """Return True when path includes a parent part."""

    return path.parent != Path(".")


def resolve_user_artifact_path(
    value: str | os.PathLike[str],
    *,
    kind: str,
    create_parent: bool = False,
) -> Path:
    """
    Resolve a user-provided export path.

    Rules
    -----
    - absolute path: use as given
    - relative path with parent: respect that folder
    - bare filename/stem: place under scripts/figs or
      scripts/out
    """

    path = Path(value).expanduser()
    kind_norm = kind.strip().lower()

    if path.is_absolute():
        out = path
    elif _has_explicit_parent(path):
        out = path.resolve()
    elif kind_norm in {"fig", "figs", "figure", "figures"}:
        out = fig_dir(create=False) / path
    elif kind_norm == "out":
        out = out_dir(create=False) / path
    else:
        raise ValueError(
            "kind must be 'fig', 'figs', or 'out'"
        )

    if create_parent:
        out.parent.mkdir(parents=True, exist_ok=True)

    return out


def resolve_export_path(
    kind: str,
    *parts: str | os.PathLike[str],
    create_parent: bool = True,
) -> Path:
    """Resolve a path under fig/ or out/."""

    path = resolve_user_artifact_path(
        Path(*map(str, parts)),
        kind=kind,
        create_parent=create_parent,
    )
    return path


def export_env_summary() -> dict[str, str]:
    """Return resolved export environment values."""

    return {
        ROOT_ENV: str(scripts_root(create=False)),
        FIG_DIR_ENV: str(fig_dir(create=False)),
        OUT_DIR_ENV: str(out_dir(create=False)),
    }


# ---------------------------------------------------------
# Filesystem layout (v3.2)
# ---------------------------------------------------------
MODULE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = MODULE_DIR
ARTIFACT_ROOT = scripts_root()
FIG_DIR = fig_dir()
FIGS_DIR = FIG_DIR
OUT_DIR = out_dir()

# ---------------------------------------------------------
# Common discovery patterns
# (src may be a directory or a file)
# ---------------------------------------------------------
# NOTE:
# - We search recursively under src when src is a
#   directory.
# - For GeoPrior eval JSON we prefer the
#   interpretable one when available.
# - It may carry human units such as "mm".
PATTERNS = {
    # GeoPrior evaluation JSON
    "phys_json": (
        "geoprior_eval_phys_*_interpretable.json",
        "geoprior_eval_phys_*.json",
    ),
    # Eval diagnostics fallback JSON
    "eval_diag_json": (
        "*GeoPriorSubsNet_eval_diagnostics*_calibrated.json",
        "*GeoPriorSubsNet_eval_diagnostics*.json",
        "eval_diagnostics*.json",
        "diagnostics*.json",
        "*eval*diagnostic*.json",
    ),
    # --- TestSet: EVAL
    # (has subsidence_actual) ---
    "forecast_test_csv": (
        "*forecast*TestSet*eval_calibrated*.csv",
        "*forecast_TestSet*_eval_calibrated.csv",
        "*_forecast_TestSet_*_eval_calibrated.csv",
        "*TestSet*eval_calibrated*.csv",
    ),
    # --- TestSet: FUTURE
    # (no subsidence_actual) ---
    "forecast_test_future_csv": (
        "*forecast*TestSet*future*.csv",
        "*forecast_TestSet*_future.csv",
        "*_forecast_TestSet_*_future.csv",
        "*TestSet*future*.csv",
    ),
    # Typical calibrated validation forecasts
    "forecast_val_csv": (
        "*forecast*Validation*calibrated*.csv",
        "*forecast*Validation*Fallback*calibrated*.csv",
        "*Validation*calibrated*.csv",
        "*_calibrated.csv",
    ),
    # Typical future forecasts
    "forecast_future_csv": (
        "*future*.csv",
        "*H*_future*.csv",
        "*forecast*future*.csv",
    ),
    # Physics payloads
    "physics_payload": (
        "*physics_payload*.npz",
        "*phys_payload*.npz",
        "*payload*.npz",
    ),
    # Optional coords file
    # (used by some physics maps)
    "coords_npz": (
        "*coords*.npz",
        "*xy*.npz",
        "*lonlat*.npz",
        "*val_inputs*.npz",
        "*test_inputs*.npz",
        "*train_inputs*.npz",
        "*oos_time_inputs*.npz",
    ),
    # Ablation records (Supplement S6/S7)
    "ablation_record_jsonl": (
        "ablation_records/ablation_record.updated*.jsonl",
        "ablation_records/ablation_record*.jsonl",
        "ablation_record*.jsonl",
        "ablation_record.updated*.jsonl",
    ),
    "boundary_shp": (
        "*boundary*.shp",
        "*coast*.shp",
        "*admin*.shp",
        "*outline*.shp",
        "*border*.shp",
    ),
}

# ---------------------------------------------------------
# Plot metric metadata
# (titles / labels / format)
# Used by multi-panel figures to avoid duplicated
# strings.
# ---------------------------------------------------------
PLOT_METRIC_META = {
    "r2": {
        "title": r"$R^2$ (↑)",
        "ylabel": r"$R^2$",
        "fmt": "{:.2f}",
    },
    "mae": {
        "title": "MAE (↓, {unit})",
        "ylabel": "MAE ({unit})",
        "fmt": "{:.2f}",
        "unit": "mm",
    },
    "rmse": {
        "title": "RMSE (↓, {unit})",
        "ylabel": "RMSE ({unit})",
        "fmt": "{:.2f}",
        "unit": "mm",
    },
    "mse": {
        "title": "MSE (↓, {unit})",
        "ylabel": "MSE ({unit})",
        "fmt": "{:.2f}",
        "unit": "mm²",
    },
    "coverage80": {
        "title": "Coverage (80% PI; target 0.80)",
        "ylabel": "Coverage",
        "fmt": "{:.3f}",
    },
    "sharpness80": {
        "title": "Sharpness (80% PI; ↓, {unit})",
        "ylabel": "Sharpness ({unit})",
        "fmt": "{:.3f}",
        "unit": "mm",
    },
}

# ---------------------------------------------------------
# Matplotlib defaults (paper-friendly)
# ---------------------------------------------------------
PAPER_DPI = 600
PAPER_FONT = 8

# ---------------------------------------------------------
# City styling
# (keep consistent across all scripts)
# (Used repeatedly across figures)
# ---------------------------------------------------------
CITY_CANON = {
    "nansha": "Nansha",
    "zhongshan": "Zhongshan",
    "ns": "Nansha",
    "zh": "Zhongshan",
}

CITY_COLORS = {
    "Nansha": "#1F78B4",
    "Zhongshan": "#E31A1C",
}

# ---------------------------------------------------------
# Units: payload + drivers + targets
# ---------------------------------------------------------
# Keep *units* separate from *labels* so plots can
# do:
#   label = LABELS[key]
#   unit = UNITS.get(key)
#   ax.set_xlabel(
#       f"{label} ({unit})" if unit else label
#   )
UNITS = {
    # Drivers / coordinates
    "GWL": "m",
    "rainfall_mm": "mm/yr",
    "soil_thickness": "m",
    "normalized_urban_load_proxy": "1",
    "longitude": "deg",
    "latitude": "deg",
    "year": "year",
    # Targets
    "subsidence": "mm/yr",
    "subsidence_cum": "mm",
    # Forecast CSV canonical columns
    "forecast_step": "step",
    "sample_idx": "id",
}

LABELS = {
    "GWL": "Groundwater level",
    "rainfall_mm": "Rainfall",
    "soil_thickness": "Soil thickness",
    "normalized_urban_load_proxy": "Urban load",
    "longitude": "Longitude",
    "latitude": "Latitude",
    "year": "Year",
    "subsidence": "Subsidence",
    "subsidence_cum": "Cumulative subsidence",
    "lithology_class": "Lithology class",
    "city": "City",
    "forecast_step": "Forecast step",
    "sample_idx": "Sample id",
}

UNITS.update(
    {
        "GWL_depth_bgs_m": "m",
        "GWL_depth_bgs": "m",
        "GWL_depth_bgs_raw": "m",
        "soil_thickness_eff": "m",
        "soil_thickness_imputed": "m",
        "urban_load_global": "1",
        "head_m": "m",
        "z_surf_m": "m",
        "z_surf": "m",
    }
)

LABELS.update(
    {
        "GWL_depth_bgs_m": "Depth to water table",
        "GWL_depth_bgs": "Depth to water table",
        "GWL_depth_bgs_raw": "Depth to water table",
        "soil_thickness_eff": "Effective soil thickness",
        "soil_thickness_imputed": "Imputed soil thickness",
        "urban_load_global": "Urban load",
        "head_m": "Hydraulic head",
        "z_surf_m": "Surface elevation",
        "z_surf": "Surface elevation",
    }
)

COLUMN_ALIASES = {
    # Drivers
    "GWL_depth_bgs_m": (
        "GWL_depth_bgs_m",
        "GWL_depth_bgs",
        "GWL",
        "GWL_depth_bgs_raw",
    ),
    "rainfall_mm": (
        "rainfall_mm",
        "rainfall",
        "rainfall_m",
    ),
    "soil_thickness_eff": (
        "soil_thickness_eff",
        "soil_thickness_imputed",
        "soil_thickness",
    ),
    "urban_load_global": (
        "urban_load_global",
        "normalized_urban_load_proxy",
    ),
    # Responses
    "subsidence_cum": ("subsidence_cum",),
    "subsidence": ("subsidence",),
    "head_m": ("head_m",),
}

DRIVER_RESPONSE_DEFAULT_DRIVERS = (
    "GWL_depth_bgs_m",
    "rainfall_mm",
    "soil_thickness_eff",
    "urban_load_global",
)

DRIVER_RESPONSE_DEFAULT_RESPONSE = "subsidence_cum"

# ---------------------------------------------------------
# Units: GeoPrior physics fields / diagnostics
# ---------------------------------------------------------
# Used for physics maps and sanity plots.
PHYS_UNITS = {
    "K": "m/s",
    "Ss": "1/m",
    "Hd": "m",
    "H": "m",
    "tau": "s",
    "tau_prior": "s",
    "log10_K": "log10(m/s)",
    "log10_Ss": "log10(1/m)",
    "log10_tau": "log10(s)",
    # Diagnostics
    "epsilon_prior": "1",
    "epsilon_cons": "1",
    "epsilon_gw": "1",
    "epsilon_cons_raw": "varies",
    "epsilon_gw_raw": "varies",
}

PHYS_LABELS = {
    "K": "Hydraulic conductivity",
    "Ss": "Specific storage",
    "Hd": "Drainage thickness",
    "H": "Effective thickness",
    "tau": "Consolidation time scale",
    "tau_prior": "Prior time scale",
    "log10_K": "log10 K",
    "log10_Ss": "log10 Ss",
    "log10_tau": "log10 τ",
    "epsilon_prior": r"$\epsilon_{\mathrm{prior}}$",
    "epsilon_cons": r"$\epsilon_{\mathrm{cons}}$",
    "epsilon_gw": r"$\epsilon_{\mathrm{gw}}$",
    "epsilon_cons_raw": (r"$\epsilon_{\mathrm{cons,raw}}$"),
    "epsilon_gw_raw": r"$\epsilon_{\mathrm{gw,raw}}$",
}

# ---------------------------------------------------------
# Time + scaling
# (used to harmonize JSON variants)
# ---------------------------------------------------------
SECONDS_PER_YEAR = 31556952.0

# ---------------------------------------------------------
# Closure strings for captions / legends
# (math "single source")
# ---------------------------------------------------------
CLOSURES = {
    "tau_prior": (
        r"$\tau_{\mathrm{prior}}"
        r"\approx H_d^2\,S_s"
        r"/(\pi^2\,\kappa_b\,K)$"
    ),
    "s_eq": r"$s_{\mathrm{eq}}\approx S_s\,\Delta h\,H$",
    "R_cons": (
        r"$R_{\mathrm{cons}}="
        r"\partial_t \bar{s}"
        r"-(s_{\mathrm{eq}}(\bar{h})-\bar{s})/\tau$"
    ),
    "R_gw": (
        r"$R_{\mathrm{gw}}="
        r"S_s\,\partial_t \bar{h}"
        r"-\nabla\cdot(K\nabla\bar{h})$"
    ),
}

# ---------------------------------------------------------
# Column canonicalization
# ---------------------------------------------------------
_BASE_ALIASES = {
    "sample_idx": ("sample_idx", "sample_id", "sample"),
    "forecast_step": ("forecast_step", "forecast_s", "h"),
    "coord_t": ("coord_t", "year", "t"),
    "coord_x": ("coord_x", "lon", "longitude", "x"),
    "coord_y": ("coord_y", "lat", "latitude", "y"),
    "subsidence_actual": ("subsidence_actual", "actual"),
    "subsidence_q50": ("subsidence_q50", "q50", "p50"),
    "subsidence_unit": ("subsidence_unit", "unit"),
}

_CALIB_REQUIRED = [
    "sample_idx",
    "forecast_step",
    "coord_t",
    "coord_x",
    "coord_y",
    "subsidence_actual",
    "subsidence_q50",
]

_FUT_REQUIRED = [
    "sample_idx",
    "forecast_step",
    "coord_t",
    "coord_x",
    "coord_y",
    "subsidence_q50",
]

_CAL_ORDER = ("none", "source", "target")

_CAL_LABEL = {
    "none": "None",
    "source": "Source",
    "target": "Target",
}

_CAL_MARKER = {
    "none": "o",
    "source": "^",
    "target": "s",
}

_STRAT_DEFAULT = ("baseline", "xfer", "warm")

_STRAT_LABEL = {
    "baseline": "Baseline (target)",
    "xfer": "Transfer",
    "warm": "Warm-start",
}

_STRAT_HATCH = {
    "baseline": "//",
    "xfer": "",
    "warm": "xx",
}

_STRAT_EDGE = {
    "baseline": "#111111",
    "xfer": "#111111",
    "warm": "#111111",
}

_STRAT_LINESTYLE = {
    "baseline": ":",
    "xfer": "--",
    "warm": "-",
}

_STRAT_MARKER = {
    "baseline": "o",
    "xfer": "^",
    "warm": "s",
}

_BASELINE_MAP = {
    "A_to_B": "B_to_B",
    "B_to_A": "A_to_A",
}

_METRIC_DEF = {
    "mae": ("overall_mae", "MAE (mm)", "min"),
    "mse": ("overall_mse", "MSE (mm^2)", "min"),
    "rmse": ("overall_rmse", "RMSE (mm)", "min"),
    "r2": ("overall_r2", r"$R^2$", "max"),
}
