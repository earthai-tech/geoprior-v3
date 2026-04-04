# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>
r"""CLI for initializing GeoPrior configuration files."""

from __future__ import annotations

import argparse
from pathlib import Path
from string import Template

from ..utils.nat_utils import (
    ensure_config_json,
    get_config_paths,
    get_natcom_dir,
)

_DEFAULT_TEMPLATE = """# SPDX-License-Identifier: Apache-2.0
# Auto-generated NATCOM config for GeoPrior-v3.
# Edit values here, then rerun:
#   geoprior-init
# or
#   python -m geoprior.cli init-config

# ---------------------------------------------------------
# 1) Dataset identity and file naming
# ---------------------------------------------------------
CITY_NAME = "$CITY_NAME"
MODEL_NAME = "$MODEL_NAME"
DATA_DIR = "$DATA_DIR"
DATASET_VARIANT = "$DATASET_VARIANT"

# Input file templates. These are resolved into BIG_FN and
# SMALL_FN by geoprior.utils.nat_utils.
BIG_FN_TEMPLATE = "{city}_${BIG_STEM}_{variant}.csv"
SMALL_FN_TEMPLATE = "{city}_${SMALL_STEM}_{variant}.csv"

BIG_FN = BIG_FN_TEMPLATE.format(
    city=CITY_NAME,
    variant=DATASET_VARIANT,
)
SMALL_FN = SMALL_FN_TEMPLATE.format(
    city=CITY_NAME,
    variant=DATASET_VARIANT,
)

ALL_CITIES_PARQUET = "$ALL_CITIES_PARQUET"

# ---------------------------------------------------------
# 2) Time layout
# ---------------------------------------------------------
TRAIN_END_YEAR = $TRAIN_END_YEAR
FORECAST_START_YEAR = $FORECAST_START_YEAR
FORECAST_HORIZON_YEARS = $FORECAST_HORIZON_YEARS
TIME_STEPS = $TIME_STEPS
MODE = "$MODE"

# ---------------------------------------------------------
# 3) Required columns
# ---------------------------------------------------------
TIME_COL = "$TIME_COL"
LON_COL = "$LON_COL"
LAT_COL = "$LAT_COL"
SUBSIDENCE_COL = "$SUBSIDENCE_COL"
GWL_COL = "$GWL_COL"

# Optional extra columns used by some workflows
H_FIELD_COL_NAME = "$H_FIELD_COL_NAME"
Z_SURF_COL = "$Z_SURF_COL"
HEAD_COL = "$HEAD_COL"
INCLUDE_Z_SURF_AS_STATIC = $INCLUDE_Z_SURF_AS_STATIC

# ---------------------------------------------------------
# 4) Physical interpretation of GWL
# ---------------------------------------------------------
GWL_KIND = "$GWL_KIND"
GWL_SIGN = "$GWL_SIGN"
USE_HEAD_PROXY = $USE_HEAD_PROXY

# ---------------------------------------------------------
# 5) Feature groups
# ---------------------------------------------------------
FUTURE_DRIVER_FEATURES = [
$FUTURE_DRIVER_FEATURES
]

OPTIONAL_NUMERIC_FEATURES = [
$OPTIONAL_NUMERIC_FEATURES
]

# ---------------------------------------------------------
# 6) Forecast / model defaults
# ---------------------------------------------------------
QUANTILES = [0.10, 0.50, 0.90]
PDE_MODE_CONFIG = "$PDE_MODE_CONFIG"
NORMALIZE_COORDS = $NORMALIZE_COORDS
KEEP_COORDS_RAW = $KEEP_COORDS_RAW
SCALE_GWL = $SCALE_GWL
SCALE_H_FIELD = $SCALE_H_FIELD
SCALE_Z_SURF = $SCALE_Z_SURF

# ---------------------------------------------------------
# 7) Training defaults
# ---------------------------------------------------------
BATCH_SIZE = $BATCH_SIZE
EPOCHS = $EPOCHS
LEARNING_RATE = $LEARNING_RATE
HIDDEN_UNITS = $HIDDEN_UNITS
LSTM_UNITS = [$LSTM_UNITS]
ATTENTION_UNITS = $ATTENTION_UNITS
NUMBER_HEADS = $NUMBER_HEADS
DROPOUT_RATE = $DROPOUT_RATE
USE_BATCH_NORM = $USE_BATCH_NORM
USE_VSN = $USE_VSN

# ---------------------------------------------------------
# 8) Scaling / bookkeeping
# ---------------------------------------------------------
SCALES = {
    "subsidence": "$SUBS_SCALE",
    "gwl": "$GWL_SCALE",
}
AUDIT_STAGES = ["stage1", "stage2", "stage3"]
VERBOSE = 1
"""

_PROMPTS = (
    ("CITY_NAME", "City name", "nansha"),
    (
        "MODEL_NAME",
        "Model name",
        "GeoPriorSubsNet",
    ),
    ("DATA_DIR", "Data dir", "."),
    (
        "DATASET_VARIANT",
        "Dataset variant",
        "with_zsurf",
    ),
    (
        "ALL_CITIES_PARQUET",
        "All-cities parquet",
        "",
    ),
    (
        "TRAIN_END_YEAR",
        "Train end year",
        "2022",
    ),
    (
        "FORECAST_START_YEAR",
        "Forecast start year",
        "2023",
    ),
    (
        "FORECAST_HORIZON_YEARS",
        "Forecast horizon years",
        "3",
    ),
    ("TIME_STEPS", "Input time steps", "5"),
    ("MODE", "Mode", "tft_like"),
    ("TIME_COL", "Time column", "year"),
    ("LON_COL", "Longitude column", "longitude"),
    ("LAT_COL", "Latitude column", "latitude"),
    (
        "SUBSIDENCE_COL",
        "Subsidence column",
        "subsidence_cum",
    ),
    (
        "GWL_COL",
        "GWL column",
        "GWL_depth_bgs_m",
    ),
    (
        "H_FIELD_COL_NAME",
        "H-field thickness column",
        "soil_thickness",
    ),
    ("Z_SURF_COL", "Surface elev. col", "z_surf_m"),
    ("HEAD_COL", "Head column", "head_m"),
    (
        "INCLUDE_Z_SURF_AS_STATIC",
        "Include z_surf as static [True/False]",
        "True",
    ),
    ("GWL_KIND", "GWL kind", "depth_bgs"),
    (
        "GWL_SIGN",
        "GWL sign",
        "down_positive",
    ),
    (
        "USE_HEAD_PROXY",
        "Use head proxy [True/False]",
        "False",
    ),
    (
        "PDE_MODE_CONFIG",
        "PDE mode",
        "on",
    ),
    (
        "NORMALIZE_COORDS",
        "Normalize coords [True/False]",
        "True",
    ),
    (
        "KEEP_COORDS_RAW",
        "Keep raw coords [True/False]",
        "True",
    ),
    (
        "SCALE_GWL",
        "Scale GWL [True/False]",
        "False",
    ),
    (
        "SCALE_H_FIELD",
        "Scale H-field [True/False]",
        "False",
    ),
    (
        "SCALE_Z_SURF",
        "Scale z_surf [True/False]",
        "False",
    ),
    ("BATCH_SIZE", "Batch size", "32"),
    ("EPOCHS", "Epochs", "100"),
    (
        "LEARNING_RATE",
        "Learning rate",
        "0.001",
    ),
    ("HIDDEN_UNITS", "Hidden units", "128"),
    (
        "LSTM_UNITS",
        "LSTM units (csv)",
        "128,64",
    ),
    (
        "ATTENTION_UNITS",
        "Attention units",
        "128",
    ),
    ("NUMBER_HEADS", "Attention heads", "4"),
    (
        "DROPOUT_RATE",
        "Dropout rate",
        "0.10",
    ),
    (
        "USE_BATCH_NORM",
        "Use batch norm [True/False]",
        "False",
    ),
    ("USE_VSN", "Use VSN [True/False]", "True"),
    (
        "SUBS_SCALE",
        "Subsidence scale",
        "auto",
    ),
    ("GWL_SCALE", "GWL scale", "auto"),
)

_DRIVER_DEFAULTS = (
    "rainfall",
    "u_star",
    "h_eff",
)

_OPTIONAL_NUMERIC_DEFAULTS = (
    "building_density",
    "seismic_count",
    "seismic_risk_score",
)


def _parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="init-config",
        description=(
            "Create nat.com/config.py "
            "for the GeoPrior pipeline."
        ),
    )
    p.add_argument(
        "--root",
        default="nat.com",
        help="Config folder relative to project root.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing config.py.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Use defaults for missing answers.",
    )
    p.add_argument(
        "--template",
        default=None,
        help=(
            "Optional template file. "
            "Falls back to built-in template."
        ),
    )
    p.add_argument("--city", default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument(
        "--dataset-variant",
        default=None,
    )
    p.add_argument(
        "--train-end-year",
        type=int,
        default=None,
    )
    p.add_argument(
        "--forecast-start-year",
        type=int,
        default=None,
    )
    p.add_argument(
        "--forecast-horizon-years",
        type=int,
        default=None,
    )
    p.add_argument(
        "--time-steps",
        type=int,
        default=None,
    )
    return p.parse_args(argv)


def _ask(
    label: str,
    default: str,
    *,
    yes: bool,
) -> str:
    if yes:
        return default

    raw = input(f"{label} [{default}]: ").strip()
    return raw or default


def _as_bool_literal(value: str) -> str:
    return "True" if str(value).lower() == "true" else "False"


def _quote_list(items: tuple[str, ...]) -> str:
    return ",\n".join(f'    "{item}"' for item in items)


def _normalize_cli(
    args: argparse.Namespace,
    values: dict[str, str],
) -> None:
    cli_map = {
        "city": "CITY_NAME",
        "model_name": "MODEL_NAME",
        "data_dir": "DATA_DIR",
        "dataset_variant": "DATASET_VARIANT",
        "train_end_year": "TRAIN_END_YEAR",
        "forecast_start_year": "FORECAST_START_YEAR",
        "forecast_horizon_years": ("FORECAST_HORIZON_YEARS"),
        "time_steps": "TIME_STEPS",
    }

    for src, dst in cli_map.items():
        value = getattr(args, src, None)
        if value is not None:
            values[dst] = str(value)


def _resolve_template(
    args: argparse.Namespace,
    project_root: Path,
) -> str:
    candidates: list[Path] = []

    if args.template:
        candidates.append(Path(args.template))

    candidates.extend(
        [
            project_root
            / "scripts"
            / "templates"
            / "natcom_config_template.py",
            project_root
            / "geoprior"
            / "resources"
            / "natcom_config_template.py",
            project_root / "nat.com" / "config.template.py",
        ]
    )

    for path in candidates:
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if "$CITY_NAME" in text:
                return text

    return _DEFAULT_TEMPLATE


def _build_values(
    args: argparse.Namespace,
) -> dict[str, str]:
    values: dict[str, str] = {}

    for key, label, default in _PROMPTS:
        values[key] = _ask(
            label,
            default,
            yes=args.yes,
        )

    _normalize_cli(args, values)

    values["CITY_NAME"] = values["CITY_NAME"].strip().lower()
    values["USE_HEAD_PROXY"] = _as_bool_literal(
        values["USE_HEAD_PROXY"]
    )
    values["INCLUDE_Z_SURF_AS_STATIC"] = _as_bool_literal(
        values["INCLUDE_Z_SURF_AS_STATIC"]
    )
    values["NORMALIZE_COORDS"] = _as_bool_literal(
        values["NORMALIZE_COORDS"]
    )
    values["KEEP_COORDS_RAW"] = _as_bool_literal(
        values["KEEP_COORDS_RAW"]
    )
    values["SCALE_GWL"] = _as_bool_literal(
        values["SCALE_GWL"]
    )
    values["SCALE_H_FIELD"] = _as_bool_literal(
        values["SCALE_H_FIELD"]
    )
    values["SCALE_Z_SURF"] = _as_bool_literal(
        values["SCALE_Z_SURF"]
    )
    values["USE_BATCH_NORM"] = _as_bool_literal(
        values["USE_BATCH_NORM"]
    )
    values["USE_VSN"] = _as_bool_literal(values["USE_VSN"])

    values["BIG_STEM"] = "big"
    values["SMALL_STEM"] = "small"

    if not values["ALL_CITIES_PARQUET"].strip():
        values["ALL_CITIES_PARQUET"] = ""

    values["FUTURE_DRIVER_FEATURES"] = _quote_list(
        _DRIVER_DEFAULTS
    )
    values["OPTIONAL_NUMERIC_FEATURES"] = _quote_list(
        _OPTIONAL_NUMERIC_DEFAULTS
    )

    return values


def _write_config(
    text: str,
    *,
    root: str,
    force: bool,
) -> Path:
    nat_dir = Path(get_natcom_dir(root=root))
    nat_dir.mkdir(parents=True, exist_ok=True)

    cfg_path_str, _ = get_config_paths(root=root)
    cfg_path = Path(cfg_path_str)

    if cfg_path.exists() and not force:
        raise FileExistsError(
            f"{cfg_path} already exists. "
            "Use --force to overwrite it."
        )

    cfg_path.write_text(text, encoding="utf-8")
    return cfg_path


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    nat_dir = Path(get_natcom_dir(root=args.root))
    project_root = nat_dir.parent

    values = _build_values(args)
    template_text = _resolve_template(
        args,
        project_root,
    )
    rendered = Template(template_text).substitute(values)

    try:
        cfg_path = _write_config(
            rendered,
            root=args.root,
            force=args.force,
        )
    except FileExistsError as exc:
        print(f"[OK] {exc}")
        _, json_path = ensure_config_json(root=args.root)
        print(
            f"[OK] Reused existing config and refreshed: "
            f"{json_path}"
        )
        return

    _, json_path = ensure_config_json(root=args.root)

    print(f"[OK] Created: {cfg_path}")
    print(f"[OK] Created: {json_path}")
    print("")
    print("Next commands:")
    print("  geoprior-run preprocess")
    print("  geoprior-run train")
    print("  geoprior-run tune")
    print("  geoprior-run infer --help")
    print("  geoprior-run transfer --help")
    print("")
    print("  # or")
    print("  python -m geoprior.cli preprocess")


if __name__ == "__main__":
    main()
