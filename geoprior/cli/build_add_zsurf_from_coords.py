# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""Build harmonized datasets enriched with surface elevation.

This command merges a main tabular dataset with a coordinate-to-
elevation lookup on rounded longitude/latitude pairs, then optionally
computes hydraulic head when a depth-below-ground-surface column is
available.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geoprior.cli._config import (
    add_city_arg,
    add_config_args,
    add_outdir_arg,
    bootstrap_runtime_config,
    ensure_outdir,
)


@dataclass(frozen=True)
class CityPaths:
    """Resolved input paths for one city."""

    city: str
    main_csv: Path
    elev_csv: Path


def _pair_items_to_map(
    items: list[str] | None,
    *,
    label: str,
) -> dict[str, Path]:
    """Parse repeated CITY=PATH items into a mapping."""
    out: dict[str, Path] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(
                f"Each {label} must be CITY=PATH. Got: {item!r}"
            )
        city, raw = item.split("=", 1)
        city = city.strip()
        if not city:
            raise SystemExit(
                f"Invalid empty city in {label}: {item!r}"
            )
        out[city] = Path(raw).expanduser().resolve()
    return out


def _append_suffix(path: Path, suffix: str) -> Path:
    """Append a suffix before the final extension."""
    if (
        suffix.endswith(".csv")
        and path.suffix.lower() == ".csv"
    ):
        stem = path.name[: -len(path.suffix)]
        return path.with_name(f"{stem}{suffix}")
    return path.with_name(f"{path.name}{suffix}")


def _round_coords(
    df: pd.DataFrame,
    *,
    lon_col: str,
    lat_col: str,
    decimals: int,
) -> pd.DataFrame:
    """Return a frame with rounded numeric coordinates."""
    out = df.copy()
    out[lon_col] = pd.to_numeric(
        out[lon_col],
        errors="coerce",
    ).round(decimals)
    out[lat_col] = pd.to_numeric(
        out[lat_col],
        errors="coerce",
    ).round(decimals)
    return out


def _resolve_city_paths(
    args: argparse.Namespace,
    cfg: dict[str, object],
) -> list[CityPaths]:
    """Resolve per-city main/elevation input paths."""
    main_map = _pair_items_to_map(
        args.main_csvs,
        label="--main-csv",
    )
    elev_map = _pair_items_to_map(
        args.elev_csvs,
        label="--elev-csv",
    )

    cities = list(args.cities or [])
    if not cities:
        shared = sorted(set(main_map) & set(elev_map))
        if shared:
            cities = shared

    if not cities:
        cfg_city = cfg.get("CITY_NAME")
        if isinstance(cfg_city, str) and cfg_city.strip():
            cities = [cfg_city.strip()]

    if not cities:
        raise SystemExit(
            "No city was resolved. Provide --city, or matched "
            "CITY=PATH pairs via --main-csv and --elev-csv."
        )

    data_root = None
    coords_root = None
    if args.data_root:
        data_root = (
            Path(args.data_root).expanduser().resolve()
        )
    elif isinstance(cfg.get("DATA_ROOT"), str):
        data_root = (
            Path(str(cfg["DATA_ROOT"])).expanduser().resolve()
        )

    if args.coords_root:
        coords_root = (
            Path(args.coords_root).expanduser().resolve()
        )
    elif isinstance(cfg.get("COORDS_ROOT"), str):
        coords_root = (
            Path(str(cfg["COORDS_ROOT"]))
            .expanduser()
            .resolve()
        )

    out: list[CityPaths] = []
    for city in cities:
        city = str(city).strip()
        if not city:
            continue

        main_csv = main_map.get(city)
        if main_csv is None:
            if data_root is None:
                raise SystemExit(
                    f"Missing main CSV for {city!r}. Provide "
                    "--main-csv city=PATH or --data-root."
                )
            main_csv = data_root / args.main_pattern.format(
                city=city,
            )

        elev_csv = elev_map.get(city)
        if elev_csv is None:
            if coords_root is None:
                raise SystemExit(
                    f"Missing elevation CSV for {city!r}. Provide "
                    "--elev-csv city=PATH or --coords-root."
                )
            elev_csv = coords_root / args.elev_pattern.format(
                city=city,
            )

        out.append(
            CityPaths(
                city=city,
                main_csv=main_csv,
                elev_csv=elev_csv,
            )
        )
    return out


def _reduce_elevation(
    elev: pd.DataFrame,
    *,
    lon_col: str,
    lat_col: str,
    elev_col: str,
    reducer: str,
) -> pd.DataFrame:
    """Deduplicate coordinate-elevation rows."""
    grouped = elev.groupby(
        [lon_col, lat_col],
        as_index=False,
    )[elev_col]
    if reducer == "mean":
        return grouped.mean()
    if reducer == "median":
        return grouped.median()
    if reducer == "first":
        return grouped.first()
    raise ValueError(
        "Unsupported reducer. Expected one of: "
        "mean, median, first."
    )


def _pick_depth_col(
    merged: pd.DataFrame,
    args: argparse.Namespace,
) -> str | None:
    """Choose the depth-bgs column used to compute head."""
    if args.depth_col:
        for col in args.depth_col:
            if col in merged.columns:
                return col
        return None

    for col in (
        "GWL_depth_bgs",
        "GWL_depth_bgs_m",
        "gwl_depth_bgs",
    ):
        if col in merged.columns:
            return col
    return None


def enrich_city_dataset(
    paths: CityPaths,
    *,
    lon_col: str,
    lat_col: str,
    elev_col: str,
    zsurf_col: str,
    round_decimals: int,
    reducer: str,
    compute_head: bool,
    head_col: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Merge one city dataset with its surface elevation lookup."""
    if not paths.main_csv.exists():
        raise FileNotFoundError(
            f"Main dataset not found: {paths.main_csv}"
        )
    if not paths.elev_csv.exists():
        raise FileNotFoundError(
            f"Elevation dataset not found: {paths.elev_csv}"
        )

    main = pd.read_csv(paths.main_csv)
    elev = pd.read_csv(paths.elev_csv)

    for frame, label in ((main, "main"), (elev, "elev")):
        missing = {lon_col, lat_col} - set(frame.columns)
        if missing:
            raise ValueError(
                f"{label} is missing coordinate columns: "
                f"{sorted(missing)}"
            )

    if elev_col not in elev.columns:
        raise ValueError(
            f"Elevation file is missing {elev_col!r}."
        )

    main_r = _round_coords(
        main,
        lon_col=lon_col,
        lat_col=lat_col,
        decimals=round_decimals,
    )
    elev_r = _round_coords(
        elev,
        lon_col=lon_col,
        lat_col=lat_col,
        decimals=round_decimals,
    )
    elev_r = _reduce_elevation(
        elev_r,
        lon_col=lon_col,
        lat_col=lat_col,
        elev_col=elev_col,
        reducer=reducer,
    )

    merged = main_r.merge(
        elev_r,
        on=[lon_col, lat_col],
        how="left",
        validate="m:1",
    )
    merged = merged.rename(columns={elev_col: zsurf_col})

    depth_col = None
    if compute_head:
        depth_col = _pick_depth_col(merged, args)
        if depth_col is not None:
            merged[head_col] = pd.to_numeric(
                merged[zsurf_col],
                errors="coerce",
            ) - pd.to_numeric(
                merged[depth_col],
                errors="coerce",
            )

    miss_rate = float(merged[zsurf_col].isna().mean())
    n_miss = int(merged[zsurf_col].isna().sum())
    n_rows = int(len(merged))
    diag = {
        "city": paths.city,
        "main_csv": str(paths.main_csv),
        "elev_csv": str(paths.elev_csv),
        "rows": n_rows,
        "missing_zsurf": n_miss,
        "missing_rate": miss_rate,
        "computed_head": depth_col is not None,
        "depth_col": depth_col,
        "round_decimals": round_decimals,
        "reducer": reducer,
    }
    return merged, diag


def _default_output_path(
    main_csv: Path,
    *,
    outdir: Path | None,
    out_suffix: str,
) -> Path:
    """Return the default output path for one city."""
    base = outdir or main_csv.parent
    target = base / main_csv.name
    return _append_suffix(target, out_suffix)


def build_add_zsurf_main(
    argv: list[str] | None = None,
) -> None:
    """CLI entry point for z_surf enrichment."""
    ap = argparse.ArgumentParser(
        prog="add-zsurf-from-coords",
        description=(
            "Merge coordinate elevation into harmonized main "
            "datasets and optionally compute hydraulic head."
        ),
    )
    add_config_args(ap)
    add_city_arg(
        ap,
        required=False,
        action="append",
        dest="cities",
        help=(
            "City name. Repeat to process multiple cities. "
            "When omitted, the command tries config CITY_NAME "
            "or the shared keys from --main-csv and --elev-csv."
        ),
    )
    ap.add_argument(
        "--data-root",
        "--data-dir",
        dest="data_root",
        help=(
            "Root directory used to resolve main dataset files "
            "through --main-pattern."
        ),
    )
    ap.add_argument(
        "--coords-root",
        dest="coords_root",
        help=(
            "Root directory used to resolve elevation lookup files "
            "through --elev-pattern."
        ),
    )
    ap.add_argument(
        "--main-pattern",
        default="{city}_final_main_std.harmonized.csv",
        help=(
            "Filename pattern under --data-root. Use {city} as "
            "the placeholder."
        ),
    )
    ap.add_argument(
        "--elev-pattern",
        default="{city}_coords_with_elevation.csv",
        help=(
            "Filename pattern under --coords-root. Use {city} as "
            "the placeholder."
        ),
    )
    ap.add_argument(
        "--main-csv",
        dest="main_csvs",
        action="append",
        help=(
            "Explicit CITY=PATH mapping for a main dataset CSV. "
            "Repeat as needed."
        ),
    )
    ap.add_argument(
        "--elev-csv",
        dest="elev_csvs",
        action="append",
        help=(
            "Explicit CITY=PATH mapping for an elevation lookup "
            "CSV. Repeat as needed."
        ),
    )
    add_outdir_arg(
        ap,
        required=False,
        help=(
            "Optional output directory. When omitted, each output "
            "is written next to its main dataset."
        ),
    )
    ap.add_argument(
        "--out-suffix",
        default=".with_zsurf.csv",
        help=(
            "Suffix appended to each input main CSV name when "
            "building output files."
        ),
    )
    ap.add_argument(
        "--summary-json",
        help=(
            "Optional JSON path for merge diagnostics across all "
            "processed cities."
        ),
    )
    ap.add_argument(
        "--longitude-col",
        "--lon-col",
        dest="lon_col",
        default="longitude",
        help="Longitude column name shared by both tables.",
    )
    ap.add_argument(
        "--latitude-col",
        "--lat-col",
        dest="lat_col",
        default="latitude",
        help="Latitude column name shared by both tables.",
    )
    ap.add_argument(
        "--elevation-col",
        dest="elev_col",
        default="elevation",
        help="Elevation column name in the lookup CSV.",
    )
    ap.add_argument(
        "--zsurf-col",
        default="z_surf",
        help="Output column name for surface elevation.",
    )
    ap.add_argument(
        "--head-col",
        default="head_m",
        help="Output hydraulic head column name.",
    )
    ap.add_argument(
        "--depth-col",
        action="append",
        help=(
            "Candidate depth-below-ground-surface column. Repeat "
            "to provide multiple fallback names."
        ),
    )
    ap.add_argument(
        "--round-decimals",
        type=int,
        default=6,
        help="Coordinate rounding decimals before merging.",
    )
    ap.add_argument(
        "--reducer",
        choices=("mean", "median", "first"),
        default="mean",
        help="Reducer used for duplicate lookup coordinates.",
    )
    ap.add_argument(
        "--no-head",
        action="store_true",
        help="Disable hydraulic-head computation.",
    )

    args = ap.parse_args(argv)
    cfg = bootstrap_runtime_config(args, field_map={})

    city_paths = _resolve_city_paths(args, cfg)
    outdir = None
    if args.outdir:
        outdir = ensure_outdir(args.outdir)

    diagnostics: list[dict[str, object]] = []
    written: list[str] = []
    for paths in city_paths:
        merged, diag = enrich_city_dataset(
            paths,
            lon_col=args.lon_col,
            lat_col=args.lat_col,
            elev_col=args.elev_col,
            zsurf_col=args.zsurf_col,
            round_decimals=args.round_decimals,
            reducer=args.reducer,
            compute_head=not args.no_head,
            head_col=args.head_col,
            args=args,
        )
        out_path = _default_output_path(
            paths.main_csv,
            outdir=outdir,
            out_suffix=args.out_suffix,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)

        diag["output_csv"] = str(out_path)
        diagnostics.append(diag)
        written.append(str(out_path))

        print(f"[{paths.city}] rows: {diag['rows']}")
        print(
            f"[{paths.city}] missing {args.zsurf_col}: "
            f"{diag['missing_zsurf']} "
            f"({diag['missing_rate']:.2%})"
        )
        if diag["computed_head"]:
            print(
                f"[{paths.city}] computed {args.head_col} "
                f"from {diag['depth_col']}"
            )
        else:
            print(f"[{paths.city}] head not computed")
        print(f"[{paths.city}] saved: {out_path}")

    if args.summary_json:
        summary_path = (
            Path(args.summary_json).expanduser().resolve()
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(diagnostics, indent=2),
            encoding="utf-8",
        )
        print(f"[OK] wrote summary: {summary_path}")

    print("[OK] completed z_surf enrichment for:")
    for item in written:
        print(f"  - {item}")


def main(argv: list[str] | None = None) -> None:
    """Command alias entry point."""
    build_add_zsurf_main(argv)


if __name__ == "__main__":
    main()
