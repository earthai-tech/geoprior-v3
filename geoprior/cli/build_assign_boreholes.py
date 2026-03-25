# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""Assign boreholes to the nearest city point cloud.

This command classifies each borehole row against one or more city
coordinate clouds built from Stage-1 processed CSV files.

Supported city sources
----------------------
1. Explicit processed CSVs via ``--city-csv CITY=PATH``.
2. Explicit Stage-1 directories via ``--city-stage1 CITY=DIR``.
3. Plain Stage-1 directories via repeated ``--stage1-dir DIR`` with
   automatic city-name inference.
4. Repeated city names via ``--cities ...`` together with
   ``--results-dir`` and optionally ``--model``.

Outputs
-------
- one classified CSV containing all boreholes and distance columns
- optional per-city split CSVs
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ._config import (
    add_config_args,
    add_model_arg,
    add_outdir_arg,
    add_results_dir_arg,
    bootstrap_runtime_config,
    ensure_outdir,
)


def _parse_named_path(
    item: str,
    *,
    option: str,
) -> tuple[str, Path]:
    """Parse ``NAME=PATH`` CLI items."""
    if "=" not in item:
        raise SystemExit(
            f"{option} expects CITY=PATH. Got: {item!r}"
        )
    name, raw_path = item.split("=", 1)
    city = name.strip()
    if not city:
        raise SystemExit(
            f"{option} received an empty city name."
        )
    path = Path(raw_path).expanduser().resolve()
    return city, path


def _infer_city_from_proc_name(path: Path) -> str | None:
    """Infer city name from a processed CSV filename."""
    stem = path.stem
    pats = (
        r"^(?P<city>.+?)_\d+_\d+_proc$",
        r"^(?P<city>.+?)_proc$",
    )
    for pat in pats:
        m = re.match(pat, stem)
        if m:
            return str(m.group("city"))
    return None


def _find_proc_csv(
    stage1_dir: Path,
    *,
    city: str | None = None,
) -> Path:
    """Find the processed CSV under a Stage-1 directory."""
    root = stage1_dir.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(
            f"Stage-1 directory not found: {root}"
        )

    cands = sorted(root.glob("*_proc.csv"))
    if not cands:
        cands = sorted(root.rglob("*_proc.csv"))
    if not cands:
        raise FileNotFoundError(
            "Could not find any '*_proc.csv' file under "
            f"{root}."
        )

    if city:
        pref = f"{city}_"
        named = [p for p in cands if p.name.startswith(pref)]
        if len(named) == 1:
            return named[0]
        if len(named) > 1:
            raise RuntimeError(
                "Multiple processed CSVs matched city "
                f"{city!r} under {root}."
            )

    if len(cands) == 1:
        return cands[0]

    inferred = []
    for cand in cands:
        tag = _infer_city_from_proc_name(cand)
        inferred.append((cand, tag))

    valid = [(p, t) for p, t in inferred if t]
    if city and valid:
        matched = [p for p, t in valid if t == city]
        if len(matched) == 1:
            return matched[0]

    raise RuntimeError(
        "Multiple processed CSVs were found under "
        f"{root}. Pass --city-stage1 CITY=DIR or "
        "--city-csv CITY=PATH to disambiguate."
    )


def _infer_city_from_stage1_dir(
    stage1_dir: Path,
    *,
    model: str | None = None,
) -> str:
    """Infer a city name from a Stage-1 directory."""
    proc = None
    try:
        proc = _find_proc_csv(stage1_dir)
    except Exception:
        proc = None
    if proc is not None:
        city = _infer_city_from_proc_name(proc)
        if city:
            return city

    name = stage1_dir.expanduser().resolve().name
    if model and name.endswith(f"_{model}_stage1"):
        return name[: -len(f"_{model}_stage1")]
    if name.endswith("_stage1"):
        return name[: -len("_stage1")]
    return name


def _resolve_stage1_dir_for_city(
    city: str,
    *,
    results_dir: str | None,
    model: str | None,
) -> Path:
    """Resolve a Stage-1 directory for one city."""
    if not results_dir:
        raise FileNotFoundError(
            "--cities requires --results-dir when a Stage-1 "
            "directory is not given explicitly."
        )

    root = Path(results_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"Results directory not found: {root}"
        )

    if model:
        direct = root / f"{city}_{model}_stage1"
        if direct.exists() and direct.is_dir():
            return direct

    cands = sorted(root.glob(f"{city}_*_stage1"))
    dirs = [p for p in cands if p.is_dir()]
    if len(dirs) == 1:
        return dirs[0]
    if not dirs:
        raise FileNotFoundError(
            "Could not resolve a Stage-1 directory for city "
            f"{city!r} under {root}."
        )
    raise RuntimeError(
        f"Multiple Stage-1 directories matched city {city!r}: "
        + ", ".join(str(p.name) for p in dirs)
    )


def _resolve_borehole_csv(
    args: argparse.Namespace,
    cfg: dict,
) -> Path:
    """Resolve the borehole CSV path."""
    raw = args.borehole_csv
    if raw:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Borehole CSV not found: {path}"
            )
        return path

    for key in (
        "BOREHOLE_PUMPING_VALIDATION_CSV",
        "EXTERNAL_VALIDATION_CSV",
        "VALIDATION_CSV",
    ):
        val = cfg.get(key)
        if isinstance(val, str) and val.strip():
            path = Path(val).expanduser().resolve()
            if path.exists():
                return path

    raise SystemExit(
        "Missing borehole table. Pass --borehole-csv or set one of "
        "BOREHOLE_PUMPING_VALIDATION_CSV / EXTERNAL_VALIDATION_CSV / "
        "VALIDATION_CSV in config."
    )


def _collect_city_sources(
    args: argparse.Namespace,
    cfg: dict,
) -> list[tuple[str, Path]]:
    """Collect city names and processed CSV paths."""
    out: list[tuple[str, Path]] = []
    seen: dict[str, Path] = {}

    def add(city: str, path: Path) -> None:
        city0 = str(city).strip()
        path0 = path.expanduser().resolve()
        if not path0.exists():
            raise FileNotFoundError(
                f"City CSV not found for {city0!r}: {path0}"
            )
        prev = seen.get(city0)
        if prev is not None and prev != path0:
            raise RuntimeError(
                f"City {city0!r} was provided multiple times with "
                "different sources."
            )
        seen[city0] = path0

    for item in args.city_csvs or []:
        city, path = _parse_named_path(
            item,
            option="--city-csv",
        )
        add(city, path)

    for item in args.city_stage1s or []:
        city, stage1_dir = _parse_named_path(
            item,
            option="--city-stage1",
        )
        add(city, _find_proc_csv(stage1_dir, city=city))

    model = args.model or cfg.get("MODEL_NAME")
    for raw in args.stage1_dirs or []:
        stage1_dir = Path(raw).expanduser().resolve()
        city = _infer_city_from_stage1_dir(
            stage1_dir,
            model=model,
        )
        add(city, _find_proc_csv(stage1_dir, city=city))

    results_dir = args.results_dir or cfg.get("RESULTS_DIR")
    for city in args.cities or []:
        stage1_dir = _resolve_stage1_dir_for_city(
            city,
            results_dir=results_dir,
            model=model,
        )
        add(city, _find_proc_csv(stage1_dir, city=city))

    out = sorted(seen.items(), key=lambda kv: kv[0])
    if not out:
        raise SystemExit(
            "No city inputs were provided. Use one or more of "
            "--city-csv, --city-stage1, --stage1-dir, or --cities."
        )
    if len(out) < 2:
        raise SystemExit(
            "At least two city clouds are required to classify "
            "boreholes by nearest city."
        )
    return out


def _load_city_cloud(
    path: Path,
    *,
    x_col: str,
    y_col: str,
) -> np.ndarray:
    """Load one city coordinate cloud."""
    df = pd.read_csv(path)
    missing = [
        c for c in (x_col, y_col) if c not in df.columns
    ]
    if missing:
        raise KeyError(
            f"Missing columns {missing!r} in {path}"
        )
    arr = df[[x_col, y_col]].dropna().to_numpy(dtype=float)
    if arr.size == 0:
        raise ValueError(
            f"No valid coordinates found in {path}"
        )
    return arr


def _nearest_distance(
    point_xy: np.ndarray,
    city_xy: np.ndarray,
) -> float:
    """Return the nearest Euclidean distance to one city cloud."""
    d2 = np.sum((city_xy - point_xy) ** 2, axis=1)
    return float(np.sqrt(d2.min()))


def _classify_boreholes(
    boreholes: pd.DataFrame,
    *,
    city_clouds: dict[str, np.ndarray],
    x_col: str,
    y_col: str,
    tie_label: str,
    tie_tol: float,
) -> pd.DataFrame:
    """Classify each borehole to the nearest city cloud."""
    rows: list[dict] = []

    for _, row in boreholes.iterrows():
        pt = np.array(
            [float(row[x_col]), float(row[y_col])],
            dtype=float,
        )
        dists = {
            city: _nearest_distance(pt, xy)
            for city, xy in city_clouds.items()
        }
        best = min(dists.values())
        winners = [
            city
            for city, dist in dists.items()
            if abs(dist - best) <= tie_tol
        ]

        out = dict(row)
        for city, dist in dists.items():
            out[f"dist_to_{city}_m"] = dist
        out["assigned_city"] = (
            winners[0] if len(winners) == 1 else tie_label
        )
        rows.append(out)

    return pd.DataFrame(rows)


def _default_outdir(
    args: argparse.Namespace,
    *,
    cfg: dict,
    borehole_csv: Path,
) -> Path:
    """Resolve a sensible output directory."""
    if args.outdir:
        return ensure_outdir(args.outdir)

    results_dir = args.results_dir or cfg.get("RESULTS_DIR")
    if isinstance(results_dir, str) and results_dir.strip():
        return ensure_outdir(
            Path(results_dir).expanduser().resolve()
            / "borehole_assignment"
        )

    return ensure_outdir(
        borehole_csv.parent / "borehole_assignment"
    )


def build_assign_boreholes_main(
    argv: list[str] | None = None,
) -> None:
    """CLI entry point for borehole-to-city assignment."""
    parser = argparse.ArgumentParser(
        prog="assign-boreholes",
        description=(
            "Assign boreholes to the nearest city cloud and "
            "write combined plus per-city CSV outputs."
        ),
    )
    add_config_args(parser)
    add_results_dir_arg(parser)
    add_model_arg(parser)
    add_outdir_arg(parser)

    parser.add_argument(
        "--borehole-csv",
        type=str,
        default=None,
        help="Validation borehole CSV with point coordinates.",
    )
    parser.add_argument(
        "--city-csv",
        dest="city_csvs",
        action="append",
        default=[],
        metavar="CITY=CSV",
        help=(
            "Explicit processed city CSV. Repeat as needed, for "
            "example --city-csv nansha=path/to/nansha_proc.csv."
        ),
    )
    parser.add_argument(
        "--city-stage1",
        dest="city_stage1s",
        action="append",
        default=[],
        metavar="CITY=DIR",
        help=(
            "Explicit Stage-1 directory for one city. Repeat as "
            "needed. The command resolves the *_proc.csv file "
            "under that directory."
        ),
    )
    parser.add_argument(
        "--stage1-dir",
        dest="stage1_dirs",
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "Stage-1 directory with one city processed CSV. The "
            "city name is inferred automatically. Repeat as needed."
        ),
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help=(
            "City names to resolve from results layout using "
            "--results-dir and optionally --model."
        ),
    )
    parser.add_argument(
        "--borehole-x-col",
        "--x-col",
        dest="borehole_x_col",
        default="x",
        help="Borehole X-coordinate column.",
    )
    parser.add_argument(
        "--borehole-y-col",
        "--y-col",
        dest="borehole_y_col",
        default="y",
        help="Borehole Y-coordinate column.",
    )
    parser.add_argument(
        "--city-x-col",
        default="x_m",
        help="Processed city CSV X-coordinate column.",
    )
    parser.add_argument(
        "--city-y-col",
        default="y_m",
        help="Processed city CSV Y-coordinate column.",
    )
    parser.add_argument(
        "--tie-label",
        type=str,
        default="tie",
        help="Assigned-city label when distances tie.",
    )
    parser.add_argument(
        "--tie-tol",
        type=float,
        default=0.0,
        help=(
            "Absolute tolerance in metres used to declare a tie."
        ),
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="boreholes",
        help="Base stem for generated CSV files.",
    )
    parser.add_argument(
        "--classified-out",
        type=str,
        default=None,
        help="Optional explicit path for the combined CSV.",
    )
    parser.add_argument(
        "--no-split-files",
        action="store_true",
        help="Skip per-city split CSV outputs.",
    )

    args = parser.parse_args(argv)
    cfg = bootstrap_runtime_config(
        args,
        field_map={
            "results_dir": "RESULTS_DIR",
            "model": "MODEL_NAME",
        },
    )

    borehole_csv = _resolve_borehole_csv(args, cfg)
    sources = _collect_city_sources(args, cfg)

    bh = pd.read_csv(borehole_csv)
    missing = [
        c
        for c in (
            args.borehole_x_col,
            args.borehole_y_col,
        )
        if c not in bh.columns
    ]
    if missing:
        raise KeyError(
            f"Missing borehole columns {missing!r} in {borehole_csv}"
        )

    bh = bh.dropna(
        subset=[args.borehole_x_col, args.borehole_y_col]
    ).copy()
    if bh.empty:
        raise RuntimeError(
            "The borehole table became empty after dropping rows "
            "with missing coordinates."
        )

    city_clouds = {
        city: _load_city_cloud(
            path,
            x_col=args.city_x_col,
            y_col=args.city_y_col,
        )
        for city, path in sources
    }
    classified = _classify_boreholes(
        bh,
        city_clouds=city_clouds,
        x_col=args.borehole_x_col,
        y_col=args.borehole_y_col,
        tie_label=args.tie_label,
        tie_tol=float(args.tie_tol),
    )

    outdir = _default_outdir(
        args,
        cfg=cfg,
        borehole_csv=borehole_csv,
    )
    stem = str(args.output_stem).strip() or "boreholes"

    if args.classified_out:
        classified_out = (
            Path(args.classified_out).expanduser().resolve()
        )
        classified_out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
    else:
        classified_out = outdir / f"{stem}_classified.csv"

    classified.to_csv(classified_out, index=False)
    print(f"[OK] wrote: {classified_out}")

    if not args.no_split_files:
        for city in sorted(city_clouds):
            part = classified[
                classified["assigned_city"] == city
            ].copy()
            out_csv = outdir / f"{stem}_{city}.csv"
            part.to_csv(out_csv, index=False)
            print(f"[OK] wrote: {out_csv} ({len(part)} rows)")

    cols = [
        c
        for c in (
            "well_id",
            args.borehole_x_col,
            args.borehole_y_col,
        )
        if c in classified.columns
    ]
    cols += [
        f"dist_to_{city}_m" for city in sorted(city_clouds)
    ]
    cols.append("assigned_city")
    print("")
    print(classified[cols].to_string(index=False))


main = build_assign_boreholes_main


if __name__ == "__main__":
    build_assign_boreholes_main()
