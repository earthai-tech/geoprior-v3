# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""Build a combined SM3 summary table from a suite directory.

This command scans an SM3 suite root, collects per-regime summary CSV
files, and writes combined CSV and JSON outputs.

Examples
--------
Use the versatile root entry point::

    geoprior build sm3-collect-summaries \
        --suite-root results/sm3_tau_suite_20260303-120000

Use the family-specific entry point::

    geoprior-build sm3-summaries \
        --results-dir results \
        --outdir results/sm3_tau_suite_20260303-120000
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from ._config import (
    add_city_arg,
    add_config_args,
    add_model_arg,
    add_outdir_arg,
    add_output_stem_arg,
    add_results_dir_arg,
    bootstrap_runtime_config,
    ensure_outdir,
    find_latest_dir,
)

DEFAULT_SUMMARY_NAME = "sm3_synth_summary.csv"
DEFAULT_STEM = "sm3_combined_summary"
DEFAULT_PATTERN = r"sm3_(?:tau|both)_(.+?)_50$"


def infer_regime(
    folder_name: str,
    *,
    pattern: str = DEFAULT_PATTERN,
) -> str:
    """Infer regime label from a run directory name."""
    match = re.search(pattern, folder_name)
    if match:
        return str(match.group(1))
    return folder_name


def discover_suite_root(
    results_dir: str | Path,
    *,
    summary_name: str = DEFAULT_SUMMARY_NAME,
) -> Path | None:
    """Return the newest suite-like directory under results root."""
    root = Path(results_dir).expanduser().resolve()
    if not root.exists():
        return None

    direct = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if any(child.rglob(summary_name)):
            direct.append(child)

    if direct:
        return max(direct, key=lambda p: p.stat().st_mtime)

    latest = find_latest_dir(
        root,
        pattern="*",
        must_contain=summary_name,
    )
    if latest is not None:
        return latest

    if any(root.rglob(summary_name)):
        return root
    return None


def resolve_suite_root(
    args: argparse.Namespace,
    cfg: dict,
) -> Path:
    """Resolve suite root from explicit args, config, or results root."""
    if args.suite_root:
        path = Path(args.suite_root).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Suite root not found: {path}"
            )
        return path

    for key in ("SM3_SUITE_ROOT", "SUITE_ROOT"):
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            path = Path(value).expanduser().resolve()
            if path.exists():
                return path

    results_dir = args.results_dir or cfg.get("RESULTS_DIR")
    if results_dir:
        guess = discover_suite_root(
            results_dir,
            summary_name=args.summary_name,
        )
        if guess is not None:
            return guess

    raise FileNotFoundError(
        "Could not resolve suite root. Pass --suite-root or "
        "provide a results root that contains an SM3 suite."
    )


def collect_rows(
    suite_root: Path,
    *,
    summary_name: str,
    pattern: str,
    strict: bool,
) -> list[pd.DataFrame]:
    """Collect per-regime summary tables from a suite root."""
    rows: list[pd.DataFrame] = []

    for path in suite_root.rglob(summary_name):
        run_dir = path.parent
        regime = infer_regime(run_dir.name, pattern=pattern)

        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            msg = f"failed to read {path}: {exc}"
            if strict:
                raise RuntimeError(msg) from exc
            print(f"[skip] {msg}")
            continue

        if frame.empty or "metric" not in frame.columns:
            msg = f"unexpected format: {path}"
            if strict:
                raise RuntimeError(msg)
            print(f"[skip] {msg}")
            continue

        frame = frame.copy()
        frame.insert(0, "regime", regime)
        frame.insert(1, "run_dir", str(run_dir))
        rows.append(frame)

    return rows


def resolve_outputs(
    args: argparse.Namespace,
    suite_root: Path,
) -> tuple[Path, Path]:
    """Resolve output CSV and JSON paths."""
    base_dir = (
        ensure_outdir(args.outdir)
        if args.outdir
        else suite_root
    )
    stem = args.output_stem or DEFAULT_STEM

    out_csv = (
        Path(args.out_csv).expanduser().resolve()
        if args.out_csv
        else base_dir / f"{stem}.csv"
    )
    out_json = (
        Path(args.out_json).expanduser().resolve()
        if args.out_json
        else base_dir / f"{stem}.json"
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    return out_csv, out_json


def build_sm3_collect_parser() -> argparse.ArgumentParser:
    """Build parser for SM3 summary collection."""
    parser = argparse.ArgumentParser(
        prog="sm3-collect-summaries",
        description=(
            "Collect SM3 per-regime summary CSV files into "
            "combined CSV and JSON outputs."
        ),
    )
    add_config_args(parser)
    add_city_arg(parser)
    add_model_arg(parser)
    add_results_dir_arg(parser)
    add_outdir_arg(parser)
    add_output_stem_arg(parser, default=DEFAULT_STEM)

    parser.add_argument(
        "--suite-root",
        type=str,
        default=None,
        help=(
            "SM3 suite root. If omitted, the command tries to "
            "discover the newest suite under --results-dir."
        ),
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default=DEFAULT_SUMMARY_NAME,
        help="Summary CSV filename to collect.",
    )
    parser.add_argument(
        "--regime-pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help=(
            "Regular expression used to infer the regime from "
            "each run directory name."
        ),
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Explicit combined CSV output path.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Explicit combined JSON output path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail on unreadable or malformed summary files "
            "instead of skipping them."
        ),
    )
    return parser


def run_sm3_collect_summaries(
    args: argparse.Namespace,
) -> None:
    """Execute the SM3 collection workflow."""
    cfg = bootstrap_runtime_config(
        args,
        field_map={
            "city": "CITY_NAME",
            "model": "MODEL_NAME",
            "results_dir": "RESULTS_DIR",
        },
    )

    suite_root = resolve_suite_root(args, cfg)
    rows = collect_rows(
        suite_root,
        summary_name=args.summary_name,
        pattern=args.regime_pattern,
        strict=bool(args.strict),
    )
    if not rows:
        raise RuntimeError(
            "No summary CSV files were found under the suite "
            "root."
        )

    frame = pd.concat(rows, ignore_index=True)
    sort_cols = [
        col
        for col in ["metric", "regime"]
        if col in frame.columns
    ]
    if sort_cols:
        frame = frame.sort_values(sort_cols).reset_index(
            drop=True
        )

    out_csv, out_json = resolve_outputs(args, suite_root)
    frame.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(frame.to_dict("records"), handle, indent=2)

    print("[OK] suite_root:", str(suite_root))
    print("[OK] wrote:", str(out_csv))
    print("[OK] wrote:", str(out_json))
    print("[OK] rows:", len(frame))


def build_sm3_collect_main(
    argv: list[str] | None = None,
) -> None:
    """CLI entry point for SM3 summary collection."""
    parser = build_sm3_collect_parser()
    args = parser.parse_args(argv)
    run_sm3_collect_summaries(args)


def main(argv: list[str] | None = None) -> None:
    """Alias for the command entry point."""
    build_sm3_collect_main(argv)


if __name__ == "__main__":  # pragma: no cover
    main()
