# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

from __future__ import annotations

import argparse
from pathlib import Path

from ..utils.spatial_utils import (
    make_forecast_ready_sample,
)
from .utils import (
    add_data_reader_args,
    load_dataframe_from_args,
    write_dataframe,
)


def _parse_sample_size(value: str) -> int | float:
    text = str(value).strip()
    if not text:
        raise argparse.ArgumentTypeError(
            "--sample-size cannot be empty."
        )

    low = text.lower()
    if any(ch in low for ch in (".", "e")):
        try:
            out = float(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "--sample-size must be int or float."
            ) from exc
        return out

    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "--sample-size must be int or float."
            ) from exc


def _parse_bins(
    values: list[str] | tuple[str, ...],
) -> int | tuple[int, int]:
    raw = [str(v).strip() for v in values if str(v).strip()]
    if not raw:
        return 10

    if len(raw) == 1 and "," in raw[0]:
        raw = [
            part.strip()
            for part in raw[0].split(",")
            if part.strip()
        ]

    try:
        nums = [int(v) for v in raw]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--spatial-bins must be integers."
        ) from exc

    if len(nums) == 1:
        return nums[0]

    if len(nums) == 2:
        return (nums[0], nums[1])

    raise argparse.ArgumentTypeError(
        "--spatial-bins accepts one or two integers."
    )


def _build_parser(
    prog: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Build a compact forecast-ready panel sample "
            "from one or many tabular inputs."
        ),
    )

    add_data_reader_args(parser, nargs="+", paths_metavar="PATH")

    parser.add_argument(
        "--out",
        "-o",
        required=True,
        help=(
            "Output table path. Extension controls the "
            "format, e.g. .csv, .parquet, .xlsx."
        ),
    )
    parser.add_argument(
        "--time-col",
        default="year",
        help="Time column name. Default: year.",
    )
    parser.add_argument(
        "--spatial-cols",
        nargs="+",
        default=["longitude", "latitude"],
        help=(
            "Spatial coordinate columns. Default: "
            "longitude latitude."
        ),
    )
    parser.add_argument(
        "--group-cols",
        nargs="+",
        default=None,
        help=(
            "Grouping columns. Default: use spatial-cols."
        ),
    )
    parser.add_argument(
        "--stratify-by",
        nargs="*",
        default=None,
        help=(
            "Optional group-level columns used for "
            "stratified group sampling."
        ),
    )
    parser.add_argument(
        "--spatial-bins",
        nargs="+",
        default=["10"],
        help=(
            "One or two spatial bin counts. Examples: "
            "--spatial-bins 10 or --spatial-bins 10 12 "
            'or --spatial-bins "10,12".'
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=_parse_sample_size,
        default=0.05,
        help=(
            "Group sample size. Float means fraction; "
            "int means absolute number of groups."
        ),
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=3,
        help="Lookback window length. Default: 3.",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=1,
        help="Forecast horizon length. Default: 1.",
    )
    parser.add_argument(
        "--keep-years",
        type=int,
        default=None,
        help=(
            "Optional number of years to retain per "
            "sampled group."
        ),
    )
    parser.add_argument(
        "--year-mode",
        choices=[
            "all",
            "latest",
            "earliest",
            "random",
        ],
        default="latest",
        help=(
            "How to keep years when --keep-years is "
            "set. Default: latest."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["abs", "absolute", "relative"],
        default="abs",
        help="Sampling method. Default: abs.",
    )
    parser.add_argument(
        "--min-relative-ratio",
        type=float,
        default=0.01,
        help=(
            "Minimum relative ratio used when "
            "--method=relative."
        ),
    )
    parser.add_argument(
        "--min-groups",
        type=int,
        default=5,
        help=(
            "Minimum eligible groups required before "
            "sampling."
        ),
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Optional hard cap on sampled groups.",
    )
    parser.add_argument(
        "--columns-to-keep",
        nargs="*",
        default=None,
        help=(
            "Optional subset of columns to keep in the "
            "final output."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    
    parser.add_argument(
        "--require-consecutive",
        dest="require_consecutive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require a consecutive run of length "
            "time_steps + forecast_horizon."
        ),
    )
    parser.add_argument(
        "--sort-output",
        dest="sort_output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sort final output by group and time.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level. Default: 1.",
    )

    return parser


def build_forecast_ready_sample_main(
    argv: list[str] | None = None,
    *,
    prog: str | None = None,
) -> None:
    parser = _build_parser(prog=prog)
    args = parser.parse_args(argv)

    df = load_dataframe_from_args(args)

    spatial_bins = _parse_bins(args.spatial_bins)

    out_df = make_forecast_ready_sample(
        data=df,
        sample_size=args.sample_size,
        time_col=args.time_col,
        spatial_cols=args.spatial_cols,
        group_cols=args.group_cols,
        stratify_by=args.stratify_by,
        spatial_bins=spatial_bins,
        time_steps=args.time_steps,
        forecast_horizon=args.forecast_horizon,
        require_consecutive=args.require_consecutive,
        keep_years=args.keep_years,
        year_mode=args.year_mode,
        min_groups=args.min_groups,
        max_groups=args.max_groups,
        columns_to_keep=args.columns_to_keep,
        method=args.method,
        min_relative_ratio=args.min_relative_ratio,
        random_state=args.random_state,
        savefile=None,
        export_path=None,
        export_format=None,
        sort_output=args.sort_output,
        verbose=args.verbose,
    )

    out_path = write_dataframe(
        out_df,
        args.out,
        excel_engine=args.excel_engine,
        index=False,
    )

    n_groups = out_df.loc[
        :,
        list(args.group_cols or args.spatial_cols),
    ].drop_duplicates().shape[0]

    print("")
    print("[OK] forecast-ready sample written")
    print(f"  path   : {Path(out_path).resolve()}")
    print(f"  rows   : {len(out_df):,}")
    print(f"  groups : {n_groups:,}")
    print(
        "  window : "
        f"T={args.time_steps}, H={args.forecast_horizon}"
    )


if __name__ == "__main__":
    build_forecast_ready_sample_main()