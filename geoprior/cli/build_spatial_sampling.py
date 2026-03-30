# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""CLI wrapper for ``spatial_sampling``.

This command reads one or many tabular files, merges them into a
single DataFrame, and then applies
``geoprior.utils.spatial_utils.spatial_sampling``.
"""

from __future__ import annotations

import argparse
from typing import Any

from ..utils.spatial_utils import spatial_sampling
from .utils import (
    add_data_reader_args,
    load_dataframe_from_args,
    write_dataframe,
)


def _parse_sample_size(
    value: str,
) -> int | float:
    text = str(value).strip()
    if not text:
        raise argparse.ArgumentTypeError(
            "sample size cannot be empty."
        )

    try:
        fval = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "sample size must be an int or float."
        ) from exc

    if fval.is_integer() and fval >= 1.0:
        ival = int(fval)
        if ival <= 0:
            raise argparse.ArgumentTypeError(
                "integer sample size must be positive."
            )
        return ival

    if not 0.0 < fval < 1.0:
        raise argparse.ArgumentTypeError(
            "float sample size must be in (0, 1)."
        )
    return fval


def _normalize_bins(
    values: list[int] | None,
) -> int | tuple[int, ...]:
    if not values:
        return 10
    if len(values) == 1:
        return int(values[0])
    return tuple(int(v) for v in values)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spatial-sampling",
        description=(
            "Sample one combined spatial dataset from one "
            "or many input tables."
        ),
    )
    add_data_reader_args(parser)

    parser.add_argument(
        "--sample-size",
        type=_parse_sample_size,
        default=0.01,
        help=(
            "Sampling size as a fraction or absolute count."
        ),
    )
    parser.add_argument(
        "--stratify-by",
        nargs="+",
        default=None,
        help="Optional columns used for stratification.",
    )
    parser.add_argument(
        "--spatial-bins",
        nargs="+",
        type=int,
        default=None,
        help=("One integer or one value per spatial column."),
    )
    parser.add_argument(
        "--spatial-cols",
        nargs="+",
        default=None,
        help=(
            "Spatial columns. Default: auto-detect longitude "
            "and latitude."
        ),
    )
    parser.add_argument(
        "--method",
        default="abs",
        choices=("abs", "absolute", "relative"),
        help="Sampling policy used by spatial_sampling.",
    )
    parser.add_argument(
        "--min-relative-ratio",
        type=float,
        default=0.01,
        help=("Minimum fraction used by relative sampling."),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help=(
            "Output table path. Supported: csv, tsv, "
            "parquet, xlsx, json, feather, pickle."
        ),
    )
    parser.add_argument(
        "--excel-output-sheet",
        default="Sheet1",
        help="Sheet name when writing Excel output.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level passed to the sampler.",
    )
    return parser


def run_build_spatial_sampling(
    **kwargs: Any,
):
    """Run the spatial-sampling workflow from keyword args."""
    df = load_dataframe_from_args(
        argparse.Namespace(**kwargs)
    )

    sampled = spatial_sampling(
        data=df,
        sample_size=kwargs["sample_size"],
        stratify_by=kwargs["stratify_by"],
        spatial_bins=_normalize_bins(
            kwargs.get("spatial_bins")
        ),
        spatial_cols=kwargs.get("spatial_cols"),
        method=kwargs["method"],
        min_relative_ratio=kwargs["min_relative_ratio"],
        random_state=kwargs["random_state"],
        verbose=kwargs["verbose"],
    )

    out = write_dataframe(
        sampled,
        kwargs["output"],
        excel_sheet_name=kwargs["excel_output_sheet"],
        excel_engine=kwargs.get("excel_engine"),
        index=False,
    )

    print(
        f"[OK] loaded {len(df):,} row(s) and wrote "
        f"{len(sampled):,} sampled row(s) to {out}"
    )
    return sampled


def build_spatial_sampling_main(
    argv: list[str] | None = None,
) -> None:
    args = _build_parser().parse_args(argv)
    run_build_spatial_sampling(**vars(args))


def main(
    argv: list[str] | None = None,
) -> None:
    build_spatial_sampling_main(argv)


if __name__ == "__main__":
    build_spatial_sampling_main()
