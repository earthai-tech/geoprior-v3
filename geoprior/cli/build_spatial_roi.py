# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""CLI wrapper for ``extract_spatial_roi``.

This command reads one or many tabular files, merges them into a
single DataFrame, and then applies
``geoprior.utils.spatial_utils.extract_spatial_roi``.
"""

from __future__ import annotations

import argparse
from typing import Any

from ..utils.spatial_utils import extract_spatial_roi
from .utils import (
    add_data_reader_args,
    load_dataframe_from_args,
    write_dataframe,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spatial-roi",
        description=(
            "Extract a rectangular spatial region of interest "
            "from one combined input table."
        ),
    )
    add_data_reader_args(parser)

    parser.add_argument(
        "--x-range",
        nargs=2,
        type=float,
        required=True,
        metavar=("XMIN", "XMAX"),
        help="Lower and upper bounds for the x coordinate.",
    )
    parser.add_argument(
        "--y-range",
        nargs=2,
        type=float,
        required=True,
        metavar=("YMIN", "YMAX"),
        help="Lower and upper bounds for the y coordinate.",
    )
    parser.add_argument(
        "--x-col",
        default="longitude",
        help="X coordinate column name.",
    )
    parser.add_argument(
        "--y-col",
        default="latitude",
        help="Y coordinate column name.",
    )
    parser.add_argument(
        "--no-snap-to-closest",
        action="store_true",
        help=(
            "Use the exact bounds instead of snapping them "
            "to the nearest available coordinates."
        ),
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
    return parser


def run_build_spatial_roi(**kwargs: Any):
    """Run the ROI extraction workflow from keyword args."""
    df = load_dataframe_from_args(
        argparse.Namespace(**kwargs)
    )

    roi = extract_spatial_roi(
        df=df,
        x_range=tuple(kwargs["x_range"]),
        y_range=tuple(kwargs["y_range"]),
        x_col=kwargs["x_col"],
        y_col=kwargs["y_col"],
        snap_to_closest=not kwargs["no_snap_to_closest"],
    )

    out = write_dataframe(
        roi,
        kwargs["output"],
        excel_sheet_name=kwargs["excel_output_sheet"],
        excel_engine=kwargs.get("excel_engine"),
        index=False,
    )

    print(
        f"[OK] loaded {len(df):,} row(s) and wrote "
        f"{len(roi):,} ROI row(s) to {out}"
    )
    return roi


def build_spatial_roi_main(
    argv: list[str] | None = None,
) -> None:
    args = _build_parser().parse_args(argv)
    run_build_spatial_roi(**vars(args))


def main(
    argv: list[str] | None = None,
) -> None:
    build_spatial_roi_main(argv)


if __name__ == "__main__":
    build_spatial_roi_main()
