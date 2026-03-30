# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""CLI wrapper for ``extract_zones_from``.

This command reads one or many tabular files, merges them into a
single DataFrame, and then applies
``geoprior.utils.spatial_utils.extract_zones_from``.
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from ..utils.spatial_utils import extract_zones_from
from .utils import (
    add_data_reader_args,
    load_dataframe_from_args,
    write_dataframe,
)


def _parse_threshold(
    values: list[str],
) -> str | float | tuple[float, float]:
    if not values:
        raise argparse.ArgumentTypeError(
            "--threshold requires one or two values."
        )

    if len(values) == 1:
        text = str(values[0]).strip()
        if not text:
            raise argparse.ArgumentTypeError(
                "threshold cannot be empty."
            )
        if text.lower() == "auto":
            return "auto"
        try:
            return float(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "threshold must be 'auto', a float, "
                "or two floats."
            ) from exc

    if len(values) == 2:
        try:
            low = float(values[0])
            high = float(values[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "two-value thresholds must both be floats."
            ) from exc
        return (low, high)

    raise argparse.ArgumentTypeError(
        "threshold accepts one value or exactly two values."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract-zones",
        description=(
            "Extract rows that satisfy a value-threshold "
            "zone criterion."
        ),
    )
    add_data_reader_args(parser)

    parser.add_argument(
        "--z-col",
        required=True,
        help="Column used to define the zone criterion.",
    )
    parser.add_argument(
        "--threshold",
        nargs="+",
        required=True,
        metavar="VALUE",
        help=(
            "Use 'auto', one numeric value, or two numeric "
            "values for a between-range filter."
        ),
    )
    parser.add_argument(
        "--condition",
        default="auto",
        choices=("auto", "above", "below", "between"),
        help="Threshold condition.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=10.0,
        help=(
            "Percentile used when threshold is set to auto."
        ),
    )
    parser.add_argument(
        "--positive-criteria",
        action="store_true",
        help=(
            "When threshold is auto, use an above-threshold "
            "criterion instead of the default negative one."
        ),
    )
    parser.add_argument(
        "--x-col",
        default=None,
        help="Optional x column to include in the output.",
    )
    parser.add_argument(
        "--y-col",
        default=None,
        help="Optional y column to include in the output.",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Display the diagnostic plot.",
    )
    parser.add_argument(
        "--plot-type",
        default="scatter",
        help="Plot type used when --view is enabled.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(8.0, 6.0),
        metavar=("W", "H"),
        help="Figure size used when --view is enabled.",
    )
    parser.add_argument(
        "--axis-off",
        action="store_true",
        help="Hide plot axes when plotting.",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Disable grid lines in the diagnostic plot.",
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


def run_build_extract_zones(
    **kwargs: Any,
):
    """Run the extract-zones workflow from keyword args."""
    df = load_dataframe_from_args(
        argparse.Namespace(**kwargs)
    )

    result = extract_zones_from(
        z=kwargs["z_col"],
        threshold=_parse_threshold(list(kwargs["threshold"])),
        condition=kwargs["condition"],
        use_negative_criteria=not kwargs["positive_criteria"],
        percentile=kwargs["percentile"],
        x=kwargs.get("x_col"),
        y=kwargs.get("y_col"),
        data=df,
        view=kwargs["view"],
        plot_type=kwargs["plot_type"],
        figsize=tuple(kwargs["figsize"]),
        axis_off=kwargs["axis_off"],
        show_grid=not kwargs["no_grid"],
    )

    if isinstance(result, pd.Series):
        name = kwargs["z_col"] or "value"
        result_df = result.to_frame(name=name)
    else:
        result_df = result.copy()

    out = write_dataframe(
        result_df,
        kwargs["output"],
        excel_sheet_name=kwargs["excel_output_sheet"],
        excel_engine=kwargs.get("excel_engine"),
        index=False,
    )

    print(
        f"[OK] loaded {len(df):,} row(s) and wrote "
        f"{len(result_df):,} extracted row(s) to {out}"
    )
    return result_df


def build_extract_zones_main(
    argv: list[str] | None = None,
) -> None:
    args = _build_parser().parse_args(argv)
    run_build_extract_zones(**vars(args))


def main(
    argv: list[str] | None = None,
) -> None:
    build_extract_zones_main(argv)


if __name__ == "__main__":
    build_extract_zones_main()
