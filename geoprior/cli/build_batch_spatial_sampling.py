# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""CLI wrapper for ``batch_spatial_sampling``.

This command reads one or many tabular files, merges them into one
DataFrame, and applies
``geoprior.utils.spatial_utils.batch_spatial_sampling``.

By default it writes one stacked table containing all sampled batches
with an extra batch identifier column. It can also optionally write
one file per batch into a directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.spatial_utils import batch_spatial_sampling
from .build_spatial_sampling import (
    _normalize_bins,
    _parse_sample_size,
)
from .utils import (
    add_data_reader_args,
    load_dataframe_from_args,
    normalize_output_format,
    write_dataframe,
)


def _stack_batches(
    batches: list[pd.DataFrame],
    *,
    batch_col: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for idx, batch in enumerate(batches, start=1):
        frame = batch.copy()
        frame.insert(0, batch_col, idx)
        frames.append(frame)

    if not frames:
        return pd.DataFrame({batch_col: pd.Series(dtype=int)})

    return pd.concat(frames, ignore_index=True)


def _write_split_batches(
    batches: list[pd.DataFrame],
    *,
    out_dir: str | Path,
    prefix: str,
    suffix: str,
    excel_sheet_name: str,
    excel_engine: str | None,
) -> list[Path]:
    root = Path(out_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for idx, batch in enumerate(batches, start=1):
        path = root / f"{prefix}{idx:03d}{suffix}"
        out = write_dataframe(
            batch,
            path,
            excel_sheet_name=excel_sheet_name,
            excel_engine=excel_engine,
            index=False,
        )
        written.append(out)

    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batch-spatial-sampling",
        description=(
            "Build many non-overlapping spatial sample batches "
            "from one combined input table."
        ),
    )
    add_data_reader_args(parser)

    parser.add_argument(
        "--sample-size",
        type=_parse_sample_size,
        default=0.1,
        help=(
            "Total sampling size as a fraction or absolute "
            "count across all batches."
        ),
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=10,
        help="Number of sampled batches to create.",
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
        help="Sampling policy used by batch_spatial_sampling.",
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
        "--batch-col",
        default="batch_id",
        help="Column name added to the stacked output.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help=(
            "Stacked output table path. Supported: csv, tsv, "
            "parquet, xlsx, json, feather, pickle."
        ),
    )
    parser.add_argument(
        "--split-dir",
        default=None,
        help=(
            "Optional directory where one file per batch will "
            "also be written."
        ),
    )
    parser.add_argument(
        "--split-prefix",
        default="batch_",
        help="Filename prefix used inside --split-dir.",
    )
    parser.add_argument(
        "--split-format",
        default="auto",
        choices=(
            "auto",
            "csv",
            "tsv",
            "parquet",
            "xlsx",
            "json",
            "feather",
            "pickle",
        ),
        help=(
            "Format for files written to --split-dir. By "
            "default it follows the main output extension."
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


def run_build_batch_spatial_sampling(
    **kwargs: Any,
) -> pd.DataFrame:
    """Run the batch spatial-sampling workflow."""
    df = load_dataframe_from_args(
        argparse.Namespace(**kwargs)
    )

    batches = batch_spatial_sampling(
        data=df,
        sample_size=kwargs["sample_size"],
        n_batches=kwargs["n_batches"],
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

    stacked = _stack_batches(
        batches,
        batch_col=kwargs["batch_col"],
    )
    out = write_dataframe(
        stacked,
        kwargs["output"],
        excel_sheet_name=kwargs["excel_output_sheet"],
        excel_engine=kwargs.get("excel_engine"),
        index=False,
    )

    written_split: list[Path] = []
    if kwargs.get("split_dir"):
        fmt = kwargs["split_format"]
        if fmt == "auto":
            fmt = normalize_output_format(kwargs["output"])

        suffix_map = {
            "csv": ".csv",
            "tsv": ".tsv",
            "parquet": ".parquet",
            "excel": ".xlsx",
            "json": ".json",
            "feather": ".feather",
            "pickle": ".pkl",
        }
        suffix = suffix_map[fmt]
        written_split = _write_split_batches(
            batches,
            out_dir=kwargs["split_dir"],
            prefix=kwargs["split_prefix"],
            suffix=suffix,
            excel_sheet_name=kwargs["excel_output_sheet"],
            excel_engine=kwargs.get("excel_engine"),
        )

    total_rows = sum(len(batch) for batch in batches)
    print(
        f"[OK] loaded {len(df):,} row(s), created "
        f"{len(batches):,} batch(es), and wrote "
        f"{total_rows:,} sampled row(s) to {out}"
    )
    if written_split:
        print(
            f"[OK] also wrote {len(written_split):,} "
            f"per-batch file(s) to {kwargs['split_dir']}"
        )

    return stacked


def build_batch_spatial_sampling_main(
    argv: list[str] | None = None,
) -> None:
    args = _build_parser().parse_args(argv)
    run_build_batch_spatial_sampling(**vars(args))


def main(
    argv: list[str] | None = None,
) -> None:
    build_batch_spatial_sampling_main(argv)


if __name__ == "__main__":
    build_batch_spatial_sampling_main()
