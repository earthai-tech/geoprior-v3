# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""CLI wrapper for ``create_spatial_clusters``.

This command reads one or many tabular files, merges them into a
single DataFrame, and then applies
``geoprior.utils.spatial_utils.create_spatial_clusters``.
"""

from __future__ import annotations

import argparse
from typing import Any

from ..utils.spatial_utils import create_spatial_clusters
from .utils import (
    add_data_reader_args,
    load_dataframe_from_args,
    write_dataframe,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spatial-clusters",
        description=(
            "Assign spatial cluster labels to one combined "
            "input table."
        ),
    )
    add_data_reader_args(parser)

    parser.add_argument(
        "--spatial-cols",
        nargs=2,
        default=["longitude", "latitude"],
        metavar=("XCOL", "YCOL"),
        help="Two spatial columns used for clustering.",
    )
    parser.add_argument(
        "--cluster-col",
        default="region",
        help="Output column name storing cluster labels.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help=(
            "Optional number of clusters. For kmeans, "
            "the function can auto-detect it when omitted."
        ),
    )
    parser.add_argument(
        "--algorithm",
        default="kmeans",
        choices=("kmeans", "dbscan", "agglo"),
        help="Clustering backend.",
    )
    parser.add_argument(
        "--no-auto-scale",
        action="store_true",
        help="Disable standard scaling before clustering.",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Display the diagnostic cluster plot.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(14.0, 10.0),
        metavar=("W", "H"),
        help="Figure size used when --view is enabled.",
    )
    parser.add_argument(
        "--marker-size",
        type=int,
        default=60,
        help="Marker size used when --view is enabled.",
    )
    parser.add_argument(
        "--plot-style",
        default="default",
        help="Matplotlib style used for plotting.",
    )
    parser.add_argument(
        "--cmap",
        default="tab20",
        help="Colormap used for plotting.",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Disable grid lines in the cluster plot.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level passed to the clustering helper.",
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


def run_build_spatial_clusters(
    **kwargs: Any,
):
    """Run the spatial-clustering workflow from keyword args."""
    df = load_dataframe_from_args(
        argparse.Namespace(**kwargs)
    )

    clustered = create_spatial_clusters(
        df=df,
        spatial_cols=list(kwargs["spatial_cols"]),
        cluster_col=kwargs["cluster_col"],
        n_clusters=kwargs.get("n_clusters"),
        algorithm=kwargs["algorithm"],
        view=kwargs["view"],
        figsize=tuple(kwargs["figsize"]),
        s=kwargs["marker_size"],
        plot_style=kwargs["plot_style"],
        cmap=kwargs["cmap"],
        show_grid=not kwargs["no_grid"],
        auto_scale=not kwargs["no_auto_scale"],
        verbose=kwargs["verbose"],
    )

    out = write_dataframe(
        clustered,
        kwargs["output"],
        excel_sheet_name=kwargs["excel_output_sheet"],
        excel_engine=kwargs.get("excel_engine"),
        index=False,
    )

    labels = clustered[kwargs["cluster_col"]]
    n_unique = int(labels.nunique(dropna=False))
    print(
        f"[OK] loaded {len(df):,} row(s), created "
        f"{n_unique:,} cluster label(s), and wrote "
        f"{len(clustered):,} row(s) to {out}"
    )
    return clustered


def build_spatial_clusters_main(
    argv: list[str] | None = None,
) -> None:
    args = _build_parser().parse_args(argv)
    run_build_spatial_clusters(**vars(args))


def main(
    argv: list[str] | None = None,
) -> None:
    build_spatial_clusters_main(argv)


if __name__ == "__main__":
    build_spatial_clusters_main()
