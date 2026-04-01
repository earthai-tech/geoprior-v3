import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


def _make_spatial_df():
    rows = []
    for year in [2024, 2025]:
        for i in range(8):
            rows.append(
                {
                    "coord_x": float(i),
                    "coord_y": float(i % 4),
                    "coord_t": year,
                    "value": float(i + year / 1000.0),
                    "value2": float((i + 1) * 0.5),
                }
            )
    return pd.DataFrame(rows)


def test_plot_spatial_smoke_and_save(tmp_path):
    from geoprior.plot.spatial import plot_spatial

    df = _make_spatial_df()
    out = tmp_path / "spatial_map.png"

    figs = plot_spatial(
        df=df,
        value_col="value",
        spatial_cols=("coord_x", "coord_y"),
        dt_col="coord_t",
        dt_values=[2024, 2025],
        show_grid=False,
        savefig=str(out),
    )

    assert isinstance(figs, list)
    assert len(figs) == 1
    assert out.exists()


def test_plot_spatial_roi_returns_single_figure():
    from geoprior.plot.spatial import plot_spatial_roi

    df = _make_spatial_df()
    figs = plot_spatial_roi(
        df=df,
        value_cols=["value", "value2"],
        x_range=(1.0, 6.0),
        y_range=(0.0, 3.0),
        spatial_cols=("coord_x", "coord_y"),
        dt_col="coord_t",
        dt_values=[2024, 2025],
        show_grid=False,
    )

    assert isinstance(figs, list)
    assert len(figs) == 1


def test_plot_spatial_contours_smoke():
    from geoprior.plot.spatial import plot_spatial_contours

    df = _make_spatial_df()
    figs = plot_spatial_contours(
        df=df,
        value_col="value",
        spatial_cols=("coord_x", "coord_y"),
        dt_col="coord_t",
        dt_values=[2024, 2025],
        grid_res=20,
        levels=[0.2, 0.5, 0.8],
        show_points=False,
        show_grid=False,
    )

    assert isinstance(figs, list)
    assert len(figs) == 1


def test_plot_hotspots_smoke():
    from geoprior.plot.spatial import plot_hotspots

    df = _make_spatial_df()
    figs = plot_hotspots(
        df=df,
        value_col="value",
        spatial_cols=("coord_x", "coord_y"),
        dt_col="coord_t",
        dt_values=[2024, 2025],
        percentile=80.0,
        show_grid=False,
    )

    assert isinstance(figs, list)
    assert len(figs) == 1


def test_plot_spatial_heatmap_grid_and_kde_smoke():
    from geoprior.plot.spatial import plot_spatial_heatmap

    df = _make_spatial_df()

    figs_grid = plot_spatial_heatmap(
        df=df,
        value_col="value",
        spatial_cols=("coord_x", "coord_y"),
        dt_col="coord_t",
        dt_values=[2024],
        grid_res=20,
        method="grid",
        show_points=False,
        show_grid=False,
    )
    assert len(figs_grid) == 1

    figs_kde = plot_spatial_heatmap(
        df=df,
        value_col="value",
        spatial_cols=("coord_x", "coord_y"),
        dt_col="coord_t",
        dt_values=[2025],
        grid_res=20,
        method="kde",
        show_points=False,
        show_grid=False,
    )
    assert len(figs_kde) == 1


def test_plot_spatial_voronoi_smoke():
    from geoprior.plot.spatial import plot_spatial_voronoi

    df = _make_spatial_df()
    figs = plot_spatial_voronoi(
        df=df,
        value_col="value",
        spatial_cols=("coord_x", "coord_y"),
        dt_col="coord_t",
        dt_values=[2024],
        show_grid=False,
    )

    assert isinstance(figs, list)
    assert len(figs) == 1
