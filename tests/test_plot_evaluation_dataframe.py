import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from geoprior.plot.evaluation import (
    plot_forecast_comparison,
    plot_metric_over_horizon,
    plot_metric_radar,
)


@pytest.fixture(autouse=True)
def _patch_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


@pytest.fixture
def forecast_df_point():
    rows = []
    rng = np.random.default_rng(0)
    cities = ["A", "B"]
    for sid in range(4):
        for step in range(1, 4):
            actual = sid + step + 0.1
            pred = actual + (0.2 if sid % 2 == 0 else -0.1)
            rows.append(
                {
                    "sample_idx": sid,
                    "forecast_step": step,
                    "target_actual": actual,
                    "target_pred": pred,
                    "city": cities[sid % 2],
                    "x": rng.uniform(0, 10),
                    "y": rng.uniform(0, 10),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def forecast_df_quantile(forecast_df_point):
    df = forecast_df_point.copy()
    df["target_q10"] = df["target_pred"] - 0.4
    df["target_q50"] = df["target_pred"]
    df["target_q90"] = df["target_pred"] + 0.4
    return df


def test_plot_metric_over_horizon_point_metrics_smoke(
    forecast_df_point,
):
    ax = plot_metric_over_horizon(
        forecast_df=forecast_df_point,
        target_name="target",
        metrics=["mae", "rmse"],
        plot_kind="bar",
    )
    assert ax is not None
    assert plt.get_fignums()


def test_plot_metric_over_horizon_quantile_grouped_line_smoke(
    forecast_df_quantile,
):
    ax = plot_metric_over_horizon(
        forecast_df=forecast_df_quantile,
        target_name="target",
        metrics=["coverage", "pinball_median"],
        quantiles=[0.1, 0.5, 0.9],
        group_by_cols=["city"],
        plot_kind="line",
    )
    assert ax is not None
    assert plt.get_fignums()


def test_plot_metric_over_horizon_requires_forecast_step(
    forecast_df_point,
):
    with pytest.raises(ValueError, match="forecast_step"):
        plot_metric_over_horizon(
            forecast_df=forecast_df_point.drop(
                columns=["forecast_step"]
            ),
            target_name="target",
        )


def test_plot_metric_radar_smoke(forecast_df_point):
    plot_metric_radar(
        forecast_df=forecast_df_point,
        segment_col="city",
        metric="rmse",
        target_name="target",
    )
    assert not plt.get_fignums()


def test_plot_forecast_comparison_temporal_quantile_smoke(
    forecast_df_quantile,
):
    plot_forecast_comparison(
        forecast_df=forecast_df_quantile,
        target_name="target",
        quantiles=[0.1, 0.5, 0.9],
        kind="temporal",
        sample_ids="first_n",
        num_samples=2,
    )
    assert plt.get_fignums()


def test_plot_forecast_comparison_spatial_point_smoke(
    forecast_df_point,
):
    plot_forecast_comparison(
        forecast_df=forecast_df_point,
        target_name="target",
        kind="spatial",
        spatial_cols=["x", "y"],
        horizon_steps=[1, 2],
    )
    assert plt.get_fignums()


def test_plot_forecast_comparison_invalid_kind_raises(
    forecast_df_point,
):
    with pytest.raises(ValueError, match="temporal"):
        plot_forecast_comparison(
            forecast_df=forecast_df_point,
            target_name="target",
            kind="bad-kind",
        )
