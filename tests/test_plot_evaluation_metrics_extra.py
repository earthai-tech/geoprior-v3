import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from geoprior.plot._metrics import _validate_qce_plot_inputs
from geoprior.plot.evaluation import (
    plot_crps,
    plot_mean_interval_width,
    plot_prediction_stability,
    plot_qce_donut,
    plot_radar_scores,
)


@pytest.fixture(autouse=True)
def _patch_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


@pytest.fixture
def ensemble_data():
    rng = np.random.default_rng(0)
    y_true = np.linspace(0.0, 1.0, 25)
    ensemble = y_true[:, None] + rng.normal(
        scale=0.15, size=(25, 8)
    )
    return y_true, ensemble


def test_plot_crps_all_kinds_smoke(ensemble_data):
    y_true, ensemble = ensemble_data
    ax1 = plot_crps(y_true, ensemble, kind="summary_bar")
    ax2 = plot_crps(y_true, ensemble, kind="scores_histogram")
    ax3 = plot_crps(
        y_true, ensemble, kind="ensemble_ecdf", sample_idx=0
    )
    assert ax1 is not None
    assert ax2 is not None
    assert ax3 is not None


def test_plot_mean_interval_width_histogram_and_summary():
    y_lower = np.array([0.0, 1.0, 2.0, 3.0])
    y_upper = np.array([1.0, 2.1, 3.2, 4.3])
    ax1 = plot_mean_interval_width(
        y_lower, y_upper, kind="summary_bar"
    )
    ax2 = plot_mean_interval_width(
        y_lower,
        y_upper,
        kind="widths_histogram",
    )
    assert ax1 is not None
    assert ax2 is not None


def test_plot_prediction_stability_histogram_and_summary():
    preds = np.array(
        [
            [1.0, 1.1, 1.0, 1.2],
            [0.9, 1.0, 1.1, 1.1],
            [1.2, 1.15, 1.1, 1.05],
        ]
    )
    ax1 = plot_prediction_stability(preds, kind="summary_bar")
    ax2 = plot_prediction_stability(
        preds, kind="scores_histogram"
    )
    assert ax1 is not None
    assert ax2 is not None


def test_validate_qce_inputs_and_plot_qce_donut_smoke():
    df = pd.DataFrame(
        {
            "actual": [0.2, 0.4, 0.6, 0.8],
            "q10": [0.1, 0.2, 0.4, 0.5],
            "q50": [0.2, 0.4, 0.6, 0.8],
            "q90": [0.3, 0.6, 0.8, 1.0],
        }
    )
    valid_df, levels = _validate_qce_plot_inputs(
        df,
        actual_col="actual",
        quantile_cols=["q10", "q50", "q90"],
        quantile_levels=[0.1, 0.5, 0.9],
    )
    assert valid_df is df
    assert np.allclose(levels, [0.1, 0.5, 0.9])

    ax = plot_qce_donut(
        df=df,
        actual_col="actual",
        quantile_cols=["q10", "q50", "q90"],
        quantile_levels=[0.1, 0.5, 0.9],
        show_legend=True,
    )
    assert ax is not None


def test_validate_qce_inputs_rejects_invalid_quantile_levels():
    df = pd.DataFrame({"actual": [1.0], "q50": [1.0]})
    with pytest.raises(ValueError):
        _validate_qce_plot_inputs(
            df,
            actual_col="actual",
            quantile_cols=["q50"],
            quantile_levels=[1.0],
        )


def test_plot_radar_scores_with_values_and_metric_mode():
    ax1 = plot_radar_scores(
        data_values={"MAE": 0.2, "RMSE": 0.3, "MAPE": 0.1},
        normalize_values=True,
    )
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 2.8, 4.2])
    ax2 = plot_radar_scores(
        y_true=y_true,
        y_pred=y_pred,
        category_names=["MAE", "RMSE", "MAPE"],
    )
    assert ax1 is not None
    assert ax2 is not None
