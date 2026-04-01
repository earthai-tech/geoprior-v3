import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from geoprior.metrics import weighted_interval_score
from geoprior.plot.evaluation import (
    plot_coverage,
    plot_quantile_calibration,
    plot_theils_u_score,
    plot_time_weighted_metric,
    plot_weighted_interval_score,
)


@pytest.fixture(autouse=True)
def _patch_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


@pytest.fixture
def probabilistic_data():
    x = np.linspace(0.0, 1.0, 20)
    y_true = x + 0.1
    y_median = x
    y_lower = x - 0.2
    y_upper = x + 0.2
    alphas = np.array([0.2])
    y_pred_quantiles = np.column_stack([x - 0.2, x, x + 0.2])
    quantiles = np.array([0.1, 0.5, 0.9])
    return {
        "y_true": y_true,
        "y_median": y_median,
        "y_lower": y_lower,
        "y_upper": y_upper,
        "alphas": alphas,
        "y_pred_quantiles": y_pred_quantiles,
        "quantiles": quantiles,
    }


def test_plot_theils_u_score_summary_bar_returns_axes():
    y_true = np.array([1.0, 1.5, 2.0, 2.5])
    y_pred = np.array([1.1, 1.4, 1.9, 2.6])
    ax = plot_theils_u_score(y_true, y_pred)
    assert ax is not None
    assert ax.get_title()


def test_plot_weighted_interval_score_summary_and_histogram(
    probabilistic_data,
):
    ax1 = plot_weighted_interval_score(
        probabilistic_data["y_true"],
        probabilistic_data["y_median"],
        probabilistic_data["y_lower"],
        probabilistic_data["y_upper"],
        probabilistic_data["alphas"],
        kind="summary_bar",
    )
    ax2 = plot_weighted_interval_score(
        probabilistic_data["y_true"],
        probabilistic_data["y_median"],
        probabilistic_data["y_lower"],
        probabilistic_data["y_upper"],
        probabilistic_data["alphas"],
        kind="scores_histogram",
    )
    assert ax1 is not None
    assert ax2 is not None


def test_plot_time_weighted_metric_summary_and_profile():
    y_true = np.array([1.0, 1.2, 1.4, 1.6])
    y_pred = np.array([1.1, 1.1, 1.5, 1.7])
    ax1 = plot_time_weighted_metric(
        metric_type="mae",
        y_true=y_true,
        y_pred=y_pred,
        kind="summary_bar",
    )
    ax2 = plot_time_weighted_metric(
        metric_type="mae",
        y_true=y_true,
        y_pred=y_pred,
        kind="time_profile",
        show_time_weights_on_profile=True,
    )
    assert ax1 is not None
    assert ax2 is not None


def test_plot_quantile_calibration_reliability_and_summary(
    probabilistic_data,
):
    ax1 = plot_quantile_calibration(
        probabilistic_data["y_true"],
        probabilistic_data["y_pred_quantiles"],
        probabilistic_data["quantiles"],
        kind="reliability_diagram",
    )
    ax2 = plot_quantile_calibration(
        probabilistic_data["y_true"],
        probabilistic_data["y_pred_quantiles"],
        probabilistic_data["quantiles"],
        kind="summary_bar",
    )
    assert ax1 is not None
    assert ax2 is not None


def test_plot_coverage_intervals_and_summary_bar(
    probabilistic_data,
):
    ax1 = plot_coverage(
        probabilistic_data["y_true"],
        probabilistic_data["y_lower"],
        probabilistic_data["y_upper"],
        kind="intervals",
    )
    ax2 = plot_coverage(
        probabilistic_data["y_true"],
        probabilistic_data["y_lower"],
        probabilistic_data["y_upper"],
        kind="summary_bar",
    )
    assert ax1 is not None
    assert ax2 is not None


def test_weighted_interval_score_accepts_1d_bounds_when_k1():
    y_true = np.array([1.0, 2.0, 3.0])
    y_median = np.array([1.1, 1.9, 3.1])
    y_lower = np.array([0.8, 1.7, 2.8])
    y_upper = np.array([1.2, 2.2, 3.2])
    alphas = np.array([0.2])

    score = weighted_interval_score(
        y_true, y_lower, y_upper, y_median, alphas
    )
    assert np.isfinite(score)


def test_weighted_interval_score_accepts_2d_bounds_when_k1():
    y_true = np.array([[1.0, 10.0], [2.0, 20.0]])
    y_median = np.array([[1.1, 10.2], [1.9, 19.8]])
    y_lower = np.array([[0.8, 9.5], [1.7, 19.0]])
    y_upper = np.array([[1.3, 10.5], [2.2, 20.5]])
    alphas = np.array([0.2])

    score = weighted_interval_score(
        y_true,
        y_lower,
        y_upper,
        y_median,
        alphas,
        multioutput="raw_values",
    )
    assert score.shape == (2,)
