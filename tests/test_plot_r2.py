import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest


def test_plot_r2_smoke_with_multiple_predictions():
    from geoprior.plot.r2 import plot_r2

    rng = np.random.default_rng(0)
    y_true = np.linspace(0, 1, 30)
    y_pred1 = y_true + rng.normal(0, 0.05, size=y_true.shape)
    y_pred2 = y_true + rng.normal(0, 0.10, size=y_true.shape)

    fig = plot_r2(
        y_true,
        y_pred1,
        y_pred2,
        titles=["model-a", "model-b"],
        other_metrics=["mae"],
        max_cols=2,
    )

    assert fig is not None
    assert len(fig.axes) >= 2


def test_plot_r2_in_smoke_with_fit_line():
    from geoprior.plot.r2 import plot_r2_in

    rng = np.random.default_rng(1)
    y_true_a = np.linspace(0, 2, 40)
    y_pred_a = y_true_a + rng.normal(
        0, 0.08, size=y_true_a.shape
    )
    y_true_b = np.linspace(1, 3, 40)
    y_pred_b = y_true_b + rng.normal(
        0, 0.10, size=y_true_b.shape
    )

    fig = plot_r2_in(
        y_true_a,
        y_pred_a,
        y_true_b,
        y_pred_b,
        titles=["A", "B"],
        other_metrics=["mae"],
        fit_eq=True,
        max_cols=2,
    )

    assert fig is not None
    assert len(fig.axes) >= 2


def test_plot_r2_in_raises_when_no_valid_pairs():
    from geoprior.plot.r2 import plot_r2_in

    with pytest.raises(ValueError):
        plot_r2_in(np.array([np.nan]), np.array([np.nan]))
