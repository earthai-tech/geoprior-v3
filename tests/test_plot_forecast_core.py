import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


def _make_quantile_long_df():
    rows = []
    for sample_idx in range(4):
        for step in [1, 2]:
            base = sample_idx + step
            rows.append(
                {
                    "sample_idx": sample_idx,
                    "forecast_step": step,
                    "coord_x": float(sample_idx),
                    "coord_y": float(step),
                    "coord_t": 2023 + step,
                    "subsidence_actual": float(base + 0.1),
                    "subsidence_q10": float(base - 0.4),
                    "subsidence_q50": float(base),
                    "subsidence_q90": float(base + 0.4),
                }
            )
    return pd.DataFrame(rows)


def _make_wide_forecast_df():
    return pd.DataFrame(
        {
            "sample_idx": [0, 1, 2, 3],
            "coord_x": [0.0, 1.0, 2.0, 3.0],
            "coord_y": [0.0, 1.0, 0.5, 1.5],
            "subsidence_2024_actual": [1.0, 1.1, 1.2, 1.3],
            "subsidence_2024_q10": [0.8, 0.9, 1.0, 1.1],
            "subsidence_2024_q50": [1.0, 1.1, 1.2, 1.3],
            "subsidence_2024_q90": [1.2, 1.3, 1.4, 1.5],
            "subsidence_2025_q10": [1.1, 1.2, 1.3, 1.4],
            "subsidence_2025_q50": [1.3, 1.4, 1.5, 1.6],
            "subsidence_2025_q90": [1.5, 1.6, 1.7, 1.8],
        }
    )


def test_plot_reliability_diagram_quantiles_smoke():
    from geoprior.plot.forecast import (
        plot_reliability_diagram,
    )

    df = _make_quantile_long_df()
    ax = plot_reliability_diagram(
        df,
        quantiles=[0.1, 0.5, 0.9],
        q_prefix="subsidence",
        actual_col="subsidence_actual",
        show_grid=False,
    )

    assert ax.get_xlabel() == "Nominal probability"
    assert ax.get_ylabel() == "Empirical frequency"


def test_plot_calibration_comparison_probability_smoke(
    monkeypatch,
):
    import geoprior.plot.forecast as forecast_mod

    df = pd.DataFrame(
        {
            "p_event": [0.1, 0.2, 0.7, 0.9, 0.8],
            "event_flag": [0, 0, 1, 1, 1],
        }
    )

    def _fake_calibrate_probability_forecast(
        df, prob_col, actual_col, method="isotonic"
    ):
        out = df.copy()
        out[f"{prob_col}_calib"] = np.clip(
            out[prob_col] * 0.95 + 0.02, 0, 1
        )
        return out

    monkeypatch.setattr(
        forecast_mod,
        "calibrate_probability_forecast",
        _fake_calibrate_probability_forecast,
    )

    ax = forecast_mod.plot_calibration_comparison(
        df,
        prob_col="p_event",
        actual_col="event_flag",
        bins=4,
        show_grid=False,
    )

    labels = [line.get_label() for line in ax.lines]
    assert "Perfect" in labels
    assert any("raw" in label for label in labels)
    assert any("calib" in label for label in labels)


def test_plot_forecast_by_step_spatial_smoke():
    from geoprior.plot.forecast import plot_forecast_by_step

    df = _make_quantile_long_df()
    plot_forecast_by_step(
        df=df,
        value_prefixes=["subsidence"],
        kind="spatial",
        steps=[1, 2],
        spatial_cols=("coord_x", "coord_y"),
        max_cols=3,
        show_grid=False,
    )


def test_plot_forecasts_temporal_and_spatial_smoke():
    from geoprior.plot.forecast import plot_forecasts

    df = _make_quantile_long_df()

    plot_forecasts(
        forecast_df=df,
        target_name="subsidence",
        quantiles=[0.1, 0.5, 0.9],
        kind="temporal",
        sample_ids="first_n",
        num_samples=2,
        show=False,
    )

    plot_forecasts(
        forecast_df=df,
        target_name="subsidence",
        quantiles=[0.1, 0.5, 0.9],
        kind="spatial",
        horizon_steps=[1, 2],
        spatial_cols=["coord_x", "coord_y"],
        cbar="uniform",
        show=False,
    )


def test_forecast_view_smoke_with_identity_formatter(
    monkeypatch,
):
    import geoprior.plot.forecast as forecast_mod

    wide_df = _make_wide_forecast_df()

    monkeypatch.setattr(
        forecast_mod,
        "format_forecast_dataframe",
        lambda df, **kwargs: df.copy(),
    )

    forecast_mod.forecast_view(
        forecast_df=wide_df,
        value_prefixes=["subsidence"],
        kind="dual",
        view_years=[2024, 2025],
        view_quantiles=[0.1, 0.5, 0.9],
        spatial_cols=("coord_x", "coord_y"),
        show=False,
    )


def test_plot_eval_future_delegates_to_forecast_view(
    monkeypatch,
):
    import geoprior.plot.forecast as forecast_mod

    eval_df = _make_quantile_long_df()
    future_df = eval_df.copy()

    calls = []

    def _fake_forecast_view(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        forecast_mod, "forecast_view", _fake_forecast_view
    )

    forecast_mod.plot_eval_future(
        df_eval=eval_df,
        df_future=future_df,
        target_name="subsidence",
        quantiles=[0.1, 0.5, 0.9],
        eval_years=[2024],
        future_years=[2025],
        show=False,
    )

    assert len(calls) == 2
    assert calls[0]["kind"] == "dual"
    assert calls[1]["kind"] == "pred_only"
