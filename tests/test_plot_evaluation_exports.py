import importlib


def test_public_evaluation_reexports_underlying_functions():
    public = importlib.import_module(
        "geoprior.plot.evaluation"
    )
    core = importlib.import_module(
        "geoprior.plot._evaluation"
    )
    metrics = importlib.import_module(
        "geoprior.plot._metrics"
    )

    expected = {
        "plot_coverage",
        "plot_crps",
        "plot_mean_interval_width",
        "plot_prediction_stability",
        "plot_quantile_calibration",
        "plot_theils_u_score",
        "plot_time_weighted_metric",
        "plot_weighted_interval_score",
        "plot_metric_radar",
        "plot_forecast_comparison",
        "plot_metric_over_horizon",
        "plot_radar_scores",
        "plot_qce_donut",
    }
    assert expected.issubset(set(public.__all__))

    assert (
        public.plot_metric_over_horizon
        is core.plot_metric_over_horizon
    )
    assert public.plot_metric_radar is core.plot_metric_radar
    assert (
        public.plot_forecast_comparison
        is core.plot_forecast_comparison
    )

    assert public.plot_coverage is metrics.plot_coverage
    assert public.plot_crps is metrics.plot_crps
    assert (
        public.plot_mean_interval_width
        is metrics.plot_mean_interval_width
    )
    assert (
        public.plot_prediction_stability
        is metrics.plot_prediction_stability
    )
    assert (
        public.plot_quantile_calibration
        is metrics.plot_quantile_calibration
    )
    assert (
        public.plot_theils_u_score
        is metrics.plot_theils_u_score
    )
    assert (
        public.plot_time_weighted_metric
        is metrics.plot_time_weighted_metric
    )
    assert (
        public.plot_weighted_interval_score
        is metrics.plot_weighted_interval_score
    )
    assert (
        public.plot_radar_scores is metrics.plot_radar_scores
    )
    assert public.plot_qce_donut is metrics.plot_qce_donut


def test_package_plot_init_exposes_evaluation_functions():
    pkg = importlib.import_module("geoprior.plot")
    public = importlib.import_module(
        "geoprior.plot.evaluation"
    )

    for name in [
        "plot_metric_over_horizon",
        "plot_metric_radar",
        "plot_forecast_comparison",
        "plot_coverage",
        "plot_qce_donut",
    ]:
        assert hasattr(pkg, name)
        assert getattr(pkg, name) is getattr(public, name)
