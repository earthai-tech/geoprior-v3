from __future__ import annotations

import json

import pandas as pd
import pytest

pytestmark = [
    pytest.mark.script_artifacts,
    pytest.mark.fast_plots,
]


@pytest.fixture
def sm3_bounds_csv(tmp_path):
    df = pd.DataFrame(
        {
            "identify": ["both"] * 6,
            "nx": [21, 21, 25, 25, 29, 29],
            "lith_idx": [0, 1, 2, 3, 0, 1],
            "K_est_med_mps": [
                1e-7,
                5e-7,
                1e-6,
                2e-6,
                2e-6,
                1e-7,
            ],
            "tau_est_med_sec": [
                5.0,
                8.0,
                10.0,
                20.0,
                5.0,
                20.0,
            ],
            "Hd_est_med": [
                10.0,
                12.0,
                15.0,
                20.0,
                20.0,
                10.0,
            ],
            "ridge_resid_q50": [0.8, 2.2, 3.1, 1.1, 2.8, 0.5],
        }
    )
    out = tmp_path / "sm3_bounds_ridge.csv"
    df.to_csv(out, index=False)
    return out


def test_plot_sm3_bounds_ridge_summary_main_writes_outputs(
    script_test_env,
    fast_script_figures,
    collect_script_outputs,
    sm3_bounds_csv,
):
    from geoprior._scripts.plot_sm3_bounds_ridge_summary import (
        plot_sm3_bounds_ridge_summary_main,
    )

    plot_sm3_bounds_ridge_summary_main(
        [
            "--csv",
            str(sm3_bounds_csv),
            "--ridge-thr",
            "2.0",
            "--out",
            "sm3_bounds_ridge_case",
            "--out-json",
            "sm3_bounds_ridge_case.json",
            "--out-csv",
            "sm3_bounds_ridge_case.csv",
            "--show-legend",
            "false",
        ]
    )

    pngs = collect_script_outputs("sm3_bounds_ridge_case.png")
    svgs = collect_script_outputs("sm3_bounds_ridge_case.svg")
    jsons = collect_script_outputs(
        "sm3_bounds_ridge_case.json"
    )
    csvs = collect_script_outputs("sm3_bounds_ridge_case.csv")

    assert pngs, "Expected a PNG artifact."
    assert svgs, "Expected an SVG artifact."
    assert jsons, "Expected a JSON summary export."
    assert csvs, "Expected a CSV category export."

    payload = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert "summary_any" in payload
    assert payload["summary_any"]["n"] == 6


def test_infer_bounds_returns_run_extrema():
    from geoprior._scripts.plot_sm3_bounds_ridge_summary import (
        infer_bounds,
    )

    df = pd.DataFrame(
        {
            "K_est_med_mps": [1e-7, 2e-6],
            "tau_est_med_sec": [5.0, 20.0],
            "Hd_est_med": [10.0, 30.0],
        }
    )

    bounds = infer_bounds(df)
    assert bounds.K_min == pytest.approx(1e-7)
    assert bounds.tau_max == pytest.approx(20.0)
    assert bounds.Hd_max == pytest.approx(30.0)
