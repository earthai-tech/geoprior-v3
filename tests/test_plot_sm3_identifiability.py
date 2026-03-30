from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytestmark = [
    pytest.mark.script_artifacts,
    pytest.mark.fast_plots,
]


@pytest.fixture
def sm3_ident_csv(tmp_path):
    n = 8
    df = pd.DataFrame(
        {
            "identify": ["tau"] * n,
            "nx": [21, 21, 21, 25, 25, 29, 29, 29],
            "lith_idx": [0, 1, 2, 3, 0, 1, 2, 3],
            "kappa_b": [1.0] * n,
            "tau_true_sec": [10, 15, 20, 25, 30, 35, 40, 45],
            "tau_prior_sec": [11, 16, 22, 24, 31, 33, 38, 47],
            "tau_true_year": [
                10 / 31556952.0,
                15 / 31556952.0,
                20 / 31556952.0,
                25 / 31556952.0,
                30 / 31556952.0,
                35 / 31556952.0,
                40 / 31556952.0,
                45 / 31556952.0,
            ],
            "tau_prior_year": [
                11 / 31556952.0,
                16 / 31556952.0,
                22 / 31556952.0,
                24 / 31556952.0,
                31 / 31556952.0,
                33 / 31556952.0,
                38 / 31556952.0,
                47 / 31556952.0,
            ],
            "tau_est_med_sec": [
                12,
                14,
                19,
                27,
                29,
                36,
                42,
                43,
            ],
            "tau_est_med_year": [
                12 / 31556952.0,
                14 / 31556952.0,
                19 / 31556952.0,
                27 / 31556952.0,
                29 / 31556952.0,
                36 / 31556952.0,
                42 / 31556952.0,
                43 / 31556952.0,
            ],
            "K_true_mps": [
                1e-7,
                2e-7,
                4e-7,
                8e-7,
                1.2e-6,
                1.6e-6,
                2e-6,
                2.4e-6,
            ],
            "K_prior_mps": [
                1.1e-7,
                2.2e-7,
                4.1e-7,
                7.5e-7,
                1.1e-6,
                1.7e-6,
                2.1e-6,
                2.5e-6,
            ],
            "K_est_med_mps": [
                1.05e-7,
                2.1e-7,
                3.8e-7,
                8.5e-7,
                1.3e-6,
                1.55e-6,
                1.95e-6,
                2.3e-6,
            ],
            "K_est_med_m_per_year": [
                1.05e-7 * 31556952.0,
                2.1e-7 * 31556952.0,
                3.8e-7 * 31556952.0,
                8.5e-7 * 31556952.0,
                1.3e-6 * 31556952.0,
                1.55e-6 * 31556952.0,
                1.95e-6 * 31556952.0,
                2.3e-6 * 31556952.0,
            ],
            "Ss_true": [1.0e-5] * n,
            "Ss_prior": [1.1e-5] * n,
            "Ss_est_med": [
                1.05e-5,
                1.02e-5,
                1.01e-5,
                1.08e-5,
                1.0e-5,
                0.99e-5,
                1.03e-5,
                1.04e-5,
            ],
            "Hd_true": [
                12.0,
                12.5,
                13.0,
                13.5,
                14.0,
                14.5,
                15.0,
                15.5,
            ],
            "Hd_prior": [
                12.0,
                12.6,
                13.1,
                13.3,
                14.2,
                14.3,
                15.1,
                15.2,
            ],
            "Hd_est_med": [
                12.1,
                12.4,
                12.9,
                13.6,
                13.9,
                14.6,
                14.9,
                15.6,
            ],
            "vs_true_delta_K_q50": [
                0.05,
                0.02,
                -0.03,
                0.04,
                0.01,
                -0.02,
                0.03,
                -0.01,
            ],
            "vs_true_delta_Ss_q50": [
                0.01,
                0.02,
                0.00,
                -0.01,
                0.01,
                0.00,
                -0.02,
                0.02,
            ],
            "vs_true_delta_Hd_q50": [
                0.00,
                0.01,
                -0.01,
                0.00,
                0.02,
                -0.01,
                0.01,
                0.00,
            ],
            "eps_prior_rms": [
                0.10,
                0.12,
                0.14,
                0.11,
                0.13,
                0.15,
                0.10,
                0.12,
            ],
            "closure_consistency_rms": [
                0.09,
                0.10,
                0.11,
                0.09,
                0.12,
                0.13,
                0.10,
                0.11,
            ],
        }
    )
    out = tmp_path / "sm3_identifiability.csv"
    df.to_csv(out, index=False)
    return out


def test_plot_sm3_identifiability_main_writes_outputs(
    script_test_env,
    fast_script_figures,
    collect_script_outputs,
    sm3_ident_csv,
):
    from geoprior._scripts.plot_sm3_identifiability import (
        plot_sm3_identifiability_main,
    )

    plot_sm3_identifiability_main(
        [
            "--csv",
            str(sm3_ident_csv),
            "--show-ci",
            "false",
            "--show-legend",
            "false",
            "--out",
            "sm3_ident_case",
            "--out-csv",
            "sm3_ident_case.csv",
            "--out-json",
            "sm3_ident_case.json",
        ]
    )

    pngs = collect_script_outputs("sm3_ident_case.png")
    svgs = collect_script_outputs("sm3_ident_case.svg")
    csvs = collect_script_outputs("sm3_ident_case.csv")
    jsons = collect_script_outputs("sm3_ident_case.json")

    assert pngs, "Expected a PNG artifact."
    assert svgs, "Expected an SVG artifact."
    assert csvs, "Expected a summary CSV export."
    assert jsons, "Expected a summary JSON export."

    payload = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert payload["meta"]["n_rows"] == 8
    assert payload["stats"]


def test_linfit_stats_returns_nan_for_constant_x():
    from geoprior._scripts.plot_sm3_identifiability import (
        _linfit_stats,
    )

    slope, intercept, r2 = _linfit_stats(
        np.array([1.0, 1.0, 1.0]),
        np.array([2.0, 3.0, 4.0]),
    )

    assert np.isnan(slope)
    assert np.isnan(intercept)
    assert np.isnan(r2)
