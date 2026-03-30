from __future__ import annotations

import pandas as pd
import pytest

pytestmark = [
    pytest.mark.script_artifacts,
    pytest.mark.fast_plots,
]


@pytest.fixture
def physics_sensitivity_csv(tmp_path):
    rows = []
    for city in ("Nansha", "Zhongshan"):
        for lam_prior in (0.1, 1.0, 10.0):
            for lam_cons in (0.1, 1.0, 10.0):
                rows.append(
                    {
                        "timestamp": (
                            f"2026-03-30T00:{len(rows):02d}:00Z"
                        ),
                        "city": city,
                        "model": "GeoPriorSubsNet",
                        "pde_mode": "both",
                        "lambda_prior": lam_prior,
                        "lambda_cons": lam_cons,
                        "epsilon_prior": (
                            0.20 + 0.01 * lam_prior + 0.005 * lam_cons
                        ),
                        "epsilon_cons": (
                            0.30 + 0.02 * lam_cons + 0.002 * lam_prior
                        ),
                        "coverage80": 0.81,
                        "sharpness80": 12.0,
                        "r2": 0.72,
                        "mae": 8.0,
                        "mse": 90.0,
                        "rmse": 9.49,
                        "pss": 0.11,
                    }
                )

    out = tmp_path / "physics_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def test_plot_physics_sensitivity_main_writes_outputs(
    script_test_env,
    fast_script_figures,
    collect_script_outputs,
    physics_sensitivity_csv,
):
    from geoprior._scripts.plot_physics_sensitivity import (
        plot_physics_sensitivity_main,
    )

    plot_physics_sensitivity_main(
        [
            "--input",
            str(physics_sensitivity_csv),
            "--cities",
            "ns,zh",
            "--render",
            "heatmap",
            "--show-legend",
            "false",
            "--out",
            "physics_sensitivity_case",
        ]
    )

    pngs = collect_script_outputs("physics_sensitivity_case.png")
    pdfs = collect_script_outputs("physics_sensitivity_case.pdf")
    csvs = collect_script_outputs("tableS7_physics_used.csv")

    assert pngs, "Expected a PNG artifact."
    assert pdfs, "Expected a PDF artifact."
    assert csvs, "Expected the used-table CSV export."

    df = pd.read_csv(csvs[0])
    assert set(df["city"]) == {"Nansha", "Zhongshan"}
    assert "epsilon_prior" in df.columns
    assert "epsilon_cons" in df.columns


def test_best_point_prefers_lower_metric_when_needed():
    from geoprior._scripts.plot_physics_sensitivity import _best_point

    df = pd.DataFrame(
        {
            "lambda_prior": [0.1, 1.0, 10.0],
            "lambda_cons": [0.1, 1.0, 10.0],
            "epsilon_prior": [0.3, 0.2, 0.4],
        }
    )

    assert _best_point(df, "epsilon_prior") == pytest.approx(
        (1.0, 1.0)
    )
