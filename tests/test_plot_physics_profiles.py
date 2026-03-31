from __future__ import annotations

import json

import pandas as pd
import pytest


def _write_ablation_jsonl(root):
    rows = []
    for city in ("Nansha", "Zhongshan"):
        for lp in (0.1, 1.0, 10.0):
            for lc in (0.1, 1.0, 10.0):
                rows.append(
                    {
                        "city": city,
                        "pde_mode": "both",
                        "lambda_prior": lp,
                        "lambda_cons": lc,
                        "epsilon_prior": 0.2
                        + lp * 0.01
                        + lc * 0.001,
                        "epsilon_cons": 0.3
                        + lc * 0.02
                        + lp * 0.001,
                        "coverage80": 0.8,
                        "sharpness80": 12.0,
                        "r2": 0.7,
                        "mae": 5.0,
                        "mse": 30.0,
                    }
                )

    records_dir = root / "case" / "ablation_records"
    records_dir.mkdir(parents=True, exist_ok=True)
    path = records_dir / "ablation_record_demo.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return path


@pytest.mark.script_artifacts
@pytest.mark.fast_plots
def test_plot_physics_profiles_main_writes_figure_and_table(
    tmp_path,
    script_test_env,
    fast_script_figures,
    collect_script_outputs,
):
    from geoprior.scripts.plot_physics_profiles import (
        figA1_phys_profiles_main,
    )

    _write_ablation_jsonl(tmp_path)

    figA1_phys_profiles_main(
        [
            "--root",
            str(tmp_path),
            "--out",
            "appendix_profiles_case",
            "--show-legend",
            "false",
        ]
    )

    pngs = collect_script_outputs(
        "appendix_profiles_case.png"
    )
    pdfs = collect_script_outputs(
        "appendix_profiles_case.pdf"
    )
    csvs = collect_script_outputs(
        "appendix_table_A1_phys_profiles_tidy.csv"
    )

    assert pngs, "Expected a PNG artifact."
    assert pdfs, "Expected a PDF artifact."
    assert csvs, "Expected the tidy CSV export."

    df = pd.read_csv(csvs[0])
    assert set(df["city"]) == {"Nansha", "Zhongshan"}
    assert "lambda_prior" in df.columns
    assert "lambda_cons" in df.columns
