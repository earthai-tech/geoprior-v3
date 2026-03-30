from __future__ import annotations

import pandas as pd
import pytest


def _make_city_df(city: str) -> pd.DataFrame:
    rows = []
    classes = [
        "Mixed Clastics",
        "Fine-Grained Soil",
        "Coarse-Grained Soil",
    ]
    for year in (2020, 2021, 2022):
        for idx in range(9):
            rows.append(
                {
                    "year": year,
                    "lithology_class": classes[idx % len(classes)],
                    "city": city,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.script_artifacts
@pytest.mark.fast_plots
def test_plot_litho_parity_main_writes_outputs(
    tmp_path,
    script_test_env,
    fast_script_figures,
    collect_script_outputs,
):
    from geoprior._scripts.plot_litho_parity import (
        figS1_lithology_parity_main,
    )

    src = tmp_path / "data"
    src.mkdir(parents=True, exist_ok=True)

    ns_file = src / "nansha.csv"
    zh_file = src / "zhongshan.csv"
    _make_city_df("Nansha").to_csv(ns_file, index=False)
    _make_city_df("Zhongshan").to_csv(zh_file, index=False)

    figS1_lithology_parity_main(
        [
            "--src",
            str(src),
            "--ns-file",
            ns_file.name,
            "--zh-file",
            zh_file.name,
            "--cities",
            "ns,zh",
            "--out",
            "litho_parity_case",
        ]
    )

    pngs = collect_script_outputs("litho_parity_case.png")
    svgs = collect_script_outputs("litho_parity_case.svg")

    assert pngs, "Expected a PNG artifact."
    assert svgs, "Expected an SVG artifact."
