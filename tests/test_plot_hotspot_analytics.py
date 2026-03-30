from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = [
    pytest.mark.script_artifacts,
    pytest.mark.fast_plots,
]


def _make_eval(city: str) -> pd.DataFrame:
    rows = []
    for sid, (x, y, base) in enumerate(
        (
            (1000.0, 2000.0, 10.0),
            (1010.0, 2010.0, 12.0),
            (1020.0, 2020.0, 14.0),
        ),
        start=1,
    ):
        rows.append(
            {
                "sample_idx": sid,
                "coord_t": 2022,
                "coord_x": x,
                "coord_y": y,
                "subsidence_actual": base,
                "subsidence_q50": base + 0.5,
                "city": city,
            }
        )
    return pd.DataFrame(rows)


def _make_future(city: str) -> pd.DataFrame:
    rows = []
    deltas = {
        2025: [
            (18.0, 20.0, 23.0),
            (19.0, 21.0, 24.0),
            (20.0, 22.0, 25.0),
        ],
        2026: [
            (22.0, 25.0, 28.0),
            (24.0, 27.0, 30.0),
            (26.0, 29.0, 32.0),
        ],
    }
    coords = {
        1: (1000.0, 2000.0),
        2: (1010.0, 2010.0),
        3: (1020.0, 2020.0),
    }

    for year, triples in deltas.items():
        for sid, (q10, q50, q90) in enumerate(
            triples,
            start=1,
        ):
            x, y = coords[sid]
            rows.append(
                {
                    "sample_idx": sid,
                    "coord_t": year,
                    "coord_x": x,
                    "coord_y": y,
                    "subsidence_q10": q10,
                    "subsidence_q50": q50,
                    "subsidence_q90": q90,
                    "city": city,
                }
            )

    return pd.DataFrame(rows)


def test_plot_hotspot_analytics_main_writes_outputs(
    tmp_path,
    script_test_env,
    fast_script_figures,
    monkeypatch,
):
    mod = pytest.importorskip(
        "geoprior._scripts.plot_hotspot_analytics"
    )

    target_utils = getattr(mod, "u", None) or mod.utils

    def _fig_out(out):
        return script_test_env["figs_dir"] / Path(str(out))

    if hasattr(target_utils, "resolve_fig_out"):
        monkeypatch.setattr(
            target_utils,
            "resolve_fig_out",
            _fig_out,
        )

    ns_eval = tmp_path / "nansha_eval.csv"
    zh_eval = tmp_path / "zhongshan_eval.csv"
    ns_fut = tmp_path / "nansha_future.csv"
    zh_fut = tmp_path / "zhongshan_future.csv"

    _make_eval("Nansha").to_csv(ns_eval, index=False)
    _make_eval("Zhongshan").to_csv(zh_eval, index=False)
    _make_future("Nansha").to_csv(ns_fut, index=False)
    _make_future("Zhongshan").to_csv(zh_fut, index=False)

    out_points = (
        script_test_env["tables_dir"]
        / "hotspot_case_points.csv"
    )
    out_years = (
        script_test_env["tables_dir"]
        / "hotspot_case_years.csv"
    )
    out_clusters = (
        script_test_env["tables_dir"]
        / "hotspot_case_clusters.csv"
    )

    mod.plot_hotspot_analytics_main(
        [
            "--ns-eval",
            str(ns_eval),
            "--zh-eval",
            str(zh_eval),
            "--ns-future",
            str(ns_fut),
            "--zh-future",
            str(zh_fut),
            "--years",
            "2025",
            "2026",
            "--focus-year",
            "2026",
            "--out",
            "hotspot_case",
            "--out-points",
            str(out_points),
            "--out-years",
            str(out_years),
            "--out-clusters",
            str(out_clusters),
            "--add-persistence",
        ]
    )

    fig_png = script_test_env["figs_dir"] / "hotspot_case.png"
    fig_svg = script_test_env["figs_dir"] / "hotspot_case.svg"
    pts = (
        script_test_env["tables_dir"]
        / "hotspot_case_points.csv"
    )
    yrs = (
        script_test_env["tables_dir"]
        / "hotspot_case_years.csv"
    )
    cls = (
        script_test_env["tables_dir"]
        / "hotspot_case_clusters.csv"
    )

    assert fig_png.exists()
    assert fig_svg.exists()
    assert pts.exists()
    assert cls.exists()

    years_df = pd.read_csv(yrs)
    assert set(years_df["city"]) == {
        "Nansha",
        "Zhongshan",
    }
    assert "n_hotspots_ever" in years_df.columns
