from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = [
    pytest.mark.script_artifacts,
    pytest.mark.fast_plots,
]


@pytest.fixture
def spatial_inputs(tmp_path: Path) -> dict[str, Path]:
    def make_calib(shift: float) -> pd.DataFrame:
        rows = []
        for sid, (x, y) in enumerate(
            (
                (113.40 + shift, 22.15 + shift),
                (113.42 + shift, 22.17 + shift),
            ),
            start=1,
        ):
            rows.append(
                {
                    "sample_idx": sid,
                    "forecast_step": 0,
                    "coord_t": 2022,
                    "coord_x": x,
                    "coord_y": y,
                    "subsidence_actual": 10.0 + sid,
                    "subsidence_q50": 10.3 + sid,
                }
            )
        return pd.DataFrame(rows)

    def make_future(shift: float) -> pd.DataFrame:
        rows = []
        for sid, (x, y) in enumerate(
            (
                (113.40 + shift, 22.15 + shift),
                (113.42 + shift, 22.17 + shift),
            ),
            start=1,
        ):
            for step, year, q50 in (
                (1, 2025, 12.0 + sid),
                (2, 2026, 13.5 + sid),
            ):
                rows.append(
                    {
                        "sample_idx": sid,
                        "forecast_step": step,
                        "coord_t": year,
                        "coord_x": x,
                        "coord_y": y,
                        "subsidence_q50": q50,
                    }
                )
        return pd.DataFrame(rows)

    ns_cal = tmp_path / "nansha_calib.csv"
    zh_cal = tmp_path / "zhongshan_calib.csv"
    ns_fut = tmp_path / "nansha_future.csv"
    zh_fut = tmp_path / "zhongshan_future.csv"

    make_calib(0.0).to_csv(ns_cal, index=False)
    make_calib(0.2).to_csv(zh_cal, index=False)
    make_future(0.0).to_csv(ns_fut, index=False)
    make_future(0.2).to_csv(zh_fut, index=False)

    return {
        "ns_cal": ns_cal,
        "zh_cal": zh_cal,
        "ns_fut": ns_fut,
        "zh_fut": zh_fut,
    }


def test_plot_spatial_forecasts_main_writes_outputs(
    script_test_env,
    fast_script_figures,
    monkeypatch,
    spatial_inputs,
):
    mod = pytest.importorskip(
        "geoprior._scripts.plot_spatial_forecasts"
    )

    monkeypatch.setattr(
        mod, "_load_calib_df", lambda path: pd.read_csv(path)
    )
    monkeypatch.setattr(
        mod, "_load_future_df", lambda path: pd.read_csv(path)
    )
    target_utils = getattr(mod, "u", None) or mod.utils

    def _fig_out(out):
        return script_test_env["figs_dir"] / Path(str(out))

    if hasattr(target_utils, "resolve_fig_out"):
        monkeypatch.setattr(
            target_utils, "resolve_fig_out", _fig_out
        )

    hotspot_out = (
        script_test_env["tables_dir"]
        / "spatial_forecast_hotspots.csv"
    )

    mod.plot_fig6_spatial_forecasts_main(
        [
            "--ns-calib",
            str(spatial_inputs["ns_cal"]),
            "--zh-calib",
            str(spatial_inputs["zh_cal"]),
            "--ns-future",
            str(spatial_inputs["ns_fut"]),
            "--zh-future",
            str(spatial_inputs["zh_fut"]),
            "--year-val",
            "2022",
            "--years-forecast",
            "2025",
            "2026",
            "--hotspot",
            "delta",
            "--hotspot-out",
            str(hotspot_out),
            "--show-legend",
            "false",
            "--out",
            "spatial_forecast_case",
        ]
    )

    assert (
        script_test_env["figs_dir"]
        / "spatial_forecast_case.png"
    ).exists()
    assert (
        script_test_env["figs_dir"]
        / "spatial_forecast_case.svg"
    ).exists()
    hs = (
        script_test_env["tables_dir"]
        / "spatial_forecast_hotspots.csv"
    )
    assert hs.exists()
    df = pd.read_csv(hs)
    assert set(df["city"]) == {"Nansha", "Zhongshan"}


def test_coord_kind_detects_lonlat_ranges():
    mod = pytest.importorskip(
        "geoprior._scripts.plot_spatial_forecasts"
    )
    df = pd.DataFrame(
        {"coord_x": [113.4, 113.5], "coord_y": [22.1, 22.2]}
    )
    assert mod._coord_kind(df) == "lonlat"
