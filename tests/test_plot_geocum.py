from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = [
    pytest.mark.script_artifacts,
    pytest.mark.fast_plots,
]


@pytest.fixture
def geocum_inputs(tmp_path: Path) -> dict[str, Path]:
    def _val(city_shift: float) -> pd.DataFrame:
        rows = []
        for sample_idx in (1, 2):
            for year, act, pred in (
                (2020, 1.0, 1.1),
                (2022, 3.2, 3.0),
            ):
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "coord_t": year,
                        "coord_x": 113.40 + city_shift + 0.01 * sample_idx,
                        "coord_y": 22.15 + city_shift + 0.01 * sample_idx,
                        "subsidence_actual": act + 0.2 * sample_idx,
                        "subsidence_q50": pred + 0.2 * sample_idx,
                    }
                )
        return pd.DataFrame(rows)

    def _future(city_shift: float) -> pd.DataFrame:
        rows = []
        for sample_idx in (1, 2):
            for step, year, pred in (
                (1, 2024, 4.2),
                (2, 2025, 5.1),
            ):
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "forecast_step": step,
                        "coord_t": year,
                        "coord_x": 113.40 + city_shift + 0.01 * sample_idx,
                        "coord_y": 22.15 + city_shift + 0.01 * sample_idx,
                        "subsidence_q50": pred + 0.2 * sample_idx,
                    }
                )
        return pd.DataFrame(rows)

    ns_val = tmp_path / "nansha_val.csv"
    zh_val = tmp_path / "zhongshan_val.csv"
    ns_future = tmp_path / "nansha_future.csv"
    zh_future = tmp_path / "zhongshan_future.csv"

    _val(0.0).to_csv(ns_val, index=False)
    _val(0.3).to_csv(zh_val, index=False)
    _future(0.0).to_csv(ns_future, index=False)
    _future(0.3).to_csv(zh_future, index=False)

    return {
        "ns_val": ns_val,
        "zh_val": zh_val,
        "ns_future": ns_future,
        "zh_future": zh_future,
    }


def _patch_outputs(monkeypatch, mod, env):
    def _save_figure(fig, out, dpi=None, **kwargs):
        del kwargs
        base = env["figs_dir"] / Path(str(out))
        if base.suffix:
            base = base.with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".png", dpi=dpi)
        fig.savefig(str(base) + ".pdf", dpi=dpi)

    monkeypatch.setattr(mod.utils, "save_figure", _save_figure)
    monkeypatch.setattr(
        mod.utils,
        "resolve_fig_out",
        lambda out: env["figs_dir"] / Path(str(out)),
    )


def test_plot_geocum_main_writes_png_and_pdf(
    script_test_env,
    fast_script_figures,
    monkeypatch,
    geocum_inputs: dict[str, Path],
):
    pytest.importorskip("geopandas")
    pytest.importorskip("contextily")
    mod = pytest.importorskip("geoprior._scripts.plot_geocum")

    _patch_outputs(monkeypatch, mod, script_test_env)
    monkeypatch.setattr(mod.cx, "add_basemap", lambda *a, **k: None)

    mod.plot_geo_cumulative_main(
        [
            "--ns-val",
            str(geocum_inputs["ns_val"]),
            "--zh-val",
            str(geocum_inputs["zh_val"]),
            "--ns-future",
            str(geocum_inputs["ns_future"]),
            "--zh-future",
            str(geocum_inputs["zh_future"]),
            "--start-year",
            "2020",
            "--year-val",
            "2022",
            "--years-forecast",
            "2024",
            "2025",
            "--coords-mode",
            "lonlat",
            "--show-legend",
            "false",
            "--out",
            "geocum-smoke",
        ],
        prog="plot-geo-cumulative",
    )

    fig_png = script_test_env["figs_dir"] / "geocum-smoke.png"
    fig_pdf = script_test_env["figs_dir"] / "geocum-smoke.pdf"

    assert fig_png.exists()
    assert fig_pdf.exists()


def test_infer_xy_crs_detects_lonlat():
    mod = pytest.importorskip("geoprior._scripts.plot_geocum")

    df = pd.DataFrame(
        {
            "coord_x": [113.4, 113.5],
            "coord_y": [22.1, 22.2],
        }
    )

    crs = mod.infer_xy_crs(
        df,
        mode="auto",
        utm_epsg=32649,
    )

    assert crs == "EPSG:4326"
