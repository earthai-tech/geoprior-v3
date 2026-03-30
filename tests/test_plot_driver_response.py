from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = [
    pytest.mark.script_artifacts,
    pytest.mark.fast_plots,
]


def _patch_fig_out(monkeypatch, mod, env):
    def _resolve(out):
        base = env["figs_dir"] / Path(str(out))
        if base.suffix:
            return base.with_suffix("")
        return base

    monkeypatch.setattr(
        mod.utils, "resolve_fig_out", _resolve
    )


@pytest.fixture
def driver_response_csv(
    tmp_path: Path,
    city_panel_df: pd.DataFrame,
) -> Path:
    ns = city_panel_df.copy()
    ns["city"] = "Nansha"

    zh = city_panel_df.copy()
    zh["city"] = "Zhongshan"
    zh["longitude"] = zh["longitude"] + 0.3
    zh["latitude"] = zh["latitude"] + 0.3
    zh["head_m"] = zh["head_m"] + 0.15

    df = pd.concat([ns, zh], ignore_index=True)
    out = tmp_path / "driver_response.csv"
    df.to_csv(out, index=False)
    return out


def test_plot_driver_response_main_writes_outputs(
    script_test_env,
    fast_script_figures,
    monkeypatch,
    driver_response_csv: Path,
):
    mod = pytest.importorskip(
        "geoprior._scripts.plot_driver_response"
    )
    _patch_fig_out(monkeypatch, mod, script_test_env)

    mod.plot_driver_response_main(
        [
            "--src",
            str(driver_response_csv),
            "--cities",
            "ns,zh",
            "--drivers",
            ("rainfall_mm,GWL_depth_bgs_m,urban_load_global"),
            "--response",
            "head_m",
            "--trend",
            "false",
            "--out",
            "driver-response-smoke",
        ],
        prog="plot-driver-response",
    )

    png = (
        script_test_env["figs_dir"]
        / "driver-response-smoke.png"
    )
    svg = (
        script_test_env["figs_dir"]
        / "driver-response-smoke.svg"
    )

    assert png.exists()
    assert svg.exists()


def test_robust_trend_returns_none_for_small_samples():
    mod = pytest.importorskip(
        "geoprior._scripts.plot_driver_response"
    )

    x = pd.Series([1.0, 2.0, 3.0])
    y = pd.Series([2.0, 3.0, 4.0])

    xi, yi = mod.robust_trend(x, y, nbins=5, min_n=2)

    assert xi is None
    assert yi is None
