from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = [
    pytest.mark.script_artifacts,
    pytest.mark.fast_plots,
]


@pytest.fixture
def core_ablation_df() -> pd.DataFrame:
    rows = [
        {
            "city": "Nansha",
            "variant": "with-phys",
            "r2": 0.92,
            "mae": 7.5,
            "mse": 72.0,
            "rmse": 8.49,
            "coverage80": 0.81,
            "sharpness80": 14.0,
        },
        {
            "city": "Nansha",
            "variant": "no-phys",
            "r2": 0.85,
            "mae": 10.2,
            "mse": 110.0,
            "rmse": 10.49,
            "coverage80": 0.76,
            "sharpness80": 17.5,
        },
        {
            "city": "Zhongshan",
            "variant": "with-phys",
            "r2": 0.89,
            "mae": 8.1,
            "mse": 80.0,
            "rmse": 8.94,
            "coverage80": 0.83,
            "sharpness80": 13.8,
        },
        {
            "city": "Zhongshan",
            "variant": "no-phys",
            "r2": 0.80,
            "mae": 11.4,
            "mse": 126.0,
            "rmse": 11.22,
            "coverage80": 0.74,
            "sharpness80": 18.1,
        },
    ]
    return pd.DataFrame(rows)


def _patch_outputs(monkeypatch, mod, env):
    def _fig_out(out):
        base = env["figs_dir"] / Path(str(out))
        if base.suffix:
            return base.with_suffix("")
        return base

    def _table_out(out):
        path = env["tables_dir"] / Path(str(out))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(
        mod.utils, "resolve_fig_out", _fig_out
    )
    monkeypatch.setattr(
        mod.utils, "resolve_out_out", _table_out
    )


def test_plot_core_ablation_main_writes_artifacts(
    script_test_env,
    fast_script_figures,
    monkeypatch,
    core_ablation_df: pd.DataFrame,
):
    mod = pytest.importorskip(
        "geoprior._scripts.plot_core_ablation"
    )
    _patch_outputs(monkeypatch, mod, script_test_env)

    monkeypatch.setattr(
        mod,
        "collect_fig3_metrics",
        lambda **_: core_ablation_df.copy(),
    )

    monkeypatch.setattr(
        pd.DataFrame,
        "to_latex",
        lambda self, *args, **kwargs: "% dummy latex table\n",
    )

    mod.plot_fig3_core_ablation_main(
        [
            "--out",
            "fig3-core-test",
            "--out-csv",
            "fig3-core-metrics.csv",
        ],
        prog="plot-core-ablation",
    )

    fig_png = (
        script_test_env["figs_dir"] / "fig3-core-test.png"
    )
    fig_svg = (
        script_test_env["figs_dir"] / "fig3-core-test.svg"
    )
    out_csv = (
        script_test_env["tables_dir"]
        / "fig3-core-metrics.csv"
    )

    assert fig_png.exists()
    assert fig_svg.exists()
    assert out_csv.exists()

    written = pd.read_csv(out_csv)
    assert set(written["city"]) == {"Nansha", "Zhongshan"}
    assert set(written["variant"]) == {
        "with-phys",
        "no-phys",
    }


def test_pick_rmse_falls_back_to_mse_square_root():
    mod = pytest.importorskip(
        "geoprior._scripts.plot_core_ablation"
    )

    rmse = mod._pick_rmse({}, {}, 25.0)

    assert rmse == pytest.approx(5.0)
