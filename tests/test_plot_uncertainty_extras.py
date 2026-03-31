from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = [
    pytest.mark.script_artifacts,
    pytest.mark.fast_plots,
]


def _make_forecast(city: str) -> pd.DataFrame:
    rows = []
    for sid in (1, 2):
        for step, year, act, q10, q50, q90 in (
            (
                1,
                2024,
                10.0 + sid,
                9.0 + sid,
                10.1 + sid,
                11.1 + sid,
            ),
            (
                2,
                2025,
                11.0 + sid,
                10.0 + sid,
                11.2 + sid,
                12.2 + sid,
            ),
            (
                3,
                2026,
                12.0 + sid,
                11.0 + sid,
                12.3 + sid,
                13.4 + sid,
            ),
        ):
            rows.append(
                {
                    "forecast_step": step,
                    "subsidence_q10": q10,
                    "subsidence_q50": q50,
                    "subsidence_q90": q90,
                    "subsidence_actual": act,
                    "city": city,
                }
            )
    return pd.DataFrame(rows)


def _patch_outputs(monkeypatch, mod, env):
    def _save_figure(fig, out, dpi=None, **kwargs):
        del kwargs
        base = env["figs_dir"] / Path(str(out))
        if base.suffix:
            base = base.with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".png", dpi=dpi)
        fig.savefig(str(base) + ".pdf", dpi=dpi)

    def _fig_out(out):
        path = env["figs_dir"] / Path(str(out))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _out_path(out):
        path = env["tables_dir"] / Path(str(out))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(
        mod.utils, "save_figure", _save_figure
    )
    monkeypatch.setattr(
        mod.utils, "resolve_fig_out", _fig_out
    )
    monkeypatch.setattr(
        mod.utils, "resolve_out_out", _out_path
    )
    monkeypatch.setattr(
        mod.utils,
        "load_forecast_csv",
        lambda p: pd.read_csv(p),
    )
    monkeypatch.setattr(
        mod.utils,
        "safe_load_json",
        lambda p: (
            json.loads(Path(p).read_text(encoding="utf-8"))
            if p
            else {}
        ),
    )


def test_plot_uncertainty_extras_main_writes_outputs(
    tmp_path: Path,
    script_test_env,
    fast_script_figures,
    monkeypatch,
):
    mod = pytest.importorskip(
        "geoprior.scripts.plot_uncertainty_extras"
    )
    _patch_outputs(monkeypatch, mod, script_test_env)

    ns_csv = tmp_path / "nansha_forecast.csv"
    zh_csv = tmp_path / "zhongshan_forecast.csv"
    ns_json = tmp_path / "nansha_phys.json"
    zh_json = tmp_path / "zhongshan_phys.json"

    _make_forecast("Nansha").to_csv(ns_csv, index=False)
    _make_forecast("Zhongshan").to_csv(zh_csv, index=False)

    meta = {
        "interval_calibration": {
            "factors_per_horizon": {
                "H1": 1.1,
                "H2": 1.0,
                "H3": 0.9,
            }
        }
    }
    ns_json.write_text(json.dumps(meta), encoding="utf-8")
    zh_json.write_text(json.dumps(meta), encoding="utf-8")

    mod.supp_figS5_uncertainty_extras_main(
        [
            "--ns-forecast",
            str(ns_csv),
            "--zh-forecast",
            str(zh_csv),
            "--ns-phys-json",
            str(ns_json),
            "--zh-phys-json",
            str(zh_json),
            "--no-legends",
            "--out",
            "uncertainty_extras_case",
            "--tables-stem",
            "uncertainty_extras_tables",
        ]
    )

    assert (
        script_test_env["figs_dir"]
        / "uncertainty_extras_case.png"
    ).exists()
    assert (
        script_test_env["figs_dir"]
        / "uncertainty_extras_case.pdf"
    ).exists()
    assert (
        script_test_env["tables_dir"]
        / "uncertainty_extras_tables.csv"
    ).exists()
    assert (
        script_test_env["tables_dir"]
        / "uncertainty_extras_tables.tex"
    ).exists()


def test_extract_factors_per_horizon_accepts_dict():
    mod = pytest.importorskip(
        "geoprior.scripts.plot_uncertainty_extras"
    )
    meta = {
        "interval_calibration": {
            "factors_per_horizon": {"H2": 0.9, "H1": 1.1}
        }
    }
    assert mod._extract_factors_per_horizon(meta) == [
        1.1,
        0.9,
    ]
