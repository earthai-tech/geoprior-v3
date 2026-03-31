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
    for sid, (x, y) in enumerate(
        ((113.40, 22.15), (113.42, 22.17)), start=1
    ):
        for step, year, act, q10, q50, q90 in (
            (
                1,
                2024,
                10.0 + sid,
                9.0 + sid,
                10.1 + sid,
                11.2 + sid,
            ),
            (
                2,
                2025,
                11.0 + sid,
                10.0 + sid,
                11.1 + sid,
                12.3 + sid,
            ),
            (
                3,
                2026,
                12.0 + sid,
                11.1 + sid,
                12.2 + sid,
                13.5 + sid,
            ),
        ):
            rows.append(
                {
                    "sample_idx": sid,
                    "forecast_step": step,
                    "coord_t": year,
                    "coord_x": x,
                    "coord_y": y,
                    "subsidence_actual": act,
                    "subsidence_q10": q10,
                    "subsidence_q50": q50,
                    "subsidence_q90": q90,
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
        fig.savefig(str(base) + ".svg", dpi=dpi)

    def _out_table(out):
        path = env["tables_dir"] / Path(str(out))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(
        mod.utils, "save_figure", _save_figure
    )
    monkeypatch.setattr(
        mod.utils, "resolve_out_out", _out_table
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
    monkeypatch.setattr(
        mod.utils, "phys_json_to_mm", lambda meta: meta or {}
    )


def test_plot_uncertainty_main_writes_outputs(
    tmp_path: Path,
    script_test_env,
    fast_script_figures,
    monkeypatch,
):
    mod = pytest.importorskip(
        "geoprior.scripts.plot_uncertainty"
    )
    _patch_outputs(monkeypatch, mod, script_test_env)

    ns_csv = tmp_path / "nansha_forecast.csv"
    zh_csv = tmp_path / "zhongshan_forecast.csv"
    ns_json = tmp_path / "nansha_phys.json"
    zh_json = tmp_path / "zhongshan_phys.json"

    _make_forecast("Nansha").to_csv(ns_csv, index=False)
    _make_forecast("Zhongshan").to_csv(zh_csv, index=False)
    ns_json.write_text("{}", encoding="utf-8")
    zh_json.write_text("{}", encoding="utf-8")

    mod.plot_fig5_uncertainty_main(
        [
            "--ns-forecast",
            str(ns_csv),
            "--zh-forecast",
            str(zh_csv),
            "--ns-phys-json",
            str(ns_json),
            "--zh-phys-json",
            str(zh_json),
            "--show-legend",
            "false",
            "--show-json-notes",
            "false",
            "--out",
            "uncertainty_case",
            "--out-csv",
            "uncertainty_case.csv",
        ]
    )

    assert (
        script_test_env["figs_dir"] / "uncertainty_case.png"
    ).exists()
    assert (
        script_test_env["figs_dir"] / "uncertainty_case.svg"
    ).exists()
    out_csv = (
        script_test_env["tables_dir"] / "uncertainty_case.csv"
    )
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert set(df["city"]) == {"Nansha", "Zhongshan"}
    assert "emp_q50" in df.columns


def test_interval_stats_returns_expected_columns():
    mod = pytest.importorskip(
        "geoprior.scripts.plot_uncertainty"
    )
    df = _make_forecast("Nansha")
    out = mod.interval_stats(df, by_horizon=False)
    assert {"coverage80", "sharpness80"}.issubset(out.columns)
