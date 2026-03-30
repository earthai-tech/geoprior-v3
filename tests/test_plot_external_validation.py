from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = [
    pytest.mark.script_artifacts,
    pytest.mark.fast_plots,
]


@pytest.fixture
def site_validation_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "well_id": ["SW1", "SW2", "SW3"],
            "x": [113.40, 113.41, 113.42],
            "y": [22.15, 22.16, 22.17],
            "matched_pixel_x": [113.40, 113.41, 113.42],
            "matched_pixel_y": [22.15, 22.16, 22.17],
            "model_H_eff_m": [11.0, 13.0, 12.5],
            "model_K_mps": [1.2e-5, 1.4e-5, 1.1e-5],
            "approx_compressible_thickness_m": [10.5, 12.8, 12.0],
            "step3_specific_capacity_Lps_per_m": [5.0, 6.0, 4.8],
            "match_distance_m": [10.0, 15.0, 8.0],
        }
    )
    out = tmp_path / "site_validation.csv"
    df.to_csv(out, index=False)
    return out


@pytest.fixture
def boundary_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "x": [113.39, 113.43, 113.43, 113.39, 113.39],
            "y": [22.14, 22.14, 22.18, 22.18, 22.14],
        }
    )
    out = tmp_path / "boundary.csv"
    df.to_csv(out, index=False)
    return out


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

    monkeypatch.setattr(mod.utils, "save_figure", _save_figure)
    monkeypatch.setattr(mod.utils, "resolve_out_out", _out_table)


def test_plot_external_validation_main_writes_plot_and_json(
    script_test_env,
    fast_script_figures,
    monkeypatch,
    site_validation_csv: Path,
    boundary_csv: Path,
):
    mod = pytest.importorskip(
        "geoprior._scripts.plot_external_validation"
    )
    _patch_outputs(monkeypatch, mod, script_test_env)

    pix = pd.DataFrame(
        {
            "x": [113.40, 113.41, 113.42],
            "y": [22.15, 22.16, 22.17],
            "H_eff_input_m": [11.2, 13.1, 12.4],
            "K_mps": [1.1e-5, 1.3e-5, 1.2e-5],
            "Hd_m": [8.0, 8.5, 8.2],
            "pixel_idx": [0, 1, 2],
        }
    )
    monkeypatch.setattr(mod, "build_pixel_table", lambda *a, **k: pix)

    mod.plot_external_validation_main(
        [
            "--site-csv",
            str(site_validation_csv),
            "--full-inputs-npz",
            "dummy_inputs.npz",
            "--full-payload-npz",
            "dummy_payload.npz",
            "--out",
            "external-validation-smoke",
            "--out-json",
            "external-validation-smoke.json",
            "--boundary",
            str(boundary_csv),
            "--paper-format",
        ],
        prog="plot-external-validation",
    )

    fig_png = (
        script_test_env["figs_dir"]
        / "external-validation-smoke.png"
    )
    fig_svg = (
        script_test_env["figs_dir"]
        / "external-validation-smoke.svg"
    )
    out_json = (
        script_test_env["tables_dir"]
        / "external-validation-smoke.json"
    )

    assert fig_png.exists()
    assert fig_svg.exists()
    assert out_json.exists()

    rec = json.loads(out_json.read_text(encoding="utf-8"))
    assert rec["city"] == "Zhongshan"
    assert rec["n_sites"] == 3
    assert "borehole_vs_H_eff" in rec


def test_load_boundary_xy_reads_simple_csv(boundary_csv: Path):
    mod = pytest.importorskip(
        "geoprior._scripts.plot_external_validation"
    )

    bx, by = mod.load_boundary_xy(str(boundary_csv))

    assert bx.shape == by.shape
    assert bx.size == 5
