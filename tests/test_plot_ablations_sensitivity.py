from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = [
    pytest.mark.script_artifacts,
    pytest.mark.fast_plots,
]


@pytest.fixture
def ablation_input_csv(tmp_path: Path) -> Path:
    rows = []
    for city in ("Nansha", "Zhongshan"):
        for lam_cons in (0.1, 1.0):
            for lam_prior in (0.1, 1.0):
                rows.append(
                    {
                        "timestamp": (
                            f"2026-03-30T00:{len(rows):02d}:00Z"
                        ),
                        "city": city,
                        "model": "GeoPriorSubsNet",
                        "pde_mode": "both",
                        "lambda_cons": lam_cons,
                        "lambda_prior": lam_prior,
                        "r2": 0.88,
                        "mae": 9.0 + lam_cons + lam_prior,
                        "rmse": 11.0,
                        "mse": 121.0,
                        "coverage80": 0.80,
                        "sharpness80": 14.0,
                        "epsilon_prior": 0.15,
                        "epsilon_cons": 0.12,
                    }
                )

    out = tmp_path / "ablation_records.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _patch_fig_save(monkeypatch, mod, env):
    def _save_figure(fig, out, dpi=None, **kwargs):
        del kwargs
        base = env["figs_dir"] / Path(str(out))
        if base.suffix:
            base = base.with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".png", dpi=dpi)
        fig.savefig(str(base) + ".pdf", dpi=dpi)

    monkeypatch.setattr(
        mod.utils, "save_figure", _save_figure
    )
    monkeypatch.setattr(
        mod.utils,
        "resolve_fig_out",
        lambda out: env["figs_dir"] / Path(str(out)),
    )


def test_plot_ablations_sensitivity_main_writes_plot_and_table(
    script_test_env,
    fast_script_figures,
    monkeypatch,
    ablation_input_csv: Path,
):
    mod = pytest.importorskip(
        "geoprior.scripts.plot_ablations_sensitivity"
    )
    _patch_fig_save(monkeypatch, mod, script_test_env)

    mod.plot_ablations_sensivity_main(
        [
            "--input",
            str(ablation_input_csv),
            "--heatmap-metric",
            "mae",
            "--no-bars",
            "true",
            "--out",
            "ablations-smoke",
            "--cities",
            "Nansha,Zhongshan",
        ],
        prog="plot-ablations-sensitivity",
    )

    fig_png = (
        script_test_env["figs_dir"] / "ablations-smoke.png"
    )
    fig_pdf = (
        script_test_env["figs_dir"] / "ablations-smoke.pdf"
    )
    tidy_csv = (
        script_test_env["figs_dir"]
        / "tableS6_ablations_used.csv"
    )

    assert fig_png.exists()
    assert fig_pdf.exists()
    assert tidy_csv.exists()

    used = pd.read_csv(tidy_csv)
    assert set(used["city"]) == {"Nansha", "Zhongshan"}
    assert set(used["pde_bucket"]) == {"both"}


def test_canon_pde_mode_normalizes_aliases():
    mod = pytest.importorskip(
        "geoprior.scripts.plot_ablations_sensitivity"
    )

    assert mod._canon_pde_mode("off") == "none"
    assert mod._canon_pde_mode("gw_flow") == "both"
