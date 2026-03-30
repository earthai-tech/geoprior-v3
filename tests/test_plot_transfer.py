from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = [
    pytest.mark.script_artifacts,
    pytest.mark.fast_plots,
]


def _make_xfer_csv(path: Path) -> Path:
    rows = []

    def add(direction, strategy, source, target, *, mae, r2, cov, shp):
        rows.append(
            {
                "strategy": strategy,
                "rescale_mode": "as_is" if strategy == "baseline" else "strict",
                "direction": direction,
                "source_city": source,
                "target_city": target,
                "split": "val",
                "calibration": "source",
                "overall_mae": mae,
                "overall_r2": r2,
                "coverage80": cov,
                "sharpness80": shp,
            }
        )

    add("A_to_A", "baseline", "nansha", "nansha", mae=8.0, r2=0.78, cov=0.82, shp=10.0)
    add("B_to_B", "baseline", "zhongshan", "zhongshan", mae=9.0, r2=0.75, cov=0.81, shp=11.0)
    add("A_to_B", "xfer", "nansha", "zhongshan", mae=10.0, r2=0.70, cov=0.79, shp=12.0)
    add("A_to_B", "warm", "nansha", "zhongshan", mae=9.5, r2=0.73, cov=0.80, shp=11.5)
    add("B_to_A", "xfer", "zhongshan", "nansha", mae=8.7, r2=0.69, cov=0.78, shp=10.7)
    add("B_to_A", "warm", "zhongshan", "nansha", mae=8.1, r2=0.72, cov=0.80, shp=10.2)

    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_plot_transfer_main_writes_outputs(
    tmp_path: Path,
    script_test_env,
    fast_script_figures,
    monkeypatch,
):
    mod = pytest.importorskip("geoprior._scripts.plot_transfer")

    monkeypatch.setattr(
        mod.u,
        "resolve_fig_out",
        lambda out: script_test_env["figs_dir"] / Path(str(out)),
    )

    csv_path = _make_xfer_csv(tmp_path / "xfer_transfer.csv")

    mod.figSx_xfer_transferability_main(
        [
            "--xfer-csv",
            str(csv_path),
            "--split",
            "val",
            "--strategies",
            "baseline",
            "xfer",
            "warm",
            "--calib-modes",
            "source",
            "--metric-top",
            "mae",
            "--metric-bottom",
            "r2",
            "--show-legend",
            "false",
            "--out",
            "transfer_case",
        ]
    )

    assert (script_test_env["figs_dir"] / "transfer_case.png").exists()
    assert (script_test_env["figs_dir"] / "transfer_case.svg").exists()


def test_transfer_canon_dir_normalizes_case():
    mod = pytest.importorskip("geoprior._scripts.plot_transfer")
    assert mod._canon_dir("a_to_b") == "A_to_B"
