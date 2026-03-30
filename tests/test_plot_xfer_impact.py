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

    def add(
        direction,
        strategy,
        source,
        target,
        *,
        mae,
        r2,
        cov,
        shp,
        h1,
        h2,
        h3,
    ):
        rows.append(
            {
                "strategy": strategy,
                "rescale_mode": "as_is"
                if strategy == "baseline"
                else "strict",
                "direction": direction,
                "source_city": source,
                "target_city": target,
                "split": "val",
                "calibration": "source",
                "overall_mae": mae,
                "overall_r2": r2,
                "coverage80": cov,
                "sharpness80": shp,
                "per_horizon_r2.H1": h1,
                "per_horizon_r2.H2": h2,
                "per_horizon_r2.H3": h3,
            }
        )

    add(
        "A_to_A",
        "baseline",
        "nansha",
        "nansha",
        mae=8.0,
        r2=0.78,
        cov=0.82,
        shp=10.0,
        h1=0.80,
        h2=0.77,
        h3=0.73,
    )
    add(
        "B_to_B",
        "baseline",
        "zhongshan",
        "zhongshan",
        mae=9.0,
        r2=0.75,
        cov=0.81,
        shp=11.0,
        h1=0.77,
        h2=0.74,
        h3=0.70,
    )
    add(
        "A_to_B",
        "xfer",
        "nansha",
        "zhongshan",
        mae=10.0,
        r2=0.70,
        cov=0.79,
        shp=12.0,
        h1=0.71,
        h2=0.68,
        h3=0.65,
    )
    add(
        "A_to_B",
        "warm",
        "nansha",
        "zhongshan",
        mae=9.4,
        r2=0.73,
        cov=0.80,
        shp=11.4,
        h1=0.74,
        h2=0.71,
        h3=0.67,
    )
    add(
        "B_to_A",
        "xfer",
        "zhongshan",
        "nansha",
        mae=8.7,
        r2=0.69,
        cov=0.78,
        shp=10.8,
        h1=0.70,
        h2=0.67,
        h3=0.63,
    )
    add(
        "B_to_A",
        "warm",
        "zhongshan",
        "nansha",
        mae=8.1,
        r2=0.72,
        cov=0.80,
        shp=10.3,
        h1=0.73,
        h2=0.70,
        h3=0.66,
    )

    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_plot_xfer_impact_main_writes_outputs(
    tmp_path: Path,
    script_test_env,
    fast_script_figures,
    monkeypatch,
):
    mod = pytest.importorskip(
        "geoprior._scripts.plot_xfer_impact"
    )

    monkeypatch.setattr(
        mod.u,
        "resolve_fig_out",
        lambda out: (
            script_test_env["figs_dir"] / Path(str(out))
        ),
    )

    csv_path = _make_xfer_csv(tmp_path / "xfer_results.csv")

    mod.figSx_xfer_impact_main(
        [
            "--xfer-csv",
            str(csv_path),
            "--split",
            "val",
            "--calib",
            "source",
            "--strategies",
            "baseline",
            "xfer",
            "warm",
            "--rescale-mode",
            "strict",
            "--baseline-rescale",
            "as_is",
            "--horizon-metric",
            "r2",
            "--add-hotspots",
            "false",
            "--show-legend",
            "false",
            "--out",
            "xfer_impact_case",
        ]
    )

    assert (
        script_test_env["figs_dir"] / "xfer_impact_case.png"
    ).exists()
    assert (
        script_test_env["figs_dir"] / "xfer_impact_case.svg"
    ).exists()
    assert (
        script_test_env["figs_dir"] / "xfer_impact_case.eps"
    ).exists()


def test_canon_dir_normalizes_lowercase_direction():
    mod = pytest.importorskip(
        "geoprior._scripts.plot_xfer_impact"
    )
    assert mod._canon_dir("a_to_b") == "A_to_B"
