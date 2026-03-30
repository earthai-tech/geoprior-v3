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
        calibration,
        *,
        mae,
        mse,
        r2,
        cov,
        shp,
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
                "calibration": calibration,
                "overall_mae": mae,
                "overall_mse": mse,
                "overall_r2": r2,
                "coverage80": cov,
                "sharpness80": shp,
            }
        )

    add(
        "a_to_a",
        "baseline",
        "nansha",
        "nansha",
        "source",
        mae=8.0,
        mse=90.0,
        r2=0.78,
        cov=0.82,
        shp=10.0,
    )
    add(
        "b_to_b",
        "baseline",
        "zhongshan",
        "zhongshan",
        "source",
        mae=9.0,
        mse=110.0,
        r2=0.75,
        cov=0.81,
        shp=11.0,
    )
    add(
        "a_to_b",
        "xfer",
        "nansha",
        "zhongshan",
        "source",
        mae=10.0,
        mse=130.0,
        r2=0.70,
        cov=0.79,
        shp=12.0,
    )
    add(
        "a_to_b",
        "warm",
        "nansha",
        "zhongshan",
        "source",
        mae=9.5,
        mse=122.0,
        r2=0.73,
        cov=0.80,
        shp=11.5,
    )
    add(
        "b_to_a",
        "xfer",
        "zhongshan",
        "nansha",
        "source",
        mae=8.7,
        mse=101.0,
        r2=0.69,
        cov=0.78,
        shp=10.7,
    )
    add(
        "b_to_a",
        "warm",
        "zhongshan",
        "nansha",
        "source",
        mae=8.1,
        mse=96.0,
        r2=0.72,
        cov=0.80,
        shp=10.2,
    )

    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_plot_xfer_transferability_main_writes_outputs(
    tmp_path: Path,
    script_test_env,
    fast_script_figures,
    monkeypatch,
):
    mod = pytest.importorskip(
        "geoprior._scripts.plot_xfer_transferability"
    )

    monkeypatch.setattr(
        mod.u,
        "resolve_fig_out",
        lambda out: (
            script_test_env["figs_dir"] / Path(str(out))
        ),
    )

    csv_path = _make_xfer_csv(
        tmp_path / "xfer_transferability.csv"
    )

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
            "mse",
            "--show-legend",
            "false",
            "--out",
            "xfer_transferability_case",
        ]
    )

    assert (
        script_test_env["figs_dir"]
        / "xfer_transferability_case.png"
    ).exists()
    assert (
        script_test_env["figs_dir"]
        / "xfer_transferability_case.svg"
    ).exists()


def test_subset_uses_target_baseline_for_cross_city_rows():
    mod = pytest.importorskip(
        "geoprior._scripts.plot_xfer_transferability"
    )

    df = pd.DataFrame(
        {
            "strategy": ["baseline", "baseline", "xfer"],
            "rescale_mode": ["as_is", "as_is", "strict"],
            "direction": ["a_to_a", "b_to_b", "a_to_b"],
            "source_city": ["nansha", "zhongshan", "nansha"],
            "target_city": [
                "nansha",
                "zhongshan",
                "zhongshan",
            ],
            "split": ["val", "val", "val"],
            "calibration": ["source", "source", "source"],
            "overall_mae": [8.0, 9.0, 10.0],
            "overall_mse": [90.0, 110.0, 130.0],
            "overall_r2": [0.78, 0.75, 0.70],
            "coverage80": [0.82, 0.81, 0.79],
            "sharpness80": [10.0, 11.0, 12.0],
        }
    )
    df = mod._canon_cols(df)

    sub = mod._subset(
        df,
        direction="a_to_b",
        strategy="baseline",
        split="val",
        calib="source",
        rescale_mode="strict",
        baseline_rescale="as_is",
    )

    assert not sub.empty
    assert set(sub["direction"]) == {"b_to_b"}
