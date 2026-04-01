from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = [pytest.mark.script_artifacts]


def _import_target(name: str):
    candidates = (
        f"geoprior.scripts.{name}",
        f"geoprior._scripts.{name}",
        name,
    )
    for modname in candidates:
        try:
            return importlib.import_module(modname)
        except ModuleNotFoundError as exc:
            name = str(getattr(exc, "name", "") or "")
            if modname == name or modname.startswith(name + "."):
                continue
            raise
    pytest.skip(f"Could not import target module for {name!r}.")


@pytest.fixture
def metrics_results_root(tmp_path: Path) -> Path:
    run_dir = tmp_path / "results" / "run_a" / "ablation_records"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_rec = {
        "timestamp": "2026-04-01T00:00:00Z",
        "city": "Nansha",
        "model": "GeoPriorSubsNet",
        "pde_mode": "both",
        "lambda_cons": 1.0,
        "lambda_gw": 1.0,
        "lambda_prior": 1.0,
        "lambda_smooth": 0.1,
        "lambda_mv": 0.0,
        "mae": 9.0,
        "mse": 81.0,
        "r2": 0.74,
        "coverage80": 0.79,
        "sharpness80": 12.5,
        "per_horizon_r2": {"1": 0.78, "2": 0.73},
        "per_horizon_mae": {"1": 8.5, "2": 9.4},
        "interval_calibration": {
            "coverage80_uncalibrated": 0.70,
            "coverage80_calibrated": 0.80,
            "sharpness80_uncalibrated": 14.0,
            "sharpness80_calibrated": 12.5,
            "factors_per_horizon": {"H1": 1.1, "H2": 1.2},
            "factors_per_horizon_from_cal_stats": {
                "eval_before": {
                    "per_horizon": {
                        "1": {"coverage": 0.69, "sharpness": 14.2},
                        "2": {"coverage": 0.71, "sharpness": 14.5},
                    }
                },
                "eval_after": {
                    "per_horizon": {
                        "1": {"coverage": 0.79, "sharpness": 12.8},
                        "2": {"coverage": 0.80, "sharpness": 13.0},
                    }
                },
            },
        },
        "units": {"subs_metrics_unit": "mm", "time_units": "year"},
    }

    plain_path = run_dir / "ablation_record.jsonl"
    updated_path = run_dir / "ablation_record.updated.jsonl"

    plain = dict(base_rec)
    plain["mae"] = 10.0
    plain["mse"] = 100.0

    updated = dict(base_rec)
    updated["mae"] = 8.0
    updated["mse"] = 64.0

    with plain_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(plain) + "\n")
    with updated_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(updated) + "\n")

    return tmp_path / "results"


def test_build_model_metrics_main_writes_wide_and_long_tables(
    tmp_path: Path,
    metrics_results_root: Path,
):
    mod = _import_target("build_model_metrics")

    mod.build_model_metrics_main(
        [
            "--src",
            str(metrics_results_root),
            "--out-dir",
            str(tmp_path),
            "--out",
            "model_metrics_case",
            "--include-long",
            "true",
            "--dedupe",
            "true",
        ],
        prog="build-model-metrics",
    )

    wide_csv = tmp_path / "model_metrics_case.csv"
    wide_json = tmp_path / "model_metrics_case.json"
    long_csv = tmp_path / "model_metrics_case_long.csv"
    long_json = tmp_path / "model_metrics_case_long.json"

    assert wide_csv.exists()
    assert wide_json.exists()
    assert long_csv.exists()
    assert long_json.exists()

    wide = pd.read_csv(wide_csv)
    assert len(wide) == 1
    assert float(wide.iloc[0]["mae"]) == 8.0
    assert "r2_H1" in wide.columns
    assert "coverage80_cal_H1" in wide.columns

    long_df = pd.read_csv(long_csv)
    assert set(long_df["horizon"]) == {"H1", "H2"}


def test_needs_si_to_mm_detects_legacy_meter_scale():
    mod = _import_target("build_model_metrics")

    rec = {
        "mae": 0.05,
        "mse": 0.0025,
        "sharpness80": 0.08,
    }

    assert mod._needs_si_to_mm(rec) is True
