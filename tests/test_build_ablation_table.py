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
def ablation_jsonl(tmp_path: Path) -> Path:
    rows = [
        {
            "timestamp": "2026-04-01T00:00:00Z",
            "city": "Nansha",
            "model": "GeoPriorSubsNet",
            "pde_mode": "both",
            "lambda_prior": 1.0,
            "lambda_cons": 0.5,
            "mae": 8.5,
            "mse": 80.0,
            "rmse": 8.94427191,
            "r2": 0.81,
            "coverage80": 0.79,
            "sharpness80": 12.0,
            "epsilon_prior": 0.10,
            "epsilon_cons": 0.20,
            "per_horizon_mae": {"H1": 7.5, "H2": 8.2},
            "per_horizon_r2": {"H1": 0.84, "H2": 0.79},
            "units": {"subs_metrics_unit": "mm"},
        },
        {
            "timestamp": "2026-04-01T01:00:00Z",
            "city": "Zhongshan",
            "model": "GeoPriorSubsNet",
            "pde_mode": "none",
            "lambda_prior": 0.0,
            "lambda_cons": 0.0,
            "mae": 9.2,
            "mse": 95.0,
            "rmse": 9.74679434,
            "r2": 0.76,
            "coverage80": 0.81,
            "sharpness80": 13.5,
            "epsilon_prior": 0.00,
            "epsilon_cons": 0.00,
            "per_horizon_mae": {"H1": 8.8, "H2": 9.4},
            "per_horizon_r2": {"H1": 0.78, "H2": 0.74},
            "units": {"subs_metrics_unit": "mm"},
        },
    ]

    path = tmp_path / "ablation_record.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def test_build_ablation_table_main_writes_csv_json_and_tex(
    tmp_path: Path,
    ablation_jsonl: Path,
):
    mod = _import_target("build_ablation_table")

    mod.build_ablation_table_main(
        [
            "--input",
            str(ablation_jsonl),
            "--out-dir",
            str(tmp_path),
            "--out",
            "ablation_case",
            "--formats",
            "csv,json,tex",
            "--for-paper",
            "--err-metric",
            "rmse",
        ],
        prog="build-ablation-table",
    )

    csv_path = tmp_path / "ablation_case.csv"
    json_path = tmp_path / "ablation_case.json"
    tex_path = tmp_path / "ablation_case.tex"

    assert csv_path.exists()
    assert json_path.exists()
    assert tex_path.exists()

    out = pd.read_csv(csv_path)
    assert not out.empty
    assert "city" in out.columns
    assert "rmse" in out.columns


def test_dedupe_prefer_best_keeps_updated_record():
    mod = _import_target("build_ablation_table")

    df = pd.DataFrame(
        [
            {
                "timestamp": "2026-04-01T00:00:00Z",
                "city": "Nansha",
                "mae": 9.0,
                "_src": "ablation_record.jsonl",
            },
            {
                "timestamp": "2026-04-01T00:00:00Z",
                "city": "Nansha",
                "mae": 8.0,
                "_src": "ablation_record.updated.jsonl",
            },
        ]
    )

    out = mod._dedupe_prefer_best(df)

    assert len(out) == 1
    assert float(out.iloc[0]["mae"]) == 8.0
