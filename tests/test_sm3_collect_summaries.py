from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sm3_collect_mod():
    try:
        return importlib.import_module(
            "geoprior.cli.sm3_collect_summaries"
        )
    except ModuleNotFoundError:
        return importlib.import_module(
            "sm3_collect_summaries"
        )


def _write_summary_csv(
    run_dir: Path,
    rows: list[dict[str, object]],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "sm3_synth_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def test_infer_regime_parses_expected_folder_names(
    sm3_collect_mod,
):
    assert (
        sm3_collect_mod.infer_regime("sm3_tau_base_50")
        == "base"
    )
    assert (
        sm3_collect_mod.infer_regime(
            "sm3_both_closure_locked_50"
        )
        == "closure_locked"
    )
    assert (
        sm3_collect_mod.infer_regime("custom_folder")
        == "custom_folder"
    )


def test_main_collects_combines_and_writes_outputs(
    sm3_collect_mod,
    tmp_path,
    monkeypatch,
    capsys,
):
    suite_root = tmp_path / "suite"
    base_dir = suite_root / "sm3_tau_base_50"
    anchored_dir = suite_root / "sm3_tau_anchored_50"

    _write_summary_csv(
        base_dir,
        [
            {"metric": "mae", "value": 0.2},
            {"metric": "rmse", "value": 0.4},
        ],
    )
    _write_summary_csv(
        anchored_dir,
        [
            {"metric": "mae", "value": 0.1},
            {"metric": "rmse", "value": 0.3},
        ],
    )

    out_csv = tmp_path / "combined" / "summary.csv"
    out_json = tmp_path / "combined" / "summary.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sm3-collect-summaries",
            "--suite-root",
            str(suite_root),
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
        ],
    )

    sm3_collect_mod.main()

    out = capsys.readouterr().out
    assert "[OK] wrote:" in out
    assert out_csv.exists()
    assert out_json.exists()

    df = pd.read_csv(out_csv)
    records = json.loads(out_json.read_text(encoding="utf-8"))

    assert list(df.columns[:2]) == ["regime", "run_dir"]
    assert df["regime"].tolist() == [
        "anchored",
        "base",
        "anchored",
        "base",
    ]
    assert df["metric"].tolist() == [
        "mae",
        "mae",
        "rmse",
        "rmse",
    ]
    assert len(records) == 4
    assert {r["regime"] for r in records} == {
        "base",
        "anchored",
    }


def test_main_skips_bad_files_and_keeps_good_ones(
    sm3_collect_mod,
    tmp_path,
    monkeypatch,
    capsys,
):
    suite_root = tmp_path / "suite"

    _write_summary_csv(
        suite_root / "sm3_tau_base_50",
        [{"metric": "mae", "value": 0.2}],
    )
    _write_summary_csv(
        suite_root / "sm3_tau_bad_50",
        [{"not_metric": "x", "value": 1.0}],
    )

    out_csv = tmp_path / "combined.csv"
    out_json = tmp_path / "combined.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sm3-collect-summaries",
            "--suite-root",
            str(suite_root),
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
        ],
    )

    sm3_collect_mod.main()

    out = capsys.readouterr().out
    assert "[skip] unexpected format:" in out

    df = pd.read_csv(out_csv)
    assert df.shape[0] == 1
    assert df.loc[0, "regime"] == "base"
    assert df.loc[0, "metric"] == "mae"


def test_main_raises_for_missing_suite_root(
    sm3_collect_mod,
    tmp_path,
    monkeypatch,
):
    missing = tmp_path / "does_not_exist"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sm3-collect-summaries",
            "--suite-root",
            str(missing),
            "--out-csv",
            str(tmp_path / "x.csv"),
            "--out-json",
            str(tmp_path / "x.json"),
        ],
    )

    with pytest.raises(
        FileNotFoundError, match="Suite root not found"
    ):
        sm3_collect_mod.main()


def test_main_raises_when_no_summary_csv_exists(
    sm3_collect_mod,
    tmp_path,
    monkeypatch,
):
    suite_root = tmp_path / "empty_suite"
    suite_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sm3-collect-summaries",
            "--suite-root",
            str(suite_root),
            "--out-csv",
            str(tmp_path / "x.csv"),
            "--out-json",
            str(tmp_path / "x.json"),
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="No sm3_synth_summary.csv files found",
    ):
        sm3_collect_mod.main()
