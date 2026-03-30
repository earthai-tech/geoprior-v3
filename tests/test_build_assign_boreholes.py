from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import geoprior.cli.build_assign_boreholes as mod

def fake_ensure_outdir(p):
    path = Path(p).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path

def test_find_proc_csv_prefers_named_city_file(
    tmp_path: Path,
) -> None:
    stage1_dir = tmp_path / "nansha_GeoPriorSubsNet_stage1"
    stage1_dir.mkdir()
    wanted = stage1_dir / "nansha_1_1_proc.csv"
    other = stage1_dir / "zhongshan_1_1_proc.csv"
    pd.DataFrame({"x_m": [0.0], "y_m": [0.0]}).to_csv(
        wanted,
        index=False,
    )
    pd.DataFrame({"x_m": [1.0], "y_m": [1.0]}).to_csv(
        other,
        index=False,
    )

    found = mod._find_proc_csv(stage1_dir, city="nansha")
    assert found == wanted


def test_build_assign_boreholes_main_writes_combined_and_split_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    boreholes = tmp_path / "boreholes.csv"
    nansha = tmp_path / "nansha_proc.csv"
    zhongshan = tmp_path / "zhongshan_proc.csv"
    outdir = tmp_path / "assigned"

    pd.DataFrame(
        {
            "well_id": ["A", "B"],
            "x": [0.2, 100.2],
            "y": [0.1, 99.9],
        }
    ).to_csv(boreholes, index=False)
    pd.DataFrame(
        {"x_m": [0.0, 1.0], "y_m": [0.0, 1.0]}
    ).to_csv(nansha, index=False)
    pd.DataFrame(
        {"x_m": [100.0, 101.0], "y_m": [100.0, 101.0]}
    ).to_csv(zhongshan, index=False)

    monkeypatch.setattr(
        mod,
        "bootstrap_runtime_config",
        lambda args, field_map=None: {},
    )
    monkeypatch.setattr(
        mod,
        "ensure_outdir",
        fake_ensure_outdir,
    )

    mod.build_assign_boreholes_main(
        [
            "--borehole-csv",
            str(boreholes),
            "--city-csv",
            f"nansha={nansha}",
            "--city-csv",
            f"zhongshan={zhongshan}",
            "--outdir",
            str(outdir),
            "--output-stem",
            "bh",
        ]
    )

    classified = pd.read_csv(outdir / "bh_classified.csv")
    assert classified["assigned_city"].tolist() == [
        "nansha",
        "zhongshan",
    ]
    assert (outdir / "bh_nansha.csv").exists()
    assert (outdir / "bh_zhongshan.csv").exists()


def test_collect_city_sources_requires_at_least_two_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    one_csv = tmp_path / "one_proc.csv"
    pd.DataFrame({"x_m": [0.0], "y_m": [0.0]}).to_csv(
        one_csv,
        index=False,
    )

    parser = mod.argparse.ArgumentParser()
    ns = parser.parse_args([])
    ns.city_csvs = [f"nansha={one_csv}"]
    ns.city_stage1s = []
    ns.stage1_dirs = []
    ns.cities = None
    ns.results_dir = None
    ns.model = None

    with pytest.raises(SystemExit, match="At least two city clouds"):
        mod._collect_city_sources(ns, {})
