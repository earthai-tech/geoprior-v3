from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

pytestmark = [pytest.mark.script_artifacts]


def _import_target(name: str):
    candidates = (
        f"geoprior.scripts.{name}",
        name,
    )
    for modname in candidates:
        try:
            return importlib.import_module(modname)
        except ModuleNotFoundError as exc:
            missing = str(getattr(exc, "name", "") or "")
            if modname == missing or modname.startswith(
                missing + "."
            ):
                continue
            raise
    pytest.skip(
        f"Could not import target module for {name!r}."
    )


@pytest.fixture
def hotspot_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        [
            {
                "city": "nansha",
                "year": 2025,
                "kind": "forecast",
                "value": 12.0,
                "metric_value": 2.0,
                "baseline_value": 10.0,
                "threshold": 1.5,
            },
            {
                "city": "nansha",
                "year": 2025,
                "kind": "forecast",
                "value": 14.0,
                "metric_value": 3.0,
                "baseline_value": 10.5,
                "threshold": 1.5,
            },
            {
                "city": "zhongshan",
                "year": 2026,
                "kind": "forecast",
                "value": 20.0,
                "metric_value": 5.0,
                "baseline_value": 16.0,
                "threshold": 2.5,
            },
        ]
    )
    path = tmp_path / "hotspots.csv"
    df.to_csv(path, index=False)
    return path


def test_summarize_hotspots_main_writes_grouped_csv(
    tmp_path: Path,
    hotspot_csv: Path,
):
    mod = _import_target("summarize_hotspots")
    out_csv = tmp_path / "hotspot_summary.csv"

    mod.summarize_hotspots_main(
        [
            "--hotspot-csv",
            str(hotspot_csv),
            "--out",
            str(out_csv),
            "--quiet",
            "true",
        ],
        prog="summarize-hotspots",
    )

    assert out_csv.exists()
    out = pd.read_csv(out_csv)
    assert set(out["city"]) == {"Nansha", "Zhongshan"}
    assert "n_hotspots" in out.columns
    assert "value_mean" in out.columns


def test_summarize_hotspots_returns_expected_group_means(
    hotspot_csv: Path,
):
    mod = _import_target("summarize_hotspots")

    summary = mod.summarize_hotspots(pd.read_csv(hotspot_csv))
    sub = summary[
        (summary["city"] == "Nansha")
        & (summary["year"] == 2025)
    ]

    assert len(sub) == 1
    assert float(sub.iloc[0]["n_hotspots"]) == 2
    assert float(sub.iloc[0]["value_mean"]) == pytest.approx(
        13.0
    )
    assert float(sub.iloc[0]["metric_mean"]) == pytest.approx(
        2.5
    )
