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
            if modname == missing or modname.startswith(missing + "."):
                continue
            raise
    pytest.skip(f"Could not import target module for {name!r}.")


@pytest.fixture
def clusters_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "city": ["Nansha", "Nansha"],
            "year": [2025, 2026],
            "cluster_id": [1, 2],
            "priority_rank": [1, 2],
            "centroid_x": [113.40, 113.45],
            "centroid_y": [22.15, 22.20],
            "metric_mean": [3.0, 4.0],
            "risk_mean": [0.8, 0.9],
        }
    )
    path = tmp_path / "clusters.csv"
    df.to_csv(path, index=False)
    return path


def test_tag_clusters_with_zones_main_writes_csv(
    tmp_path: Path,
    clusters_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _import_target("tag_clusters_with_zones")

    monkeypatch.setattr(mod, "_load_grid", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        mod,
        "_match_points_to_polys",
        lambda xs, ys, grid, nearest: (
            ["Z0101", "Z0201"],
            ["Zone Z0101", "Zone Z0201"],
        ),
    )

    out_csv = tmp_path / "cluster_zones.csv"
    mod.tag_clusters_with_zones_main(
        [
            "--clusters",
            str(clusters_csv),
            "--grid",
            str(tmp_path / "grid.geojson"),
            "--out",
            str(out_csv),
        ],
        prog="tag-clusters-with-zones",
    )

    assert out_csv.exists()
    out = pd.read_csv(out_csv)
    assert list(out["zone_id"]) == ["Z0101", "Z0201"]
    assert "zone_label" in out.columns


def test_load_clusters_coerces_numeric_fields(clusters_csv: Path):
    mod = _import_target("tag_clusters_with_zones")

    df = mod._load_clusters(str(clusters_csv))

    assert str(df["cluster_id"].dtype) == "Int64"
    assert str(df["year"].dtype) == "Int64"
    assert float(df["centroid_x"].iloc[0]) == pytest.approx(113.40)
