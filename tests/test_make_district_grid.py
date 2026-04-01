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


class _FakeGrid(pd.DataFrame):
    @property
    def _constructor(self):
        return _FakeGrid

    def to_file(self, path, driver=None):
        Path(path).write_text(
            f"driver={driver};n={len(self)}",
            encoding="utf-8",
        )


@pytest.fixture
def district_inputs(tmp_path: Path) -> dict[str, Path]:
    eval_df = pd.DataFrame(
        {
            "sample_idx": [1, 2, 3],
            "coord_x": [113.40, 113.42, 113.44],
            "coord_y": [22.15, 22.17, 22.19],
        }
    )
    future_df = pd.DataFrame(
        {
            "sample_idx": [4, 5],
            "coord_x": [113.41, 113.45],
            "coord_y": [22.16, 22.20],
        }
    )

    eval_path = tmp_path / "nansha_eval.csv"
    future_path = tmp_path / "nansha_future.csv"
    eval_df.to_csv(eval_path, index=False)
    future_df.to_csv(future_path, index=False)
    return {"eval": eval_path, "future": future_path}


def test_make_district_grid_main_writes_grid_and_assignments(
    tmp_path: Path,
    district_inputs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _import_target("make_district_grid")

    fake_grid = _FakeGrid(
        [
            {"zone_id": "Z0101", "zone_label": "Zone Z0101"},
            {"zone_id": "Z0102", "zone_label": "Zone Z0102"},
        ]
    )
    monkeypatch.setattr(mod, "_build_grid_gdf", lambda **kwargs: fake_grid)

    out_stem = tmp_path / "district_grid"
    mod.make_district_grid_main(
        [
            "--cities",
            "ns",
            "--ns-eval",
            str(district_inputs["eval"]),
            "--ns-future",
            str(district_inputs["future"]),
            "--nx",
            "2",
            "--ny",
            "2",
            "--assign-samples",
            "--format",
            "geojson",
            "--out",
            str(out_stem),
        ],
        prog="make-district-grid",
    )

    grid_path = tmp_path / "district_grid_nansha.geojson"
    assign_path = tmp_path / "district_assignments_nansha.csv"

    assert grid_path.exists()
    assert assign_path.exists()

    assigned = pd.read_csv(assign_path)
    assert set(assigned["city"]) == {"Nansha"}
    assert "zone_id" in assigned.columns
    assert "zone_present" in assigned.columns


def test_zone_id_uses_zero_padded_row_and_col():
    mod = _import_target("make_district_grid")

    zid = mod._zone_id(2, 4, ny=12, nx=12)

    assert zid == "Z0305"
