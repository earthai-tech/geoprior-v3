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


class _FakeGeoFrame:
    def __init__(self, city: str, poly):
        self.city = city
        self.poly = poly

    def to_file(self, path, driver=None):
        Path(path).write_text(
            f"city={self.city};driver={driver};poly={self.poly}",
            encoding="utf-8",
        )


@pytest.fixture
def boundary_inputs(tmp_path: Path) -> dict[str, Path]:
    eval_df = pd.DataFrame(
        {
            "coord_x": [113.40, 113.42, 113.41, None],
            "coord_y": [22.15, 22.17, 22.16, 22.18],
        }
    )
    future_df = pd.DataFrame(
        {
            "coord_x": [113.43, 113.44, 113.45],
            "coord_y": [22.18, 22.19, 22.20],
        }
    )

    eval_path = tmp_path / "nansha_eval.csv"
    future_path = tmp_path / "nansha_future.csv"
    eval_df.to_csv(eval_path, index=False)
    future_df.to_csv(future_path, index=False)
    return {"eval": eval_path, "future": future_path}


def test_make_boundary_main_writes_geojson_and_shp(
    tmp_path: Path,
    boundary_inputs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _import_target("make_boundary")

    monkeypatch.setattr(
        mod,
        "_poly_from_points",
        lambda xy, method, alpha: {
            "n_points": int(len(xy)),
            "method": method,
            "alpha": float(alpha),
        },
    )
    monkeypatch.setattr(
        mod,
        "_make_boundary_gdf",
        lambda city, poly: _FakeGeoFrame(city, poly),
    )

    out_stem = tmp_path / "boundary_case"
    written = mod.make_boundary_main(
        [
            "--cities",
            "ns",
            "--ns-eval",
            str(boundary_inputs["eval"]),
            "--ns-future",
            str(boundary_inputs["future"]),
            "--format",
            "both",
            "--out",
            str(out_stem),
        ],
        prog="make-boundary",
    )

    geojson_path = tmp_path / "boundary_case_nansha.geojson"
    shp_path = tmp_path / "boundary_case_nansha.shp"

    assert geojson_path.exists()
    assert shp_path.exists()
    assert len(written) == 2


def test_load_xy_filters_nonfinite_rows(tmp_path: Path):
    mod = _import_target("make_boundary")

    df = pd.DataFrame(
        {
            "coord_x": [113.40, None, 113.42],
            "coord_y": [22.15, 22.16, float("nan")],
        }
    )
    path = tmp_path / "points.csv"
    df.to_csv(path, index=False)

    xy = mod._load_xy(str(path))

    assert xy.shape == (1, 2)
    assert xy[0, 0] == pytest.approx(113.40)
    assert xy[0, 1] == pytest.approx(22.15)
