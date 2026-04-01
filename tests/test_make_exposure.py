from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
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
def exposure_inputs(tmp_path: Path) -> dict[str, Path]:
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


def test_make_exposure_main_writes_csv(
    tmp_path: Path,
    exposure_inputs: dict[str, Path],
):
    mod = _import_target("make_exposure")
    out_csv = tmp_path / "exposure.csv"

    mod.make_exposure_main(
        [
            "--cities",
            "ns",
            "--ns-eval",
            str(exposure_inputs["eval"]),
            "--ns-future",
            str(exposure_inputs["future"]),
            "--mode",
            "uniform",
            "--out",
            str(out_csv),
        ],
        prog="make-exposure",
    )

    assert out_csv.exists()
    out = pd.read_csv(out_csv)
    assert set(out["city"]) == {"Nansha"}
    assert np.allclose(out["exposure"], 1.0)


def test_density_exposure_is_positive_and_mean_normalized():
    mod = _import_target("make_exposure")

    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    z = mod._density_exposure(x, y, k=2)

    assert np.all(z > 0)
    assert float(np.mean(z)) == pytest.approx(1.0)
