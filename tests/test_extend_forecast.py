from __future__ import annotations

import importlib
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
def extend_inputs(tmp_path: Path) -> dict[str, Path]:
    eval_rows = []
    future_rows = []
    for sample_idx, shift in ((1, 0.0), (2, 0.5)):
        eval_rows.append(
            {
                "sample_idx": sample_idx,
                "forecast_step": 0,
                "coord_t": 2022,
                "coord_x": 113.40 + shift,
                "coord_y": 22.15 + shift,
                "subsidence_q10": 8.0 + shift,
                "subsidence_q50": 10.0 + shift,
                "subsidence_q90": 12.0 + shift,
                "subsidence_actual": 10.2 + shift,
                "subsidence_unit": "mm",
            }
        )
        for step, year, q50 in ((1, 2023, 13.0), (2, 2024, 16.0), (3, 2025, 19.0)):
            future_rows.append(
                {
                    "sample_idx": sample_idx,
                    "forecast_step": step,
                    "coord_t": year,
                    "coord_x": 113.40 + shift,
                    "coord_y": 22.15 + shift,
                    "subsidence_q10": q50 - 2.0,
                    "subsidence_q50": q50 + shift,
                    "subsidence_q90": q50 + 2.0 + shift,
                    "subsidence_unit": "mm",
                    "calibration_factor": 1.0,
                    "is_calibrated": True,
                }
            )

    eval_path = tmp_path / "nansha_eval.csv"
    future_path = tmp_path / "nansha_future.csv"
    pd.DataFrame(eval_rows).to_csv(eval_path, index=False)
    pd.DataFrame(future_rows).to_csv(future_path, index=False)
    return {"eval": eval_path, "future": future_path}


def test_extend_forecast_main_writes_extended_csv(
    tmp_path: Path,
    extend_inputs: dict[str, Path],
):
    mod = _import_target("extend_forecast")
    out_csv = tmp_path / "future_extended.csv"

    mod.extend_forecast_main(
        [
            "--cities",
            "ns",
            "--ns-eval",
            str(extend_inputs["eval"]),
            "--ns-future",
            str(extend_inputs["future"]),
            "--subsidence-kind",
            "cumulative",
            "--years",
            "2026",
            "2027",
            "--out",
            str(out_csv),
        ],
        prog="extend-forecast",
    )

    assert out_csv.exists()
    out = pd.read_csv(out_csv)

    extended = out[out["coord_t"].isin([2026, 2027])].copy()
    assert not extended.empty
    assert set(extended["coord_t"]) == {2026, 2027}
    assert extended["extended"].fillna(False).all()
    assert extended["forecast_step"].min() >= 4


def test_out_path_adds_city_suffix_for_multi_city_output():
    mod = _import_target("extend_forecast")

    out = mod._out_path(
        "future_extended",
        city="Nansha",
        multi=True,
    )

    assert out.name.endswith("nansha.csv")
