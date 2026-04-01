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
def hotspot_inputs(tmp_path: Path) -> dict[str, Path]:
    eval_df = pd.DataFrame(
        [
            {
                "sample_idx": 1,
                "coord_t": 2022,
                "subsidence_actual": 10.0,
                "subsidence_q50": 10.5,
                "subsidence_unit": "mm",
            },
            {
                "sample_idx": 2,
                "coord_t": 2022,
                "subsidence_actual": 12.0,
                "subsidence_q50": 12.3,
                "subsidence_unit": "mm",
            },
            {
                "sample_idx": 3,
                "coord_t": 2022,
                "subsidence_actual": 14.0,
                "subsidence_q50": 14.1,
                "subsidence_unit": "mm",
            },
        ]
    )
    future_df = pd.DataFrame(
        [
            {
                "sample_idx": 1,
                "coord_t": 2025,
                "subsidence_q50": 11.0,
                "subsidence_unit": "mm",
            },
            {
                "sample_idx": 2,
                "coord_t": 2025,
                "subsidence_q50": 18.0,
                "subsidence_unit": "mm",
            },
            {
                "sample_idx": 3,
                "coord_t": 2025,
                "subsidence_q50": 14.5,
                "subsidence_unit": "mm",
            },
            {
                "sample_idx": 1,
                "coord_t": 2026,
                "subsidence_q50": 11.5,
                "subsidence_unit": "mm",
            },
            {
                "sample_idx": 2,
                "coord_t": 2026,
                "subsidence_q50": 19.0,
                "subsidence_unit": "mm",
            },
            {
                "sample_idx": 3,
                "coord_t": 2026,
                "subsidence_q50": 15.0,
                "subsidence_unit": "mm",
            },
        ]
    )

    eval_path = tmp_path / "nansha_eval.csv"
    future_path = tmp_path / "nansha_future.csv"
    eval_df.to_csv(eval_path, index=False)
    future_df.to_csv(future_path, index=False)
    return {"eval": eval_path, "future": future_path}


def test_compute_hotspots_main_writes_csv_and_tex(
    tmp_path: Path,
    hotspot_inputs: dict[str, Path],
):
    mod = _import_target("compute_hotspots")
    out_stem = tmp_path / "hotspots_case"

    mod.compute_hotspots_main(
        [
            "--cities",
            "ns",
            "--ns-eval",
            str(hotspot_inputs["eval"]),
            "--ns-future",
            str(hotspot_inputs["future"]),
            "--subsidence-kind",
            "rate",
            "--baseline-year",
            "2022",
            "--years",
            "2025",
            "2026",
            "--format",
            "both",
            "--out",
            str(out_stem),
        ],
        prog="summarize-hotspots",
    )

    csv_path = out_stem.with_suffix(".csv")
    tex_path = out_stem.with_suffix(".tex")

    assert csv_path.exists()
    assert tex_path.exists()

    out = pd.read_csv(csv_path)
    assert list(out["City"].unique()) == ["Nansha"]
    assert set(out["Year"]) == {2025, 2026}
    assert "Hotspots_n" in out.columns
    assert "T_0p9" in out.columns


def test_pick_years_from_future_uses_last_n_years():
    mod = _import_target("compute_hotspots")

    future_df = pd.DataFrame(
        {
            "coord_t": [2023, 2024, 2025, 2026],
            "sample_idx": [1, 1, 1, 1],
            "subsidence_q50": [1.0, 2.0, 3.0, 4.0],
        }
    )

    years = mod._pick_years_from_future(
        future_df,
        base_year=2022,
        years=None,
        n_years=2,
    )

    assert years == [2025, 2026]
