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
def brier_inputs(tmp_path: Path) -> dict[str, Path]:
    def _df(offset: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coord_t": [2020, 2021, 2022],
                "subsidence_actual": [28.0 + offset, 45.0 + offset, 55.0 + offset],
                "subsidence_q10": [20.0 + offset, 35.0 + offset, 45.0 + offset],
                "subsidence_q50": [30.0 + offset, 45.0 + offset, 55.0 + offset],
                "subsidence_q90": [40.0 + offset, 55.0 + offset, 65.0 + offset],
            }
        )

    ns = tmp_path / "nansha_eval.csv"
    zh = tmp_path / "zhongshan_eval.csv"
    _df(0.0).to_csv(ns, index=False)
    _df(5.0).to_csv(zh, index=False)
    return {"ns": ns, "zh": zh}


def test_brier_exceedance_main_writes_expected_rows(
    tmp_path: Path,
    brier_inputs: dict[str, Path],
):
    mod = _import_target("compute_brier_exceedance")
    out_csv = tmp_path / "brier_scores.csv"

    mod.brier_exceedance_main(
        [
            "--ns-csv",
            str(brier_inputs["ns"]),
            "--zh-csv",
            str(brier_inputs["zh"]),
            "--thresholds",
            "30,50",
            "--years",
            "2020,2021,2022",
            "--out",
            str(out_csv),
            "--quiet",
            "true",
        ],
        prog="brier-exceedance",
    )

    assert out_csv.exists()
    out = pd.read_csv(out_csv)
    assert len(out) == 4
    assert set(out["city"]) == {"Nansha", "Zhongshan"}
    assert set(out["threshold_mm_per_yr"]) == {30.0, 50.0}
    assert out["brier_score"].notna().all()


def test_exceed_prob_from_quantiles_is_half_at_q50():
    mod = _import_target("compute_brier_exceedance")

    q10 = np.array([10.0])
    q50 = np.array([20.0])
    q90 = np.array([30.0])

    p = mod.exceed_prob_from_quantiles(
        q10,
        q50,
        q90,
        threshold=20.0,
    )

    assert p.shape == (1,)
    assert p[0] == pytest.approx(0.5)
