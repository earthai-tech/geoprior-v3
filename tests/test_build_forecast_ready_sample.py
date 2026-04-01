from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _import_target(name: str):
    candidates = (
        f"geoprior.cli.{name}",
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

pytestmark = [pytest.mark.stage_artifacts]

def _simple_write_dataframe(df, path, **kwargs):
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=kwargs.get("index", False))
    return out


def test_build_forecast_ready_sample_main_writes_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _import_target("build_forecast_ready_sample")

    dummy_input = tmp_path / "input.csv"
    dummy_input.write_text("placeholder\n", encoding="utf-8")

    df = pd.DataFrame(
        {
            "longitude": [113.4, 113.4, 113.4, 113.5, 113.5, 113.5],
            "latitude": [22.1, 22.1, 22.1, 22.2, 22.2, 22.2],
            "year": [2020, 2021, 2022, 2020, 2021, 2022],
            "value": [1, 2, 3, 4, 5, 6],
        }
    )

    monkeypatch.setattr(mod, "load_dataframe_from_args", lambda args: df.copy())
    monkeypatch.setattr(
        mod,
        "make_forecast_ready_sample",
        lambda **kwargs: kwargs["data"].copy(),
    )
    monkeypatch.setattr(mod, "write_dataframe", _simple_write_dataframe)

    out_csv = tmp_path / "forecast_ready.csv"
    mod.build_forecast_ready_sample_main(
        [
            str(dummy_input),
            "--out",
            str(out_csv),
            "--time-steps",
            "2",
            "--forecast-horizon",
            "1",
        ]
    )

    assert out_csv.exists()
    out_df = pd.read_csv(out_csv)
    assert len(out_df) == len(df)


def test_parse_bins_and_sample_size_helpers():
    mod = _import_target("build_forecast_ready_sample")

    assert mod._parse_sample_size("5") == 5
    assert mod._parse_sample_size("0.25") == pytest.approx(0.25)
    assert mod._parse_bins(["10"]) == 10
    assert mod._parse_bins(["10", "12"]) == (10, 12)
    assert mod._parse_bins(["10,12"]) == (10, 12)
