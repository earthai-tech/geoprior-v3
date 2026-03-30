from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from geoprior.cli import build_spatial_roi as mod


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "longitude": [113.4, 113.41, 113.42],
            "latitude": [22.1, 22.11, 22.12],
            "value": [1.0, 2.0, 3.0],
        }
    )


def test_run_build_spatial_roi_calls_helper_and_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    df = _df()
    roi = df.iloc[[1]].copy()
    calls: dict[str, object] = {}

    def fake_load(ns: Namespace) -> pd.DataFrame:
        return df

    def fake_roi(**kwargs):
        calls["roi"] = kwargs
        return roi

    def fake_write(
        frame: pd.DataFrame,
        output: str,
        **kwargs,
    ) -> Path:
        calls["write"] = (frame.copy(), output, kwargs)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        return out

    monkeypatch.setattr(
        mod, "load_dataframe_from_args", fake_load
    )
    monkeypatch.setattr(mod, "extract_spatial_roi", fake_roi)
    monkeypatch.setattr(mod, "write_dataframe", fake_write)

    out = tmp_path / "roi.csv"
    result = mod.run_build_spatial_roi(
        input_paths=["dummy.csv"],
        input_format="csv",
        input_sheet=0,
        read_kwargs={},
        source_col=None,
        x_range=[113.405, 113.415],
        y_range=[22.105, 22.115],
        x_col="longitude",
        y_col="latitude",
        no_snap_to_closest=True,
        output=str(out),
        excel_output_sheet="Sheet1",
    )

    pd.testing.assert_frame_equal(result, roi)
    assert out.exists()

    got = calls["roi"]
    assert got["df"].equals(df)
    assert got["x_range"] == (113.405, 113.415)
    assert got["y_range"] == (22.105, 22.115)
    assert got["x_col"] == "longitude"
    assert got["y_col"] == "latitude"
    assert got["snap_to_closest"] is False


def test_build_spatial_roi_main_uses_parser_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class DummyParser:
        def parse_args(self, argv):
            seen["argv"] = list(argv)
            return Namespace(alpha=9)

    def fake_run(**kwargs) -> None:
        seen["kwargs"] = kwargs

    monkeypatch.setattr(
        mod, "_build_parser", lambda: DummyParser()
    )
    monkeypatch.setattr(
        mod, "run_build_spatial_roi", fake_run
    )

    mod.build_spatial_roi_main(["--demo"])

    assert seen["argv"] == ["--demo"]
    assert seen["kwargs"] == {"alpha": 9}


def test_main_alias_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_main(argv=None) -> None:
        seen["argv"] = argv

    monkeypatch.setattr(
        mod, "build_spatial_roi_main", fake_main
    )
    mod.main(["--flag"])
    assert seen["argv"] == ["--flag"]
