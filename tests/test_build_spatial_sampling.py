from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from geoprior.cli import build_spatial_sampling as mod


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "longitude": [113.4, 113.41, 113.42],
            "latitude": [22.1, 22.11, 22.12],
            "region": ["a", "a", "b"],
            "value": [1.0, 2.0, 3.0],
        }
    )


def test_parse_sample_size_accepts_int_and_fraction() -> None:
    assert mod._parse_sample_size("5") == 5
    assert mod._parse_sample_size("5.0") == 5
    assert mod._parse_sample_size("0.25") == 0.25


@pytest.mark.parametrize(
    "raw",
    ["", "0", "-2", "1.5", "abc"],
)
def test_parse_sample_size_rejects_bad_values(
    raw: str,
) -> None:
    with pytest.raises(Exception):
        mod._parse_sample_size(raw)


def test_normalize_bins_handles_none_scalar_and_many() -> (
    None
):
    assert mod._normalize_bins(None) == 10
    assert mod._normalize_bins([7]) == 7
    assert mod._normalize_bins([3, 5]) == (3, 5)


def test_run_build_spatial_sampling_calls_helper_and_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    df = _sample_df()
    sampled = df.iloc[[0, 2]].copy()
    calls: dict[str, object] = {}

    def fake_load(ns: Namespace) -> pd.DataFrame:
        assert isinstance(ns, Namespace)
        return df

    def fake_sampling(**kwargs):
        calls["sampling"] = kwargs
        return sampled

    def fake_write(
        frame: pd.DataFrame,
        output: str,
        **kwargs,
    ) -> Path:
        calls["write"] = {
            "frame": frame.copy(),
            "output": output,
            "kwargs": kwargs,
        }
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        return out

    monkeypatch.setattr(
        mod, "load_dataframe_from_args", fake_load
    )
    monkeypatch.setattr(
        mod, "spatial_sampling", fake_sampling
    )
    monkeypatch.setattr(mod, "write_dataframe", fake_write)

    out = tmp_path / "sampled.csv"
    result = mod.run_build_spatial_sampling(
        input_paths=["dummy.csv"],
        input_format="csv",
        input_sheet=0,
        read_kwargs={},
        source_col=None,
        sample_size=0.4,
        stratify_by=["region"],
        spatial_bins=[4, 6],
        spatial_cols=["longitude", "latitude"],
        method="relative",
        min_relative_ratio=0.05,
        random_state=7,
        output=str(out),
        excel_output_sheet="Sheet1",
        verbose=2,
    )

    pd.testing.assert_frame_equal(result, sampled)
    assert out.exists()

    got = calls["sampling"]
    assert got["data"].equals(df)
    assert got["sample_size"] == 0.4
    assert got["stratify_by"] == ["region"]
    assert got["spatial_bins"] == (4, 6)
    assert got["spatial_cols"] == ["longitude", "latitude"]
    assert got["method"] == "relative"
    assert got["min_relative_ratio"] == 0.05
    assert got["random_state"] == 7
    assert got["verbose"] == 2

    written = calls["write"]
    pd.testing.assert_frame_equal(written["frame"], sampled)
    assert written["output"] == str(out)


def test_build_spatial_sampling_main_uses_parser_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class DummyParser:
        def parse_args(self, argv):
            seen["argv"] = list(argv)
            return Namespace(alpha=1, beta="x")

    def fake_run(**kwargs) -> None:
        seen["kwargs"] = kwargs

    monkeypatch.setattr(
        mod, "_build_parser", lambda: DummyParser()
    )
    monkeypatch.setattr(
        mod, "run_build_spatial_sampling", fake_run
    )

    mod.build_spatial_sampling_main(["--anything", "value"])

    assert seen["argv"] == ["--anything", "value"]
    assert seen["kwargs"] == {"alpha": 1, "beta": "x"}


def test_main_alias_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_main(argv=None) -> None:
        seen["argv"] = argv

    monkeypatch.setattr(
        mod, "build_spatial_sampling_main", fake_main
    )
    mod.main(["--flag"])
    assert seen["argv"] == ["--flag"]
