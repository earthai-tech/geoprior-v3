from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from geoprior.cli import build_batch_spatial_sampling as mod


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "longitude": [113.4, 113.41, 113.42, 113.43],
            "latitude": [22.1, 22.11, 22.12, 22.13],
            "region": ["a", "a", "b", "b"],
            "value": [1, 2, 3, 4],
        }
    )


def test_stack_batches_adds_batch_id() -> None:
    b1 = pd.DataFrame({"x": [1, 2]})
    b2 = pd.DataFrame({"x": [3]})

    got = mod._stack_batches([b1, b2], batch_col="batch_id")

    assert list(got.columns) == ["batch_id", "x"]
    assert got["batch_id"].tolist() == [1, 1, 2]
    assert got["x"].tolist() == [1, 2, 3]


def test_stack_batches_handles_empty_input() -> None:
    got = mod._stack_batches([], batch_col="batch_id")
    assert list(got.columns) == ["batch_id"]
    assert got.empty


def test_run_build_batch_spatial_sampling_writes_stacked_and_splits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    df = _df()
    batches = [df.iloc[:2].copy(), df.iloc[2:].copy()]
    calls: dict[str, object] = {"writes": []}

    def fake_load(ns: Namespace) -> pd.DataFrame:
        return df

    def fake_batch_sampling(**kwargs):
        calls["sampling"] = kwargs
        return batches

    def fake_norm(_output: str) -> str:
        return "csv"

    def fake_write(
        frame: pd.DataFrame,
        output: str | Path,
        **kwargs,
    ) -> Path:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        calls["writes"].append((frame.copy(), out, kwargs))
        return out

    monkeypatch.setattr(
        mod, "load_dataframe_from_args", fake_load
    )
    monkeypatch.setattr(
        mod,
        "batch_spatial_sampling",
        fake_batch_sampling,
    )
    monkeypatch.setattr(
        mod, "normalize_output_format", fake_norm
    )
    monkeypatch.setattr(mod, "write_dataframe", fake_write)

    stacked_out = tmp_path / "stacked.csv"
    split_dir = tmp_path / "splits"
    result = mod.run_build_batch_spatial_sampling(
        input_paths=["dummy.csv"],
        input_format="csv",
        input_sheet=0,
        read_kwargs={},
        source_col=None,
        sample_size=0.5,
        n_batches=2,
        stratify_by=["region"],
        spatial_bins=[5],
        spatial_cols=["longitude", "latitude"],
        method="abs",
        min_relative_ratio=0.01,
        random_state=13,
        batch_col="batch_id",
        output=str(stacked_out),
        split_dir=str(split_dir),
        split_prefix="part_",
        split_format="auto",
        excel_output_sheet="Sheet1",
        verbose=1,
    )

    assert stacked_out.exists()
    assert (split_dir / "part_001.csv").exists()
    assert (split_dir / "part_002.csv").exists()

    assert result["batch_id"].tolist() == [1, 1, 2, 2]
    assert list(result.columns)[0] == "batch_id"

    got = calls["sampling"]
    assert got["data"].equals(df)
    assert got["sample_size"] == 0.5
    assert got["n_batches"] == 2
    assert got["spatial_bins"] == 5
    assert got["spatial_cols"] == ["longitude", "latitude"]

    assert len(calls["writes"]) == 3


def test_build_batch_spatial_sampling_main_uses_parser_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class DummyParser:
        def parse_args(self, argv):
            seen["argv"] = list(argv)
            return Namespace(alpha=3, beta="y")

    def fake_run(**kwargs) -> None:
        seen["kwargs"] = kwargs

    monkeypatch.setattr(
        mod, "_build_parser", lambda: DummyParser()
    )
    monkeypatch.setattr(
        mod,
        "run_build_batch_spatial_sampling",
        fake_run,
    )

    mod.build_batch_spatial_sampling_main(["--x"])

    assert seen["argv"] == ["--x"]
    assert seen["kwargs"] == {"alpha": 3, "beta": "y"}


def test_main_alias_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_main(argv=None) -> None:
        seen["argv"] = argv

    monkeypatch.setattr(
        mod,
        "build_batch_spatial_sampling_main",
        fake_main,
    )
    mod.main(["--flag"])
    assert seen["argv"] == ["--flag"]
