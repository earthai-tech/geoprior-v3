from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from geoprior.cli import build_spatial_clusters as mod


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "longitude": [113.4, 113.41, 113.42],
            "latitude": [22.1, 22.11, 22.12],
            "value": [1.0, 2.0, 3.0],
        }
    )


def test_run_build_spatial_clusters_calls_helper_and_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    df = _df()
    clustered = df.copy()
    clustered["region"] = [0, 1, 1]
    calls: dict[str, object] = {}

    def fake_load(ns: Namespace) -> pd.DataFrame:
        return df

    def fake_clusters(**kwargs):
        calls["clusters"] = kwargs
        return clustered

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

    monkeypatch.setattr(mod, "load_dataframe_from_args", fake_load)
    monkeypatch.setattr(
        mod,
        "create_spatial_clusters",
        fake_clusters,
    )
    monkeypatch.setattr(mod, "write_dataframe", fake_write)

    out = tmp_path / "clusters.csv"
    result = mod.run_build_spatial_clusters(
        input_paths=["dummy.csv"],
        input_format="csv",
        input_sheet=0,
        read_kwargs={},
        source_col=None,
        spatial_cols=["longitude", "latitude"],
        cluster_col="region",
        n_clusters=2,
        algorithm="kmeans",
        no_auto_scale=False,
        view=False,
        figsize=[10.0, 8.0],
        marker_size=25,
        plot_style="default",
        cmap="tab20",
        no_grid=False,
        verbose=2,
        output=str(out),
        excel_output_sheet="Sheet1",
    )

    pd.testing.assert_frame_equal(result, clustered)
    assert out.exists()

    got = calls["clusters"]
    assert got["df"].equals(df)
    assert got["spatial_cols"] == ["longitude", "latitude"]
    assert got["cluster_col"] == "region"
    assert got["n_clusters"] == 2
    assert got["algorithm"] == "kmeans"
    assert got["view"] is False
    assert got["figsize"] == (10.0, 8.0)
    assert got["s"] == 25
    assert got["plot_style"] == "default"
    assert got["cmap"] == "tab20"
    assert got["show_grid"] is True
    assert got["auto_scale"] is True
    assert got["verbose"] == 2


def test_build_spatial_clusters_main_uses_parser_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class DummyParser:
        def parse_args(self, argv):
            seen["argv"] = list(argv)
            return Namespace(alpha=2)

    def fake_run(**kwargs) -> None:
        seen["kwargs"] = kwargs

    monkeypatch.setattr(mod, "_build_parser", lambda: DummyParser())
    monkeypatch.setattr(mod, "run_build_spatial_clusters", fake_run)

    mod.build_spatial_clusters_main(["--demo"])

    assert seen["argv"] == ["--demo"]
    assert seen["kwargs"] == {"alpha": 2}


def test_main_alias_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_main(argv=None) -> None:
        seen["argv"] = argv

    monkeypatch.setattr(mod, "build_spatial_clusters_main", fake_main)
    mod.main(["--flag"])
    assert seen["argv"] == ["--flag"]
