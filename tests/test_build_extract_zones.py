from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import geoprior.cli.build_extract_zones as mod


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["auto"], "auto"),
        (["1.5"], 1.5),
        (["1", "2.5"], (1.0, 2.5)),
    ],
)
def test_parse_threshold_accepts_supported_forms(
    raw: list[str],
    expected: str | float | tuple[float, float],
) -> None:
    assert mod._parse_threshold(raw) == expected


def test_parse_threshold_rejects_invalid_forms() -> None:
    with pytest.raises(Exception):
        mod._parse_threshold([])

    with pytest.raises(Exception):
        mod._parse_threshold(["a", "b"])

    with pytest.raises(Exception):
        mod._parse_threshold(["1", "2", "3"])


def test_run_build_extract_zones_writes_result(
    monkeypatch: pytest.MonkeyPatch,
    city_panel_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    loaded = city_panel_df.head(4).copy()
    calls: dict[str, object] = {}

    def fake_load_dataframe_from_args(ns):
        calls["namespace"] = ns
        return loaded

    def fake_extract_zones_from(**kwargs):
        calls["extract_kwargs"] = kwargs
        return kwargs["data"].iloc[[0, 2]].copy()

    def fake_write_dataframe(
        df: pd.DataFrame,
        output: str,
        **kwargs,
    ) -> str:
        calls["written_df"] = df.copy()
        calls["write_kwargs"] = kwargs
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        return str(out)

    monkeypatch.setattr(
        mod,
        "load_dataframe_from_args",
        fake_load_dataframe_from_args,
    )
    monkeypatch.setattr(
        mod,
        "extract_zones_from",
        fake_extract_zones_from,
    )
    monkeypatch.setattr(
        mod,
        "write_dataframe",
        fake_write_dataframe,
    )

    output = tmp_path / "zones.csv"
    result = mod.run_build_extract_zones(
        input_files=["dummy.csv"],
        input_format=None,
        excel_sheet=None,
        excel_engine=None,
        csv_sep=",",
        csv_encoding="utf-8",
        z_col="subsidence_cum",
        threshold=["10", "20"],
        condition="between",
        percentile=15.0,
        positive_criteria=True,
        x_col="longitude",
        y_col="latitude",
        view=False,
        plot_type="scatter",
        figsize=(7.0, 4.0),
        axis_off=True,
        no_grid=True,
        output=str(output),
        excel_output_sheet="Sheet1",
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert output.exists()

    extract_kwargs = calls["extract_kwargs"]
    assert extract_kwargs["z"] == "subsidence_cum"
    assert extract_kwargs["threshold"] == (10.0, 20.0)
    assert extract_kwargs["condition"] == "between"
    assert extract_kwargs["use_negative_criteria"] is False
    assert extract_kwargs["show_grid"] is False
    assert extract_kwargs["axis_off"] is True
    assert extract_kwargs["figsize"] == (7.0, 4.0)


def test_build_extract_zones_main_parses_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_build_extract_zones(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"x": [1]})

    monkeypatch.setattr(
        mod,
        "run_build_extract_zones",
        fake_run_build_extract_zones,
    )

    input_csv = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "subsidence_cum": [1.0, 2.0],
            "longitude": [113.4, 113.5],
            "latitude": [22.1, 22.2],
        }
    ).to_csv(input_csv, index=False)

    mod.build_extract_zones_main(
        [
            str(input_csv),
            "--z-col",
            "subsidence_cum",
            "--threshold",
            "auto",
            "--output",
            str(tmp_path / "zones.csv"),
        ]
    )

    assert captured["paths"] == [str(input_csv)]
    assert captured["z_col"] == "subsidence_cum"
    assert captured["threshold"] == ["auto"]
    assert captured["output"] == str(tmp_path / "zones.csv")
    assert captured["condition"] == "auto"
