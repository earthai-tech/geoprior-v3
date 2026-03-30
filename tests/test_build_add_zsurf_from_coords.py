from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import geoprior.cli.build_add_zsurf_from_coords as mod


def test_enrich_city_dataset_merges_zsurf_and_head(
    tmp_path: Path,
) -> None:
    main_csv = tmp_path / "nansha_main.csv"
    elev_csv = tmp_path / "nansha_elev.csv"

    pd.DataFrame(
        {
            "longitude": [113.4, 113.5],
            "latitude": [22.1, 22.2],
            "GWL_depth_bgs_m": [1.5, 2.0],
            "value": [10, 20],
        }
    ).to_csv(main_csv, index=False)
    pd.DataFrame(
        {
            "longitude": [113.4, 113.5],
            "latitude": [22.1, 22.2],
            "elevation": [5.0, 6.0],
        }
    ).to_csv(elev_csv, index=False)

    args = mod._build_parser().parse_args if hasattr(mod, '_build_parser') else None
    # The module exposes argparse assembly inline, so a light namespace is enough.
    ns = type(
        "Args",
        (),
        {"depth_col": ["GWL_depth_bgs_m"]},
    )()

    merged, diag = mod.enrich_city_dataset(
        mod.CityPaths(
            city="nansha",
            main_csv=main_csv,
            elev_csv=elev_csv,
        ),
        lon_col="longitude",
        lat_col="latitude",
        elev_col="elevation",
        zsurf_col="z_surf_m",
        round_decimals=6,
        reducer="mean",
        compute_head=True,
        head_col="head_m",
        args=ns,
    )

    assert "z_surf_m" in merged.columns
    assert "head_m" in merged.columns
    assert merged["head_m"].tolist() == [3.5, 4.0]
    assert diag["computed_head"] is True
    assert diag["missing_zsurf"] == 0


def test_build_add_zsurf_main_writes_outputs_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "out"
    data_root = tmp_path / "data"
    coords_root = tmp_path / "coords"
    data_root.mkdir()
    coords_root.mkdir()

    for city, z0 in (("nansha", 5.0), ("zhongshan", 7.0)):
        pd.DataFrame(
            {
                "longitude": [113.4],
                "latitude": [22.1],
                "GWL_depth_bgs_m": [1.0],
            }
        ).to_csv(
            data_root / f"{city}_final_main_std.harmonized.csv",
            index=False,
        )
        pd.DataFrame(
            {
                "longitude": [113.4],
                "latitude": [22.1],
                "elevation": [z0],
            }
        ).to_csv(
            coords_root / f"{city}_coords_with_elevation.csv",
            index=False,
        )

    monkeypatch.setattr(
        mod,
        "bootstrap_runtime_config",
        lambda args, field_map=None: {},
    )
    monkeypatch.setattr(
        mod,
        "ensure_outdir",
        lambda p: Path(p).expanduser().resolve(),
    )

    summary_json = tmp_path / "summary.json"
    mod.build_add_zsurf_main(
        [
            "--city",
            "nansha",
            "--city",
            "zhongshan",
            "--data-root",
            str(data_root),
            "--coords-root",
            str(coords_root),
            "--outdir",
            str(outdir),
            "--summary-json",
            str(summary_json),
        ]
    )

    n_out = outdir / "nansha_final_main_std.harmonized.with_zsurf.csv"
    z_out = outdir / "zhongshan_final_main_std.harmonized.with_zsurf.csv"
    assert n_out.exists()
    assert z_out.exists()
    assert summary_json.exists()

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert {item["city"] for item in summary} == {
        "nansha",
        "zhongshan",
    }
