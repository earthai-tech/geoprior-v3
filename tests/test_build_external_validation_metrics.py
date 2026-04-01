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
            if modname == missing or modname.startswith(
                missing + "."
            ):
                continue
            raise
    pytest.skip(
        f"Could not import target module for {name!r}."
    )


pytestmark = [pytest.mark.stage_artifacts]


def test_compute_metrics_writes_site_table_and_metrics_json(
    tmp_path: Path,
):
    mod = _import_target("build_external_validation_metrics")

    stage1_manifest = tmp_path / "manifest.json"
    stage1_manifest.write_text(
        __import__("json").dumps(
            {"artifacts": {"encoders": {}}}
        ),
        encoding="utf-8",
    )

    coords = np.array(
        [
            [[0.0, 10.0, 100.0]],
            [[0.0, 20.0, 200.0]],
        ],
        dtype=np.float32,
    )
    h_field = np.array([[[5.0]], [[7.0]]], dtype=np.float32)
    payload_k = np.array([[[0.1]], [[0.2]]], dtype=np.float32)
    payload_hd = np.array(
        [[[4.0]], [[6.0]]], dtype=np.float32
    )

    inputs_npz = tmp_path / "inputs.npz"
    payload_npz = tmp_path / "payload.npz"
    np.savez(inputs_npz, coords=coords, H_field=h_field)
    np.savez(payload_npz, K=payload_k, Hd=payload_hd)

    validation = pd.DataFrame(
        {
            "well_id": ["BH1", "BH2"],
            "x": [10.0, 20.0],
            "y": [100.0, 200.0],
            "approx_compressible_thickness_m": [5.0, 7.0],
            "step3_specific_capacity_Lps_per_m": [0.1, 0.2],
        }
    )
    validation_csv = tmp_path / "validation.csv"
    validation.to_csv(validation_csv, index=False)

    outdir = tmp_path / "metrics_out"
    site_df, metrics = mod.compute_metrics(
        validation_csv=str(validation_csv),
        stage1_manifest_path=str(stage1_manifest),
        outdir=str(outdir),
        split="test",
        inputs_npz=str(inputs_npz),
        physics_payload=str(payload_npz),
        coord_scaler=None,
        stage2_manifest_path=None,
        productivity_col="step3_specific_capacity_Lps_per_m",
        horizon_reducer="mean",
        site_reducer="median",
        max_distance_m=50000.0,
        min_unique_pixels=1,
    )

    assert len(site_df) == 2
    assert metrics["n_sites"] == 2
    assert (
        outdir / "site_level_external_validation.csv"
    ).exists()
    assert (
        outdir / "external_validation_metrics.json"
    ).exists()


def test_nearest_match_can_prefer_swapped_xy():
    mod = _import_target("build_external_validation_metrics")

    pixels = pd.DataFrame(
        {
            "x": [1.0, 10.0],
            "y": [2.0, 20.0],
            "pixel_idx": [0, 1],
        }
    )
    match = mod.nearest_match(
        pixels,
        sx=2.0,
        sy=1.0,
        well_id="BH1",
    )

    assert match.match_mode == "swapped_xy"
    assert match.pixel_idx == 0
