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


def _write_npz(path: Path, **arrays) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def test_compute_external_metrics_writes_site_table_and_json(tmp_path: Path):
    mod = _import_target("build_external_validation_fullcity")

    coords = np.array(
        [
            [[0.0, 10.0, 100.0]],
            [[0.0, 20.0, 200.0]],
        ],
        dtype=np.float32,
    )
    h_field = np.array([[[5.0]], [[7.0]]], dtype=np.float32)
    payload_k = np.array([[[0.1]], [[0.2]]], dtype=np.float32)
    payload_hd = np.array([[[4.0]], [[6.0]]], dtype=np.float32)
    payload_h = np.array([[[5.5]], [[7.5]]], dtype=np.float32)

    inputs_npz = _write_npz(
        tmp_path / "full_inputs.npz",
        coords=coords,
        H_field=h_field,
    )
    payload_npz = _write_npz(
        tmp_path / "payload.npz",
        K=payload_k,
        Hd=payload_hd,
        H=payload_h,
    )

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

    outdir = tmp_path / "out"
    site_df, metrics = mod.compute_external_metrics(
        validation_csv=str(validation_csv),
        full_inputs_npz=str(inputs_npz),
        full_payload_npz=str(payload_npz),
        coord_scaler_path=None,
        outdir=str(outdir),
        x_col="x",
        y_col="y",
        productivity_col="step3_specific_capacity_Lps_per_m",
        thickness_col="approx_compressible_thickness_m",
        horizon_reducer="mean",
        site_reducer="median",
        max_match_distance_m=50000.0,
        min_unique_pixels=1,
        allow_swapped_xy=False,
    )

    assert len(site_df) == 2
    assert metrics["n_sites"] == 2
    assert (outdir / "site_level_external_validation_fullcity.csv").exists()
    assert (outdir / "external_validation_metrics_fullcity.json").exists()


def test_make_full_inputs_npz_concatenates_matching_keys(tmp_path: Path):
    mod = _import_target("build_external_validation_fullcity")

    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    out = tmp_path / "full.npz"

    np.savez(a, coords=np.array([[1.0]]), H_field=np.array([[2.0]]))
    np.savez(b, coords=np.array([[3.0]]), H_field=np.array([[4.0]]))

    result = mod.make_full_inputs_npz([str(a), str(b)], str(out))
    merged = np.load(result)

    assert merged["coords"].shape[0] == 2
    assert merged["H_field"].shape[0] == 2
