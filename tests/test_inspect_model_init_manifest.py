from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest


def _import_target(name: str):
    candidates = (
        f"geoprior.utils.inspect.{name}",
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


def test_model_init_manifest_payload_frames_and_summary():
    mod = _import_target("model_init_manifest")
    payload = mod.default_model_init_manifest_payload(
        identifiability_regime="closure_locked",
    )

    summary = mod.summarize_model_init_manifest(payload)
    assert summary["model_class"] == "GeoPriorSubsNet"
    assert summary["forecast_horizon"] == 3
    assert summary["n_static_features"] > 0
    assert summary["n_dynamic_features"] > 0
    assert summary["n_bounds"] > 0
    assert (
        summary["identifiability_regime"] == "closure_locked"
    )

    dims = mod.model_init_dims_frame(payload)
    arch = mod.model_init_architecture_frame(payload)
    gp = mod.model_init_geoprior_frame(payload)
    scaling = mod.model_init_scaling_overview_frame(payload)
    groups = mod.model_init_feature_groups_frame(payload)

    assert "forecast_horizon" in set(dims["name"])
    assert "embed_dim" in set(arch["key"])
    assert "init_kappa" in set(gp["key"])
    assert "time_units" in set(scaling["key"])
    assert set(groups["group"]) == {
        "static",
        "dynamic",
        "future",
    }


def test_model_init_manifest_roundtrip_and_inspect_bundle(
    tmp_path: Path,
):
    mod = _import_target("model_init_manifest")
    pkg = importlib.import_module("geoprior.utils.inspect")

    out = tmp_path / "model_init_manifest.json"
    mod.generate_model_init_manifest(
        out,
        overrides={
            "model_class": "PoroElasticSubsNet",
            "dims": {"forecast_horizon": 5},
        },
    )

    record = mod.load_model_init_manifest(out)
    assert (
        record.payload["model_class"] == "PoroElasticSubsNet"
    )
    assert record.payload["dims"]["forecast_horizon"] == 5

    fig_dir = tmp_path / "figs"
    bundle = mod.inspect_model_init_manifest(
        record,
        output_dir=fig_dir,
        save_figures=True,
    )
    assert (
        bundle["summary"]["model_class"]
        == "PoroElasticSubsNet"
    )
    assert not bundle["frames"]["dims"].empty
    assert not bundle["frames"]["architecture"].empty
    assert not bundle["frames"]["geoprior"].empty
    assert not bundle["frames"]["scaling_overview"].empty
    assert not bundle["frames"]["feature_groups"].empty
    assert not bundle["frames"]["flattened"].empty
    assert len(bundle["figure_paths"]) == 5
    for path in bundle["figure_paths"].values():
        assert Path(path).exists()

    assert (
        pkg.default_model_init_manifest_payload
        is mod.default_model_init_manifest_payload
    )
    assert (
        pkg.inspect_model_init_manifest
        is mod.inspect_model_init_manifest
    )


def test_model_init_manifest_plotting_smoke():
    mod = _import_target("model_init_manifest")
    payload = mod.default_model_init_manifest_payload()

    plotters = [
        mod.plot_model_init_dims,
        mod.plot_model_init_architecture,
        mod.plot_model_init_geoprior,
        mod.plot_model_init_feature_group_sizes,
        mod.plot_model_init_boolean_summary,
    ]

    for fn in plotters:
        fig, ax = plt.subplots(figsize=(6, 4))
        out_ax = fn(payload, ax=ax)
        assert out_ax is ax
        assert ax.get_title()
        plt.close(fig)
