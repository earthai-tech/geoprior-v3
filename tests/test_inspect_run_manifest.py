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


def test_run_manifest_payload_frames_and_summary():
    mod = _import_target("run_manifest")
    payload = mod.default_run_manifest_payload(
        city="zhongshan",
        identifiability_regime="anchored",
    )

    summary = mod.summarize_run_manifest(payload)
    assert summary["city"] == "zhongshan"
    assert summary["model"] == "GeoPriorSubsNet"
    assert summary["path_count"] >= 5
    assert summary["artifact_count"] >= 1
    assert summary["dynamic_feature_count"] > 0
    assert summary["has_final_keras"] is True

    ident = mod.run_manifest_identity_frame(payload)
    cfg = mod.run_manifest_config_frame(payload)
    scaling = mod.run_manifest_scaling_overview_frame(payload)
    paths = mod.run_manifest_paths_frame(payload)
    arts = mod.run_manifest_artifacts_frame(payload)

    assert list(ident["key"]) == ["stage", "city", "model"]
    assert "TIME_STEPS" in set(cfg["key"])
    assert "dynamic_feature_count" in set(scaling["key"])
    assert "final_keras" in set(paths["key"])
    assert "training_summary_json" in set(arts["key"])


def test_run_manifest_roundtrip_and_inspect_bundle(
    tmp_path: Path,
):
    mod = _import_target("run_manifest")
    pkg = importlib.import_module("geoprior.utils.inspect")

    out = tmp_path / "run_manifest.json"
    mod.generate_run_manifest(
        out,
        overrides={
            "city": "nansha",
            "config": {"PDE_MODE_CONFIG": "off"},
        },
    )

    record = mod.load_run_manifest(out)
    assert record.payload["city"] == "nansha"
    assert (
        record.payload["config"]["PDE_MODE_CONFIG"] == "off"
    )

    bundle = mod.inspect_run_manifest(record)
    assert bundle["summary"]["city"] == "nansha"
    assert not bundle["identity"].empty
    assert not bundle["config"].empty
    assert not bundle["scaling_overview"].empty
    assert not bundle["paths"].empty
    assert not bundle["artifacts"].empty

    assert (
        pkg.default_run_manifest_payload
        is mod.default_run_manifest_payload
    )
    assert (
        pkg.inspect_run_manifest is mod.inspect_run_manifest
    )


def test_run_manifest_plotting_smoke():
    mod = _import_target("run_manifest")
    payload = mod.default_run_manifest_payload()

    plotters = [
        mod.plot_run_manifest_path_inventory,
        mod.plot_run_manifest_feature_group_sizes,
        mod.plot_run_manifest_coord_ranges,
        mod.plot_run_manifest_boolean_summary,
    ]

    for fn in plotters:
        fig, ax = plt.subplots(figsize=(6, 4))
        out_ax = fn(ax, payload)
        assert out_ax is ax
        assert ax.get_title()
        plt.close(fig)
