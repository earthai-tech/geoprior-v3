from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib.pyplot as plt
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


def test_manifest_payload_frames_and_summary():
    mod = _import_target("manifest")
    payload = mod.default_manifest_payload()

    summary = mod.summarize_manifest(payload)
    assert summary["schema_version"] == "3.2"
    assert summary["stage"] == "stage1"
    assert summary["static_feature_count"] > 0
    assert summary["dynamic_feature_count"] > 0
    assert summary["artifact_path_count"] > 0
    assert summary["has_holdout"] is True

    ident = mod.manifest_identity_frame(payload)
    cfg = mod.manifest_config_frame(payload)
    groups = mod.manifest_feature_groups_frame(payload)
    holdout = mod.manifest_holdout_frame(payload)
    arts = mod.manifest_artifacts_frame(payload)
    paths = mod.manifest_paths_frame(payload)
    vers = mod.manifest_versions_frame(payload)
    shapes = mod.manifest_shapes_frame(payload)

    assert "schema_version" in set(ident["key"])
    assert "TIME_STEPS" in set(cfg["key"])
    assert set(groups["group"]) >= {
        "static",
        "dynamic",
        "future",
    }
    assert "train_groups" in set(holdout["key"])
    assert not arts.empty
    assert "run_dir" in set(paths["key"])
    assert "python" in set(vers["key"])
    assert "train_inputs" in set(shapes["split"])
    assert shapes["rank"].max() >= 2


def test_manifest_roundtrip_and_inspect_bundle(
    tmp_path: Path,
):
    mod = _import_target("manifest")
    pkg = importlib.import_module("geoprior.utils.inspect")

    out = tmp_path / "manifest.json"
    mod.generate_manifest(
        out,
        overrides={
            "city": "zhongshan",
            "config": {
                "TIME_STEPS": 7,
                "features": {
                    "future": [
                        "rainfall_mm",
                        "temperature_c",
                    ],
                },
            },
        },
    )

    record = mod.load_manifest(out)
    assert record.payload["city"] == "zhongshan"
    assert record.payload["config"]["TIME_STEPS"] == 7

    bundle = mod.inspect_manifest(record)
    assert bundle["summary"]["city"] == "zhongshan"
    assert not bundle["identity"].empty
    assert not bundle["config"].empty
    assert not bundle["feature_groups"].empty
    assert not bundle["holdout"].empty
    assert not bundle["artifacts"].empty
    assert not bundle["paths"].empty
    assert not bundle["versions"].empty
    assert not bundle["shapes"].empty

    assert (
        pkg.default_manifest_payload
        is mod.default_manifest_payload
    )
    assert pkg.inspect_manifest is mod.inspect_manifest


def test_manifest_plotting_smoke():
    mod = _import_target("manifest")
    payload = mod.default_manifest_payload()

    plotters = [
        mod.plot_manifest_artifact_inventory,
        mod.plot_manifest_feature_group_sizes,
        mod.plot_manifest_holdout_counts,
        mod.plot_manifest_coord_ranges,
        mod.plot_manifest_boolean_summary,
    ]

    for fn in plotters:
        fig, ax = plt.subplots(figsize=(6, 4))
        out_ax = fn(ax, payload)
        assert out_ax is ax
        assert ax.get_title()
        plt.close(fig)
