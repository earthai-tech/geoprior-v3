from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest


def _import_target(name: str):
    candidates = (
        f"geoprior.utils.inspect.{name}",
        f"geoprior.utils.{name}",
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


def test_stage1_audit_public_reexports_exist():
    inspect_pkg = importlib.import_module(
        "geoprior.utils.inspect"
    )
    root_pkg = importlib.import_module("geoprior.utils")

    assert hasattr(
        inspect_pkg, "default_stage1_audit_payload"
    )
    assert hasattr(inspect_pkg, "inspect_stage1_audit")
    assert hasattr(root_pkg, "default_stage1_audit_payload")
    assert hasattr(root_pkg, "inspect_stage1_audit")


def test_build_stage1_feature_split_and_summary():
    mod = _import_target("stage1_audit")

    split = mod.build_stage1_feature_split(
        dynamic_features=[
            "a__si",
            "rainfall_mm",
            "other_dyn",
        ],
        static_features=["z_surf_m__si", "cat"],
        future_features=["rainfall_mm"],
        scaled_ml_numeric_cols=["rainfall_mm"],
    )

    assert split["dynamic_features"]["count"] == 3
    assert split["dynamic_features"][
        "scaled_by_main_scaler"
    ] == ["rainfall_mm"]
    assert split["dynamic_features"]["si_unscaled"] == [
        "a__si"
    ]
    assert split["static_features"]["si_unscaled"] == [
        "z_surf_m__si"
    ]

    payload = mod.default_stage1_audit_payload()
    summary = mod.summarize_stage1_audit(payload)

    assert summary["brief"]["kind"] == "stage1_audit"
    assert summary["checks"]["coords_normalized"] is True
    assert summary["checks"]["has_feature_split"] is True
    assert summary["feature_counts"]["dynamic_features"] == 5
    assert summary["scaled_ml_numeric_cols_count"] == 2


def test_stage1_audit_frames_roundtrip_and_inspect(
    tmp_path: Path,
):
    mod = _import_target("stage1_audit")

    out = tmp_path / "stage1_audit.json"
    written = mod.generate_stage1_audit(
        output_path=out,
        overrides={
            "city": "zhongshan",
            "config": {"TIME_STEPS": 7},
        },
    )
    assert Path(written).exists()

    record = mod.load_stage1_audit(out)
    assert record.kind == "stage1_audit"
    assert record.payload["city"] == "zhongshan"

    coord_frame = mod.stage1_coord_ranges_frame(record)
    feature_frame = mod.stage1_feature_split_frame(record)
    physics_stats = mod.stage1_stats_frame(
        record, section="physics_df_stats"
    )
    target_stats = mod.stage1_stats_frame(
        record, section="targets_stats"
    )

    assert set(coord_frame["coord"]) == {"t", "x", "y"}
    assert set(feature_frame["group"]) >= {
        "dynamic_features",
        "static_features",
        "future_features",
    }
    assert not physics_stats.empty
    assert not target_stats.empty

    fig_dir = tmp_path / "figs"
    bundle = mod.inspect_stage1_audit(
        record, output_dir=fig_dir
    )
    assert set(bundle["frames"]) == {
        "coord_ranges",
        "feature_split",
        "physics_df_stats",
        "targets_stats",
    }
    assert len(bundle["figure_paths"]) == 5
    for path in bundle["figure_paths"].values():
        assert Path(path).exists()


def test_stage1_audit_plot_functions_smoke():
    mod = _import_target("stage1_audit")
    payload = mod.default_stage1_audit_payload()

    calls = [
        lambda ax: mod.plot_stage1_coord_ranges(
            payload, ax=ax
        ),
        lambda ax: mod.plot_stage1_feature_split(
            payload, ax=ax
        ),
        lambda ax: mod.plot_stage1_variable_stats(
            payload,
            section="physics_df_stats",
            stat="mean",
            ax=ax,
        ),
        lambda ax: mod.plot_stage1_target_stats(
            payload, stat="mean", ax=ax
        ),
        lambda ax: mod.plot_stage1_boolean_summary(
            payload, ax=ax
        ),
    ]

    for call in calls:
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        out_ax = call(ax)
        assert out_ax is ax
        plt.close(fig)
