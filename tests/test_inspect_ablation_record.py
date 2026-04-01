from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")


def _import_target():
    candidates = (
        "geoprior.utils.inspect.ablation_record",
        "ablation_record",
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
        "Could not import geoprior.utils.inspect.ablation_record."
    )


def test_default_payload_has_expected_record_shape():
    mod = _import_target()

    records = mod.default_ablation_record_payload()

    assert isinstance(records, list)
    assert len(records) >= 2

    first = records[0]
    assert "metrics" in first
    assert "units" in first
    assert "per_horizon_mae" in first
    assert "per_horizon_r2" in first
    assert "mae" in first
    assert "rmse" in first
    assert "r2" in first
    assert "lambda_gw" in first


def test_generate_and_load_ablation_record_jsonl_roundtrip(
    tmp_path: Path,
):
    mod = _import_target()

    out = tmp_path / "ablation_record.jsonl"
    written = mod.generate_ablation_record(out)

    assert Path(written).exists()

    records = mod.load_ablation_record(out)
    assert isinstance(records, list)
    assert len(records) == len(
        mod.default_ablation_record_payload()
    )

    first = records[0]
    assert first["city"] == "nansha"
    assert "metrics" in first
    assert "units" in first


def test_generate_ablation_record_accepts_dict_and_list_overrides(
    tmp_path: Path,
):
    mod = _import_target()

    out1 = tmp_path / "ablation_all.jsonl"
    mod.generate_ablation_record(
        out1,
        overrides={
            "city": "zhongshan",
            "lambda_gw": 0.25,
            "metrics": {
                "subs_metrics_unit": "ignored_by_schema_ok"
            },
        },
    )
    records1 = mod.load_ablation_record(out1)
    assert all(rec["city"] == "zhongshan" for rec in records1)
    assert all(
        float(rec["lambda_gw"]) == 0.25 for rec in records1
    )

    out2 = tmp_path / "ablation_list.jsonl"
    mod.generate_ablation_record(
        out2,
        overrides=[
            {"ablation": "custom_a", "seed": 101},
            {"ablation": "custom_b", "seed": 202},
        ],
    )
    records2 = mod.load_ablation_record(out2)
    assert records2[0]["ablation"] == "custom_a"
    assert records2[0]["seed"] == 101
    assert records2[1]["ablation"] == "custom_b"
    assert records2[1]["seed"] == 202


def test_frames_and_summary_are_consistent():
    mod = _import_target()
    records = mod.default_ablation_record_payload(
        city="nansha"
    )

    runs = mod.ablation_record_runs_frame(records)
    metrics = mod.ablation_metrics_frame(records)
    per_h = mod.ablation_per_horizon_frame(records)
    flags = mod.ablation_record_flags_frame(records)
    config = mod.ablation_config_frame(records)
    summary = mod.summarize_ablation_record(records)

    assert not runs.empty
    assert not metrics.empty
    assert not per_h.empty
    assert not flags.empty
    assert not config.empty

    assert "variant" in runs.columns
    assert "rmse" in runs.columns
    assert "subs_metrics_unit" in runs.columns

    assert set(metrics["metric"]) >= {"mae", "rmse", "r2"}
    assert set(per_h["metric"]) >= {"mae", "r2"}
    assert set(flags["flag"]) == {"use_effective_h"}

    assert summary["record_count"] == len(records)
    assert summary["variant_count"] >= 1
    assert summary["has_metrics"] is True
    assert summary["has_per_horizon"] is True
    assert summary["has_units"] is True
    assert summary["checks"]["has_records"] is True
    assert summary["checks"]["has_core_metrics"] is True
    assert summary["best_by_rmse"] is not None
    assert summary["best_by_r2"] is not None


def test_plot_helpers_smoke():
    mod = _import_target()
    records = mod.default_ablation_record_payload()

    ax = mod.plot_ablation_run_counts(records)
    assert ax.get_title()

    ax = mod.plot_ablation_metric_by_variant(
        records, metric="rmse"
    )
    assert "rmse" in ax.get_title().lower()

    ax = mod.plot_ablation_lambda_weights(records)
    assert ax.get_title()

    ax = mod.plot_ablation_per_horizon_metric(
        records, metric="mae"
    )
    assert "mae" in ax.get_title().lower()

    ax = mod.plot_ablation_top_variants(
        records, metric="r2", top_n=3
    )
    assert ax.get_title()

    ax = mod.plot_ablation_boolean_summary(records)
    assert ax.get_title()


def test_inspect_ablation_record_bundle_and_saved_figures(
    tmp_path: Path,
):
    mod = _import_target()
    records = mod.default_ablation_record_payload()

    bundle = mod.inspect_ablation_record(
        records,
        output_dir=tmp_path,
        stem="ablation_case",
        save_figures=True,
    )

    assert "summary" in bundle
    assert "frames" in bundle
    assert "figure_paths" in bundle

    frames = bundle["frames"]
    assert set(frames) >= {
        "runs",
        "metrics",
        "per_horizon",
        "flags",
        "config",
    }

    fig_paths = bundle["figure_paths"]
    assert len(fig_paths) == 6
    for path in fig_paths.values():
        assert Path(path).exists()


def test_load_ablation_record_from_path_matches_frame_builders(
    tmp_path: Path,
):
    mod = _import_target()

    out = tmp_path / "ablation.jsonl"
    mod.generate_ablation_record(out)

    loaded = mod.load_ablation_record(out)
    runs = mod.ablation_record_runs_frame(out)
    metrics = mod.ablation_metrics_frame(out)

    assert len(loaded) == len(runs)
    assert not metrics.empty
    assert runs["record_id"].tolist() == list(
        range(1, len(runs) + 1)
    )
