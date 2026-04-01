from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest


def _import_target():
    candidates = (
        "geoprior.utils.inspect.training_summary",
        "training_summary",
    )
    for modname in candidates:
        try:
            return importlib.import_module(modname)
        except ModuleNotFoundError:
            continue
    pytest.skip("Could not import training_summary module")


def test_default_payload_has_expected_sections():
    mod = _import_target()

    payload = mod.default_training_summary_payload()
    assert payload["city"] == "demo_city"
    assert "metrics_at_best" in payload
    assert "final_epoch_metrics" in payload
    assert "compile" in payload
    assert "hp_init" in payload
    assert "paths" in payload


def test_generate_and_load_training_summary_roundtrip(
    tmp_path: Path,
):
    mod = _import_target()

    out = tmp_path / "training_summary.json"
    mod.generate_training_summary(
        output_path=out,
        overrides={
            "city": "nansha",
            "best_epoch": 12,
        },
    )

    record = mod.load_training_summary(out)
    assert record.kind == "training_summary"
    assert record.city == "nansha"
    assert record.payload["best_epoch"] == 12


def test_training_frames_are_nonempty():
    mod = _import_target()

    payload = mod.default_training_summary_payload()

    best = mod.training_metrics_frame(
        payload,
        section="metrics_at_best",
        split="val",
    )
    final = mod.training_metrics_frame(
        payload,
        section="final_epoch_metrics",
        split="train",
    )
    env = mod.training_env_frame(payload)
    compile_frame = mod.training_compile_frame(payload)
    hp = mod.training_hp_frame(payload)
    paths = mod.training_paths_frame(payload)

    assert not best.empty
    assert not final.empty
    assert not env.empty
    assert not compile_frame.empty
    assert not hp.empty
    assert not paths.empty
    assert {"section", "split", "metric", "value"} <= set(
        best.columns
    )
    assert {"key", "value"} <= set(paths.columns)


def test_summarize_training_summary_reports_core_checks():
    mod = _import_target()

    payload = mod.default_training_summary_payload()
    summary = mod.summarize_training_summary(payload)

    assert summary["brief"]["kind"] == "training_summary"
    assert summary["checks"]["has_best_metrics"] is True
    assert summary["checks"]["has_final_metrics"] is True
    assert summary["checks"]["has_optimizer"] is True
    assert "best_val_loss" in summary["core_metrics"]


def test_inspect_training_summary_bundle_and_saved_figures(
    tmp_path: Path,
):
    mod = _import_target()

    payload = mod.default_training_summary_payload()
    bundle = mod.inspect_training_summary(
        payload,
        output_dir=tmp_path,
        stem="ts",
        save_figures=True,
    )

    assert "summary" in bundle
    assert "frames" in bundle
    assert "figure_paths" in bundle
    assert "metrics_at_best" in bundle["frames"]
    assert len(bundle["figure_paths"]) == 5

    for path in bundle["figure_paths"].values():
        assert Path(path).exists()


def test_training_plot_functions_smoke(tmp_path: Path):
    mod = _import_target()
    payload = mod.default_training_summary_payload()

    for idx, fn in enumerate(
        [
            lambda ax: mod.plot_training_best_metrics(
                payload, split="val", ax=ax
            ),
            lambda ax: mod.plot_training_final_metrics(
                payload, split="val", ax=ax
            ),
            lambda ax: mod.plot_training_metric_deltas(
                payload, split="val", ax=ax
            ),
            lambda ax: mod.plot_training_loss_family(
                payload,
                section="metrics_at_best",
                split="val",
                ax=ax,
            ),
            lambda ax: mod.plot_training_boolean_summary(
                payload, ax=ax
            ),
        ],
        start=1,
    ):
        fig, ax = plt.subplots()
        fn(ax)
        out = tmp_path / f"plot_{idx}.png"
        fig.savefig(out)
        plt.close(fig)
        assert out.exists()
