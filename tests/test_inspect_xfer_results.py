from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest


def _import_target():
    candidates = (
        "geoprior.utils.inspect.xfer_results",
        "xfer_results",
    )
    for modname in candidates:
        try:
            return importlib.import_module(modname)
        except ModuleNotFoundError:
            continue
    pytest.skip("Could not import xfer_results module")


def test_default_xfer_results_payload_has_two_records():
    mod = _import_target()

    payload = mod.default_xfer_results_payload()
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert {rec["direction"] for rec in payload} == {
        "A_to_B",
        "B_to_A",
    }


def test_generate_and_load_xfer_results_roundtrip(
    tmp_path: Path,
):
    mod = _import_target()

    out = tmp_path / "xfer_results.json"
    mod.generate_xfer_results(
        out,
        overrides={"strategy": "baseline"},
    )

    records = mod.load_xfer_results(out)
    assert len(records) == 2
    assert all(
        rec["strategy"] == "baseline" for rec in records
    )


def test_xfer_frames_are_nonempty_and_well_formed():
    mod = _import_target()

    records = mod.default_xfer_results_payload()
    overall = mod.xfer_overall_frame(records)
    per_h = mod.xfer_per_horizon_frame(records)
    schema = mod.xfer_schema_frame(records)
    warm = mod.xfer_warm_frame(records)

    assert not overall.empty
    assert not per_h.empty
    assert not schema.empty
    assert not warm.empty
    assert {"label", "direction", "overall_rmse"} <= set(
        overall.columns
    )
    assert {"metric", "horizon", "value"} <= set(
        per_h.columns
    )
    assert {"kind", "name", "value"} <= set(schema.columns)
    assert {"warm_split", "warm_epochs"} <= set(warm.columns)


def test_summarize_xfer_results_reports_best_records():
    mod = _import_target()

    records = mod.default_xfer_results_payload()
    summary = mod.summarize_xfer_results(records)

    assert summary["n_records"] == 2
    assert "best_overall_rmse" in summary
    assert "best_overall_r2" in summary
    assert "schema_pass_rates" in summary
    assert "mean_coverage80" in summary


def test_inspect_xfer_results_bundle():
    mod = _import_target()

    records = mod.default_xfer_results_payload()
    bundle = mod.inspect_xfer_results(records)

    assert set(bundle) == {
        "summary",
        "overall",
        "per_horizon",
        "schema",
        "warm",
    }
    assert not bundle["overall"].empty


def test_xfer_plot_functions_smoke(tmp_path: Path):
    mod = _import_target()
    records = mod.default_xfer_results_payload()

    plotters = [
        lambda: mod.plot_xfer_overall_metrics(records),
        lambda: mod.plot_xfer_direction_metric(
            records,
            metric="overall_rmse",
        ),
        lambda: mod.plot_xfer_per_horizon_metrics(
            records,
            metric="rmse",
        ),
        lambda: mod.plot_xfer_schema_counts(records),
        lambda: mod.plot_xfer_boolean_summary(records),
    ]

    for idx, make in enumerate(plotters, start=1):
        fig, _ = make()
        out = tmp_path / f"xfer_plot_{idx}.png"
        fig.savefig(out)
        plt.close(fig)
        assert out.exists()
