from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

MODULE_CANDIDATES = (
    "geoprior.utils.inspect.calibration_stats",
    "calibration_stats",
)


PACKAGE_CANDIDATES = (
    "geoprior.utils.inspect",
    "geoprior.utils",
)


def _import_module():
    for name in MODULE_CANDIDATES:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            missing = str(getattr(exc, "name", "") or "")
            if name == missing or name.startswith(
                missing + "."
            ):
                continue
            raise
    pytest.skip("Could not import calibration_stats module.")


@pytest.fixture
def mod():
    return _import_module()


def test_public_exports_include_calibration_stats_helpers(
    mod,
):
    for pkg_name in PACKAGE_CANDIDATES:
        try:
            pkg = importlib.import_module(pkg_name)
        except ModuleNotFoundError:
            continue
        assert hasattr(
            pkg, "default_calibration_stats_payload"
        )
        assert hasattr(pkg, "generate_calibration_stats")
        assert hasattr(pkg, "summarize_calibration_stats")


def test_default_payload_frames_and_summary(mod):
    payload = mod.default_calibration_stats_payload()

    factors = mod.calibration_stats_factors_frame(payload)
    overall = mod.calibration_stats_overall_frame(payload)
    per_before = mod.calibration_stats_per_horizon_frame(
        payload, which="eval_before"
    )
    per_after = mod.calibration_stats_per_horizon_frame(
        payload, which="eval_after"
    )
    summary = mod.summarize_calibration_stats(payload)

    assert list(factors["horizon"]) == ["1", "2", "3"]
    assert set(overall["which"]) == {
        "eval_before",
        "eval_after",
    }
    assert not per_before.empty
    assert not per_after.empty
    assert summary["n_horizons"] == 3
    assert summary["has_eval_before"] is True
    assert summary["has_eval_after"] is True
    assert summary["has_factors"] is True
    assert summary["coverage_error_improved"] is True


def test_load_can_extract_nested_payload_from_eval_physics_json(
    tmp_path: Path, mod
):
    eval_physics_mod = importlib.import_module(
        "geoprior.utils.inspect.eval_physics"
    )
    nested_payload = (
        eval_physics_mod.default_eval_physics_payload()
    )
    path = tmp_path / "eval_physics_nested.json"
    eval_physics_mod.write_json(nested_payload, path)

    record = mod.load_calibration_stats(path)

    assert record.kind == "calibration_stats"
    assert record.meta["has_eval_before"] is True
    assert "factors" in record.payload


def test_generate_and_inspect_calibration_stats(
    tmp_path: Path, mod
):
    out_json = tmp_path / "calibration_stats.json"
    out = mod.generate_calibration_stats(
        out_json,
        overrides={
            "target": 0.85,
            "factors": {"1": 1.0, "2": 1.1, "3": 1.2},
        },
    )

    assert out == out_json
    assert out_json.exists()

    bundle = mod.inspect_calibration_stats(out_json)
    assert "summary" in bundle
    assert "overall" in bundle
    assert "factors" in bundle
    assert bundle["summary"]["target"] == 0.85


def test_plot_helpers_return_axes(mod):
    payload = mod.default_calibration_stats_payload()
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    axes = axes.ravel()

    out_axes = [
        mod.plot_calibration_factors(axes[0], payload),
        mod.plot_calibration_overall_metrics(
            axes[1], payload
        ),
        mod.plot_calibration_per_horizon_coverage(
            axes[2], payload
        ),
        mod.plot_calibration_per_horizon_sharpness(
            axes[3], payload
        ),
        mod.plot_calibration_boolean_summary(
            axes[4], payload
        ),
    ]

    assert all(hasattr(ax, "plot") for ax in out_axes)
    plt.close(fig)
