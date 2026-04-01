from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

MODULE_CANDIDATES = (
    "geoprior.utils.inspect.eval_physics",
    "eval_physics",
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
    pytest.skip("Could not import eval_physics module.")


@pytest.fixture
def mod():
    return _import_module()


def test_public_exports_include_eval_physics_helpers(mod):
    for pkg_name in PACKAGE_CANDIDATES:
        try:
            pkg = importlib.import_module(pkg_name)
        except ModuleNotFoundError:
            continue
        assert hasattr(pkg, "default_eval_physics_payload")
        assert hasattr(pkg, "generate_eval_physics")
        assert hasattr(pkg, "summarize_eval_physics")


def test_default_payload_frames_and_summary(mod):
    payload = mod.default_eval_physics_payload(horizon=3)

    metrics = mod.eval_physics_metrics_frame(payload)
    point = mod.eval_physics_point_metrics_frame(payload)
    units = mod.eval_physics_units_frame(payload)
    censor = mod.eval_physics_censor_frame(payload)
    per_h = mod.eval_physics_per_horizon_frame(payload)
    cal = mod.eval_physics_calibration_frame(payload)
    cal_h = mod.eval_physics_calibration_per_horizon_frame(
        payload
    )
    summary = mod.summarize_eval_physics(payload)

    assert not metrics.empty
    assert not point.empty
    assert not units.empty
    assert not censor.empty
    assert set(per_h["metric"]) >= {"mae", "r2"}
    assert not cal.empty
    assert not cal_h.empty
    assert summary["brief"]["kind"] == "eval_physics"
    assert summary["checks"]["has_metrics_evaluate"] is True
    assert (
        summary["checks"]["has_physics_diagnostics"] is True
    )
    assert (
        summary["checks"]["has_interval_calibration"] is True
    )
    assert summary["checks"]["reported_unit_present"] is True


def test_generate_load_and_inspect_eval_physics(
    tmp_path: Path, mod
):
    out_json = tmp_path / "eval_physics.json"
    out_dir = tmp_path / "inspect"

    path = mod.generate_eval_physics(
        output_path=out_json,
        overrides={
            "metrics_evaluate": {
                "physics_loss": 0.25,
                "physics_loss_scaled": 0.125,
            },
            "physics_diagnostics": {
                "epsilon_prior": 1e-4,
                "epsilon_cons": 2e-5,
                "epsilon_gw": 3e-6,
            },
        },
    )

    assert path == out_json
    assert out_json.exists()

    record = mod.load_eval_physics(out_json)
    assert record.kind == "eval_physics"

    bundle = mod.inspect_eval_physics(
        out_json,
        output_dir=out_dir,
        save_figures=True,
    )

    assert "metrics_evaluate" in bundle["frames"]
    assert (
        "interval_calibration_per_horizon" in bundle["frames"]
    )
    assert len(bundle["figures"]) == 6
    for path in bundle["figures"].values():
        assert Path(path).exists()


def test_plot_helpers_return_axes(mod):
    payload = mod.default_eval_physics_payload()
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    axes = axes.ravel()

    out_axes = [
        mod.plot_eval_physics_metrics(payload, ax=axes[0]),
        mod.plot_eval_physics_epsilons(payload, ax=axes[1]),
        mod.plot_eval_physics_calibration_factors(
            payload, ax=axes[2]
        ),
        mod.plot_eval_physics_point_metrics(
            payload, ax=axes[3]
        ),
        mod.plot_eval_physics_per_horizon_metrics(
            payload, ax=axes[4]
        ),
        mod.plot_eval_physics_boolean_summary(
            payload, ax=axes[5]
        ),
    ]

    assert all(hasattr(ax, "plot") for ax in out_axes)
    plt.close(fig)
