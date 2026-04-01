from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

MODULE_CANDIDATES = (
    "geoprior.utils.inspect.eval_diagnostics",
    "eval_diagnostics",
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
    pytest.skip("Could not import eval_diagnostics module.")


@pytest.fixture
def mod():
    return _import_module()


def test_public_exports_include_eval_diagnostics_helpers(mod):
    for pkg_name in PACKAGE_CANDIDATES:
        try:
            pkg = importlib.import_module(pkg_name)
        except ModuleNotFoundError:
            continue
        assert hasattr(
            pkg, "default_eval_diagnostics_payload"
        )
        assert hasattr(pkg, "generate_eval_diagnostics")
        assert hasattr(pkg, "summarize_eval_diagnostics")


def test_default_payload_frames_and_summary(mod):
    payload = mod.default_eval_diagnostics_payload(
        years=[2020, 2021, 2022],
        per_horizon_mae=[2.0, 4.0, 6.0],
        per_horizon_mse=[4.0, 16.0, 36.0],
        per_horizon_rmse=[2.0, 4.0, 6.0],
        per_horizon_r2=[0.9, 0.8, 0.7],
        coverage80=[0.81, 0.82, 0.83],
        sharpness80=[10.0, 11.0, 12.0],
        pss=[1.0, 2.0, 3.0],
    )

    years = mod.eval_years_frame(payload)
    overall = mod.eval_overall_frame(payload)
    per_h = mod.eval_per_horizon_frame(payload)
    summary = mod.summarize_eval_diagnostics(payload)

    assert list(years["year"].astype(int)) == [
        2020,
        2021,
        2022,
    ]
    assert list(per_h["horizon"].astype(int)) == [1, 2, 3]
    assert overall.loc[0, "n_horizons"] == 3
    assert overall.loc[0, "n_year_blocks"] == 3
    assert summary["brief"]["kind"] == "eval_diagnostics"
    assert summary["checks"]["has_overall_block"] is True
    assert summary["checks"]["has_year_blocks"] is True
    assert (
        summary["checks"]["overall_has_per_horizon_mae"]
        is True
    )
    assert (
        summary["checks"]["horizon_count_matches_year_count"]
        is True
    )


def test_generate_load_and_inspect_eval_diagnostics(
    tmp_path: Path, mod
):
    out_json = tmp_path / "eval_diagnostics.json"
    out_dir = tmp_path / "inspect"

    payload = mod.generate_eval_diagnostics(
        output_path=out_json,
        overrides={
            "__overall__": {
                "overall_mae": 12.5,
                "per_horizon_mae": {
                    "1": 5.0,
                    "2": 10.0,
                    "3": 15.0,
                },
            }
        },
    )

    assert out_json.exists()
    assert payload == out_json

    record = mod.load_eval_diagnostics(out_json)
    assert record.kind == "eval_diagnostics"

    bundle = mod.inspect_eval_diagnostics(
        out_json,
        output_dir=out_dir,
        save_figures=True,
    )

    assert set(bundle["frames"]) == {
        "years",
        "overall",
        "per_horizon",
    }
    assert bundle["summary"]["brief"]["n_year_blocks"] >= 1
    assert len(bundle["figure_paths"]) == 4
    for path in bundle["figure_paths"].values():
        assert Path(path).exists()


def test_plot_helpers_return_axes(mod):
    payload = mod.default_eval_diagnostics_payload()
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.ravel()

    out_axes = [
        mod.plot_eval_overall_metrics(payload, ax=axes[0]),
        mod.plot_eval_year_metric_trend(payload, ax=axes[1]),
        mod.plot_eval_per_horizon_metrics(
            payload, ax=axes[2]
        ),
        mod.plot_eval_boolean_summary(payload, ax=axes[3]),
    ]

    assert all(hasattr(ax, "plot") for ax in out_axes)
    plt.close(fig)
