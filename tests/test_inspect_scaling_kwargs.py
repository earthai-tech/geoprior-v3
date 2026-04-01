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


def test_scaling_kwargs_public_reexports_exist():
    inspect_pkg = importlib.import_module(
        "geoprior.utils.inspect"
    )
    root_pkg = importlib.import_module("geoprior.utils")

    assert hasattr(
        inspect_pkg, "default_scaling_kwargs_payload"
    )
    assert hasattr(inspect_pkg, "inspect_scaling_kwargs")
    assert hasattr(root_pkg, "default_scaling_kwargs_payload")
    assert hasattr(root_pkg, "inspect_scaling_kwargs")


def test_default_scaling_kwargs_summary_and_frames():
    mod = _import_target("scaling_kwargs")

    payload = mod.default_scaling_kwargs_payload()
    summary = mod.summarize_scaling_kwargs(payload)

    assert summary["time_units"] == "year"
    assert summary["coords_normalized"] is True
    assert summary["coords_in_degrees"] is False
    assert summary["dynamic_features"] == 5
    assert summary["future_features"] == 1
    assert summary["static_features"] == 12
    assert summary["has_bounds"] is True
    assert "subs_scale_si" in summary["affine_maps"]

    affine = mod.scaling_kwargs_affine_frame(payload)
    coords = mod.scaling_kwargs_coord_frame(payload)
    bounds = mod.scaling_kwargs_bounds_frame(payload)
    features = mod.scaling_kwargs_feature_channels_frame(
        payload
    )
    schedule = mod.scaling_kwargs_schedule_frame(payload)

    assert isinstance(affine, pd.DataFrame)
    assert set(affine["parameter"]) >= {
        "subs_scale_si",
        "head_scale_si",
        "H_scale_si",
    }
    assert set(coords["section"]) >= {
        "coord_meta",
        "coord_ranges",
    }
    assert set(
        coords.loc[
            coords["section"].eq("coord_ranges"), "name"
        ]
    ) == {"t", "x", "y"}
    assert "H_min" in set(bounds["bound"])
    assert "dynamic_feature_names" in set(features["name"])
    assert "gwl_dyn_index" in set(features["name"])
    assert "mv_weight" in set(schedule["name"])


def test_generate_load_and_inspect_scaling_kwargs(
    tmp_path: Path,
):
    mod = _import_target("scaling_kwargs")

    out = tmp_path / "scaling_kwargs.json"
    written = mod.generate_scaling_kwargs(
        out,
        overrides={
            "time_units": "month",
            "coord_ranges": {"t": 9.0, "x": 10.0, "y": 11.0},
        },
    )

    assert Path(written).exists()

    record = mod.load_scaling_kwargs(out)
    assert record.kind == "scaling_kwargs"
    assert record.payload["time_units"] == "month"
    assert record.payload["coord_ranges"]["t"] == 9.0

    bundle = mod.inspect_scaling_kwargs(record)
    assert set(bundle) == {
        "summary",
        "affine",
        "coords",
        "bounds",
        "features",
        "schedule",
    }
    assert bundle["summary"]["time_units"] == "month"


def test_scaling_kwargs_plot_functions_smoke():
    mod = _import_target("scaling_kwargs")
    payload = mod.default_scaling_kwargs_payload()

    plotters = [
        mod.plot_scaling_kwargs_affine_maps,
        mod.plot_scaling_kwargs_coord_ranges,
        mod.plot_scaling_kwargs_bounds,
        mod.plot_scaling_kwargs_schedule_scalars,
        mod.plot_scaling_kwargs_feature_group_sizes,
        mod.plot_scaling_kwargs_boolean_summary,
    ]

    for fn in plotters:
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        out_ax = fn(payload, ax=ax)
        assert out_ax is ax
        plt.close(fig)
