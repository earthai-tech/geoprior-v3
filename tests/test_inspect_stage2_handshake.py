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


def test_stage2_handshake_public_reexports_exist():
    inspect_pkg = importlib.import_module(
        "geoprior.utils.inspect"
    )
    root_pkg = importlib.import_module("geoprior.utils")

    assert hasattr(
        inspect_pkg, "default_stage2_handshake_payload"
    )
    assert hasattr(inspect_pkg, "inspect_stage2_handshake")
    assert hasattr(
        root_pkg, "default_stage2_handshake_payload"
    )
    assert hasattr(root_pkg, "inspect_stage2_handshake")


def test_default_stage2_handshake_summary_and_frames():
    mod = _import_target("stage2_handshake")

    payload = mod.default_stage2_handshake_payload()
    summary = mod.summarize_stage2_handshake(payload)

    assert summary["brief"]["kind"] == "stage2_handshake"
    assert summary["expected"]["mode"] == "tft_like"
    assert summary["checks"]["coords_normalized"] is True
    assert summary["checks"]["has_scaling_summary"] is True

    layout = mod.stage2_layout_frame(payload)
    finite = mod.stage2_finite_frame(payload)
    coord_norm = mod.stage2_coord_stats_frame(
        payload, section="coord_stats_norm"
    )
    coord_raw = mod.stage2_coord_stats_frame(
        payload, section="coord_stats_raw"
    )
    coord_ranges = mod.stage2_coord_range_frame(payload)
    scaling = mod.stage2_scaling_frame(payload)

    assert "coords.shape" in set(layout["key"])
    assert not finite.empty
    assert set(coord_norm["coord"]) == {"t", "x", "y"}
    assert set(coord_raw["coord"]) == {"t", "x", "y"}
    assert set(coord_ranges["coord"]) == {"t", "x", "y"}
    assert "coords_normalized" in set(scaling["key"])


def test_stage2_handshake_generate_load_and_inspect(
    tmp_path: Path,
):
    mod = _import_target("stage2_handshake")

    out = tmp_path / "stage2_handshake.json"
    written = mod.generate_stage2_handshake(
        output_path=out,
        overrides={
            "city": "nansha",
            "got": {"N_train": 1500, "N_val": 450},
        },
    )
    assert Path(written).exists()

    record = mod.load_stage2_handshake(out)
    assert record.kind == "stage2_handshake"
    assert record.payload["city"] == "nansha"

    fig_dir = tmp_path / "figs"
    bundle = mod.inspect_stage2_handshake(
        record, output_dir=fig_dir
    )
    assert set(bundle["frames"]) == {
        "layout",
        "finite",
        "coord_stats_norm",
        "coord_stats_raw",
        "coord_ranges",
        "scaling_summary",
    }
    assert len(bundle["figure_paths"]) == 7
    for path in bundle["figure_paths"].values():
        assert Path(path).exists()


def test_stage2_handshake_plot_functions_smoke():
    mod = _import_target("stage2_handshake")
    payload = mod.default_stage2_handshake_payload()

    calls = [
        lambda ax: mod.plot_stage2_sample_sizes(
            payload, ax=ax
        ),
        lambda ax: mod.plot_stage2_finite_ratios(
            payload, ax=ax
        ),
        lambda ax: mod.plot_stage2_coord_stats(
            payload,
            section="coord_stats_norm",
            stat="mean",
            ax=ax,
        ),
        lambda ax: mod.plot_stage2_coord_range_errors(
            payload, ax=ax
        ),
        lambda ax: mod.plot_stage2_scaling_summary(
            payload, ax=ax
        ),
        lambda ax: mod.plot_stage2_boolean_summary(
            payload, ax=ax
        ),
    ]

    for call in calls:
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        out_ax = call(ax)
        assert out_ax is ax
        plt.close(fig)
