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


def test_physics_payload_meta_payload_frames_and_summary():
    mod = _import_target("physics_payload_meta")
    payload = mod.default_physics_payload_meta_payload()

    summary = mod.summarize_physics_payload_meta(payload)
    assert summary["brief"]["kind"] == "physics_payload_meta"
    assert summary["brief"]["n_pde_modes"] >= 1
    assert summary["conventions"]["time_units"] == "year"
    assert summary["closure"]["tau_prior_definition"]
    assert summary["checks"]["has_units"] is True

    ident = mod.physics_payload_meta_identity_frame(payload)
    closure = mod.physics_payload_meta_closure_frame(payload)
    units = mod.physics_payload_meta_units_frame(payload)
    metrics = mod.physics_payload_meta_metrics_frame(payload)

    assert "model_name" in set(ident["field"])
    assert "tau_closure_formula" in set(closure["field"])
    assert "K" in set(units["quantity"])
    assert not metrics.empty


def test_physics_payload_meta_roundtrip_and_inspect_bundle(
    tmp_path: Path,
):
    mod = _import_target("physics_payload_meta")
    pkg = importlib.import_module("geoprior.utils.inspect")

    out = tmp_path / "physics_payload_meta.json"
    mod.generate_physics_payload_meta(
        out,
        overrides={
            "city": "nansha",
            "payload_metrics": {
                "eps_prior_rms": 1.23e-4,
                "eps_cons_scaled_rms": 4.56e-6,
            },
        },
    )

    record = mod.load_physics_payload_meta(out)
    assert record.payload["city"] == "nansha"
    assert record.payload["payload_metrics"][
        "eps_prior_rms"
    ] == pytest.approx(1.23e-4)

    bundle = mod.inspect_physics_payload_meta(record)
    assert bundle["summary"]["brief"]["city"] == "nansha"
    assert not bundle["identity_frame"].empty
    assert not bundle["closure_frame"].empty
    assert not bundle["units_frame"].empty
    assert not bundle["metrics_frame"].empty

    assert (
        pkg.default_physics_payload_meta_payload
        is mod.default_physics_payload_meta_payload
    )
    assert (
        pkg.inspect_physics_payload_meta
        is mod.inspect_physics_payload_meta
    )


def test_physics_payload_meta_plotting_smoke():
    mod = _import_target("physics_payload_meta")
    payload = mod.default_physics_payload_meta_payload()

    plotters = [
        mod.plot_physics_payload_meta_core_scalars,
        mod.plot_physics_payload_meta_payload_metrics,
        mod.plot_physics_payload_meta_boolean_summary,
    ]

    for fn in plotters:
        fig, ax = plt.subplots(figsize=(6, 4))
        out_ax = fn(payload, ax=ax)
        assert out_ax is ax
        assert ax.get_title()
        plt.close(fig)
