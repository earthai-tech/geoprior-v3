from __future__ import annotations

import json

import numpy as np
import pytest


def _write_payload(tmp_path, *, stem: str = "payload_case"):
    n = 64
    x = np.linspace(1000.0, 1200.0, n)
    y = np.linspace(2000.0, 2200.0, n)

    payload_path = tmp_path / f"{stem}.npz"
    np.savez(
        payload_path,
        x=x,
        y=y,
        K=np.linspace(1e-6, 5e-6, n),
        Ss=np.linspace(1e-5, 3e-5, n),
        Hd=np.linspace(8.0, 24.0, n),
        tau=np.linspace(10.0, 50.0, n),
        tau_prior=np.linspace(12.0, 55.0, n),
        censored=np.zeros(n, dtype=float),
    )

    meta = {
        "units": {
            "log10_K": "log10(m/s)",
            "log10_Ss": "log10(1/m)",
            "Hd": "m",
            "log10_tau": "log10(s)",
        },
        "tau_prior_human_name": "tau_prior",
    }
    payload_path.with_suffix(".npz.meta.json").write_text(
        json.dumps(meta),
        encoding="utf-8",
    )
    return payload_path


@pytest.mark.script_artifacts
@pytest.mark.fast_plots
def test_plot_physics_maps_main_writes_outputs(
    tmp_path,
    script_test_env,
    fast_script_figures,
    collect_script_outputs,
):
    from geoprior.scripts.plot_physics_maps import (
        plot_physics_maps_main,
    )

    payload = _write_payload(tmp_path)

    plot_physics_maps_main(
        [
            "--payload",
            str(payload),
            "--city",
            "Nansha",
            "--coord-kind",
            "xy",
            "--grid",
            "24",
            "--show-legend",
            "false",
            "--out",
            "physics_maps_case",
            "--out-json",
            "physics_maps_case.json",
        ]
    )

    pngs = collect_script_outputs("physics_maps_case.png")
    svgs = collect_script_outputs("physics_maps_case.svg")
    jsons = collect_script_outputs("physics_maps_case.json")

    assert pngs, "Expected a PNG artifact."
    assert svgs, "Expected an SVG artifact."
    assert jsons, "Expected a JSON sidecar."

    record = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert record["city"] == "Nansha"
    assert "ranges" in record
    assert set(record["ranges"]) >= {
        "log10_K",
        "log10_Ss",
        "Hd",
        "log10_tau",
        "log10_tau_p",
        "delta_log10_tau",
    }
