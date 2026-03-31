from __future__ import annotations

import json

import numpy as np
import pytest


def _write_payload(
    tmp_path, *, stem: str = "physics_fields_case"
):
    n = 81
    gx = np.repeat(np.linspace(113.2, 113.4, 9), 9)
    gy = np.tile(np.linspace(22.0, 22.2, 9), 9)

    payload_path = tmp_path / f"{stem}.npz"
    np.savez(
        payload_path,
        lon=gx,
        lat=gy,
        K=np.linspace(1e-6, 8e-6, n),
        Ss=np.linspace(1e-5, 6e-5, n),
        Hd=np.linspace(6.0, 18.0, n),
        tau=np.linspace(8.0, 80.0, n),
        tau_prior=np.linspace(10.0, 70.0, n),
        censored=np.where(np.arange(n) % 7 == 0, 1.0, 0.0),
    )

    meta = {
        "units": {
            "log10_K": "log10(m/s)",
            "log10_Ss": "log10(1/m)",
            "Hd": "m",
            "log10_tau": "log10(s)",
        },
        "tau_prior_human_name": "tau_closure",
    }
    payload_path.with_suffix(".npz.meta.json").write_text(
        json.dumps(meta),
        encoding="utf-8",
    )
    return payload_path


@pytest.mark.script_artifacts
@pytest.mark.fast_plots
def test_plot_physics_fields_main_writes_outputs(
    tmp_path,
    script_test_env,
    fast_script_figures,
    collect_script_outputs,
):
    from geoprior.scripts.plot_physics_fields import (
        plot_physics_fields_main,
    )

    payload = _write_payload(tmp_path)

    plot_physics_fields_main(
        [
            "--payload",
            str(payload),
            "--city",
            "Zhongshan",
            "--coord-kind",
            "lonlat",
            "--render",
            "grid",
            "--grid",
            "30",
            "--export-pdf",
            "true",
            "--show-legend",
            "false",
            "--out",
            "physics_fields_case",
            "--out-json",
            "physics_fields_case.json",
        ]
    )

    pngs = collect_script_outputs("physics_fields_case.png")
    svgs = collect_script_outputs("physics_fields_case.svg")
    pdfs = collect_script_outputs("physics_fields_case.pdf")
    jsons = collect_script_outputs("physics_fields_case.json")

    assert pngs, "Expected a PNG artifact."
    assert svgs, "Expected an SVG artifact."
    assert pdfs, "Expected a PDF artifact."
    assert jsons, "Expected a JSON sidecar."

    record = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert record["city"] == "Zhongshan"
    assert record["payload_keys"]
    assert len(record["figures"]) == 3
