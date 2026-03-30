from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.script_artifacts,
    pytest.mark.fast_plots,
]


def _write_payload(path: Path, *, shift: float) -> Path:
    n = 64
    tau = np.linspace(8.0 + shift, 80.0 + shift, n)
    tau_prior = np.linspace(10.0 + shift, 70.0 + shift, n)
    K = np.linspace(1.0e-6, 6.0e-6, n)
    Ss = np.linspace(1.0e-5, 5.0e-5, n)
    Hd = np.linspace(8.0, 20.0, n)
    cons = np.linspace(-2.0e-5, 2.0e-5, n)

    np.savez(
        path,
        tau=tau,
        tau_prior=tau_prior,
        K=K,
        Ss=Ss,
        Hd=Hd,
        cons_res_vals=cons,
    )
    meta = {
        "units": {
            "cons_res_vals": "m/s",
        },
        "tau_prior_human_name": "tau_prior",
    }
    path.with_suffix(".npz.meta.json").write_text(
        json.dumps(meta),
        encoding="utf-8",
    )
    return path


def test_plot_physics_sanity_main_writes_outputs(
    tmp_path: Path,
    script_test_env,
    fast_script_figures,
    collect_script_outputs,
):
    from geoprior._scripts.plot_physics_sanity import (
        plot_physics_sanity_main,
    )

    ns = _write_payload(tmp_path / "nansha_payload.npz", shift=0.0)
    zh = _write_payload(
        tmp_path / "zhongshan_payload.npz",
        shift=5.0,
    )

    plot_physics_sanity_main(
        [
            "--src",
            str(ns),
            "--src",
            str(zh),
            "--city",
            "Nansha",
            "--city",
            "Zhongshan",
            "--plot-mode",
            "joint",
            "--tau-scale",
            "log10",
            "--show-legend",
            "false",
            "--out",
            "physics_sanity_case",
            "--out-json",
            "physics_sanity_case.json",
        ]
    )

    pngs = collect_script_outputs("physics_sanity_case.png")
    svgs = collect_script_outputs("physics_sanity_case.svg")
    jsons = collect_script_outputs("physics_sanity_case.json")

    assert pngs, "Expected a PNG artifact."
    assert svgs, "Expected an SVG artifact."
    assert jsons, "Expected a summary JSON artifact."

    stats = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert set(stats) == {"Nansha", "Zhongshan"}
    assert "r2_logtau" in stats["Nansha"]
    assert "eps_cons_rms" in stats["Zhongshan"]


def test_eps_prior_ln_is_zero_for_matching_arrays():
    from geoprior._scripts.plot_physics_sanity import _eps_prior_ln

    tau = np.array([1.0, 2.0, 4.0])
    tp = np.array([1.0, 2.0, 4.0])

    assert _eps_prior_ln(tau, tp) == pytest.approx(0.0)
