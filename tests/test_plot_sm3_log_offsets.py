from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [
    pytest.mark.script_artifacts,
    pytest.mark.fast_plots,
]


def _write_payload(path: Path) -> Path:
    n = 50
    np.savez(
        path,
        tau=np.linspace(8.0, 60.0, n),
        tau_prior=np.linspace(10.0, 55.0, n),
        K=np.linspace(1.0e-6, 5.0e-6, n),
        Ss=np.linspace(1.0e-5, 4.0e-5, n),
        Hd=np.linspace(10.0, 25.0, n),
    )
    path.with_suffix(".npz.meta.json").write_text(
        json.dumps({"source": "test"}),
        encoding="utf-8",
    )
    return path


def test_plot_sm3_log_offsets_main_writes_outputs(
    tmp_path: Path,
    script_test_env,
    fast_script_figures,
    collect_script_outputs,
):
    from geoprior._scripts.plot_sm3_log_offsets import (
        plot_sm3_log_offsets_main,
    )

    payload = _write_payload(
        tmp_path / "sm3_offsets_payload.npz"
    )

    plot_sm3_log_offsets_main(
        [
            "--payload",
            str(payload),
            "--K-prior",
            "1e-6",
            "--Ss-prior",
            "1e-5",
            "--Hd-prior",
            "10.0",
            "--out",
            "sm3_offsets_case",
            "--out-raw-csv",
            "sm3_offsets_case_raw.csv",
            "--out-summary-csv",
            "sm3_offsets_case_summary.csv",
            "--out-json",
            "sm3_offsets_case.json",
        ]
    )

    raw_csv = collect_script_outputs(
        "sm3_offsets_case_raw.csv"
    )
    sum_csv = collect_script_outputs(
        "sm3_offsets_case_summary.csv"
    )
    jsons = collect_script_outputs("sm3_offsets_case.json")
    hist_png = collect_script_outputs(
        "sm3_offsets_case-hists.png"
    )
    tau_png = collect_script_outputs(
        "sm3_offsets_case-tau-scatter.png"
    )

    assert raw_csv, "Expected the raw CSV export."
    assert sum_csv, "Expected the summary CSV export."
    assert jsons, "Expected the JSON export."
    assert hist_png, "Expected the histogram PNG artifact."
    assert tau_png, "Expected the tau-scatter PNG artifact."

    raw = pd.read_csv(raw_csv[0])
    assert "delta_log_tau" in raw.columns
    assert "delta_logK" in raw.columns


def test_summarise_offsets_requires_delta_columns():
    from geoprior._scripts.plot_sm3_log_offsets import (
        summarise_offsets,
    )

    with pytest.raises(RuntimeError):
        summarise_offsets(pd.DataFrame({"x": [1, 2, 3]}))
