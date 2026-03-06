from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests._helpers import DummyModel, import_module_group

mod = import_module_group("payloads")


def _payload():
    return {
        "tau": np.array([2.0, 4.0, 8.0], dtype=float),
        "tau_prior": np.array([2.0, 4.0, 8.0], dtype=float),
        "tau_closure": np.array([2.0, 4.0, 8.0], dtype=float),
        "K": np.array([1e-5, 1e-5, 1e-5], dtype=float),
        "Ss": np.array([1e-4, 1e-4, 1e-4], dtype=float),
        "Hd": np.array([10.0, 10.0, 10.0], dtype=float),
        "cons_res_vals": np.array([0.0, 0.0, 0.0], dtype=float),
        "log10_tau": np.log10(np.array([2.0, 4.0, 8.0], dtype=float)),
        "log10_tau_prior": np.log10(np.array([2.0, 4.0, 8.0], dtype=float)),
        "metrics": {"eps_prior_rms": 0.0},
    }


def test_maybe_subsample_keeps_row_aligned_arrays(monkeypatch):
    payload = {
        "tau": np.arange(5),
        "K": np.arange(5) + 10,
        "meta": {"x": 1},
        "matrix": np.arange(6).reshape(3, 2),
    }
    monkeypatch.setattr(mod.np.random, "choice", lambda n, size, replace: np.array([1, 3]))

    out = mod._maybe_subsample(payload, 0.4)

    assert out["tau"].tolist() == [1, 3]
    assert out["K"].tolist() == [11, 13]
    assert out["meta"] == {"x": 1}
    assert out["matrix"].shape == (3, 2)


def test_default_meta_from_model_uses_aliases_and_formula():
    model = DummyModel(
        scaling_kwargs={"time_units": "year"},
        use_effective_thickness=True,
        Hd_factor=0.5,
        lambda_offsets=2.0,
        pde_modes_active=["both"],
        kappa_mode="kb",
        kappa_b=1.25,
    )

    meta = mod.default_meta_from_model(model)

    assert meta["time_units"] == "year"
    assert meta["use_effective_h"] is True
    assert meta["Hd_factor"] == 0.5
    assert meta["lambda_offsets"] == 2.0
    assert meta["kappa_value"] == 1.25
    assert "Hd^2 * Ss" in meta["tau_closure_formula"]


def test_identifiability_diagnostics_from_payload_returns_expected_blocks():
    payload = _payload()
    out = mod.identifiability_diagnostics_from_payload(
        payload,
        tau_true=4.0,
        K_true=1e-5,
        Ss_true=1e-4,
        Hd_true=10.0,
        K_prior=1e-5,
        Ss_prior=1e-4,
        Hd_prior=10.0,
    )

    assert set(out) == {"tau_rel_error", "closure_log_resid", "offsets"}
    assert out["tau_rel_error"]["q50"] >= 0.0
    assert abs(out["closure_log_resid"]["q50"]) < 1e-12
    assert abs(out["offsets"]["vs_true"]["delta_K"]["q50"]) < 1e-12


def test_save_and_load_physics_payload_roundtrip_npz(tmp_path: Path):
    payload = _payload()
    meta = {"city": "nansha"}
    path = tmp_path / "physics_payload.npz"

    saved = mod.save_physics_payload(payload, meta, str(path), format="npz", overwrite=True)
    loaded, meta2 = mod.load_physics_payload(saved)

    assert Path(saved).exists()
    assert meta2["city"] == "nansha"
    assert "tau_closure" in loaded
    np.testing.assert_allclose(loaded["tau_prior"], payload["tau_prior"])
    np.testing.assert_allclose(loaded["tau_closure"], payload["tau_prior"])


def test_save_and_load_physics_payload_roundtrip_csv(tmp_path: Path):
    payload = _payload()
    meta = {"city": "zhongshan"}
    path = tmp_path / "physics_payload.csv"

    saved = mod.save_physics_payload(payload, meta, str(path), format="csv", overwrite=True)
    loaded, meta2 = mod.load_physics_payload(saved)

    assert meta2["city"] == "zhongshan"
    np.testing.assert_allclose(loaded["tau_prior"], payload["tau_prior"])
    np.testing.assert_allclose(loaded["tau_closure"], payload["tau_prior"])


def test_load_physics_payload_adds_compat_aliases(tmp_path: Path):
    path = tmp_path / "legacy.csv"
    rows = {
        "tau": [1.0, 2.0],
        "tau_closure": [1.5, 2.5],
        "K": [1e-5, 1e-5],
        "Ss": [1e-4, 1e-4],
        "Hd": [10.0, 10.0],
        "cons_res_vals": [0.0, 0.0],
        "log10_tau": [0.0, 0.3],
        "log10_tau_closure": [0.1, 0.4],
    }
    import pandas as pd

    pd.DataFrame(rows).to_csv(path, index=False)
    (tmp_path / "legacy.csv.meta.json").write_text(
        json.dumps({"ok": True}),
        encoding="utf-8",
    )

    loaded, meta = mod.load_physics_payload(str(path))

    assert meta["ok"] is True
    assert "tau_prior" in loaded
    assert "log10_tau_prior" in loaded
    np.testing.assert_allclose(loaded["tau_prior"], loaded["tau_closure"])
