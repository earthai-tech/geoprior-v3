from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

MODULE_CANDIDATES = [
    "geoprior.models.subsidence.payloads",
]


def _import_target():
    last = None
    for name in MODULE_CANDIDATES:
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover
            last = exc
    raise last  # type: ignore[misc]


@pytest.fixture(scope="module")
def mod():
    return _import_target()


class _DummyModel:
    def __init__(self, *, kappa_mode="kb", kappa_value=2.0):
        self.scaling_kwargs = {"time_units": "year"}
        self.use_effective_h = True
        self.hd_factor = 0.6
        self.lambda_offset = 1.25
        self.kappa_mode = kappa_mode
        self.kappa_value = kappa_value
        self.pde_modes_active = ["consolidation", "gw_flow"]

    def evaluate_physics(self, inputs, return_maps=True):
        n = len(inputs["tau"])
        return {
            "tau": np.asarray(inputs["tau"], dtype=float),
            "tau_prior": np.asarray(
                inputs["tau_prior"], dtype=float
            ),
            "K": np.asarray(inputs["K"], dtype=float),
            "Ss": np.asarray(inputs["Ss"], dtype=float),
            "Hd": np.asarray(inputs["Hd"], dtype=float),
            "H": np.asarray(inputs["H"], dtype=float),
            "R_cons": np.asarray(
                inputs["R_cons"], dtype=float
            ),
            "R_cons_scaled": np.full(n, 0.25, dtype=float),
            "cons_scale": np.full(n, 2.0, dtype=float),
        }


def _base_payload():
    tau = np.array([10.0, 20.0, 40.0], dtype=np.float32)
    tau_prior = np.array([11.0, 19.0, 41.0], dtype=np.float32)
    return {
        "tau": tau,
        "tau_prior": tau_prior,
        "K": np.array([1e-5, 2e-5, 3e-5], dtype=np.float32),
        "Ss": np.array(
            [1e-4, 1.2e-4, 1.4e-4], dtype=np.float32
        ),
        "Hd": np.array([10.0, 10.0, 10.0], dtype=np.float32),
        "H": np.array([12.0, 12.0, 12.0], dtype=np.float32),
        "cons_res_vals": np.array(
            [0.1, -0.2, 0.3], dtype=np.float32
        ),
        "log10_tau": np.log10(tau).astype(np.float32),
        "log10_tau_prior": np.log10(tau_prior).astype(
            np.float32
        ),
        "metrics": {"eps_prior_rms": 0.1},
        "K_from_tau": np.array(
            [9.0, 8.0, 7.0], dtype=np.float32
        ),
        "K_from_tau_m_per_year": np.array(
            [1.0, 2.0, 3.0], dtype=np.float32
        ),
        "log10_K_from_tau_m_per_year": np.array(
            [0.0, 0.3, 0.48], dtype=np.float32
        ),
        "cons_res_scaled": np.array(
            [0.01, 0.02, 0.03], dtype=np.float32
        ),
        "cons_scale": np.array(
            [2.0, 2.0, 2.0], dtype=np.float32
        ),
    }


def test_maybe_subsample_validates_fraction_and_preserves_non_row_arrays(
    mod,
):
    payload = {
        "tau": np.arange(10, dtype=np.float32),
        "K": np.arange(10, dtype=np.float32) + 1,
        "grid": np.arange(6, dtype=np.float32).reshape(2, 3),
    }

    with pytest.raises(ValueError):
        mod._maybe_subsample(payload, 0.0)

    out = mod._maybe_subsample(payload, 0.4)
    assert 1 <= out["tau"].shape[0] <= 10
    assert out["tau"].shape[0] == out["K"].shape[0]
    np.testing.assert_array_equal(
        out["grid"], payload["grid"]
    )


def test_default_meta_from_model_switches_tau_formula_with_kappa_mode(
    mod,
):
    meta_bar = mod.default_meta_from_model(
        _DummyModel(kappa_mode="bar", kappa_value=3.0)
    )
    meta_kb = mod.default_meta_from_model(
        _DummyModel(kappa_mode="kb", kappa_value=4.0)
    )

    assert meta_bar["kappa_value"] == pytest.approx(3.0)
    assert (
        "kappa_bar * H^2 * Ss"
        in meta_bar["tau_closure_formula"]
    )
    assert meta_kb["kappa_value"] == pytest.approx(4.0)
    assert (
        "Hd^2 * Ss / (pi^2 * kappa_b * K)"
        == meta_kb["tau_closure_formula"]
    )


def test_compute_identifiability_summary_uses_pi_squared_closure(
    mod,
):
    eff = {"tau": 8.0, "K": 2.0, "Ss": 3.0, "Hd": 4.0}
    true = {"tau": 10.0, "K": 1.5, "Ss": 2.5, "Hd": 3.5}
    prior = {"K": 1.0, "Ss": 2.0, "Hd": 3.0}

    out = mod.compute_identifiability_summary(
        eff, true, prior, kappa_b=2.0
    )

    expected_closure = (4.0**2) * 3.0 / (np.pi**2 * 2.0 * 2.0)
    expected_log_resid = np.log(expected_closure) - np.log(
        10.0
    )
    assert out["rel_err_tau"] == pytest.approx(0.2)
    assert out["log_closure_resid"] == pytest.approx(
        expected_log_resid
    )
    assert out["delta_Hd_prior"] == pytest.approx(
        np.log(4.0) - np.log(3.0)
    )


def test_identifiability_diagnostics_from_payload_returns_quantile_stats(
    mod,
):
    payload = {
        "tau": np.array([9.0, 10.0, 11.0], dtype=float),
        "tau_prior": np.array([8.0, 10.0, 12.0], dtype=float),
        "K": np.array([1.0, 1.1, 1.2], dtype=float),
        "Ss": np.array([2.0, 2.1, 2.2], dtype=float),
        "Hd": np.array([3.0, 3.1, 3.2], dtype=float),
    }

    out = mod.identifiability_diagnostics_from_payload(
        payload,
        tau_true=10.0,
        K_true=1.0,
        Ss_true=2.0,
        Hd_true=3.0,
        K_prior=0.9,
        Ss_prior=1.9,
        Hd_prior=2.9,
    )

    assert "q50" in out["tau_rel_error"]
    assert "q95" in out["closure_log_resid"]
    assert set(out["offsets"]) == {"vs_true", "vs_prior"}
    assert set(out["offsets"]["vs_true"]) == {
        "delta_K",
        "delta_Ss",
        "delta_Hd",
    }


def test_gather_and_save_load_payload_roundtrip_with_optional_fields(
    mod, tmp_path: Path
):
    dataset = [
        {
            "tau": np.array([10.0, 20.0]),
            "tau_prior": np.array([11.0, 19.0]),
            "K": np.array([1e-5, 2e-5]),
            "Ss": np.array([1e-4, 1.2e-4]),
            "Hd": np.array([10.0, 10.0]),
            "H": np.array([12.0, 12.0]),
            "R_cons": np.array([0.1, -0.1]),
        },
        (
            {
                "tau": np.array([30.0]),
                "tau_prior": np.array([31.0]),
                "K": np.array([3e-5]),
                "Ss": np.array([1.4e-4]),
                "Hd": np.array([10.0]),
                "H": np.array([12.0]),
                "R_cons": np.array([0.2]),
            },
            None,
        ),
    ]
    model = _DummyModel(kappa_mode="kb", kappa_value=2.0)

    payload = mod.gather_physics_payload(
        model, dataset, max_batches=2
    )
    assert "tau_closure" in payload
    assert "K_from_tau" in payload
    assert payload["metrics"][
        "kappa_used_for_closure"
    ] == pytest.approx(2.0)
    assert payload["metrics"][
        "eps_cons_scaled_rms"
    ] == pytest.approx(0.25)

    save_path = tmp_path / "physics_payload.npz"
    meta = mod.default_meta_from_model(model)
    written = mod.save_physics_payload(
        payload, meta, str(save_path), overwrite=True
    )
    loaded, loaded_meta = mod.load_physics_payload(written)

    assert Path(written).exists()
    np.testing.assert_allclose(loaded["tau"], payload["tau"])
    np.testing.assert_allclose(
        loaded["tau_prior"], payload["tau_prior"]
    )
    assert loaded_meta["payload_metrics"][
        "kappa_used_for_closure"
    ] == pytest.approx(2.0)
    assert (
        loaded_meta["tau_prior_definition"]
        == "tau_closure_from_learned_fields"
    )


def test_save_and_load_csv_aliases_tau_closure(
    mod, tmp_path: Path
):
    payload = _base_payload()
    meta = {"note": "csv-roundtrip"}
    path = tmp_path / "payload.csv"
    mod.save_physics_payload(
        payload, meta, str(path), format="csv", overwrite=True
    )
    loaded, loaded_meta = mod.load_physics_payload(str(path))

    assert "tau_prior" in loaded and "tau_closure" in loaded
    np.testing.assert_allclose(
        loaded["tau_closure"], loaded["tau_prior"]
    )
    assert loaded_meta["note"] == "csv-roundtrip"
