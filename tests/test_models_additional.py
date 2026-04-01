from __future__ import annotations

import importlib

import numpy as np
import pytest

MODULE_CANDIDATES = [
    "geoprior.models.subsidence.models",
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


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def test_targets_for_loss_is_strict_by_default_and_warns_only_once(
    monkeypatch, mod
):
    model = object.__new__(mod.GeoPriorSubsNet)
    model.output_names = ["subs_pred", "gwl_pred"]
    model.scaling_kwargs = {}

    y_pred = {
        "subs_pred": mod.tf_constant(
            [[1.0]], dtype=mod.tf_float32
        ),
        "gwl_pred": mod.tf_constant(
            [[2.0]], dtype=mod.tf_float32
        ),
    }

    with pytest.raises(KeyError):
        model._targets_for_loss({"subs_pred": 1.0}, y_pred)

    model.scaling_kwargs = {"allow_missing_targets": True}
    model._warned_missing_targets = False
    calls = []
    monkeypatch.setattr(
        mod.logger,
        "warning",
        lambda *a, **k: calls.append((a, k)),
    )

    first = model._targets_for_loss(
        {"subs_pred": 1.0}, y_pred
    )
    second = model._targets_for_loss(
        {"subs_pred": 1.0}, y_pred
    )

    assert first["subs_pred"] == 1.0
    np.testing.assert_allclose(
        _to_numpy(first["gwl_pred"]),
        _to_numpy(y_pred["gwl_pred"]),
    )
    np.testing.assert_allclose(
        _to_numpy(second["gwl_pred"]),
        _to_numpy(y_pred["gwl_pred"]),
    )
    assert len(calls) == 1


def test_split_data_predictions_slices_last_axis(mod):
    model = object.__new__(mod.GeoPriorSubsNet)
    model.output_subsidence_dim = 2
    model.output_gwl_dim = 1

    data = mod.tf_constant(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
        dtype=mod.tf_float32,
    )
    subs, gwl = model.split_data_predictions(data)

    np.testing.assert_allclose(
        _to_numpy(subs), [[[1.0, 2.0], [4.0, 5.0]]]
    )
    np.testing.assert_allclose(
        _to_numpy(gwl), [[[3.0], [6.0]]]
    )


def test_split_physics_predictions_handles_disabled_or_missing_q(
    mod,
):
    model = object.__new__(mod.GeoPriorSubsNet)
    model.output_K_dim = 1
    model.output_Ss_dim = 1
    model.output_tau_dim = 1
    model.output_Q_dim = 0

    phys = mod.tf_constant(
        np.arange(12).reshape(2, 2, 3), dtype=mod.tf_float32
    )
    K, Ss, dlogtau, Q = model.split_physics_predictions(phys)
    assert _to_numpy(Q).shape == (2, 2, 1)
    np.testing.assert_allclose(_to_numpy(Q), 0.0)

    model.output_Q_dim = 1
    K2, Ss2, dlogtau2, Q2 = model.split_physics_predictions(
        phys
    )
    assert _to_numpy(Q2).shape == (2, 2, 1)
    np.testing.assert_allclose(_to_numpy(Q2), 0.0)
    np.testing.assert_allclose(_to_numpy(K2), _to_numpy(K))
    np.testing.assert_allclose(_to_numpy(Ss2), _to_numpy(Ss))
    np.testing.assert_allclose(
        _to_numpy(dlogtau2), _to_numpy(dlogtau)
    )


def test_physics_off_and_loss_multiplier_modes(mod):
    model = object.__new__(mod.GeoPriorSubsNet)
    model.pde_modes_active = ["none"]
    assert model._physics_off() is True

    model2 = object.__new__(mod.GeoPriorSubsNet)
    model2._lambda_offset = mod.tf_constant(
        2.0, dtype=mod.tf_float32
    )
    model2._physics_off = lambda: False
    model2.offset_mode = "mul"
    assert float(
        model2._physics_loss_multiplier().numpy()
    ) == pytest.approx(2.0)

    model2.offset_mode = "log10"
    assert float(
        model2._physics_loss_multiplier().numpy()
    ) == pytest.approx(100.0)

    model2._physics_off = lambda: True
    assert float(
        model2._physics_loss_multiplier().numpy()
    ) == pytest.approx(1.0)


def test_poroelastic_defaults_merge_bounds_before_super_init(
    monkeypatch, mod
):
    captured = {}

    def fake_super_init(self, *args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        mod.GeoPriorSubsNet, "__init__", fake_super_init
    )

    mod.PoroElasticSubsNet(
        static_input_dim=1,
        dynamic_input_dim=1,
        future_input_dim=1,
        scaling_kwargs={"bounds": {"H_min": 10.0}},
    )

    bounds = captured["scaling_kwargs"]["bounds"]
    assert bounds["H_min"] == 10.0
    assert "logK_min" in bounds
    assert "logSs_max" in bounds
    assert captured["pde_mode"] == "consolidation"
    assert captured["kappa_mode"] == "bar"
