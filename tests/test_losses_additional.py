from __future__ import annotations

import importlib

import numpy as np
import pytest

MODULE_CANDIDATES = [
    "geoprior.models.subsidence.losses",
]


def _import_target():
    last = None
    for name in MODULE_CANDIDATES:
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover
            last = exc
    raise last  # type: ignore[misc]


class _Metric:
    def __init__(self, value=0.0):
        self.value = value
        self.built = True
        self.calls = []

    def update_state(self, value):
        if hasattr(value, "numpy"):
            value = float(value.numpy())
        self.calls.append(value)
        self.value = value

    def result(self):
        return self.value


class _DummyModel:
    def __init__(
        self, mod, *, physics_off=False, log_when_off=False
    ):
        self.lambda_cons = 1.0
        self.lambda_gw = 2.0
        self.lambda_prior = 3.0
        self.lambda_smooth = 4.0
        self.lambda_bounds = 5.0
        self.lambda_mv = 6.0
        self.lambda_q = 7.0
        self._scale_mv_with_offset = False
        self._scale_q_with_offset = True
        self._lambda_offset = mod.tf_constant(
            2.5, dtype=mod.tf_float32
        )
        self.output_names = ["subs_pred", "gwl_pred"]
        self.scaling_kwargs = {
            "log_physics_when_off": log_when_off,
            "log_q_diagnostics": True,
        }
        self.eps_prior_metric = _Metric()
        self.eps_cons_metric = _Metric()
        self.eps_gw_metric = _Metric()
        self._physics_off_flag = physics_off
        self.log_fn = None

    def _physics_off(self):
        return self._physics_off_flag

    def _physics_loss_multiplier(self):
        return 10.0


@pytest.fixture(scope="module")
def mod():
    return _import_target()


def _to_float(x):
    if hasattr(x, "numpy"):
        return float(x.numpy())
    return float(x)


def test_should_log_physics_respects_switches(mod):
    assert mod.should_log_physics(object()) is True
    assert (
        mod.should_log_physics(
            _DummyModel(mod, physics_off=False)
        )
        is True
    )
    assert (
        mod.should_log_physics(
            _DummyModel(
                mod, physics_off=True, log_when_off=False
            )
        )
        is False
    )
    assert (
        mod.should_log_physics(
            _DummyModel(
                mod, physics_off=True, log_when_off=True
            )
        )
        is True
    )


def test_assemble_physics_loss_respects_mv_and_q_offset_scaling(
    mod,
):
    model = _DummyModel(mod)
    raw, scaled, phys_mult, terms = mod.assemble_physics_loss(
        model,
        loss_cons=1.0,
        loss_gw=1.0,
        loss_prior=1.0,
        loss_smooth=1.0,
        loss_mv=1.0,
        loss_q_reg=1.0,
        loss_bounds=1.0,
    )

    # raw = 1+2+3+4+5 + 6 + 7 = 28
    assert _to_float(raw) == pytest.approx(28.0)
    # scaled core = 10*(1+2+3+4+5)=150 ; mv unscaled=6 ; q scaled=70
    assert _to_float(scaled) == pytest.approx(226.0)
    assert _to_float(phys_mult) == pytest.approx(10.0)
    assert _to_float(terms["mv"]) == pytest.approx(6.0)
    assert _to_float(terms["q"]) == pytest.approx(70.0)

    model._scale_mv_with_offset = True
    raw2, scaled2, _, terms2 = mod.assemble_physics_loss(
        model,
        loss_cons=1.0,
        loss_gw=1.0,
        loss_prior=1.0,
        loss_smooth=1.0,
        loss_mv=1.0,
        loss_q_reg=1.0,
        loss_bounds=1.0,
    )
    assert _to_float(raw2) == pytest.approx(28.0)
    assert _to_float(scaled2) == pytest.approx(280.0)
    assert _to_float(terms2["mv"]) == pytest.approx(60.0)


def test_zero_and_build_physics_bundle_cover_defaults(mod):
    model = _DummyModel(mod)
    zero = mod.zero_physics_bundle(model)
    assert _to_float(zero["physics_mult"]) == pytest.approx(
        1.0
    )
    assert _to_float(zero["lambda_offset"]) == pytest.approx(
        2.5
    )

    built = mod.build_physics_bundle(
        model,
        physics_loss_raw=1.0,
        physics_loss_scaled=2.0,
        phys_mult=3.0,
        loss_cons=4.0,
        loss_gw=5.0,
        loss_prior=6.0,
        loss_smooth=7.0,
        loss_mv=8.0,
        loss_q_reg=9.0,
        q_rms=10.0,
        q_gate=11.0,
        subs_resid_gate=12.0,
        loss_bounds=13.0,
        eps_prior=14.0,
        eps_cons=15.0,
        eps_gw=16.0,
        eps_cons_raw=None,
        eps_gw_raw=None,
    )
    assert _to_float(
        built["physics_loss_scaled"]
    ) == pytest.approx(2.0)
    assert _to_float(
        built["epsilon_cons_raw"]
    ) == pytest.approx(0.0)
    assert _to_float(
        built["epsilon_gw_raw"]
    ) == pytest.approx(0.0)


def test_pack_eval_physics_uses_zero_bundle_or_empty_dict(
    mod,
):
    model_off = _DummyModel(
        mod, physics_off=True, log_when_off=False
    )
    assert (
        mod.pack_eval_physics(model_off, physics=None) == {}
    )

    model_log = _DummyModel(
        mod, physics_off=True, log_when_off=True
    )
    packed = mod.pack_eval_physics(model_log, physics=None)
    assert "physics_loss_raw" in packed
    assert _to_float(
        packed["physics_loss_raw"]
    ) == pytest.approx(0.0)

    existing = {"physics_loss_raw": 9.0}
    assert (
        mod.pack_eval_physics(model_log, physics=existing)
        is existing
    )


def test_pack_step_results_merges_compiled_metrics_manual_and_physics(
    monkeypatch, mod
):
    model = _DummyModel(mod, physics_off=False)
    physics = mod.zero_physics_bundle(model)
    physics["loss_q_reg"] = 1.5
    physics["q_rms"] = 2.5
    physics["q_gate"] = 0.75
    physics["subs_resid_gate"] = 0.5

    monkeypatch.setattr(
        mod,
        "ensure_targets_for_outputs",
        lambda **kwargs: kwargs["targets"],
    )
    monkeypatch.setattr(
        mod,
        "update_compiled_metrics",
        lambda model, **kwargs: None,
    )
    # # or more permissive
    # monkeypatch.setattr(
    #     mod,
    #     "update_compiled_metrics",
    #     lambda *args, **kwargs: None,
    # )

    monkeypatch.setattr(
        mod,
        "compiled_metrics_dict",
        lambda model, dtype=None: {
            "subs_pred_mae_q50": mod.tf_constant(
                0.3, dtype=mod.tf_float32
            ),
            "epsilon_prior": mod.tf_constant(
                999.0, dtype=mod.tf_float32
            ),
            "loss": mod.tf_constant(
                777.0, dtype=mod.tf_float32
            ),
        },
    )

    manual = {"custom_tracker": _Metric(0.8)}
    results = mod.pack_step_results(
        model,
        total_loss=mod.tf_constant(3.0, dtype=mod.tf_float32),
        data_loss=mod.tf_constant(2.0, dtype=mod.tf_float32),
        targets={
            "subs_pred": mod.tf_constant(
                [[1.0]], dtype=mod.tf_float32
            )
        },
        y_pred={
            "subs_pred": mod.tf_constant(
                [[1.0]], dtype=mod.tf_float32
            ),
            "gwl_pred": mod.tf_constant(
                [[2.0]], dtype=mod.tf_float32
            ),
        },
        physics=physics,
        manual_trackers=manual,
    )

    assert _to_float(results["loss"]) == pytest.approx(3.0)
    assert _to_float(results["data_loss"]) == pytest.approx(
        2.0
    )
    assert _to_float(
        results["subs_pred_mae_q50"]
    ) == pytest.approx(0.3)
    assert _to_float(
        results["custom_tracker"]
    ) == pytest.approx(0.8)
    assert _to_float(
        results["epsilon_prior"]
    ) == pytest.approx(0.0)
    assert _to_float(results["q_reg_loss"]) == pytest.approx(
        1.5
    )
    assert _to_float(results["q_gate"]) == pytest.approx(0.75)
    assert _to_float(
        results["subs_resid_gate"]
    ) == pytest.approx(0.5)
    assert (
        model.eps_prior_metric.calls
    )  # updated through update_epsilon_metrics
