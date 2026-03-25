from __future__ import annotations

from ._helpers import (
    DummyModel,
    import_module_group,
    to_scalar,
)

mod = import_module_group("losses")


def _make_model(scale_mv=False, scale_q=True):
    return DummyModel(
        lambda_cons=2.0,
        lambda_gw=3.0,
        lambda_prior=5.0,
        lambda_smooth=7.0,
        lambda_bounds=11.0,
        lambda_mv=13.0,
        lambda_q=17.0,
        _scale_mv_with_offset=scale_mv,
        _scale_q_with_offset=scale_q,
        scaling_kwargs={},
        _physics_loss_multiplier=lambda: 10.0,
        _physics_off=lambda: True,
    )


def test_should_log_physics_depends_on_flag_when_physics_off():
    model = _make_model()
    assert mod.should_log_physics(model) is False

    model.scaling_kwargs = {"log_physics_when_off": True}
    assert mod.should_log_physics(model) is True

    model._physics_off = lambda: False
    assert mod.should_log_physics(model) is True


def test_assemble_physics_loss_respects_default_scaling_policy():
    model = _make_model(scale_mv=False, scale_q=True)
    out = mod.assemble_physics_loss(
        model,
        loss_cons=1.0,
        loss_gw=1.0,
        loss_prior=1.0,
        loss_smooth=1.0,
        loss_mv=1.0,
        loss_q_reg=1.0,
        loss_bounds=1.0,
    )
    raw, scaled, mult, terms = out

    assert to_scalar(mult) == 10.0
    assert to_scalar(raw) == (2 + 3 + 5 + 7 + 11 + 13 + 17)
    assert (
        to_scalar(scaled)
        == 10 * (2 + 3 + 5 + 7 + 11) + 13 + 10 * 17
    )
    assert to_scalar(terms["mv"]) == 13.0
    assert to_scalar(terms["q"]) == 170.0


def test_assemble_physics_loss_can_scale_mv_and_q_independently():
    model = _make_model(scale_mv=True, scale_q=False)
    raw, scaled, mult, terms = mod.assemble_physics_loss(
        model,
        loss_cons=1.0,
        loss_gw=0.0,
        loss_prior=0.0,
        loss_smooth=0.0,
        loss_mv=2.0,
        loss_q_reg=3.0,
        loss_bounds=0.0,
    )

    assert to_scalar(mult) == 10.0
    assert to_scalar(raw) == 2 + 26 + 51
    assert to_scalar(scaled) == 10 * 2 + 10 * 26 + 51
    assert to_scalar(terms["cons"]) == 20.0
    assert to_scalar(terms["mv"]) == 260.0
    assert to_scalar(terms["q"]) == 51.0
