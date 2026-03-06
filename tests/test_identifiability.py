from __future__ import annotations

from tests._helpers import DummyLayer, DummyModel, import_module_group

mod = import_module_group("identifiability")


def test_get_ident_profile_normalizes_and_falls_back_to_base():
    regime, profile = mod.get_ident_profile(None)
    assert regime is None
    assert profile is None

    regime, profile = mod.get_ident_profile("closure_locked")
    assert regime == "closure_locked"
    assert profile["locks"]["tau_head"] is True

    regime, profile = mod.get_ident_profile("unknown-profile")
    assert regime == "base"
    assert profile["sk"]["freeze_physics_fields_over_time"] is True


def test_init_identifiability_preserves_user_values_and_adds_bounds_loss():
    scaling_kwargs = {
        "bounds_loss_kind": "residual",
        "physics_warmup_steps": 12,
        "time_units": "year",
    }

    regime, profile, merged = mod.init_identifiability(
        "anchored",
        scaling_kwargs,
    )

    assert regime == "anchored"
    assert profile is not None
    assert merged["bounds_loss_kind"] == "residual"
    assert merged["physics_warmup_steps"] == 12
    assert merged["bounds_loss"]["kind"] == "residual"
    assert merged["bounds_tau_w"] == 2.0


def test_apply_ident_locks_only_changes_locked_heads():
    model = DummyModel(
        tau_head=DummyLayer(True, "tau_head"),
        K_head=DummyLayer(True, "K_head"),
        Ss_head=DummyLayer(True, "Ss_head"),
    )
    profile = {"locks": {"tau_head": True, "K_head": False}}

    mod.apply_ident_locks(model, profile)

    assert model.tau_head.trainable is False
    assert model.K_head.trainable is True
    assert model.Ss_head.trainable is True


def test_resolve_compile_weights_prefers_explicit_then_profile_then_defaults():
    profile = {
        "compile": {
            "lambda_cons": 50.0,
            "lambda_prior": 10.0,
            "lambda_bounds": 3.0,
        }
    }

    out = mod.resolve_compile_weights(
        profile,
        lambda_cons=None,
        lambda_gw=None,
        lambda_prior=7.0,
        lambda_smooth=None,
        lambda_mv=None,
        lambda_bounds=None,
        lambda_q=None,
    )

    assert out["lambda_cons"] == 50.0
    assert out["lambda_prior"] == 7.0
    assert out["lambda_bounds"] == 3.0
    assert out["lambda_gw"] == 1.0
    assert out["lambda_mv"] == 0.0
    assert out["lambda_q"] == 0.0


def test_ident_audit_dict_reports_effective_state():
    profile = {
        "sk": {
            "freeze_physics_fields_over_time": True,
            "bounds_loss_kind": "barrier",
        },
        "compile": {"lambda_cons": 50.0},
        "locks": {"tau_head": True},
    }
    model = DummyModel(
        identifiability_regime="closure_locked",
        _ident_profile=profile,
        scaling_kwargs={
            "freeze_physics_fields_over_time": True,
            "bounds_loss_kind": "barrier",
            "bounds_loss": {"kind": "barrier"},
            "time_units": "year",
        },
        tau_head=DummyLayer(False, "tau_head"),
        K_head=DummyLayer(True, "K_head"),
        Ss_head=DummyLayer(True, "Ss_head"),
        lambda_cons=50.0,
        lambda_gw=0.0,
        lambda_prior=100.0,
        lambda_smooth=1.0,
        lambda_bounds=10.0,
        lambda_mv=0.0,
        lambda_q=0.0,
    )

    audit = mod.ident_audit_dict(
        model,
        extra_sk_keys=["time_units"],
    )

    assert audit["enabled"] is True
    assert audit["locks"]["tau_head"] is True
    assert audit["heads"]["tau_head"]["trainable"] is False
    assert audit["lambdas"]["lambda_prior"] == 100.0
    assert audit["sk_effective"]["bounds_loss_kind"] == "barrier"
    assert audit["bounds_loss"]["kind"] == "barrier"
