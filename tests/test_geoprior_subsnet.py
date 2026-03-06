import importlib
from pathlib import Path

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")


def _import_geoprior_subsnet():
    candidates = [
        ("geoprior.models", "GeoPriorSubsNet"),
        ("geoprior.models.subsidence", "GeoPriorSubsNet"),
        ("geoprior.models.subsidence.models", "GeoPriorSubsNet"),
    ]
    errors = []
    for module_name, attr in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr)
        except Exception as exc:  # pragma: no cover
            errors.append(f"{module_name}: {exc!r}")
    raise ImportError(
        "Could not import GeoPriorSubsNet from any known path.\n"
        + "\n".join(errors)
    )


GeoPriorSubsNet = _import_geoprior_subsnet()


@pytest.fixture(scope="session", autouse=True)
def _set_seed():
    tf.keras.utils.set_random_seed(7)


@pytest.fixture()
def dims():
    return {
        "batch": 4,
        "horizon": 3,
        "static_dim": 2,
        "dynamic_dim": 4,
        "future_dim": 2,
    }


@pytest.fixture()
def scaling_kwargs():
    return {
        "time_units": "year",
        "gwl_kind": "head",
        "gwl_sign": "up_positive",
        "use_head_proxy": False,
        "subsidence_kind": "cumulative",
        "coords_normalized": False,
        "track_aux_metrics": False,
        "allow_subs_residual": False,
    }


@pytest.fixture()
def batch_inputs(dims):
    b = dims["batch"]
    h = dims["horizon"]
    s = dims["static_dim"]
    d = dims["dynamic_dim"]
    f = dims["future_dim"]

    t = np.linspace(0.0, 1.0, h, dtype=np.float32)
    x = np.linspace(10.0, 11.0, h, dtype=np.float32)
    y = np.linspace(20.0, 20.5, h, dtype=np.float32)
    coords = np.stack([t, x, y], axis=-1)
    coords = np.broadcast_to(coords[None, ...], (b, h, 3)).copy()

    dynamic = np.arange(b * h * d, dtype=np.float32).reshape(b, h, d)
    dynamic = dynamic / (dynamic.max() + 1.0)

    future = np.arange(b * h * f, dtype=np.float32).reshape(b, h, f)
    future = future / (future.max() + 1.0)

    static = np.arange(b * s, dtype=np.float32).reshape(b, s)
    static = static / (static.max() + 1.0)

    h_field = np.full((b, h, 1), 5.0, dtype=np.float32)

    return {
        "static_features": static,
        "dynamic_features": dynamic,
        "future_features": future,
        "coords": coords,
        "H_field": h_field,
    }


@pytest.fixture()
def targets(batch_inputs):
    b, h, _ = batch_inputs["coords"].shape
    subs = np.zeros((b, h, 1), dtype=np.float32)
    gwl = np.zeros((b, h, 1), dtype=np.float32)
    return {"subs_pred": subs, "gwl_pred": gwl}


@pytest.fixture()
def dataset(batch_inputs, targets):
    ds = tf.data.Dataset.from_tensor_slices((batch_inputs, targets))
    return ds.batch(2)


def _make_model(
    dims,
    scaling_kwargs,
    **overrides,
):
    kwargs = dict(
        static_input_dim=dims["static_dim"],
        dynamic_input_dim=dims["dynamic_dim"],
        future_input_dim=dims["future_dim"],
        forecast_horizon=dims["horizon"],
        output_subsidence_dim=1,
        output_gwl_dim=1,
        embed_dim=8,
        hidden_units=8,
        lstm_units=8,
        attention_units=8,
        num_heads=2,
        max_window_size=4,
        memory_size=8,
        scales=[1],
        dropout_rate=0.0,
        use_batch_norm=False,
        use_vsn=False,
        pde_mode="consolidation",
        h_ref=0.0,
        mode="pihal_like",
        scaling_kwargs=dict(scaling_kwargs),
        verbose=0,
    )
    kwargs.update(overrides)
    return GeoPriorSubsNet(**kwargs)


def _compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={
            "subs_pred": tf.keras.losses.MeanSquaredError(),
            "gwl_pred": tf.keras.losses.MeanSquaredError(),
        },
        run_eagerly=True,
        lambda_cons=0.1,
        lambda_gw=0.0,
        lambda_prior=0.1,
        lambda_smooth=0.0,
        lambda_mv=0.0,
        lambda_bounds=0.0,
        lambda_q=0.0,
        lambda_offset=1.0,
    )
    return model


def test_build_and_call_return_expected_outputs(dims, scaling_kwargs, batch_inputs):
    model = _make_model(dims, scaling_kwargs)
    model.build(
        {
            "static_features": (None, dims["static_dim"]),
            "dynamic_features": (None, dims["horizon"], dims["dynamic_dim"]),
            "future_features": (None, dims["horizon"], dims["future_dim"]),
            "coords": (None, dims["horizon"], 3),
            "H_field": (None, dims["horizon"], 1),
        }
    )

    y = model(batch_inputs, training=False)

    assert list(y.keys()) == ["subs_pred", "gwl_pred"]
    assert y["subs_pred"].shape == (dims["batch"], dims["horizon"], 1)
    assert y["gwl_pred"].shape == (dims["batch"], dims["horizon"], 1)
    assert model.K_head is not None
    assert model.Ss_head is not None
    assert model.tau_head is not None


def test_forward_with_aux_exposes_physics_tensors(dims, scaling_kwargs, batch_inputs):
    model = _make_model(dims, scaling_kwargs)

    y_pred, aux = model.forward_with_aux(batch_inputs, training=False)

    assert set(y_pred) == {"subs_pred", "gwl_pred"}
    assert {"data_final", "data_mean_raw", "phys_mean_raw", "phys_features_raw_3d"}.issubset(aux)
    assert aux["phys_mean_raw"].shape[0] == dims["batch"]
    assert aux["phys_mean_raw"].shape[1] == dims["horizon"]
    assert aux["phys_mean_raw"].shape[-1] in (3, 4)


def test_closure_locked_regime_freezes_tau_head(dims, scaling_kwargs):
    model = _make_model(
        dims,
        scaling_kwargs,
        identifiability_regime="closure_locked",
    )
    model.build(
        {
            "static_features": (None, dims["static_dim"]),
            "dynamic_features": (None, dims["horizon"], dims["dynamic_dim"]),
            "future_features": (None, dims["horizon"], dims["future_dim"]),
            "coords": (None, dims["horizon"], 3),
            "H_field": (None, dims["horizon"], 1),
        }
    )

    assert model.identifiability_regime == "closure_locked"
    assert model.tau_head.trainable is False


def test_missing_coords_raises(dims, scaling_kwargs, batch_inputs):
    model = _make_model(dims, scaling_kwargs)
    bad_inputs = dict(batch_inputs)
    bad_inputs.pop("coords")

    with pytest.raises(Exception, match="coord|coords"):
        model(bad_inputs, training=False)


def test_hard_bounds_without_bounds_raise(dims, scaling_kwargs):
    model = _make_model(dims, scaling_kwargs, bounds_mode="hard")

    with pytest.raises(ValueError, match="requires bounds"):
        model.build(
            {
                "static_features": (None, dims["static_dim"]),
                "dynamic_features": (None, dims["horizon"], dims["dynamic_dim"]),
                "future_features": (None, dims["horizon"], dims["future_dim"]),
                "coords": (None, dims["horizon"], 3),
                "H_field": (None, dims["horizon"], 1),
            }
        )


def test_split_helpers_cover_point_and_legacy_q_fallback(dims, scaling_kwargs):
    model = _make_model(dims, scaling_kwargs)

    point = tf.zeros((2, dims["horizon"], 2), dtype=tf.float32)
    subs, gwl = model.split_data_predictions(point)
    assert subs.shape == (2, dims["horizon"], 1)
    assert gwl.shape == (2, dims["horizon"], 1)

    legacy_phys = tf.zeros((2, dims["horizon"], 3), dtype=tf.float32)
    k, ss, dlogtau, q = model.split_physics_predictions(legacy_phys)
    assert k.shape[-1] == 1
    assert ss.shape[-1] == 1
    assert dlogtau.shape[-1] == 1
    assert q.shape[-1] == 1
    np.testing.assert_allclose(q.numpy(), 0.0)


def test_config_roundtrip_reconstructs_model(dims, scaling_kwargs, batch_inputs):
    model = _make_model(dims, scaling_kwargs, identifiability_regime="anchored")
    _ = model(batch_inputs, training=False)

    cfg = model.get_config()
    clone = GeoPriorSubsNet.from_config(cfg)
    out = clone(batch_inputs, training=False)

    assert isinstance(clone, GeoPriorSubsNet)
    assert clone.forecast_horizon == model.forecast_horizon
    assert clone.identifiability_regime == "anchored"
    assert list(out.keys()) == ["subs_pred", "gwl_pred"]


def test_compile_rejects_non_positive_mul_lambda_offset(dims, scaling_kwargs):
    model = _make_model(dims, scaling_kwargs, offset_mode="mul")

    with pytest.raises(ValueError, match="lambda_offset must be > 0"):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss={"subs_pred": "mse", "gwl_pred": "mse"},
            run_eagerly=True,
            lambda_offset=0.0,
        )


def test_allow_missing_targets_flag_controls_test_step(
    dims,
    scaling_kwargs,
    batch_inputs,
    targets,
):
    strict_model = _compile_model(_make_model(dims, scaling_kwargs))
    with pytest.raises(KeyError, match="Missing targets"):
        strict_model.test_step((batch_inputs, {"subs_pred": targets["subs_pred"]}))

    relaxed_scaling = dict(scaling_kwargs)
    relaxed_scaling["allow_missing_targets"] = True
    relaxed_model = _compile_model(_make_model(dims, relaxed_scaling))
    logs = relaxed_model.test_step((batch_inputs, {"subs_pred": targets["subs_pred"]}))

    assert "data_loss" in logs
    assert "total_loss" in logs


def test_evaluate_physics_and_export_payload_smoke(
    dims,
    scaling_kwargs,
    batch_inputs,
    dataset,
    tmp_path,
):
    model = _compile_model(_make_model(dims, scaling_kwargs))
    _ = model(batch_inputs, training=False)

    phys = model.evaluate_physics(batch_inputs, return_maps=True)
    assert "epsilon_prior" in phys
    assert "epsilon_cons" in phys
    assert "tau" in phys
    assert "tau_prior" in phys

    save_path = tmp_path / "physics_payload.npz"
    payload = model.export_physics_payload(
        dataset,
        max_batches=1,
        save_path=str(save_path),
        format="npz",
        overwrite=True,
    )

    assert "tau" in payload
    assert "tau_prior" in payload
    assert "K" in payload
    assert "Ss" in payload
    assert "Hd" in payload
    assert "metrics" in payload
    assert save_path.exists()
    assert Path(str(save_path) + ".meta.json").exists()
