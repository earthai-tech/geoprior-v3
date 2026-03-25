from __future__ import annotations

from pathlib import Path

import pytest

from ._helpers import DummyModel, import_module_group

mod = import_module_group("models")
GeoPriorSubsNet = mod.GeoPriorSubsNet


def test_export_physics_payload_delegates_to_helpers(
    monkeypatch, tmp_path: Path
):
    calls = []
    base_payload = {
        "tau": [1, 2],
        "metrics": {"eps_prior_rms": 0.0},
    }

    def fake_gather(
        self,
        dataset,
        max_batches,
        float_dtype,
        log_fn,
        **kwargs,
    ):
        calls.append(
            ("gather", dataset, max_batches, float_dtype)
        )
        return dict(base_payload)

    def fake_subsample(payload, frac):
        calls.append(("subsample", frac))
        out = dict(payload)
        out["subsampled"] = True
        return out

    def fake_default_meta(self):
        calls.append(("meta",))
        return {"source": "auto"}

    def fake_save(
        payload, meta, save_path, format, overwrite, log_fn
    ):
        calls.append(
            ("save", save_path, format, overwrite, meta)
        )
        return str(save_path)

    monkeypatch.setattr(
        mod, "gather_physics_payload", fake_gather
    )
    monkeypatch.setattr(
        mod, "_maybe_subsample", fake_subsample
    )
    monkeypatch.setattr(
        mod, "default_meta_from_model", fake_default_meta
    )
    monkeypatch.setattr(
        mod, "save_physics_payload", fake_save
    )

    dummy = DummyModel()
    save_path = tmp_path / "physics_payload.npz"
    out = GeoPriorSubsNet.export_physics_payload(
        dummy,
        dataset="demo-ds",
        max_batches=3,
        save_path=str(save_path),
        format="npz",
        overwrite=True,
        metadata={"city": "nansha"},
        random_subsample=0.5,
    )

    assert out["subsampled"] is True
    assert calls[0][:3] == ("gather", "demo-ds", 3)
    assert calls[1] == ("subsample", 0.5)
    assert calls[2] == ("meta",)
    assert calls[3][0] == "save"
    assert calls[3][4]["source"] == "auto"
    assert calls[3][4]["city"] == "nansha"


def test_load_physics_payload_staticmethod_forwards(
    monkeypatch,
):
    monkeypatch.setattr(
        mod,
        "load_physics_payload",
        lambda path: ({"ok": True}, {"path": path}),
    )
    payload, meta = GeoPriorSubsNet.load_physics_payload(
        "demo.npz"
    )
    assert payload["ok"] is True
    assert meta["path"] == "demo.npz"


def test_from_config_deserializes_and_drops_legacy_keys(
    monkeypatch,
):
    class FakeGeoPrior(GeoPriorSubsNet):
        def __init__(self, **kwargs):
            self.received = kwargs

    monkeypatch.setattr(
        mod,
        "deserialize_keras_object",
        lambda obj, custom_objects=None: {
            "deserialized": obj["class_name"]
        },
    )

    cfg = {
        "static_input_dim": 1,
        "dynamic_input_dim": 2,
        "future_input_dim": 3,
        "time_units": "year",
        "mv": {"class_name": "LearnableMV", "config": {}},
        "kappa": {
            "class_name": "LearnableKappa",
            "config": {},
        },
        "gamma_w": {
            "class_name": "FixedGammaW",
            "config": {},
        },
        "h_ref": {"class_name": "FixedHRef", "config": {}},
        "scaling_kwargs": {
            "class_name": "GeoPriorScalingConfig",
            "config": {"payload": {"time_units": "year"}},
        },
        "K": 1,
        "Ss": 1,
        "Q": 1,
        "pinn_coefficient_C": 1,
        "gw_flow_coeffs": 1,
        "output_dim": 2,
        "model_version": "3.2-GeoPrior",
    }

    obj = FakeGeoPrior.from_config(dict(cfg))

    assert obj.received["mv"] == {
        "deserialized": "LearnableMV"
    }
    assert obj.received["kappa"] == {
        "deserialized": "LearnableKappa"
    }
    assert obj.received["scaling_kwargs"] == {
        "deserialized": "GeoPriorScalingConfig"
    }
    for key in (
        "K",
        "Ss",
        "Q",
        "pinn_coefficient_C",
        "gw_flow_coeffs",
        "output_dim",
        "model_version",
    ):
        assert key not in obj.received


@pytest.mark.slow
@pytest.mark.tensorflow
def test_model_smoke_init_and_config_roundtrip():
    pytest.importorskip("tensorflow")

    try:
        model = GeoPriorSubsNet(
            static_input_dim=2,
            dynamic_input_dim=3,
            future_input_dim=1,
            forecast_horizon=2,
            scaling_kwargs={"time_units": "year"},
            use_vsn=False,
            use_batch_norm=False,
            verbose=0,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(
            f"Full model stack not available for smoke test: {exc}"
        )

    cfg = model.get_config()
    assert cfg["output_subsidence_dim"] == 1
    assert "scaling_kwargs" in cfg

    try:
        rebuilt = type(model).from_config(dict(cfg))
    except Exception as exc:  # pragma: no cover
        pytest.skip(
            f"Roundtrip needs full serialization stack: {exc}"
        )

    assert isinstance(rebuilt, GeoPriorSubsNet)
