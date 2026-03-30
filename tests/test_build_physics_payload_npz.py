from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import geoprior.cli.build_physics_payload_npz as mod


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def export_physics_payload(
        self,
        ds,
        *,
        max_batches,
        save_path,
        format,
        overwrite,
        metadata,
    ):
        self.calls.append(
            {
                "ds": ds,
                "max_batches": max_batches,
                "save_path": save_path,
                "format": format,
                "overwrite": overwrite,
                "metadata": metadata,
            }
        )
        payload = {
            "K": np.ones((4,), dtype=np.float32),
            "Ss": np.ones((4,), dtype=np.float32) * 2,
            "tau": np.ones((4,), dtype=np.float32) * 3,
            "H": np.ones((4,), dtype=np.float32) * 4,
        }
        np.savez_compressed(save_path, **payload)
        return payload


def test_choose_inputs_prefers_full_inputs_npz(
    mini_stage1_bundle: dict[str, object],
) -> None:
    manifest_path = Path(mini_stage1_bundle["manifest_path"])
    payload = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    art_dir = manifest_path.parent / "artifacts"
    full_npz = art_dir / "full_inputs.npz"

    train = mini_stage1_bundle["arrays"]["train_inputs"]
    val = mini_stage1_bundle["arrays"]["val_inputs"]
    test = mini_stage1_bundle["arrays"]["test_inputs"]
    merged = {
        key: np.concatenate(
            [train[key], val[key], test[key]],
            axis=0,
        )
        for key in train
    }
    np.savez_compressed(full_npz, **merged)

    x_np, label, src = mod._choose_inputs(
        payload=payload,
        manifest_path=manifest_path,
        inputs_npz=None,
        splits=None,
    )

    assert label == "full_city_union"
    assert src == str(full_npz)
    assert x_np["coords"].shape == merged["coords"].shape


def test_build_physics_payload_npz_exports_with_fake_model(
    monkeypatch: pytest.MonkeyPatch,
    mini_stage1_bundle: dict[str, object],
) -> None:
    bundle = mini_stage1_bundle
    run_dir = Path(bundle["run_dir"])
    manifest_path = Path(bundle["manifest_path"])
    art_dir = Path(bundle["artifacts_dir"])
    model_path = run_dir / "mock_best.keras"
    model_path.write_text("stub", encoding="utf-8")

    fake_model = FakeModel()
    make_ds_calls: list[dict[str, object]] = []

    def fake_make_tf_dataset(
        x_np,
        y_dummy,
        **kwargs,
    ):
        make_ds_calls.append(
            {
                "x_np": x_np,
                "y_dummy": y_dummy,
                "kwargs": kwargs,
            }
        )
        return {"dataset": "fake", "n": x_np["coords"].shape[0]}

    monkeypatch.setattr(mod, "make_tf_dataset", fake_make_tf_dataset)
    monkeypatch.setattr(mod, "_load_model", lambda path: fake_model)

    out, payload = mod.build_physics_payload_npz(
        manifest=str(manifest_path),
        stage1_dir=str(run_dir),
        model_path=str(model_path),
        splits=["train", "val"],
        source_label="train_val",
        batch_size=16,
        max_batches=2,
    )

    assert out.exists()
    assert sorted(payload) == ["H", "K", "Ss", "tau"]
    assert make_ds_calls

    ds_call = make_ds_calls[0]
    assert ds_call["kwargs"]["batch_size"] == 16
    assert ds_call["kwargs"]["forecast_horizon"] == 3
    assert ds_call["y_dummy"]["subs_pred"].shape[1] == 3

    export_call = fake_model.calls[0]
    assert export_call["max_batches"] == 2
    assert export_call["format"] == "npz"
    assert export_call["metadata"]["split"] == "train_val"
    assert export_call["metadata"]["stage1_manifest"] == str(
        manifest_path
    )
    assert export_call["metadata"]["source_inputs_npz"] is not None

    expected_n = (
        bundle["arrays"]["train_inputs"]["coords"].shape[0]
        + bundle["arrays"]["val_inputs"]["coords"].shape[0]
    )
    assert ds_call["x_np"]["coords"].shape[0] == expected_n


def test_build_physics_payload_main_delegates(
    monkeypatch: pytest.MonkeyPatch,
    mini_stage1_bundle: dict[str, object],
    write_natcom_config,
) -> None:
    config = dict(mini_stage1_bundle["config"])
    config["RESULTS_DIR"] = str(Path(config["BASE_OUTPUT_DIR"]))
    paths = write_natcom_config(config)

    captured: dict[str, object] = {}

    def fake_build_physics_payload_npz(**kwargs):
        captured.update(kwargs)
        out = Path(config["BASE_OUTPUT_DIR"]) / "payload.npz"
        np.savez_compressed(out, K=np.zeros((1,)))
        return out, {"K": np.zeros((1,))}

    monkeypatch.setattr(
        mod,
        "build_physics_payload_npz",
        fake_build_physics_payload_npz,
    )

    mod.build_physics_payload_main(
        [
            "--config-root",
            str(paths["root"]),
            "--source-label",
            "full_city_union",
            "--set",
            "CITY_NAME='zhongshan'",
        ]
    )

    cfg_json = json.loads(
        Path(paths["config_json"]).read_text(encoding="utf-8")
    )
    assert cfg_json["config"]["CITY_NAME"] == "zhongshan"
    assert captured["city"] == "zhongshan"
    assert captured["model"] == config["MODEL_NAME"]
    assert captured["source_label"] == "full_city_union"
