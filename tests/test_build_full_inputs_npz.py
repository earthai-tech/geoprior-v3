from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import geoprior.cli.build_full_inputs_npz as mod


def test_build_full_inputs_npz_merges_stage1_splits(
    mini_stage1_bundle: dict[str, object],
) -> None:
    manifest_path = mini_stage1_bundle["manifest_path"]
    out = mod.build_full_inputs_npz(
        manifest=str(manifest_path),
    )

    assert out.exists()

    with np.load(out, allow_pickle=False) as z:
        merged = {k: z[k] for k in z.files}

    arrays = mini_stage1_bundle["arrays"]
    expected_n = sum(
        arrays[f"{split}_inputs"]["coords"].shape[0]
        for split in ("train", "val", "test")
    )
    assert merged["coords"].shape[0] == expected_n
    assert set(merged) == set(arrays["train_inputs"])


def test_merge_inputs_rejects_mismatched_keys() -> None:
    split_arrays = {
        "train": {
            "coords": np.zeros((2, 3, 3)),
            "H_field": np.zeros((2, 3, 1)),
        },
        "val": {
            "coords": np.zeros((1, 3, 3)),
            "extra": np.zeros((1, 3, 1)),
        },
    }

    with pytest.raises(
        ValueError, match="Input-key mismatch"
    ):
        mod._merge_inputs(split_arrays, strict_keys=True)


def test_build_full_inputs_main_uses_runtime_defaults(
    monkeypatch: pytest.MonkeyPatch,
    mini_stage1_bundle: dict[str, object],
    write_natcom_config,
) -> None:
    config = dict(mini_stage1_bundle["config"])
    config["RESULTS_DIR"] = str(
        Path(config["BASE_OUTPUT_DIR"])
    )
    paths = write_natcom_config(config)

    captured: dict[str, object] = {}

    def fake_build_full_inputs_npz(**kwargs):
        captured.update(kwargs)
        out = (
            Path(config["BASE_OUTPUT_DIR"])
            / "synthetic_full.npz"
        )
        np.savez_compressed(out, coords=np.zeros((1, 1, 1)))
        return out

    monkeypatch.setattr(
        mod,
        "build_full_inputs_npz",
        fake_build_full_inputs_npz,
    )

    mod.build_full_inputs_main(
        [
            "--config-root",
            str(paths["root"]),
            "--set",
            "CITY_NAME='zhongshan'",
        ]
    )

    cfg_json = json.loads(
        Path(paths["config_json"]).read_text(encoding="utf-8")
    )
    assert cfg_json["config"]["CITY_NAME"] == "zhongshan"
    assert captured["results_dir"] == str(
        config["BASE_OUTPUT_DIR"]
    )
    assert captured["city"] == "zhongshan"
    assert captured["model"] == config["MODEL_NAME"]
    assert captured["strict_keys"] is True
