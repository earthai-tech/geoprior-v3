from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests._helpers import import_module_group

mod = import_module_group("scaling")


def test_jsonify_converts_numpy_scalars_and_sets():
    out = mod._jsonify({1: np.float32(2.5), "tags": {3, 1}})
    assert out["1"] == 2.5
    assert out["tags"] == [1, 3]


def test_from_any_handles_none_mapping_and_existing_instance(tmp_path: Path):
    cfg0 = mod.GeoPriorScalingConfig.from_any(None)
    assert cfg0.payload == {}

    cfg1 = mod.GeoPriorScalingConfig.from_any({"time_units": "year"})
    assert cfg1.payload["time_units"] == "year"

    cfg2 = mod.GeoPriorScalingConfig.from_any(cfg1)
    assert cfg2 is cfg1

    p = tmp_path / "scaling.json"
    p.write_text(json.dumps({"time_units": "year"}), encoding="utf-8")
    cfg3 = mod.GeoPriorScalingConfig.from_any(str(p))
    assert cfg3.source == str(p)
    assert cfg3.payload["time_units"] == "year"


def test_resolve_runs_canonicalization_and_validation():
    cfg = mod.GeoPriorScalingConfig.from_any(
        {
            "time_unit": "year",
            "coords_norm": True,
            "coord_range": {"t": 5.0, "x": 1.0, "y": 1.0},
        }
    )
    out = cfg.resolve()

    assert out["time_units"] == "year"
    assert out["coords_normalized"] is True
    assert out["coord_ranges"]["t"] == 5.0


def test_get_config_from_config_roundtrip_is_json_safe():
    cfg = mod.GeoPriorScalingConfig(
        payload={
            "time_units": "year",
            "weights": np.float32(1.5),
            "tags": {"a", "b"},
        },
        source="demo.json",
        schema_version="3",
    )

    ser = cfg.get_config()
    assert ser["payload"]["weights"] == 1.5
    assert ser["payload"]["tags"] == ["a", "b"]

    cfg2 = mod.GeoPriorScalingConfig.from_config(ser)
    assert cfg2.source == "demo.json"
    assert cfg2.schema_version == "3"
    assert cfg2.payload["time_units"] == "year"
