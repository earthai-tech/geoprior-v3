from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._helpers import import_module_group

mod = import_module_group("utils")


def test_canonicalize_scaling_kwargs_promotes_alias_without_mutating_source():
    src = {"gwl_index": 4, "time_units": "year"}
    out = mod.canonicalize_scaling_kwargs(src, copy=True)

    assert "gwl_dyn_index" in out
    assert out["gwl_dyn_index"] == 4
    assert "gwl_dyn_index" not in src


def test_get_sk_tries_aliases_and_casts_values():
    sk = {"gwl_index": "7", "time_units": "year"}
    assert mod.get_sk(sk, "gwl_dyn_index", cast=int) == 7
    assert mod.get_sk(sk, "missing", default="x") == "x"


def test_load_scaling_kwargs_supports_mapping_json_string_and_path(tmp_path: Path):
    payload = {"time_units": "year", "coords_normalized": False}
    p = tmp_path / "scaling.json"
    p.write_text(json.dumps(payload), encoding="utf-8")

    assert mod.load_scaling_kwargs(payload)["time_units"] == "year"
    assert mod.load_scaling_kwargs(json.dumps(payload))["time_units"] == "year"
    assert mod.load_scaling_kwargs(p)["time_units"] == "year"
    assert mod.load_scaling_kwargs(str(p))["time_units"] == "year"


def test_load_scaling_kwargs_rejects_bad_inputs(tmp_path: Path):
    bad_json = "[1, 2, 3]"
    with pytest.raises(ValueError):
        mod.load_scaling_kwargs(bad_json)

    with pytest.raises(FileNotFoundError):
        mod.load_scaling_kwargs(str(tmp_path / "missing.json"))

    with pytest.raises(TypeError):
        mod.load_scaling_kwargs(object())


def test_enforce_scaling_alias_consistency_warns_or_raises():
    sk_warn = {
        "gwl_dyn_index": 2,
        "gwl_index": 3,
        "scaling_error_policy": "warn",
        "time_units": "year",
    }
    with pytest.warns(RuntimeWarning):
        mod.enforce_scaling_alias_consistency(sk_warn)

    sk_raise = {
        "gwl_dyn_index": 2,
        "gwl_index": 3,
        "scaling_error_policy": "raise",
        "time_units": "year",
    }
    with pytest.raises(ValueError):
        mod.enforce_scaling_alias_consistency(
            sk_raise,
            where="validate",
        )


def test_validate_scaling_kwargs_requires_time_units_and_coord_rules():
    with pytest.raises(ValueError, match="time_units"):
        mod.validate_scaling_kwargs({"coords_normalized": False})

    with pytest.raises(ValueError, match="coord_ranges"):
        mod.validate_scaling_kwargs(
            {
                "time_units": "year",
                "coords_normalized": True,
            }
        )

    with pytest.raises(ValueError, match="deg_to_m_lon"):
        mod.validate_scaling_kwargs(
            {
                "time_units": "year",
                "coords_in_degrees": True,
            }
        )

    ok = {
        "time_units": "year",
        "coords_normalized": True,
        "coord_ranges": {"t": 3.0, "x": 1.0, "y": 1.0},
    }
    mod.validate_scaling_kwargs(ok)
