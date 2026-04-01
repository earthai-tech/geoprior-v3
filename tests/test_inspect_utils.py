from __future__ import annotations

import importlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


def _import_target():
    candidates = (
        "geoprior.utils.inspect.utils",
        "utils",
    )
    for modname in candidates:
        try:
            return importlib.import_module(modname)
        except ModuleNotFoundError:
            continue
    pytest.skip(
        "Could not import geoprior.utils.inspect.utils"
    )


def test_json_roundtrip_and_helpers(tmp_path: Path):
    mod = _import_target()

    payload = {
        "a": 1,
        "b": {
            "c": np.float64(2.5),
            "d": [1, 2, np.int64(3)],
        },
        "nan_val": float("nan"),
    }
    path = tmp_path / "demo.json"

    written = mod.write_json(payload, path)
    assert written == path.resolve()

    loaded = mod.read_json(path)
    assert loaded["a"] == 1
    assert loaded["b"]["c"] == 2.5
    assert loaded["b"]["d"] == [1, 2, 3]
    assert loaded["nan_val"] is None

    assert mod.as_path(path) == path.resolve()
    assert mod.nested_get(loaded, "b", "c") == 2.5
    assert (
        mod.nested_get(loaded, "missing", default="x") == "x"
    )


def test_read_json_rejects_non_object(tmp_path: Path):
    mod = _import_target()

    path = tmp_path / "list.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError):
        mod.read_json(path)


def test_deep_update_clone_flatten_and_numeric_helpers():
    mod = _import_target()

    base = {
        "a": 1,
        "b": {"x": 10, "y": 20},
        "flag": True,
    }
    updates = {
        "b": {"y": 99, "z": 5},
        "c": 7,
    }

    merged = mod.deep_update(base, updates)
    assert merged["a"] == 1
    assert merged["b"]["x"] == 10
    assert merged["b"]["y"] == 99
    assert merged["b"]["z"] == 5
    assert merged["c"] == 7
    assert base["b"]["y"] == 20

    cloned = mod.clone_artifact(base, overrides={"a": 2})
    assert cloned["a"] == 2
    assert base["a"] == 1

    flat = mod.flatten_dict({"x": {"y": 1}, "z": [1, 2]})
    assert flat["x.y"] == 1
    assert flat["z"] == [1, 2]

    nums = mod.numeric_items(
        {"a": 1, "b": 2.5, "c": "x", "d": False}
    )
    assert nums == {"a": 1.0, "b": 2.5}

    assert mod.is_number(3)
    assert mod.is_number(np.float64(1.2))
    assert not mod.is_number("3")


def test_infer_artifact_kind_and_load_artifact(
    tmp_path: Path,
):
    mod = _import_target()

    payload = {
        "metrics_at_best": {"loss": 1.0},
        "final_epoch_metrics": {"loss": 0.9},
        "compile": {},
        "hp_init": {},
        "city": "demo_city",
        "model": "GeoPriorSubsNet",
    }
    path = tmp_path / "demo_training_summary.json"
    mod.write_json(payload, path)

    assert (
        mod.infer_artifact_kind(path, payload)
        == "training_summary"
    )

    record = mod.load_artifact(path)
    assert record.kind == "training_summary"
    assert record.city == "demo_city"
    assert record.model == "GeoPriorSubsNet"
    brief = mod.artifact_brief(record)
    assert brief["kind"] == "training_summary"
    assert brief["city"] == "demo_city"


def test_metrics_and_bool_frames():
    mod = _import_target()

    mframe = mod.metrics_frame(
        {"b": 2.0, "a": 1.0, "flag": True},
        section="demo",
    )
    assert list(mframe.columns) == [
        "section",
        "metric",
        "value",
    ]
    assert list(mframe["metric"]) == ["a", "b"]

    bframe = mod.bool_checks_frame(
        {"x": True, "y": False, "z": 3},
        section="checks",
    )
    assert list(bframe.columns) == ["section", "check", "ok"]
    assert set(bframe["check"]) == {"x", "y"}


def test_plot_helpers_smoke(tmp_path: Path):
    mod = _import_target()

    fig1, ax1 = plt.subplots()
    mod.plot_metric_bars(
        ax1, {"a": 1.0, "b": 2.0}, title="demo"
    )
    fig1.savefig(tmp_path / "metric_bars.png")
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    mod.plot_boolean_checks(ax2, {"ok1": True, "ok2": False})
    fig2.savefig(tmp_path / "bools.png")
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    mod.plot_series_map(ax3, {"1": 1.0, "2": 2.0, "3": 1.5})
    fig3.savefig(tmp_path / "series.png")
    plt.close(fig3)

    assert (tmp_path / "metric_bars.png").exists()
    assert (tmp_path / "bools.png").exists()
    assert (tmp_path / "series.png").exists()
