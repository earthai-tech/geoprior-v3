# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Shared helpers for ``geoprior.utils.inspect``.

This module is intentionally generic.
It does not know the full semantics of each
artifact family. Instead, it provides the
small reusable pieces that inspection
submodules can share:

- robust JSON loading and writing,
- nested mapping access,
- safe flattening for summaries,
- artifact kind inference,
- demo payload cloning for gallery examples,
- compact tabular conversion for metrics,
- generic plotting helpers.

Submodules such as ``stage1_audit.py`` or
``training_summary.py`` should build the
artifact-specific logic on top of these
helpers rather than reimplementing them.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PathLike = str | Path

__all__ = [
    "ArtifactRecord",
    "artifact_brief",
    "as_path",
    "bool_checks_frame",
    "clone_artifact",
    "deep_update",
    "ensure_parent_dir",
    "flatten_dict",
    "infer_artifact_kind",
    "is_number",
    "json_ready",
    "load_artifact",
    "metrics_frame",
    "nested_get",
    "numeric_items",
    "plot_boolean_checks",
    "plot_metric_bars",
    "plot_series_map",
    "read_json",
    "write_json",
]


@dataclass(slots=True)
class ArtifactRecord:
    """
    Lightweight normalized artifact container.

    Parameters
    ----------
    path : pathlib.Path
        Artifact path.
    kind : str
        Inferred or explicit artifact kind.
    payload : dict[str, Any]
        Loaded JSON payload.
    stage : str or None
        Stage if available.
    city : str or None
        City if available.
    model : str or None
        Model if available.
    meta : dict[str, Any]
        Extra extracted metadata.
    """

    path: Path
    kind: str
    payload: dict[str, Any]
    stage: str | None = None
    city: str | None = None
    model: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def as_path(path: PathLike) -> Path:
    """Return ``path`` as resolved ``Path``."""
    return Path(path).expanduser().resolve()


def ensure_parent_dir(path: PathLike) -> Path:
    """Create parent directory for ``path``."""
    p = as_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: PathLike) -> dict[str, Any]:
    """Read a JSON file into a dictionary."""
    p = as_path(path)
    with p.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    if not isinstance(data, dict):
        raise ValueError(
            "Expected a JSON object at "
            f"{p!s}, got {type(data).__name__}."
        )
    return data


def write_json(
    payload: dict[str, Any],
    path: PathLike,
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> Path:
    """Write ``payload`` as UTF-8 JSON."""
    p = ensure_parent_dir(path)
    safe = json_ready(payload)
    with p.open("w", encoding="utf-8") as stream:
        json.dump(
            safe,
            stream,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=False,
        )
        stream.write("\n")
    return p


def is_number(value: Any) -> bool:
    """Return True for finite or non-finite scalars."""
    return isinstance(value, (int, float, np.number))


def json_ready(value: Any) -> Any:
    """
    Convert nested values into JSON-safe objects.

    Notes
    -----
    - ``NaN`` and ``Inf`` are converted to ``None``.
    - numpy scalars are converted to Python scalars.
    - arrays become lists.
    """
    if isinstance(value, dict):
        return {
            str(k): json_ready(v) for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]

    if isinstance(value, np.ndarray):
        return [json_ready(v) for v in value.tolist()]

    if isinstance(value, np.generic):
        return json_ready(value.item())

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    return value


def deep_update(
    base: dict[str, Any],
    updates: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Recursively update ``base`` with ``updates``.

    Returns a new dictionary.
    """
    out = copy.deepcopy(base)
    if not updates:
        return out

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(
            out.get(key), dict
        ):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def clone_artifact(
    template: dict[str, Any],
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Clone a template payload and apply overrides.

    This is useful for Sphinx-Gallery examples
    where we want a realistic artifact with a few
    controlled changes.
    """
    return deep_update(template, overrides)


def nested_get(
    mapping: dict[str, Any] | None,
    *keys: str,
    default: Any = None,
) -> Any:
    """
    Safely traverse nested dictionaries.

    Examples
    --------
    ``nested_get(d, "config", "scaling_kwargs")``
    """
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def flatten_dict(
    mapping: dict[str, Any],
    *,
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """
    Flatten nested dictionaries.

    Non-dict values are kept as they are.
    Lists and arrays are not expanded.
    """
    items: dict[str, Any] = {}

    for key, value in mapping.items():
        new_key = (
            f"{parent_key}{sep}{key}"
            if parent_key
            else str(key)
        )
        if isinstance(value, dict):
            items.update(
                flatten_dict(
                    value,
                    parent_key=new_key,
                    sep=sep,
                )
            )
        else:
            items[new_key] = value
    return items


def numeric_items(
    mapping: dict[str, Any] | None,
    *,
    drop_bools: bool = True,
) -> dict[str, float]:
    """Extract numeric scalar items from a mapping."""
    out: dict[str, float] = {}
    if not mapping:
        return out

    for key, value in mapping.items():
        if drop_bools and isinstance(value, bool):
            continue
        if is_number(value):
            out[str(key)] = float(value)
    return out


def infer_artifact_kind(
    path: PathLike,
    payload: dict[str, Any] | None = None,
) -> str:
    """
    Infer artifact kind from file name and keys.

    The rules are intentionally simple and stable.
    Artifact-specific readers can still override
    the inferred kind if needed.
    """
    p = as_path(path)
    name = p.name.lower()
    data = payload or {}
    keys = set(data)

    if "scaling_audit" in name:
        return "stage1_audit"
    if "handshake_audit" in name:
        return "stage2_handshake"
    if name.endswith(".npz.meta.json"):
        return "physics_payload_meta"
    if "training_summary" in name:
        return "training_summary"
    if "eval_diagnostics" in name:
        return "eval_diagnostics"
    if "calibration_stats" in name:
        return "calibration_stats"
    if "scaling_kwargs" in name:
        return "scaling_kwargs"
    if "model_init_manifest" in name:
        return "model_init_manifest"
    if "run_manifest" in name:
        return "run_manifest"
    if name == "manifest.json":
        return "manifest"

    if {
        "metrics_at_best",
        "final_epoch_metrics",
    }.issubset(keys):
        return "training_summary"

    if {
        "expected",
        "got",
        "finite",
    }.issubset(keys):
        return "stage2_handshake"

    if {
        "provenance",
        "coord_scaler",
        "feature_split",
    }.issubset(keys):
        return "stage1_audit"

    if {
        "schema_version",
        "stage",
        "artifacts",
        "paths",
    }.issubset(keys):
        return "manifest"

    if {
        "config",
        "dims",
        "model_class",
    }.issubset(keys):
        return "model_init_manifest"

    if {
        "metrics_evaluate",
        "physics_diagnostics",
    }.issubset(keys):
        return "eval_physics"

    if {
        "target",
        "interval",
        "factors",
        "eval_before",
        "eval_after",
    }.issubset(keys):
        return "calibration_stats"

    if {
        "created_utc",
        "units",
        "payload_metrics",
    }.issubset(keys):
        return "physics_payload_meta"

    if {
        "stage",
        "city",
        "model",
        "config",
        "paths",
        "artifacts",
    }.issubset(keys):
        return "run_manifest"

    return "json_artifact"


def load_artifact(
    path: PathLike,
    *,
    kind: str | None = None,
) -> ArtifactRecord:
    """Load a JSON artifact into ``ArtifactRecord``."""
    p = as_path(path)
    payload = read_json(p)
    artifact_kind = kind or infer_artifact_kind(p, payload)

    stage = nested_get(payload, "stage")
    city = nested_get(payload, "city")
    model = (
        nested_get(payload, "model")
        or nested_get(payload, "model_name")
        or nested_get(payload, "model_class")
    )

    meta = {
        "top_keys": list(payload),
        "n_top_keys": len(payload),
        "has_config": "config" in payload,
        "has_paths": "paths" in payload,
        "has_artifacts": "artifacts" in payload,
    }

    return ArtifactRecord(
        path=p,
        kind=artifact_kind,
        payload=payload,
        stage=stage,
        city=city,
        model=model,
        meta=meta,
    )


def artifact_brief(record: ArtifactRecord) -> dict[str, Any]:
    """
    Return a compact artifact header summary.
    """
    return {
        "path": str(record.path),
        "kind": record.kind,
        "stage": record.stage,
        "city": record.city,
        "model": record.model,
        "n_top_keys": record.meta.get("n_top_keys"),
        "top_keys": record.meta.get("top_keys"),
    }


def metrics_frame(
    mapping: dict[str, Any] | None,
    *,
    section: str | None = None,
    sort: bool = True,
) -> pd.DataFrame:
    """
    Convert scalar metrics into a tidy DataFrame.
    """
    pairs = numeric_items(mapping)
    frame = pd.DataFrame(
        {
            "metric": list(pairs),
            "value": list(pairs.values()),
        }
    )
    if section is not None and not frame.empty:
        frame.insert(0, "section", section)
    if sort and not frame.empty:
        frame = frame.sort_values("metric")
    return frame.reset_index(drop=True)


def bool_checks_frame(
    mapping: dict[str, Any] | None,
    *,
    section: str | None = None,
) -> pd.DataFrame:
    """
    Convert boolean checks into a tidy DataFrame.
    """
    rows: list[dict[str, Any]] = []
    for key, value in (mapping or {}).items():
        if isinstance(value, bool):
            rows.append(
                {
                    "check": str(key),
                    "ok": value,
                }
            )
    frame = pd.DataFrame(rows)
    if section is not None and not frame.empty:
        frame.insert(0, "section", section)
    return frame.reset_index(drop=True)


def plot_metric_bars(
    ax: plt.Axes,
    metrics: dict[str, Any] | pd.DataFrame,
    *,
    title: str = "Metrics",
    top_n: int | None = None,
    sort_by_value: bool = False,
    absolute: bool = False,
    annotate: bool = True,
) -> plt.Axes:
    """
    Plot a compact horizontal metric bar chart.
    """
    if isinstance(metrics, pd.DataFrame):
        frame = metrics.copy()
    else:
        frame = metrics_frame(metrics)

    if frame.empty:
        ax.set_title(title)
        ax.text(
            0.5,
            0.5,
            "No numeric metrics",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    if absolute:
        frame["plot_value"] = frame["value"].abs()
    else:
        frame["plot_value"] = frame["value"]

    if sort_by_value:
        frame = frame.sort_values("plot_value")
    else:
        frame = frame.sort_values("metric")

    if top_n is not None:
        frame = frame.tail(int(top_n))

    ax.barh(frame["metric"], frame["plot_value"])
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.grid(axis="x", alpha=0.25)

    if annotate:
        for idx, row in frame.reset_index(
            drop=True
        ).iterrows():
            ax.text(
                row["plot_value"],
                idx,
                f" {row['value']:.4g}",
                va="center",
            )

    return ax


def plot_boolean_checks(
    ax: plt.Axes,
    checks: dict[str, Any] | pd.DataFrame,
    *,
    title: str = "Checks",
) -> plt.Axes:
    """
    Plot boolean pass/fail checks as a bar view.
    """
    if isinstance(checks, pd.DataFrame):
        frame = checks.copy()
    else:
        frame = bool_checks_frame(checks)

    if frame.empty:
        ax.set_title(title)
        ax.text(
            0.5,
            0.5,
            "No boolean checks",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    frame["score"] = frame["ok"].astype(int)
    labels = frame["check"]
    vals = frame["score"]

    ax.barh(labels, vals)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["fail", "pass"])
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    return ax


def plot_series_map(
    ax: plt.Axes,
    series_map: dict[str, Any],
    *,
    title: str = "Series",
    xlabel: str = "key",
    ylabel: str = "value",
    marker: str = "o",
) -> plt.Axes:
    """
    Plot a string-keyed numeric mapping as a line.

    This is useful for horizons such as
    ``{"1": 0.9, "2": 0.8, "3": 0.7}``.
    """
    pairs = numeric_items(series_map)

    if not pairs:
        ax.set_title(title)
        ax.text(
            0.5,
            0.5,
            "No numeric series",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    def _key(v: str) -> tuple[int, Any]:
        try:
            return (0, float(v))
        except Exception:
            return (1, v)

    keys = sorted(pairs, key=_key)
    x = np.arange(len(keys))
    y = np.array([pairs[k] for k in keys], dtype=float)

    ax.plot(x, y, marker=marker)
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    return ax
