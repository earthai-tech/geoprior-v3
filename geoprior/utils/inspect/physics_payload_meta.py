# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Physics-payload metadata generation and inspection helpers.

This module focuses on the JSON sidecar written next to
exported physics payload artifacts, typically files such as
``physics_payload.npz.meta.json``.

Unlike the full payload archive, this metadata file is light
and semantic. It records:

- model and run identity,
- active PDE modes,
- unit conventions,
- closure / tau-prior definitions,
- groundwater interpretation choices,
- compact payload-level epsilon summaries.

The functions are designed for two common uses:

1. Sphinx-Gallery examples that need a realistic metadata
   sidecar without rebuilding a full physics payload.
2. Real workflow inspection when a user wants to confirm the
   saved units, closure assumptions, and core payload metrics
   before opening the heavier payload itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .utils import (
    ArtifactRecord,
    clone_artifact,
    deep_update,
    load_artifact,
    metrics_frame,
    plot_boolean_checks,
    plot_metric_bars,
    read_json,
    write_json,
)

PathLike = str | Path
PhysicsPayloadMetaLike = (
    ArtifactRecord | Mapping[str, Any] | str | Path
)

__all__ = [
    "default_physics_payload_meta_payload",
    "generate_physics_payload_meta",
    "inspect_physics_payload_meta",
    "load_physics_payload_meta",
    "physics_payload_meta_closure_frame",
    "physics_payload_meta_identity_frame",
    "physics_payload_meta_metrics_frame",
    "physics_payload_meta_units_frame",
    "plot_physics_payload_meta_boolean_summary",
    "plot_physics_payload_meta_core_scalars",
    "plot_physics_payload_meta_payload_metrics",
    "summarize_physics_payload_meta",
]

_CORE_SCALARS = [
    "hd_factor",
    "Hd_factor",
    "lambda_offset",
    "lambda_offsets",
    "kappa_value",
]

_PAYLOAD_METRIC_KEYS = [
    "eps_prior_rms",
    "eps_cons_scaled_rms",
    "closure_consistency_rms",
    "r2_logtau",
    "kappa_used_for_closure",
]


def _as_payload(
    payload: PhysicsPayloadMetaLike,
) -> dict[str, Any]:
    """Return a plain physics-payload metadata mapping."""
    if isinstance(payload, ArtifactRecord):
        return dict(payload.payload)

    if isinstance(payload, Mapping):
        return dict(payload)

    data = read_json(payload)
    return dict(data)


def _try_float(
    value: Any,
) -> float | None:
    """Return ``value`` as float when possible."""
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _string_or_none(
    value: Any,
) -> str | None:
    """Return a compact string or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _numeric_subset(
    mapping: dict[str, Any] | None,
    *,
    keys: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Return selected numeric scalar items."""
    src = mapping or {}
    keep = None if keys is None else {str(k) for k in keys}

    out: dict[str, float] = {}
    for key, value in src.items():
        if isinstance(value, bool):
            continue
        num = _try_float(value)
        if num is None:
            continue
        if keep is not None and str(key) not in keep:
            continue
        out[str(key)] = float(num)
    return out


def default_physics_payload_meta_payload(
    *,
    city: str = "demo_city",
    model_name: str = "GeoPriorSubsNet",
    split: str = "TestSet",
    created_utc: str = "2026-02-22T13:50:48Z",
    saved_utc: str | None = None,
    pde_modes_active: list[str] | None = None,
    time_units: str = "year",
    time_coord_units: str = "year",
    gwl_kind: str = "depth_bgs",
    gwl_sign: str = "down_positive",
    use_head_proxy: bool = True,
    kappa_mode: str = "kb",
    use_effective_h: bool = True,
    hd_factor: float = 0.6,
    lambda_offset: float = 1.0,
    kappa_value: float = 0.046392478,
    tau_prior_definition: str = (
        "tau_closure_from_learned_fields"
    ),
    tau_prior_human_name: str = "tau_closure",
    tau_prior_source: str = (
        "model.evaluate_physics() closure head"
    ),
    tau_closure_formula: str = (
        "Hd^2 * Ss / (pi^2 * kappa_b * K)"
    ),
    units: dict[str, str] | None = None,
    payload_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a realistic default physics-payload meta payload.

    The payload is intentionally lightweight and template-based.
    It mirrors the structure of the real ``*.npz.meta.json``
    sidecar without trying to reconstruct the full physics
    payload archive.
    """
    if saved_utc is None:
        saved_utc = created_utc

    return {
        "created_utc": created_utc,
        "model_name": model_name,
        "pde_modes_active": list(
            pde_modes_active
            or [
                "consolidation",
                "gw_flow",
            ]
        ),
        "kappa_mode": kappa_mode,
        "use_effective_h": bool(use_effective_h),
        "hd_factor": float(hd_factor),
        "lambda_offset": float(lambda_offset),
        "use_effective_thickness": bool(use_effective_h),
        "Hd_factor": float(hd_factor),
        "lambda_offsets": float(lambda_offset),
        "time_units": time_units,
        "time_coord_units": time_coord_units,
        "units": dict(
            units
            or {
                "H": "m",
                "Hd": "m",
                "Ss": "1/m",
                "K": "m/s",
                "tau": "s",
                "tau_prior": "s",
                "cons_res_vals": "m/s",
                "kappa": "dimensionless",
            }
        ),
        "tau_prior_definition": tau_prior_definition,
        "tau_prior_human_name": tau_prior_human_name,
        "tau_prior_source": tau_prior_source,
        "kappa_value": float(kappa_value),
        "tau_closure_formula": tau_closure_formula,
        "city": city,
        "split": split,
        "gwl_kind": gwl_kind,
        "gwl_sign": gwl_sign,
        "use_head_proxy": bool(use_head_proxy),
        "saved_utc": saved_utc,
        "payload_metrics": dict(
            payload_metrics
            or {
                "eps_prior_rms": 3.57e-4,
                "r2_logtau": 1.0,
                "eps_cons_scaled_rms": 2.61e-6,
                "closure_consistency_rms": None,
                "kappa_used_for_closure": None,
            }
        ),
    }


def generate_physics_payload_meta(
    path: PathLike,
    *,
    template: PhysicsPayloadMetaLike | None = None,
    overrides: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Path:
    """
    Generate and save a physics-payload meta artifact.

    Parameters
    ----------
    path : path-like
        Output JSON path.
    template : mapping, artifact, or path, optional
        Template payload to clone. When omitted, the function
        starts from ``default_physics_payload_meta_payload``.
    overrides : dict, optional
        Deep updates applied after loading the template.
    **kwargs : Any
        Forwarded to the default payload builder when no
        template is provided.
    """
    if template is None:
        payload = default_physics_payload_meta_payload(
            **kwargs
        )
    else:
        payload = clone_artifact(
            _as_payload(template),
            overrides=kwargs,
        )

    if overrides:
        payload = deep_update(payload, overrides)

    return write_json(payload, path)


def load_physics_payload_meta(
    path: PathLike,
) -> ArtifactRecord:
    """Load a physics-payload meta artifact."""
    return load_artifact(path, kind="physics_payload_meta")


def physics_payload_meta_identity_frame(
    payload: PhysicsPayloadMetaLike,
) -> pd.DataFrame:
    """Return a tidy identity / convention frame."""
    data = _as_payload(payload)
    rows = [
        {
            "field": "created_utc",
            "value": data.get("created_utc"),
        },
        {
            "field": "saved_utc",
            "value": data.get("saved_utc"),
        },
        {
            "field": "model_name",
            "value": data.get("model_name"),
        },
        {
            "field": "city",
            "value": data.get("city"),
        },
        {
            "field": "split",
            "value": data.get("split"),
        },
        {
            "field": "time_units",
            "value": data.get("time_units"),
        },
        {
            "field": "time_coord_units",
            "value": data.get("time_coord_units"),
        },
        {
            "field": "gwl_kind",
            "value": data.get("gwl_kind"),
        },
        {
            "field": "gwl_sign",
            "value": data.get("gwl_sign"),
        },
        {
            "field": "use_head_proxy",
            "value": data.get("use_head_proxy"),
        },
        {
            "field": "kappa_mode",
            "value": data.get("kappa_mode"),
        },
        {
            "field": "pde_modes_active",
            "value": ", ".join(
                str(v)
                for v in (data.get("pde_modes_active") or [])
            ),
        },
    ]
    return pd.DataFrame(rows)


def physics_payload_meta_closure_frame(
    payload: PhysicsPayloadMetaLike,
) -> pd.DataFrame:
    """Return a tidy closure / tau-prior frame."""
    data = _as_payload(payload)
    rows = [
        {
            "field": "tau_prior_definition",
            "value": data.get("tau_prior_definition"),
        },
        {
            "field": "tau_prior_human_name",
            "value": data.get("tau_prior_human_name"),
        },
        {
            "field": "tau_prior_source",
            "value": data.get("tau_prior_source"),
        },
        {
            "field": "tau_closure_formula",
            "value": data.get("tau_closure_formula"),
        },
        {
            "field": "use_effective_h",
            "value": data.get("use_effective_h"),
        },
        {
            "field": "use_effective_thickness",
            "value": data.get("use_effective_thickness"),
        },
        {
            "field": "hd_factor",
            "value": data.get("hd_factor"),
        },
        {
            "field": "Hd_factor",
            "value": data.get("Hd_factor"),
        },
        {
            "field": "kappa_value",
            "value": data.get("kappa_value"),
        },
    ]
    return pd.DataFrame(rows)


def physics_payload_meta_units_frame(
    payload: PhysicsPayloadMetaLike,
) -> pd.DataFrame:
    """Return the units block as a tidy frame."""
    data = _as_payload(payload)
    units = data.get("units", {}) or {}
    rows = [
        {
            "quantity": str(key),
            "unit": value,
        }
        for key, value in units.items()
    ]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("quantity")
    return frame.reset_index(drop=True)


def physics_payload_meta_metrics_frame(
    payload: PhysicsPayloadMetaLike,
) -> pd.DataFrame:
    """Return compact payload metrics as a tidy frame."""
    data = _as_payload(payload)
    metrics = _numeric_subset(
        data.get("payload_metrics", {}),
        keys=_PAYLOAD_METRIC_KEYS,
    )
    return metrics_frame(
        metrics,
        section="payload_metrics",
    )


def summarize_physics_payload_meta(
    payload: PhysicsPayloadMetaLike,
) -> dict[str, Any]:
    """
    Build a compact semantic summary for inspection.
    """
    data = _as_payload(payload)
    units = data.get("units", {}) or {}
    payload_metrics = data.get("payload_metrics", {}) or {}
    pde_modes = list(data.get("pde_modes_active", []) or [])

    core_scalars = _numeric_subset(data, keys=_CORE_SCALARS)
    metric_scalars = _numeric_subset(
        payload_metrics,
        keys=_PAYLOAD_METRIC_KEYS,
    )

    summary_map = {
        "brief": {
            "kind": "physics_payload_meta",
            "created_utc": data.get("created_utc"),
            "saved_utc": data.get("saved_utc"),
            "model_name": data.get("model_name"),
            "city": data.get("city"),
            "split": data.get("split"),
            "pde_modes_active": pde_modes,
            "n_pde_modes": len(pde_modes),
        },
        "conventions": {
            "time_units": _string_or_none(
                data.get("time_units")
            ),
            "time_coord_units": _string_or_none(
                data.get("time_coord_units")
            ),
            "gwl_kind": _string_or_none(data.get("gwl_kind")),
            "gwl_sign": _string_or_none(data.get("gwl_sign")),
            "use_head_proxy": bool(
                data.get("use_head_proxy")
            ),
            "kappa_mode": _string_or_none(
                data.get("kappa_mode")
            ),
        },
        "closure": {
            "tau_prior_definition": _string_or_none(
                data.get("tau_prior_definition")
            ),
            "tau_prior_human_name": _string_or_none(
                data.get("tau_prior_human_name")
            ),
            "tau_prior_source": _string_or_none(
                data.get("tau_prior_source")
            ),
            "tau_closure_formula": _string_or_none(
                data.get("tau_closure_formula")
            ),
            "use_effective_h": bool(
                data.get("use_effective_h")
            ),
            "use_effective_thickness": bool(
                data.get("use_effective_thickness")
            ),
            "hd_factor": _try_float(data.get("hd_factor")),
            "Hd_factor": _try_float(data.get("Hd_factor")),
            "kappa_value": _try_float(
                data.get("kappa_value")
            ),
        },
        "payload_metrics": metric_scalars,
        "units": {
            "n_units": len(units),
            "quantities": sorted(str(k) for k in units),
            "tau_unit": units.get("tau"),
            "tau_prior_unit": units.get("tau_prior"),
            "K_unit": units.get("K"),
            "Ss_unit": units.get("Ss"),
        },
        "checks": {
            "has_model_name": bool(
                _string_or_none(data.get("model_name"))
            ),
            "has_city": bool(
                _string_or_none(data.get("city"))
            ),
            "has_split": bool(
                _string_or_none(data.get("split"))
            ),
            "has_pde_modes": bool(pde_modes),
            "has_units": bool(units),
            "has_payload_metrics": bool(payload_metrics),
            "has_tau_prior_definition": bool(
                _string_or_none(
                    data.get("tau_prior_definition")
                )
            ),
            "has_tau_closure_formula": bool(
                _string_or_none(
                    data.get("tau_closure_formula")
                )
            ),
            "time_units_present": bool(
                _string_or_none(data.get("time_units"))
            ),
            "time_coord_units_present": bool(
                _string_or_none(data.get("time_coord_units"))
            ),
            "kappa_value_present": (
                _try_float(data.get("kappa_value"))
                is not None
            ),
            "lambda_offset_present": (
                _try_float(data.get("lambda_offset"))
                is not None
            ),
            "hd_factor_present": (
                _try_float(data.get("hd_factor")) is not None
            ),
            "payload_eps_prior_present": (
                _try_float(
                    payload_metrics.get("eps_prior_rms")
                )
                is not None
            ),
            "payload_eps_cons_present": (
                _try_float(
                    payload_metrics.get("eps_cons_scaled_rms")
                )
                is not None
            ),
            "uses_head_proxy": bool(
                data.get("use_head_proxy")
            ),
        },
        "core_scalars": core_scalars,
    }
    return summary_map


def plot_physics_payload_meta_core_scalars(
    payload: PhysicsPayloadMetaLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Physics payload meta: core scalars",
) -> plt.Axes:
    """Plot selected top-level numeric metadata values."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.8, 4.5))

    data = _as_payload(payload)
    metrics = _numeric_subset(data, keys=_CORE_SCALARS)
    plot_metric_bars(
        ax,
        metrics,
        title=title,
        sort_by_value=True,
        absolute=True,
    )
    return ax


def plot_physics_payload_meta_payload_metrics(
    payload: PhysicsPayloadMetaLike,
    *,
    ax: plt.Axes | None = None,
    title: str = ("Physics payload meta: payload metrics"),
) -> plt.Axes:
    """Plot compact payload-level metrics."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.8, 4.5))

    data = _as_payload(payload)
    metrics = _numeric_subset(
        data.get("payload_metrics", {}),
        keys=_PAYLOAD_METRIC_KEYS,
    )
    plot_metric_bars(
        ax,
        metrics,
        title=title,
        sort_by_value=True,
        absolute=True,
    )
    return ax


def plot_physics_payload_meta_boolean_summary(
    payload: PhysicsPayloadMetaLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Physics payload meta: checks",
) -> plt.Axes:
    """Plot boolean inspection checks."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8.2, 4.8))

    checks = summarize_physics_payload_meta(payload)["checks"]
    plot_boolean_checks(
        ax,
        checks,
        title=title,
    )
    return ax


def inspect_physics_payload_meta(
    payload: PhysicsPayloadMetaLike,
) -> dict[str, Any]:
    """
    Return a structured inspection bundle.
    """
    data = _as_payload(payload)
    summary_map = summarize_physics_payload_meta(data)

    return {
        "summary": summary_map,
        "identity_frame": (
            physics_payload_meta_identity_frame(data)
        ),
        "closure_frame": (
            physics_payload_meta_closure_frame(data)
        ),
        "units_frame": physics_payload_meta_units_frame(data),
        "metrics_frame": (
            physics_payload_meta_metrics_frame(data)
        ),
    }
