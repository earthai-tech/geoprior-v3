# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
Training-summary generation and inspection helpers.

This module focuses on the Stage-2 training summary
artifact. It provides:

- robust loading,
- reproducible demo-summary generation,
- compact tabular summaries,
- quick visual inspection helpers.

The functions are designed for two common uses:

1. Sphinx-Gallery examples that need a realistic
   training summary without rerunning training.
2. Real workflow inspection when a user wants to
   review best/final metrics, compile settings,
   and initialization choices at a glance.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import (
    ArtifactRecord,
    as_path,
    clone_artifact,
    deep_update,
    flatten_dict,
    load_artifact,
    metrics_frame,
    nested_get,
    plot_boolean_checks,
    plot_metric_bars,
    read_json,
    write_json,
)

PathLike = str | Path
TrainingSummaryLike = (
    ArtifactRecord | Mapping[str, Any] | str | Path
)

__all__ = [
    "default_training_summary_payload",
    "generate_training_summary",
    "inspect_training_summary",
    "load_training_summary",
    "plot_training_best_metrics",
    "plot_training_boolean_summary",
    "plot_training_final_metrics",
    "plot_training_loss_family",
    "plot_training_metric_deltas",
    "training_compile_frame",
    "training_env_frame",
    "training_hp_frame",
    "training_metrics_frame",
    "training_paths_frame",
    "summarize_training_summary",
]

_CORE_METRICS = [
    "loss",
    "data_loss",
    "physics_loss",
    "physics_loss_scaled",
    "subs_pred_mae_q50",
    "gwl_pred_mae_q50",
    "subs_pred_coverage80",
    "subs_pred_sharpness80",
    "epsilon_prior",
    "epsilon_cons",
    "epsilon_gw",
    "lambda_offset",
    "physics_mult",
]

_LOSS_FAMILY = [
    "loss",
    "data_loss",
    "physics_loss",
    "physics_loss_scaled",
    "consolidation_loss",
    "gw_flow_loss",
    "prior_loss",
    "smooth_loss",
    "mv_prior_loss",
    "bounds_loss",
    "q_reg_loss",
]


def _as_payload(
    summary: TrainingSummaryLike,
) -> dict[str, Any]:
    """Return a plain training-summary payload."""
    if isinstance(summary, ArtifactRecord):
        return dict(summary.payload)

    if isinstance(summary, Mapping):
        return dict(summary)

    payload = read_json(summary)
    return dict(payload)


def _default_metric_block(
    *,
    val_scale: float = 0.95,
) -> dict[str, float]:
    """Build a realistic core metric block."""
    base = {
        "bounds_loss": 1.0e-12,
        "consolidation_loss": 1.7e-5,
        "data_loss": 0.0568,
        "epsilon_cons": 5.8e-3,
        "epsilon_cons_raw": 4.1e-9,
        "epsilon_gw": 5.5e-7,
        "epsilon_gw_raw": 4.4e-13,
        "epsilon_prior": 7.7e-4,
        "gw_flow_loss": 1.9e-13,
        "gwl_pred_mae_q50": 0.240,
        "gwl_pred_mse_q50": 0.0777,
        "lambda_offset": 1.0,
        "loss": 0.0568,
        "mv_prior_loss": 9.0e-7,
        "physics_loss": 1.71e-5,
        "physics_loss_scaled": 1.71e-5,
        "physics_mult": 1.0,
        "prior_loss": 1.44e-7,
        "q_gate": 1.0,
        "q_reg_loss": 0.0,
        "q_rms": 4.6e-13,
        "smooth_loss": 0.0,
        "subs_pred_coverage80": 0.8000,
        "subs_pred_mae_q50": 0.0101,
        "subs_pred_mse_q50": 2.72e-4,
        "subs_pred_sharpness80": 0.0307,
        "subs_resid_gate": 1.0,
        "total_loss": 0.0568,
    }

    val = {}
    for key, value in base.items():
        if key in {
            "lambda_offset",
            "physics_mult",
            "q_gate",
            "subs_resid_gate",
        }:
            val[f"val_{key}"] = float(value)
        elif key.endswith("coverage80"):
            val[f"val_{key}"] = float(
                min(1.0, max(0.0, value * 1.02))
            )
        else:
            val[f"val_{key}"] = float(value) * float(
                val_scale
            )

    out = dict(base)
    out.update(val)
    return out


def _split_metric_name(
    name: str,
) -> tuple[str, str]:
    """Return ``(split, metric_name)`` from a metric key."""
    text = str(name)
    if text.startswith("val_"):
        return "val", text[4:]
    return "train", text


def _filter_metrics(
    mapping: dict[str, Any] | None,
    *,
    split: str = "all",
    keys: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Filter scalar metrics by split and optional key set."""
    split_mode = str(split).strip().lower()
    keep = None if keys is None else {str(k) for k in keys}

    out: dict[str, float] = {}
    for name, value in (mapping or {}).items():
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            continue

        metric_split, metric_name = _split_metric_name(name)
        if split_mode != "all" and metric_split != split_mode:
            continue
        if keep is not None and metric_name not in keep:
            continue
        out[str(metric_name)] = float(value)
    return out


def _delta_metrics(
    best: dict[str, Any] | None,
    final: dict[str, Any] | None,
    *,
    split: str = "train",
    keys: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Return ``final - best`` deltas for aligned metrics."""
    best_map = _filter_metrics(
        best,
        split=split,
        keys=keys,
    )
    final_map = _filter_metrics(
        final,
        split=split,
        keys=keys,
    )

    common = sorted(set(best_map) & set(final_map))
    return {
        name: float(final_map[name] - best_map[name])
        for name in common
    }


def default_training_summary_payload(
    *,
    city: str = "demo_city",
    model: str = "GeoPriorSubsNet",
    horizon: int = 3,
    best_epoch: int = 17,
    timestamp: str = "20260222-211635",
    optimizer: str = "Adam",
    learning_rate: float = 1e-3,
    time_steps: int = 5,
    pde_mode: str = "on",
    offset_mode: str = "mul",
    quantiles: list[float] | None = None,
    attention_levels: list[str] | None = None,
    coords_normalized: bool = True,
    coord_ranges: dict[str, float] | None = None,
    run_dir: str = "results/demo_run/train_20260222-211635",
) -> dict[str, Any]:
    """
    Build a realistic default training-summary payload.

    The payload is template-based. It is not meant to
    reproduce the full training loop. Instead, it creates
    a stable and inspectable summary artifact with the
    same broad structure as the real training summary.
    """
    q = list(quantiles or [0.1, 0.5, 0.9])
    attn = list(
        attention_levels
        or ["cross", "hierarchical", "memory"]
    )
    coord_ranges = dict(
        coord_ranges or {"t": 7.0, "x": 44447.0, "y": 39275.0}
    )

    metrics_best = _default_metric_block(val_scale=0.87)
    metrics_final = _default_metric_block(val_scale=0.90)
    metrics_final["loss"] = 0.05445
    metrics_final["data_loss"] = 0.05441
    metrics_final["physics_loss"] = 3.51e-5
    metrics_final["physics_loss_scaled"] = 3.51e-5
    metrics_final["epsilon_prior"] = 5.89e-4
    metrics_final["val_loss"] = 0.04951
    metrics_final["val_data_loss"] = 0.04948
    metrics_final["val_physics_loss"] = 6.9e-5
    metrics_final["val_physics_loss_scaled"] = 6.9e-5

    payload = {
        "timestamp": str(timestamp),
        "city": city,
        "model": model,
        "horizon": int(horizon),
        "best_epoch": int(best_epoch),
        "metrics_at_best": metrics_best,
        "final_epoch_metrics": metrics_final,
        "env": {
            "python": "3.10.19",
            "tensorflow": "2.20.0",
            "numpy": "2.0.2",
            "platform": "Windows-10-demo",
            "device": {
                "has_tf": True,
                "device_mode_requested": "auto",
                "device_mode_effective": "cpu",
                "num_cpus": 1,
                "num_gpus": 0,
                "visible_gpus": [],
                "intra_threads": None,
                "inter_threads": None,
                "gpu_memory_growth": None,
                "gpu_memory_limit_mb": None,
            },
        },
        "compile": {
            "optimizer": str(optimizer),
            "learning_rate": float(learning_rate),
            "loss_weights": {
                "subs_pred": 1.0,
                "gwl_pred": 0.8,
            },
            "metrics": {
                "subs_pred": [
                    "MAEQ50",
                    "MSEQ50",
                    "Coverage80",
                    "Sharpness80",
                ],
                "gwl_pred": ["MAEQ50", "MSEQ50"],
            },
            "physics_loss_weights": {
                "lambda_cons": 1.0,
                "lambda_gw": 0.1,
                "lambda_prior": 0.2,
                "lambda_smooth": 0.01,
                "lambda_bounds": 0.05,
                "lambda_mv": 0.01,
                "mv_lr_mult": 1.0,
                "lambda_offset": 1.0,
                "kappa_lr_mult": 5.0,
                "lambda_q": 5.0e-4,
            },
            "lambda_offset": 1.0,
        },
        "hp_init": {
            "quantiles": q,
            "subs_weights": {
                "0.1": 3.0,
                "0.5": 1.0,
                "0.9": 3.0,
            },
            "gwl_weights": {
                "0.1": 1.5,
                "0.5": 1.0,
                "0.9": 1.5,
            },
            "attention_levels": attn,
            "pde_mode": str(pde_mode),
            "time_steps": int(time_steps),
            "use_batch_norm": False,
            "use_vsn": True,
            "vsn_units": 32,
            "mode": "tft_like",
            "model_init_params": {
                "embed_dim": 32,
                "hidden_units": 64,
                "lstm_units": 64,
                "attention_units": 64,
                "num_heads": 2,
                "dropout_rate": 0.1,
                "memory_size": 50,
                "scales": [1, 2],
                "use_residuals": True,
                "use_batch_norm": False,
                "use_vsn": True,
                "vsn_units": 32,
                "mode": "tft_like",
                "attention_levels": attn,
                "scale_pde_residuals": True,
                "scaling_kwargs": {
                    "time_units": "year",
                    "coords_normalized": bool(
                        coords_normalized
                    ),
                    "coord_ranges": coord_ranges,
                    "coord_order": ["t", "x", "y"],
                    "gwl_kind": "depth_bgs",
                    "gwl_sign": "down_positive",
                    "use_head_proxy": True,
                    "Q_kind": "per_volume",
                },
                "bounds_mode": "soft",
                "mv": {
                    "type": "LearnableMV",
                    "initial_value": 1e-7,
                },
                "kappa": {
                    "type": "LearnableKappa",
                    "initial_value": 1.0,
                },
                "gamma_w": {
                    "type": "FixedGammaW",
                    "value": 9810.0,
                },
                "h_ref": {
                    "type": "FixedHRef",
                    "value": 0.0,
                },
                "kappa_mode": "kb",
                "use_effective_h": True,
                "hd_factor": 0.6,
                "offset_mode": str(offset_mode),
                "residual_method": "exact",
                "time_units": "year",
            },
            "offset_mode": str(offset_mode),
            "scaling_kwargs": {
                "bounds": {
                    "H_min": 0.1,
                    "H_max": 30.0,
                    "K_min": 1e-12,
                    "K_max": 1e-7,
                },
                "time_units": "year",
                "coords_normalized": bool(coords_normalized),
                "coord_ranges": coord_ranges,
            },
            "identifiability_regime": None,
        },
        "paths": {
            "run_dir": str(run_dir),
            "weights_h5": (
                f"{run_dir}/{city}_{model}_H{horizon}.weights.h5"
            ),
            "arch_json": (
                f"{run_dir}/{city}_{model}_architecture.json"
            ),
            "csv_log": (
                f"{run_dir}/{city}_{model}_train_log.csv"
            ),
            "best_keras": (
                f"{run_dir}/{city}_{model}_H{horizon}_best.keras"
            ),
            "best_weights": (
                f"{run_dir}/{city}_{model}_H{horizon}_best.weights.h5"
            ),
            "model_init_manifest": (
                f"{run_dir}/model_init_manifest.json"
            ),
            "final_keras": (
                f"{run_dir}/{city}_{model}_H{horizon}_final.keras"
            ),
        },
    }
    return payload


def generate_training_summary(
    *,
    output_path: PathLike | None = None,
    template: TrainingSummaryLike | None = None,
    overrides: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | Path:
    """
    Generate a training-summary payload or file.

    Parameters
    ----------
    output_path : path-like, optional
        Destination JSON path. If omitted, the payload
        is returned instead of written.
    template : mapping, ArtifactRecord, or path, optional
        Real or synthetic training-summary template used
        as the generation base.
    overrides : dict, optional
        Nested overrides applied after template/default
        payload creation.
    **kwargs : dict
        Parameters forwarded to
        ``default_training_summary_payload`` when no
        template is given.
    """
    if template is None:
        payload = default_training_summary_payload(**kwargs)
    else:
        payload = clone_artifact(
            _as_payload(template),
            overrides=None,
        )

    if overrides:
        payload = deep_update(payload, overrides)

    if output_path is None:
        return payload

    return write_json(payload, output_path)


def load_training_summary(
    path: PathLike,
) -> ArtifactRecord:
    """
    Load a training-summary artifact.

    Raises
    ------
    ValueError
        If the artifact does not look like a training
        summary payload.
    """
    record = load_artifact(path, kind="training_summary")
    needed = {
        "metrics_at_best",
        "final_epoch_metrics",
        "compile",
        "hp_init",
    }
    if not needed.issubset(record.payload):
        raise ValueError(
            "The file does not contain the expected "
            "training-summary sections."
        )
    return record


def training_metrics_frame(
    summary: TrainingSummaryLike,
    *,
    section: str = "metrics_at_best",
    split: str = "all",
) -> pd.DataFrame:
    """
    Return a tidy frame for train/validation metrics.
    """
    payload = _as_payload(summary)
    src = payload.get(section, {}) or {}

    rows: list[dict[str, Any]] = []
    for name, value in src.items():
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            continue
        metric_split, metric_name = _split_metric_name(name)
        if str(split).strip().lower() != "all":
            if metric_split != str(split).strip().lower():
                continue
        rows.append(
            {
                "section": section,
                "split": metric_split,
                "metric": metric_name,
                "value": float(value),
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["split", "metric"])
    return frame.reset_index(drop=True)


def training_env_frame(
    summary: TrainingSummaryLike,
) -> pd.DataFrame:
    """Return a tidy frame for environment info."""
    payload = _as_payload(summary)
    env = payload.get("env", {}) or {}
    flat = flatten_dict(env)

    rows = []
    for key, value in flat.items():
        rows.append(
            {
                "key": str(key),
                "value": value,
                "is_numeric": isinstance(
                    value,
                    (int, float),
                )
                and not isinstance(value, bool),
            }
        )
    return pd.DataFrame(rows)


def training_compile_frame(
    summary: TrainingSummaryLike,
) -> pd.DataFrame:
    """Return a tidy frame for compile settings."""
    payload = _as_payload(summary)
    compile_cfg = payload.get("compile", {}) or {}
    flat = flatten_dict(compile_cfg)

    rows = []
    for key, value in flat.items():
        rows.append(
            {
                "key": str(key),
                "value": value,
                "is_numeric": isinstance(
                    value,
                    (int, float),
                )
                and not isinstance(value, bool),
            }
        )
    return pd.DataFrame(rows)


def training_hp_frame(
    summary: TrainingSummaryLike,
) -> pd.DataFrame:
    """Return a tidy frame for hp/init settings."""
    payload = _as_payload(summary)
    hp = payload.get("hp_init", {}) or {}
    flat = flatten_dict(hp)

    rows = []
    for key, value in flat.items():
        rows.append(
            {
                "key": str(key),
                "value": value,
                "is_numeric": isinstance(
                    value,
                    (int, float),
                )
                and not isinstance(value, bool),
            }
        )
    return pd.DataFrame(rows)


def training_paths_frame(
    summary: TrainingSummaryLike,
) -> pd.DataFrame:
    """Return a tidy frame for output paths."""
    payload = _as_payload(summary)
    paths = payload.get("paths", {}) or {}
    return pd.DataFrame(
        {
            "key": list(paths.keys()),
            "value": list(paths.values()),
        }
    )


def summarize_training_summary(
    summary: TrainingSummaryLike,
) -> dict[str, Any]:
    """
    Build a compact semantic summary for inspection.
    """
    payload = _as_payload(summary)

    best = payload.get("metrics_at_best", {}) or {}
    final = payload.get("final_epoch_metrics", {}) or {}
    compile_cfg = payload.get("compile", {}) or {}
    hp = payload.get("hp_init", {}) or {}
    paths = payload.get("paths", {}) or {}

    best_train = _filter_metrics(best, split="train")
    best_val = _filter_metrics(best, split="val")
    final_train = _filter_metrics(final, split="train")
    final_val = _filter_metrics(final, split="val")

    saved_model_keys = {
        "best_keras",
        "best_weights",
        "final_keras",
    }

    summary_map = {
        "brief": {
            "kind": "training_summary",
            "city": payload.get("city"),
            "model": payload.get("model"),
            "timestamp": payload.get("timestamp"),
            "horizon": payload.get("horizon"),
            "best_epoch": payload.get("best_epoch"),
        },
        "core_metrics": {
            "best_train_loss": best_train.get("loss"),
            "best_val_loss": best_val.get("loss"),
            "final_train_loss": final_train.get("loss"),
            "final_val_loss": final_val.get("loss"),
            "best_train_subs_mae_q50": best_train.get(
                "subs_pred_mae_q50"
            ),
            "best_val_subs_mae_q50": best_val.get(
                "subs_pred_mae_q50"
            ),
            "best_train_gwl_mae_q50": best_train.get(
                "gwl_pred_mae_q50"
            ),
            "best_val_gwl_mae_q50": best_val.get(
                "gwl_pred_mae_q50"
            ),
            "best_val_coverage80": best_val.get(
                "subs_pred_coverage80"
            ),
            "best_val_sharpness80": best_val.get(
                "subs_pred_sharpness80"
            ),
            "delta_final_minus_best_val_loss": (
                None
                if best_val.get("loss") is None
                or final_val.get("loss") is None
                else float(
                    final_val["loss"] - best_val["loss"]
                )
            ),
        },
        "compile": {
            "optimizer": compile_cfg.get("optimizer"),
            "learning_rate": compile_cfg.get("learning_rate"),
            "lambda_offset": compile_cfg.get("lambda_offset"),
            "loss_weight_keys": list(
                (compile_cfg.get("loss_weights") or {}).keys()
            ),
        },
        "checks": {
            "has_best_metrics": bool(best),
            "has_final_metrics": bool(final),
            "has_validation_metrics": bool(best_val),
            "has_physics_metrics": all(
                key in best_train
                for key in [
                    "physics_loss",
                    "epsilon_prior",
                    "epsilon_cons",
                    "epsilon_gw",
                ]
            ),
            "best_epoch_is_positive": int(
                payload.get("best_epoch", 0) or 0
            )
            >= 0,
            "lambda_offset_stable": (
                best_train.get("lambda_offset")
                == final_train.get("lambda_offset")
            ),
            "quantiles_defined": bool(
                hp.get("quantiles", [])
            ),
            "has_scaling_kwargs": bool(
                nested_get(hp, "scaling_kwargs", default={})
                or nested_get(
                    hp,
                    "model_init_params",
                    "scaling_kwargs",
                    default={},
                )
            ),
            "has_saved_model_paths": saved_model_keys.issubset(
                paths.keys()
            ),
            "has_optimizer": bool(
                compile_cfg.get("optimizer")
            ),
        },
    }
    return summary_map


def plot_training_best_metrics(
    summary: TrainingSummaryLike,
    *,
    split: str = "val",
    keys: list[str] | tuple[str, ...] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    Plot selected metrics from ``metrics_at_best``.
    """
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(8.2, 4.8))

    payload = _as_payload(summary)
    metrics = _filter_metrics(
        payload.get("metrics_at_best", {}),
        split=split,
        keys=keys or _CORE_METRICS,
    )
    plot_metric_bars(
        ax,
        metrics,
        title=title or f"Best metrics ({split})",
        sort_by_value=True,
        top_n=12,
        absolute=True,
    )
    return ax


def plot_training_final_metrics(
    summary: TrainingSummaryLike,
    *,
    split: str = "val",
    keys: list[str] | tuple[str, ...] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    Plot selected metrics from ``final_epoch_metrics``.
    """
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(8.2, 4.8))

    payload = _as_payload(summary)
    metrics = _filter_metrics(
        payload.get("final_epoch_metrics", {}),
        split=split,
        keys=keys or _CORE_METRICS,
    )
    plot_metric_bars(
        ax,
        metrics,
        title=title or f"Final metrics ({split})",
        sort_by_value=True,
        top_n=12,
        absolute=True,
    )
    return ax


def plot_training_metric_deltas(
    summary: TrainingSummaryLike,
    *,
    split: str = "val",
    keys: list[str] | tuple[str, ...] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    Plot ``final - best`` deltas for aligned metrics.
    """
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(8.2, 4.8))

    payload = _as_payload(summary)
    deltas = _delta_metrics(
        payload.get("metrics_at_best", {}),
        payload.get("final_epoch_metrics", {}),
        split=split,
        keys=keys or _CORE_METRICS,
    )

    if not deltas:
        ax.set_title(title or f"Metric deltas ({split})")
        ax.text(
            0.5,
            0.5,
            "No aligned delta metrics",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    ax.barh(list(deltas.keys()), list(deltas.values()))
    ax.set_title(title or f"Final - best ({split})")
    ax.set_xlabel("delta")
    ax.grid(axis="x", alpha=0.25)
    return ax


def plot_training_loss_family(
    summary: TrainingSummaryLike,
    *,
    section: str = "metrics_at_best",
    split: str = "val",
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    Plot the loss-family subset for one metric section.
    """
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(8.2, 4.8))

    payload = _as_payload(summary)
    metrics = _filter_metrics(
        payload.get(section, {}),
        split=split,
        keys=_LOSS_FAMILY,
    )
    plot_metric_bars(
        ax,
        metrics,
        title=title or f"Loss family: {section} ({split})",
        sort_by_value=True,
        top_n=None,
        absolute=True,
    )
    return ax


def plot_training_boolean_summary(
    summary: TrainingSummaryLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Training summary checks",
) -> plt.Axes:
    """Plot semantic pass/fail checks."""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(8.0, 4.6))

    checks = summarize_training_summary(summary)["checks"]
    plot_boolean_checks(ax, checks, title=title)
    return ax


def inspect_training_summary(
    summary: TrainingSummaryLike,
    *,
    output_dir: PathLike | None = None,
    stem: str = "training_summary",
    save_figures: bool = True,
) -> dict[str, Any]:
    """
    Inspect a training summary and optionally save figures.

    Returns
    -------
    dict
        Bundle containing summary, tabular frames, and
        optionally written figure paths.
    """
    payload = _as_payload(summary)
    summary_map = summarize_training_summary(payload)

    bundle: dict[str, Any] = {
        "summary": summary_map,
        "frames": {
            "metrics_at_best": training_metrics_frame(
                payload,
                section="metrics_at_best",
                split="all",
            ),
            "final_epoch_metrics": training_metrics_frame(
                payload,
                section="final_epoch_metrics",
                split="all",
            ),
            "env": training_env_frame(payload),
            "compile": training_compile_frame(payload),
            "hp_init": training_hp_frame(payload),
            "paths": training_paths_frame(payload),
        },
        "figure_paths": {},
    }

    if not (output_dir and save_figures):
        return bundle

    out_dir = as_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plots = {
        f"{stem}_best_val_metrics.png": (
            plot_training_best_metrics,
            {"split": "val"},
        ),
        f"{stem}_final_val_metrics.png": (
            plot_training_final_metrics,
            {"split": "val"},
        ),
        f"{stem}_delta_val_metrics.png": (
            plot_training_metric_deltas,
            {"split": "val"},
        ),
        f"{stem}_best_val_losses.png": (
            plot_training_loss_family,
            {
                "section": "metrics_at_best",
                "split": "val",
            },
        ),
        f"{stem}_checks.png": (
            plot_training_boolean_summary,
            {},
        ),
    }

    for name, (func, kwargs) in plots.items():
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        func(payload, ax=ax, **kwargs)
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        bundle["figure_paths"][name] = str(path)

    return bundle
