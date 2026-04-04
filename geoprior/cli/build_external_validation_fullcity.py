# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>
r"""CLI for building full-city external validation artifacts."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from ..compat import load_inference_model
from ..models import (
    GeoPriorSubsNet,
    PoroElasticSubsNet,
    make_weighted_pinball,
)
from ..params import (
    FixedGammaW,
    FixedHRef,
    LearnableKappa,
    LearnableMV,
)
from ..utils.nat_utils import (
    ensure_config_json,
    get_config_paths,
    make_tf_dataset,
)

ArrayDict = dict[str, np.ndarray]


def _parse_override_value(text: str) -> Any:
    s = str(text).strip()
    if not s:
        return s
    low = s.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def _parse_set_items(
    items: list[str] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(
                f"Each --set must be KEY=VALUE. Got: {item!r}"
            )
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(
                f"Invalid empty key in: {item!r}"
            )
        out[key] = _parse_override_value(value)
    return out


def _install_user_config(
    config_path: str,
    *,
    config_root: str = "nat.com",
) -> str:
    src = Path(config_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(
            f"Config file not found: {src}"
        )

    config_py, config_json = get_config_paths(
        root=config_root
    )
    dst = Path(config_py).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src != dst:
        shutil.copy2(src, dst)

    json_path = Path(config_json)
    if json_path.exists():
        json_path.unlink()

    return str(dst)


def _persist_runtime_overrides(
    overrides: dict[str, Any] | None = None,
    *,
    config_root: str = "nat.com",
) -> dict[str, Any]:
    cfg0, config_json = ensure_config_json(root=config_root)
    cfg = dict(cfg0)

    if not overrides:
        return cfg

    cfg.update(overrides)

    payload: dict[str, Any] = {}
    cfg_json = Path(config_json)
    if cfg_json.exists():
        try:
            payload = json.loads(
                cfg_json.read_text(encoding="utf-8")
            )
        except Exception:
            payload = {}

    payload["city"] = cfg.get("CITY_NAME")
    payload["model"] = cfg.get("MODEL_NAME")
    payload["config"] = cfg
    payload.setdefault("__meta__", {})

    cfg_json.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return cfg


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def load_npz_dict(path: str | Path) -> ArrayDict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_npz_dict(
    path: str | Path,
    data: ArrayDict,
) -> None:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)


def ensure_dir(path: str | Path) -> None:
    Path(path).expanduser().resolve().mkdir(
        parents=True,
        exist_ok=True,
    )


def is_readable_file(path: str | None) -> bool:
    if not isinstance(path, str) or not path:
        return False
    try:
        return os.path.isfile(path) and os.access(
            path,
            os.R_OK,
        )
    except OSError:
        return False


def _cfg_first(
    cfg: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    for key in keys:
        val = cfg.get(key)
        if val is not None:
            return val
    return default


def _resolve_validation_csv(
    explicit: str | None,
    cfg: dict[str, Any],
) -> str | None:
    if explicit and is_readable_file(explicit):
        return str(Path(explicit).expanduser().resolve())

    cand = _cfg_first(
        cfg,
        "EXTERNAL_VALIDATION_CSV",
        "VALIDATION_CSV",
        "BOREHOLE_PUMPING_VALIDATION_CSV",
        "BOREHOLE_VALIDATION_CSV",
    )
    if isinstance(cand, str) and is_readable_file(cand):
        return str(Path(cand).expanduser().resolve())
    return None


def _resolve_manifest_path(
    *,
    manifest: str | None,
    stage1_dir: str | None,
    results_dir: str | None,
    city: str | None,
    model: str | None,
) -> Path:
    candidates: list[Path] = []

    if manifest:
        p = Path(manifest).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"Manifest not found: {p}"
            )
        return p

    if stage1_dir:
        p = (
            Path(stage1_dir).expanduser().resolve()
            / "manifest.json"
        )
        if p.exists():
            return p
        candidates.append(p)

    if results_dir and city and model:
        p = (
            Path(results_dir).expanduser().resolve()
            / f"{str(city).strip().lower()}_{model}_stage1"
            / "manifest.json"
        )
        if p.exists():
            return p
        candidates.append(p)

    if results_dir:
        root = Path(results_dir).expanduser().resolve()
        found = sorted(
            root.glob("*_stage1/manifest.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if len(found) == 1:
            return found[0]
        if found:
            preview = "\n  - ".join(str(p) for p in found[:8])
            raise SystemExit(
                "Multiple Stage-1 manifests found. "
                "Pass --manifest or --stage1-dir.\n"
                f"  - {preview}"
            )

    tried = "\n  - ".join(str(p) for p in candidates)
    msg = (
        "Could not resolve a Stage-1 manifest. "
        "Pass --manifest or --stage1-dir explicitly."
    )
    if tried:
        msg += f"\nTried:\n  - {tried}"
    raise SystemExit(msg)


def split_after_artifacts(path: str) -> str | None:
    norm = path.replace("/", "\\")
    key = "\\artifacts\\"
    low = norm.lower()
    idx = low.find(key)
    if idx < 0:
        return None
    return norm[idx + len(key) :].replace("\\", os.sep)


def resolve_stage1_artifact(
    stage1_manifest_path: str,
    recorded_path: str | None,
    explicit: str | None = None,
) -> str | None:
    if explicit and is_readable_file(explicit):
        return str(Path(explicit).expanduser().resolve())

    if is_readable_file(recorded_path):
        return str(Path(recorded_path).expanduser().resolve())

    if not isinstance(recorded_path, str):
        return None

    stage1_dir = Path(stage1_manifest_path).resolve().parent
    artifacts_dir = stage1_dir / "artifacts"

    cands: list[Path] = []
    rel = split_after_artifacts(recorded_path)
    if rel:
        cands.append(artifacts_dir / rel)

    base = Path(recorded_path).name
    cands.append(stage1_dir / base)
    cands.append(artifacts_dir / base)

    for cand in cands:
        if cand.is_file():
            return str(cand.resolve())

    return None


def resolve_stage1_inputs_paths(
    stage1_manifest_path: str,
    train_inputs: str | None,
    val_inputs: str | None,
    test_inputs: str | None,
    coord_scaler: str | None,
) -> tuple[str, str, str | None, str | None, dict[str, Any]]:
    manifest = read_json(stage1_manifest_path)
    arts = manifest.get("artifacts") or {}
    npz = arts.get("numpy") or {}
    enc = arts.get("encoders") or {}

    train_path = resolve_stage1_artifact(
        stage1_manifest_path,
        npz.get("train_inputs_npz"),
        explicit=train_inputs,
    )
    val_path = resolve_stage1_artifact(
        stage1_manifest_path,
        npz.get("val_inputs_npz"),
        explicit=val_inputs,
    )
    test_path = resolve_stage1_artifact(
        stage1_manifest_path,
        npz.get("test_inputs_npz"),
        explicit=test_inputs,
    )
    coord_path = resolve_stage1_artifact(
        stage1_manifest_path,
        enc.get("coord_scaler"),
        explicit=coord_scaler,
    )

    if not train_path:
        raise FileNotFoundError(
            "Could not resolve Stage-1 train_inputs_npz. "
            "Pass --train-inputs explicitly."
        )
    if not val_path:
        raise FileNotFoundError(
            "Could not resolve Stage-1 val_inputs_npz. "
            "Pass --val-inputs explicitly."
        )

    return (
        train_path,
        val_path,
        test_path,
        coord_path,
        manifest,
    )


def make_full_inputs_npz(
    input_paths: Sequence[str],
    out_npz: str,
) -> str:
    parts = [load_npz_dict(p) for p in input_paths if p]
    if not parts:
        raise ValueError("No input NPZs were provided.")

    key_sets = [set(d.keys()) for d in parts]
    ref = key_sets[0]
    for ks in key_sets[1:]:
        if ks != ref:
            raise KeyError(
                "Split NPZ keys are not aligned: "
                f"{[sorted(k) for k in key_sets]}"
            )

    full: ArrayDict = {}
    for key in sorted(ref):
        full[key] = np.concatenate(
            [d[key] for d in parts],
            axis=0,
        )

    save_npz_dict(out_npz, full)
    return str(Path(out_npz).expanduser().resolve())


def resolve_run_dir(
    stage2_run_dir: str | None,
    stage2_manifest_path: str | None,
) -> str | None:
    if stage2_run_dir and Path(stage2_run_dir).is_dir():
        return str(
            Path(stage2_run_dir).expanduser().resolve()
        )

    if stage2_manifest_path and is_readable_file(
        stage2_manifest_path
    ):
        manifest = read_json(stage2_manifest_path)
        run_dir = (manifest.get("paths") or {}).get("run_dir")
        if (
            isinstance(run_dir, str)
            and Path(run_dir).is_dir()
        ):
            return str(Path(run_dir).expanduser().resolve())

        man_dir = (
            Path(stage2_manifest_path)
            .expanduser()
            .resolve()
            .parent
        )
        if man_dir.is_dir():
            return str(man_dir)

    return None


def _resolve_latest_train_dir(
    stage1_dir: str | None,
) -> str | None:
    if not stage1_dir:
        return None
    root = Path(stage1_dir).expanduser().resolve()
    if not root.is_dir():
        return None
    runs = sorted(
        [p for p in root.glob("train_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        return None
    return str(runs[0])


def _search_run_dir_for_model_files(
    run_dir: str,
) -> dict[str, str | None]:
    out = {
        "final_keras": None,
        "best_keras": None,
        "weights_h5": None,
        "model_init_manifest": None,
    }
    run = Path(run_dir).expanduser().resolve()
    if not run.is_dir():
        return out

    for path in run.iterdir():
        name = path.name.lower()
        if name.endswith("_final.keras") or (
            name.endswith(".keras") and "final" in name
        ):
            out["final_keras"] = out["final_keras"] or str(
                path
            )
        elif name.endswith("_best.keras") or (
            name.endswith(".keras") and "best" in name
        ):
            out["best_keras"] = out["best_keras"] or str(path)
        elif name.endswith(".weights.h5"):
            out["weights_h5"] = out["weights_h5"] or str(path)
        elif name == "model_init_manifest.json":
            out["model_init_manifest"] = out[
                "model_init_manifest"
            ] or str(path)

    return out


def resolve_stage2_bundle(
    model_path: str | None,
    stage2_manifest_path: str | None,
    stage2_run_dir: str | None,
    stage1_dir: str | None = None,
) -> dict[str, str | None]:
    out = {
        "model_path": None,
        "weights_path": None,
        "model_init_manifest": None,
        "run_dir": None,
    }

    if model_path and is_readable_file(model_path):
        out["model_path"] = str(
            Path(model_path).expanduser().resolve()
        )

    run_dir = resolve_run_dir(
        stage2_run_dir,
        stage2_manifest_path,
    ) or _resolve_latest_train_dir(stage1_dir)
    out["run_dir"] = run_dir

    if stage2_manifest_path and is_readable_file(
        stage2_manifest_path
    ):
        manifest = read_json(stage2_manifest_path)
        paths = manifest.get("paths") or {}
        man_dir = (
            Path(stage2_manifest_path)
            .expanduser()
            .resolve()
            .parent
        )

        for key in (
            "final_keras",
            "best_keras",
            "best_weights",
            "weights_h5",
            "model_init_manifest",
        ):
            val = paths.get(key)
            direct = (
                str(val) if isinstance(val, str) else None
            )
            if is_readable_file(direct):
                hit = str(Path(direct).expanduser().resolve())
            elif isinstance(val, str):
                cand = man_dir / Path(val).name
                hit = (
                    str(cand.resolve())
                    if cand.is_file()
                    else None
                )
            else:
                hit = None

            if not hit:
                continue

            if key in {"final_keras", "best_keras"}:
                if out["model_path"] is None:
                    out["model_path"] = hit
            elif key in {"best_weights", "weights_h5"}:
                out["weights_path"] = hit
            elif key == "model_init_manifest":
                out["model_init_manifest"] = hit

    if run_dir:
        found = _search_run_dir_for_model_files(run_dir)
        if out["model_path"] is None:
            out["model_path"] = (
                found["final_keras"] or found["best_keras"]
            )
        out["weights_path"] = (
            out["weights_path"] or found["weights_h5"]
        )
        out["model_init_manifest"] = (
            out["model_init_manifest"]
            or found["model_init_manifest"]
        )

    if out["model_path"] is None:
        raise FileNotFoundError(
            "Could not resolve a Stage-2 .keras model. "
            "Pass --model-path or --stage2-run-dir / "
            "--stage2-manifest."
        )

    return out


def build_custom_objects() -> dict[str, object]:
    return {
        "GeoPriorSubsNet": GeoPriorSubsNet,
        "PoroElasticSubsNet": PoroElasticSubsNet,
        "LearnableMV": LearnableMV,
        "LearnableKappa": LearnableKappa,
        "FixedGammaW": FixedGammaW,
        "FixedHRef": FixedHRef,
        "make_weighted_pinball": make_weighted_pinball,
    }


def _model_class_from_name(name: str | None):
    name = (name or "").strip()
    if name == "PoroElasticSubsNet":
        return PoroElasticSubsNet
    return GeoPriorSubsNet


def _builder_from_init_manifest(
    model_init_manifest_path: str,
):
    init_m = read_json(model_init_manifest_path)
    dims = init_m.get("dims") or {}
    cfg = init_m.get("config") or {}
    gp = cfg.get("geoprior") or {}
    model_cls = _model_class_from_name(
        init_m.get("model_class")
    )

    def builder(_manifest: dict[str, Any]):
        return model_cls(
            static_input_dim=int(dims["static_input_dim"]),
            dynamic_input_dim=int(dims["dynamic_input_dim"]),
            future_input_dim=int(dims["future_input_dim"]),
            output_subsidence_dim=int(
                dims["output_subsidence_dim"]
            ),
            output_gwl_dim=int(dims["output_gwl_dim"]),
            forecast_horizon=int(dims["forecast_horizon"]),
            quantiles=cfg.get("quantiles"),
            pde_mode=cfg.get("pde_mode", "off"),
            mode=cfg.get("mode", "tft_like"),
            time_units=cfg.get("time_units", "year"),
            embed_dim=int(cfg.get("embed_dim", 32)),
            hidden_units=int(cfg.get("hidden_units", 64)),
            lstm_units=int(cfg.get("lstm_units", 64)),
            attention_units=int(
                cfg.get("attention_units", 64)
            ),
            num_heads=int(cfg.get("num_heads", 2)),
            dropout_rate=float(cfg.get("dropout_rate", 0.1)),
            memory_size=int(cfg.get("memory_size", 50)),
            scales=list(cfg.get("scales") or [1, 2]),
            use_residuals=bool(
                cfg.get("use_residuals", True)
            ),
            use_batch_norm=bool(
                cfg.get("use_batch_norm", False)
            ),
            use_vsn=bool(cfg.get("use_vsn", True)),
            vsn_units=int(cfg.get("vsn_units", 32))
            if cfg.get("vsn_units") is not None
            else None,
            mv=LearnableMV(
                initial_value=float(gp.get("init_mv", 1e-7))
            ),
            kappa=LearnableKappa(
                initial_value=float(gp.get("init_kappa", 1.0))
            ),
            gamma_w=FixedGammaW(
                value=float(gp.get("gamma_w", 9810.0))
            ),
            h_ref=FixedHRef(
                value=float(gp.get("h_ref_value", 0.0)),
                mode=gp.get("h_ref_mode"),
            ),
            kappa_mode=gp.get("kappa_mode", "bar"),
            use_effective_h=bool(
                gp.get("use_effective_h", True)
            ),
            hd_factor=float(gp.get("hd_factor", 0.6)),
            offset_mode=gp.get("offset_mode", "mul"),
            scaling_kwargs=dict(
                cfg.get("scaling_kwargs") or {}
            ),
            verbose=0,
        )

    return builder


def load_inference_bundle(
    bundle: dict[str, str | None],
    build_inputs: dict[str, Any],
):
    custom_objects = build_custom_objects()
    model_path = bundle["model_path"]
    weights_path = bundle["weights_path"]
    model_init_manifest = bundle["model_init_manifest"]

    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
        )
        print(f"[OK] Loaded .keras model -> {model_path}")
        return model
    except Exception as exc:
        print(f"[Warn] Direct .keras load failed: {exc}")

    if weights_path and model_init_manifest:
        builder = _builder_from_init_manifest(
            model_init_manifest
        )
        model = load_inference_model(
            keras_path=model_path,
            weights_path=weights_path,
            manifest_path=model_init_manifest,
            custom_objects=custom_objects,
            compile=False,
            builder=builder,
            build_inputs=build_inputs,
            prefer_full_model=True,
            log_fn=print,
            use_in_memory_model=False,
        )
        print("[OK] Loaded inference bundle via compat.")
        return model

    raise RuntimeError(
        "Could not load inference bundle. Provide a "
        "readable .keras model, or a valid bundle "
        "(weights + model_init_manifest)."
    )


def get_dataset_mode(
    stage1_manifest: dict[str, Any],
) -> str:
    cfg = stage1_manifest.get("config") or {}
    nested = cfg.get("model") or {}
    return (
        cfg.get("MODE")
        or cfg.get("mode")
        or nested.get("mode")
        or "tft_like"
    )


def get_forecast_horizon(
    stage1_manifest: dict[str, Any],
    fallback_h: int,
) -> int:
    cfg = stage1_manifest.get("config") or {}
    nested = cfg.get("model") or {}
    val = (
        cfg.get("FORECAST_HORIZON_YEARS")
        or cfg.get("forecast_horizon_years")
        or nested.get("forecast_horizon_years")
        or fallback_h
    )
    return int(val)


def get_feature_names(
    stage1_manifest: dict[str, Any],
) -> tuple[list[str], list[str]]:
    cfg = stage1_manifest.get("config") or {}
    features = cfg.get("features") or {}
    dyn = list(features.get("dynamic") or [])
    fut = list(features.get("future") or [])
    return dyn, fut


def export_fullcity_payload(
    stage1_manifest: dict[str, Any],
    full_inputs_npz: str,
    bundle: dict[str, str | None],
    out_payload: str,
    batch_size: int,
    metadata: dict[str, Any] | None = None,
) -> str:
    x_full = load_npz_dict(full_inputs_npz)
    n = int(x_full["coords"].shape[0])
    horizon = int(x_full["coords"].shape[1])

    dyn_names, fut_names = get_feature_names(stage1_manifest)
    mode = get_dataset_mode(stage1_manifest)
    forecast_h = get_forecast_horizon(
        stage1_manifest,
        fallback_h=horizon,
    )

    y_dummy = {
        "subs_pred": np.zeros(
            (n, horizon, 1),
            dtype=np.float32,
        ),
        "gwl_pred": np.zeros(
            (n, horizon, 1),
            dtype=np.float32,
        ),
    }

    ds_full = make_tf_dataset(
        x_full,
        y_dummy,
        batch_size=int(batch_size),
        shuffle=False,
        mode=mode,
        forecast_horizon=forecast_h,
        check_npz_finite=True,
        check_finite=True,
        dynamic_feature_names=dyn_names,
        future_feature_names=fut_names,
    )

    build_inputs = None
    for xb, _ in ds_full.take(1):
        build_inputs = xb
        break
    if build_inputs is None:
        raise RuntimeError(
            "Could not get a build batch from ds_full."
        )

    model = load_inference_bundle(
        bundle,
        build_inputs,
    )

    payload_meta = {
        "split": "full_city_union",
        "source_inputs_npz": str(
            Path(full_inputs_npz).expanduser().resolve()
        ),
    }
    if metadata:
        payload_meta.update(metadata)

    payload = model.export_physics_payload(
        ds_full,
        max_batches=None,
        save_path=out_payload,
        format="npz",
        overwrite=True,
        metadata=payload_meta,
    )

    if not Path(out_payload).is_file():
        raise RuntimeError(
            "Payload export reported success but the "
            "file was not created."
        )

    n_payload = len(np.asarray(payload["K"]).reshape(-1))
    print(
        f"[OK] Exported payload -> {out_payload} "
        f"(rows={n_payload})"
    )
    return str(Path(out_payload).expanduser().resolve())


def inverse_txy(
    coords_bh3: np.ndarray,
    coord_scaler_path: str | None,
) -> np.ndarray:
    first = np.asarray(coords_bh3[:, 0, :], dtype=float)
    if not coord_scaler_path:
        return first
    scaler = joblib.load(coord_scaler_path)
    return scaler.inverse_transform(first)


def reduce_horizon(
    arr: np.ndarray,
    n_seq: int,
    horizon: int,
    how: str,
) -> np.ndarray:
    x = np.asarray(arr, dtype=float)

    if x.size == n_seq:
        x = x.reshape(n_seq, 1)
    elif x.size == n_seq * horizon:
        x = x.reshape(n_seq, horizon)
    else:
        raise ValueError(
            "Cannot align array with sequences. "
            f"size={x.size}, n_seq={n_seq}, horizon={horizon}"
        )

    how = str(how).strip().lower()
    if how == "first":
        return x[:, 0]
    if how == "mean":
        return np.mean(x, axis=1)
    if how == "median":
        return np.median(x, axis=1)

    raise ValueError(
        "horizon reducer must be one of "
        "{'first','mean','median'}"
    )


def build_pixel_table(
    inputs_npz: str,
    payload_npz: str,
    coord_scaler_path: str | None,
    horizon_reducer: str,
    site_reducer: str,
) -> pd.DataFrame:
    x_in = load_npz_dict(inputs_npz)
    payload = load_npz_dict(payload_npz)

    coords = np.asarray(x_in["coords"], dtype=float)
    n_seq, horizon, _ = coords.shape

    txy = inverse_txy(coords, coord_scaler_path)
    x = txy[:, 1]
    y = txy[:, 2]

    h_eff = reduce_horizon(
        x_in["H_field"],
        n_seq=n_seq,
        horizon=horizon,
        how=horizon_reducer,
    )
    k_vals = reduce_horizon(
        payload["K"],
        n_seq=n_seq,
        horizon=horizon,
        how=horizon_reducer,
    )
    hd_vals = reduce_horizon(
        payload["Hd"],
        n_seq=n_seq,
        horizon=horizon,
        how=horizon_reducer,
    )

    h_payload = None
    if "H" in payload:
        h_payload = reduce_horizon(
            payload["H"],
            n_seq=n_seq,
            horizon=horizon,
            how=horizon_reducer,
        )

    df = pd.DataFrame(
        {
            "seq_idx": np.arange(n_seq, dtype=int),
            "x": x,
            "y": y,
            "H_eff_input_m": h_eff,
            "K_mps": k_vals,
            "Hd_m": hd_vals,
        }
    )
    if h_payload is not None:
        df["H_payload_m"] = h_payload

    num_cols = ["H_eff_input_m", "K_mps", "Hd_m"]
    if h_payload is not None:
        num_cols.append("H_payload_m")

    agg_fun = (
        "mean"
        if str(site_reducer).lower() == "mean"
        else "median"
    )
    pix = (
        df.groupby(["x", "y"], as_index=False)[num_cols]
        .agg(agg_fun)
        .reset_index(drop=True)
    )
    pix["pixel_idx"] = np.arange(len(pix), dtype=int)
    return pix


def nearest_match(
    pixels: pd.DataFrame,
    sx: float,
    sy: float,
    allow_swapped_xy: bool = True,
) -> tuple[str, float, float, float, int]:
    px = pixels["x"].to_numpy(dtype=float)
    py = pixels["y"].to_numpy(dtype=float)

    d_dir = np.sqrt((px - sx) ** 2 + (py - sy) ** 2)
    i_dir = int(np.argmin(d_dir))

    if not allow_swapped_xy:
        row = pixels.iloc[i_dir]
        return (
            "direct_xy",
            float(row["x"]),
            float(row["y"]),
            float(d_dir[i_dir]),
            int(row["pixel_idx"]),
        )

    d_swp = np.sqrt((px - sy) ** 2 + (py - sx) ** 2)
    i_swp = int(np.argmin(d_swp))

    if float(d_swp[i_swp]) < float(d_dir[i_dir]):
        idx = i_swp
        mode = "swapped_xy"
        dist = float(d_swp[idx])
    else:
        idx = i_dir
        mode = "direct_xy"
        dist = float(d_dir[idx])

    row = pixels.iloc[idx]
    return (
        mode,
        float(row["x"]),
        float(row["y"]),
        dist,
        int(row["pixel_idx"]),
    )


def spearman_rho(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size != y.size or x.size < 2:
        return float("nan")

    xr = (
        pd.Series(x)
        .rank(method="average")
        .to_numpy(dtype=float)
    )
    yr = (
        pd.Series(y)
        .rank(method="average")
        .to_numpy(dtype=float)
    )

    if np.ptp(xr) == 0.0 or np.ptp(yr) == 0.0:
        return float("nan")

    xs = xr - xr.mean()
    ys = yr - yr.mean()
    den = math.sqrt(float(np.sum(xs * xs) * np.sum(ys * ys)))
    if den <= 0.0:
        return float("nan")
    return float(np.sum(xs * ys) / den)


def mae(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return float(np.mean(np.abs(x[mask] - y[mask])))


def median_bias(
    obs: np.ndarray,
    pred: np.ndarray,
) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    return float(np.median(pred[mask] - obs[mask]))


def validate_site_matches(
    site_df: pd.DataFrame,
    max_distance_m: float,
    min_unique_pixels: int,
) -> None:
    n_unique = int(site_df["pixel_idx"].nunique())
    max_dist = float(site_df["match_distance_m"].max())

    if n_unique < int(min_unique_pixels):
        raise RuntimeError(
            "Too few unique matched pixels for site "
            f"validation (unique={n_unique}). This "
            "usually means CRS/order mismatch."
        )

    if max_dist > float(max_distance_m):
        raise RuntimeError(
            "Site-to-pixel matching distance is too large "
            f"(max={max_dist:.1f} m > "
            f"{float(max_distance_m):.1f} m)."
        )


def compute_external_metrics(
    validation_csv: str,
    full_inputs_npz: str,
    full_payload_npz: str,
    coord_scaler_path: str | None,
    outdir: str,
    x_col: str,
    y_col: str,
    productivity_col: str,
    thickness_col: str,
    horizon_reducer: str,
    site_reducer: str,
    max_match_distance_m: float,
    min_unique_pixels: int,
    allow_swapped_xy: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_dir(outdir)

    pixels = build_pixel_table(
        full_inputs_npz,
        full_payload_npz,
        coord_scaler_path,
        horizon_reducer,
        site_reducer,
    )

    sites = pd.read_csv(validation_csv)
    for col in (
        x_col,
        y_col,
        productivity_col,
        thickness_col,
    ):
        if col not in sites.columns:
            raise KeyError(
                f"Required column {col!r} was not found "
                "in validation CSV."
            )

    rows = []
    for _, rec in sites.iterrows():
        mode, px, py, dist, pidx = nearest_match(
            pixels,
            sx=float(rec[x_col]),
            sy=float(rec[y_col]),
            allow_swapped_xy=allow_swapped_xy,
        )
        pix = pixels.loc[pixels["pixel_idx"] == pidx].iloc[0]

        out = dict(rec)
        out.update(
            {
                "match_mode": mode,
                "matched_pixel_x": px,
                "matched_pixel_y": py,
                "match_distance_m": dist,
                "pixel_idx": pidx,
                "model_H_eff_m": float(pix["H_eff_input_m"]),
                "model_K_mps": float(pix["K_mps"]),
                "model_Hd_m": float(pix["Hd_m"]),
            }
        )
        if "H_payload_m" in pix.index:
            out["payload_H_m"] = float(pix["H_payload_m"])
        rows.append(out)

    site_df = pd.DataFrame(rows)
    site_df["dx_m"] = (
        site_df["matched_pixel_x"] - site_df[x_col]
    )
    site_df["dy_m"] = (
        site_df["matched_pixel_y"] - site_df[y_col]
    )

    validate_site_matches(
        site_df,
        max_match_distance_m,
        min_unique_pixels,
    )

    obs_h = site_df[thickness_col].to_numpy(dtype=float)
    mod_h = site_df["model_H_eff_m"].to_numpy(dtype=float)
    prod = site_df[productivity_col].to_numpy(dtype=float)
    mod_k = site_df["model_K_mps"].to_numpy(dtype=float)

    metrics = {
        "n_sites": int(len(site_df)),
        "borehole_vs_H_eff": {
            "thickness_column": thickness_col,
            "spearman_rho": spearman_rho(obs_h, mod_h),
            "mae_m": mae(obs_h, mod_h),
            "median_bias_m": median_bias(obs_h, mod_h),
        },
        "pumping_vs_K": {
            "productivity_column": productivity_col,
            "spearman_rho": spearman_rho(prod, mod_k),
        },
        "reducers": {
            "horizon_reducer": horizon_reducer,
            "site_reducer": site_reducer,
        },
        "match_summary": {
            "unique_pixels": int(
                site_df["pixel_idx"].nunique()
            ),
            "min_distance_m": float(
                site_df["match_distance_m"].min()
            ),
            "median_distance_m": float(
                site_df["match_distance_m"].median()
            ),
            "max_distance_m": float(
                site_df["match_distance_m"].max()
            ),
        },
        "files": {
            "validation_csv": str(
                Path(validation_csv).expanduser().resolve()
            ),
            "full_inputs_npz": str(
                Path(full_inputs_npz).expanduser().resolve()
            ),
            "full_payload_npz": str(
                Path(full_payload_npz).expanduser().resolve()
            ),
            "coord_scaler": (
                str(
                    Path(coord_scaler_path)
                    .expanduser()
                    .resolve()
                )
                if coord_scaler_path
                else None
            ),
        },
        "site_columns": {
            "x_col": x_col,
            "y_col": y_col,
            "thickness_col": thickness_col,
        },
    }

    site_csv = (
        Path(outdir)
        / "site_level_external_validation_fullcity.csv"
    )
    metrics_json = (
        Path(outdir)
        / "external_validation_metrics_fullcity.json"
    )

    site_df.to_csv(site_csv, index=False)
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return site_df, metrics


def _default_full_inputs_path(
    outdir: str,
    stage1_dir: str,
) -> str:
    art = Path(stage1_dir) / "artifacts" / "full_inputs.npz"
    if art.is_file():
        return str(art.resolve())
    return str((Path(outdir) / "full_inputs.npz").resolve())


def _default_payload_path(
    outdir: str,
    city: str | None,
    bundle: dict[str, str | None] | None,
) -> str:
    stem = (
        f"{str(city).strip().lower()}_phys_payload_fullcity.npz"
        if city
        else "physics_payload_fullcity.npz"
    )
    run_dir = (bundle or {}).get("run_dir")
    if run_dir:
        return str((Path(run_dir) / stem).resolve())
    return str((Path(outdir) / stem).resolve())


def _default_outdir(
    explicit: str | None,
    bundle: dict[str, str | None] | None,
    stage1_dir: str,
) -> str:
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    run_dir = (bundle or {}).get("run_dir")
    if run_dir:
        return str(
            (
                Path(run_dir) / "external_validation_fullcity"
            ).resolve()
        )
    return str(
        (
            Path(stage1_dir) / "external_validation_fullcity"
        ).resolve()
    )


def build_external_validation_fullcity(
    *,
    stage1_manifest: str,
    validation_csv: str,
    outdir: str,
    train_inputs: str | None = None,
    val_inputs: str | None = None,
    test_inputs: str | None = None,
    coord_scaler: str | None = None,
    model_path: str | None = None,
    stage2_manifest: str | None = None,
    stage2_run_dir: str | None = None,
    full_inputs_npz: str | None = None,
    full_payload_npz: str | None = None,
    batch_size: int = 256,
    x_col: str = "x",
    y_col: str = "y",
    productivity_col: str = (
        "step3_specific_capacity_Lps_per_m"
    ),
    thickness_col: str = ("approx_compressible_thickness_m"),
    horizon_reducer: str = "mean",
    site_reducer: str = "median",
    max_match_distance_m: float = 50000.0,
    min_unique_pixels: int = 3,
    allow_swapped_xy: bool = True,
    skip_export: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_dir(outdir)

    (
        train_path,
        val_path,
        test_path,
        coord_path,
        stage1_payload,
    ) = resolve_stage1_inputs_paths(
        stage1_manifest,
        train_inputs=train_inputs,
        val_inputs=val_inputs,
        test_inputs=test_inputs,
        coord_scaler=coord_scaler,
    )

    full_inputs_path = full_inputs_npz or os.path.join(
        outdir,
        "full_inputs.npz",
    )
    if not is_readable_file(full_inputs_path):
        parts = [train_path, val_path]
        if test_path:
            parts.append(test_path)
        full_inputs_path = make_full_inputs_npz(
            parts,
            full_inputs_path,
        )
        print(
            f"[OK] Saved full-city inputs -> {full_inputs_path}"
        )
    else:
        print(
            f"[OK] Reusing full-city inputs -> {full_inputs_path}"
        )

    full_payload_path = full_payload_npz or os.path.join(
        outdir,
        "physics_payload_fullcity.npz",
    )
    if not skip_export:
        bundle = resolve_stage2_bundle(
            model_path,
            stage2_manifest,
            stage2_run_dir,
            stage1_dir=str(
                Path(stage1_manifest)
                .expanduser()
                .resolve()
                .parent
            ),
        )
        print(f"[OK] Using Stage-2 bundle -> {bundle}")
        export_fullcity_payload(
            stage1_payload,
            full_inputs_path,
            bundle,
            full_payload_path,
            batch_size,
        )
    else:
        if not is_readable_file(full_payload_path):
            raise FileNotFoundError(
                "--skip-export was set but the full payload "
                "file does not exist."
            )
        print(
            f"[OK] Reusing full-city payload -> {full_payload_path}"
        )

    return compute_external_metrics(
        validation_csv=validation_csv,
        full_inputs_npz=full_inputs_path,
        full_payload_npz=full_payload_path,
        coord_scaler_path=coord_path,
        outdir=outdir,
        x_col=x_col,
        y_col=y_col,
        productivity_col=productivity_col,
        thickness_col=thickness_col,
        horizon_reducer=horizon_reducer,
        site_reducer=site_reducer,
        max_match_distance_m=max_match_distance_m,
        min_unique_pixels=min_unique_pixels,
        allow_swapped_xy=allow_swapped_xy,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="external-validation-fullcity",
        description=(
            "Build full-city union inputs, export a full-city "
            "physics payload, and compute external borehole / "
            "pumping validation metrics."
        ),
    )
    p.add_argument(
        "--manifest",
        dest="stage1_manifest",
        default=None,
        help="Path to the Stage-1 manifest.json.",
    )
    p.add_argument(
        "--stage1-dir",
        default=None,
        help=(
            "Stage-1 output directory containing "
            "manifest.json."
        ),
    )
    p.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Results root used to auto-resolve the "
            "Stage-1 directory from city/model."
        ),
    )
    p.add_argument(
        "--city",
        default=None,
        help="Dataset or city label.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Model name for Stage-1 auto-resolution.",
    )
    p.add_argument(
        "--validation-csv",
        default=None,
        help=(
            "Validation CSV with site coordinates and "
            "observed thickness/productivity columns."
        ),
    )
    p.add_argument(
        "--outdir",
        default=None,
        help=(
            "Output directory for full-city validation "
            "artifacts."
        ),
    )
    p.add_argument("--train-inputs", default=None)
    p.add_argument("--val-inputs", default=None)
    p.add_argument("--test-inputs", default=None)
    p.add_argument("--coord-scaler", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--stage2-manifest", default=None)
    p.add_argument("--stage2-run-dir", default=None)
    p.add_argument("--full-inputs-npz", default=None)
    p.add_argument("--full-payload-npz", default=None)
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    p.add_argument("--x-col", default="x")
    p.add_argument("--y-col", default="y")
    p.add_argument(
        "--productivity-col",
        default="step3_specific_capacity_Lps_per_m",
    )
    p.add_argument(
        "--thickness-col",
        default="approx_compressible_thickness_m",
        help=(
            "Observed borehole thickness column used for "
            "comparison against model H_eff."
        ),
    )
    p.add_argument(
        "--horizon-reducer",
        choices=["first", "mean", "median"],
        default="mean",
    )
    p.add_argument(
        "--site-reducer",
        choices=["mean", "median"],
        default="median",
    )
    p.add_argument(
        "--max-match-distance-m",
        type=float,
        default=50000.0,
    )
    p.add_argument(
        "--min-unique-pixels",
        type=int,
        default=3,
    )
    p.add_argument(
        "--no-swapped-xy",
        action="store_true",
        help="Disable direct-vs-swapped x/y auto matching.",
    )
    p.add_argument(
        "--skip-export",
        action="store_true",
        help=(
            "Reuse an existing physics payload instead of "
            "exporting a new one."
        ),
    )
    p.add_argument(
        "--config",
        default=None,
        help=(
            "Optional config.py to install into "
            "nat.com/config.py before resolving defaults."
        ),
    )
    p.add_argument(
        "--config-root",
        default="nat.com",
        help="Configuration root directory.",
    )
    p.add_argument(
        "--set",
        dest="set_items",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "One-off config override persisted into "
            "config.json. May be repeated."
        ),
    )
    return p


def build_external_validation_fullcity_main(
    argv: list[str] | None = None,
) -> None:
    args = _build_parser().parse_args(argv)

    if args.config:
        _install_user_config(
            args.config,
            config_root=args.config_root,
        )

    overrides = _parse_set_items(args.set_items)
    cfg = _persist_runtime_overrides(
        overrides,
        config_root=args.config_root,
    )

    city = args.city or _cfg_first(
        cfg,
        "CITY_NAME",
        "TRANSFER_CITY_B",
    )
    model = args.model or cfg.get("MODEL_NAME")
    results_dir = (
        args.results_dir
        or cfg.get("RESULTS_DIR")
        or "results"
    )

    validation_csv = _resolve_validation_csv(
        args.validation_csv,
        cfg,
    )
    if not validation_csv:
        raise SystemExit(
            "Could not resolve a validation CSV. Pass "
            "--validation-csv explicitly or define one in "
            "config.py/config.json."
        )

    manifest_path = _resolve_manifest_path(
        manifest=args.stage1_manifest,
        stage1_dir=args.stage1_dir,
        results_dir=results_dir,
        city=city,
        model=model,
    )
    stage1_dir = str(manifest_path.parent.resolve())

    bundle = None
    if not args.skip_export:
        bundle = resolve_stage2_bundle(
            args.model_path,
            args.stage2_manifest,
            args.stage2_run_dir,
            stage1_dir=stage1_dir,
        )

    outdir = _default_outdir(
        args.outdir,
        bundle,
        stage1_dir,
    )
    ensure_dir(outdir)

    full_inputs_npz = (
        str(Path(args.full_inputs_npz).expanduser().resolve())
        if args.full_inputs_npz
        else _default_full_inputs_path(outdir, stage1_dir)
    )
    full_payload_npz = (
        str(
            Path(args.full_payload_npz).expanduser().resolve()
        )
        if args.full_payload_npz
        else _default_payload_path(outdir, city, bundle)
    )

    site_df, metrics = build_external_validation_fullcity(
        stage1_manifest=str(manifest_path),
        validation_csv=validation_csv,
        outdir=outdir,
        train_inputs=args.train_inputs,
        val_inputs=args.val_inputs,
        test_inputs=args.test_inputs,
        coord_scaler=args.coord_scaler,
        model_path=args.model_path,
        stage2_manifest=args.stage2_manifest,
        stage2_run_dir=args.stage2_run_dir,
        full_inputs_npz=full_inputs_npz,
        full_payload_npz=full_payload_npz,
        batch_size=args.batch_size,
        x_col=args.x_col,
        y_col=args.y_col,
        productivity_col=args.productivity_col,
        thickness_col=args.thickness_col,
        horizon_reducer=args.horizon_reducer,
        site_reducer=args.site_reducer,
        max_match_distance_m=args.max_match_distance_m,
        min_unique_pixels=args.min_unique_pixels,
        allow_swapped_xy=(not args.no_swapped_xy),
        skip_export=args.skip_export,
    )

    print("\nPer-site summary\n")
    print(site_df.to_string(index=False))
    print("\nHeadline metrics\n")
    print(json.dumps(metrics, indent=2))


def main(argv: list[str] | None = None) -> None:
    build_external_validation_fullcity_main(argv)


if __name__ == "__main__":
    main()
