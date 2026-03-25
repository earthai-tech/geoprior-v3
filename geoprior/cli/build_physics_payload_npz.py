# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from geoprior.models import GeoPriorSubsNet
from geoprior.utils.nat_utils import (
    ensure_config_json,
    get_config_paths,
    make_tf_dataset,
)

try:  # pragma: no cover
    from geoprior.params import (
        FixedGammaW,
        FixedHRef,
        LearnableKappa,
        LearnableMV,
    )
except Exception:  # pragma: no cover
    FixedGammaW = None
    FixedHRef = None
    LearnableKappa = None
    LearnableMV = None


_CUSTOM_OBJECTS: dict[str, Any] = {
    "GeoPriorSubsNet": GeoPriorSubsNet,
}
for _name, _obj in {
    "LearnableMV": LearnableMV,
    "LearnableKappa": LearnableKappa,
    "FixedGammaW": FixedGammaW,
    "FixedHRef": FixedHRef,
}.items():
    if _obj is not None:
        _CUSTOM_OBJECTS[_name] = _obj


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


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _load_npz_dict(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _first_present(
    mapping: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _manifest_cfg_value(
    manifest: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    cfg = manifest.get("config", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    feats_cfg = cfg.get("features", {}) or {}

    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
        if key in model_cfg and model_cfg[key] is not None:
            return model_cfg[key]
        if key in feats_cfg and feats_cfg[key] is not None:
            return feats_cfg[key]
    return default


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


def _choose_input_keys(
    payload: dict[str, Any],
    splits: list[str],
) -> list[tuple[str, Path]]:
    npz_art = payload.get("artifacts", {}).get("numpy", {})
    out: list[tuple[str, Path]] = []
    for split in splits:
        key = f"{split}_inputs_npz"
        path = npz_art.get(key)
        if not path:
            raise KeyError(
                f"Missing manifest artifact key: {key}"
            )
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"Missing NPZ for split={split!r}: {p}"
            )
        out.append((split, p))
    return out


def _merge_inputs_from_splits(
    payload: dict[str, Any],
    splits: list[str],
) -> tuple[dict[str, np.ndarray], str, str | None]:
    arrays_by_split = [
        (split, _load_npz_dict(path))
        for split, path in _choose_input_keys(payload, splits)
    ]
    if not arrays_by_split:
        raise ValueError("No split arrays were found.")

    ref_keys = set(arrays_by_split[0][1])
    for split, arrs in arrays_by_split[1:]:
        keys = set(arrs)
        if keys != ref_keys:
            missing = sorted(ref_keys - keys)
            extra = sorted(keys - ref_keys)
            raise ValueError(
                "Input-key mismatch across splits: "
                f"split={split!r}, missing={missing}, "
                f"extra={extra}"
            )

    merged: dict[str, np.ndarray] = {}
    for key in sorted(ref_keys):
        merged[key] = np.concatenate(
            [arrs[key] for _, arrs in arrays_by_split],
            axis=0,
        )

    label = "_".join(splits) if splits else "union"
    src_hint = None
    if arrays_by_split:
        src_hint = ",".join(
            str(p)
            for _, p in _choose_input_keys(payload, splits)
        )
    return merged, label, src_hint


def _choose_inputs(
    *,
    payload: dict[str, Any],
    manifest_path: Path,
    inputs_npz: str | None,
    splits: list[str] | None,
) -> tuple[
    dict[str, np.ndarray],
    str,
    str | None,
]:
    if inputs_npz:
        p = Path(inputs_npz).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"Inputs NPZ not found: {p}"
            )
        return _load_npz_dict(p), p.stem, str(p)

    art_dir = manifest_path.parent / "artifacts"
    full_npz = art_dir / "full_inputs.npz"
    if full_npz.exists() and not splits:
        return (
            _load_npz_dict(full_npz),
            "full_city_union",
            str(full_npz),
        )

    split_list = list(splits or ["train", "val", "test"])
    return _merge_inputs_from_splits(payload, split_list)


def _resolve_model_path(
    model_path: str | None,
    *,
    stage1_dir: str | Path | None,
) -> Path:
    def _best_in(root: Path) -> Path | None:
        patterns = (
            "**/*_best.keras",
            "**/*.keras",
        )
        cand: list[Path] = []
        for pattern in patterns:
            cand.extend(root.glob(pattern))
            if cand:
                break
        if not cand:
            return None
        cand = [p for p in cand if p.is_file()]
        if not cand:
            return None
        cand.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return cand[0]

    if model_path:
        p = Path(model_path).expanduser().resolve()
        if p.is_file():
            if p.suffix != ".keras":
                raise SystemExit(
                    "This command currently expects a "
                    "full .keras model file. "
                    f"Got: {p}"
                )
            return p
        if p.is_dir():
            best = _best_in(p)
            if best is None:
                raise SystemExit(
                    f"No .keras model found under: {p}"
                )
            return best
        raise FileNotFoundError(f"Model path not found: {p}")

    if stage1_dir is None:
        raise SystemExit(
            "Could not resolve a model automatically. "
            "Pass --model-path explicitly."
        )

    root = Path(stage1_dir).expanduser().resolve()
    best = _best_in(root)
    if best is None:
        raise SystemExit(
            "Could not find a .keras model under the "
            f"Stage-1 directory: {root}\n"
            "Pass --model-path explicitly."
        )
    return best


def _sanitize_label(text: str) -> str:
    s = str(text).strip().replace(" ", "_")
    s = s.replace("-", "_")
    return s or "payload"


def _choose_output_path(
    *,
    output: str | None,
    output_name: str | None,
    manifest_payload: dict[str, Any],
    model_path: Path,
    source_label: str,
) -> Path:
    if output:
        out = Path(output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    run_dir = model_path.parent
    dataset_id = (
        str(
            _first_present(
                manifest_payload,
                "city",
                "dataset",
                default="dataset",
            )
        )
        .strip()
        .lower()
        or "dataset"
    )

    if output_name:
        name = output_name
    else:
        label = _sanitize_label(source_label)
        name = f"{dataset_id}_phys_payload_{label}.npz"

    out = run_dir / name
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _infer_n_samples(
    inputs: dict[str, np.ndarray],
) -> int:
    for key in (
        "dynamic_features",
        "coords",
        "static_features",
        "future_features",
    ):
        arr = inputs.get(key)
        if arr is not None:
            return int(np.asarray(arr).shape[0])
    first = next(iter(inputs.values()), None)
    if first is None:
        raise ValueError("Input NPZ is empty.")
    return int(np.asarray(first).shape[0])


def _infer_horizon(
    manifest_payload: dict[str, Any],
    inputs: dict[str, np.ndarray],
) -> int:
    h = _manifest_cfg_value(
        manifest_payload,
        "FORECAST_HORIZON_YEARS",
        "forecast_horizon_years",
        "forecast_horizon",
        default=None,
    )
    if h is not None:
        return int(h)

    for key in (
        "coords",
        "future_features",
        "dynamic_features",
    ):
        arr = inputs.get(key)
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim >= 2:
            return int(arr.shape[1])

    raise ValueError(
        "Could not infer forecast horizon from "
        "manifest or inputs."
    )


def _dummy_targets(
    *,
    n_samples: int,
    horizon: int,
    out_s_dim: int,
    out_g_dim: int,
) -> dict[str, np.ndarray]:
    return {
        "subs_pred": np.zeros(
            (n_samples, horizon, out_s_dim),
            dtype=np.float32,
        ),
        "gwl_pred": np.zeros(
            (n_samples, horizon, out_g_dim),
            dtype=np.float32,
        ),
    }


def _load_model(
    model_path: Path,
) -> GeoPriorSubsNet:
    model = tf.keras.models.load_model(
        model_path,
        custom_objects=_CUSTOM_OBJECTS,
        compile=False,
    )
    if not hasattr(model, "export_physics_payload"):
        raise TypeError(
            "Loaded model does not expose "
            "export_physics_payload(). "
            f"Path: {model_path}"
        )
    return model


def build_physics_payload_npz(
    *,
    manifest: str | None = None,
    stage1_dir: str | None = None,
    results_dir: str | None = None,
    city: str | None = None,
    model: str | None = None,
    model_path: str | None = None,
    inputs_npz: str | None = None,
    splits: list[str] | None = None,
    output: str | None = None,
    output_name: str | None = None,
    source_label: str | None = None,
    batch_size: int = 256,
    max_batches: int | None = None,
    overwrite: bool = True,
    check_npz_finite: bool = True,
    check_finite: bool = True,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = _resolve_manifest_path(
        manifest=manifest,
        stage1_dir=stage1_dir,
        results_dir=results_dir,
        city=city,
        model=model,
    )
    payload = _read_json(manifest_path)
    stage1_root = manifest_path.parent

    x_np, src_label_auto, src_hint = _choose_inputs(
        payload=payload,
        manifest_path=manifest_path,
        inputs_npz=inputs_npz,
        splits=splits,
    )

    source_label = source_label or src_label_auto
    resolved_model = _resolve_model_path(
        model_path,
        stage1_dir=stage1_root,
    )
    out_path = _choose_output_path(
        output=output,
        output_name=output_name,
        manifest_payload=payload,
        model_path=resolved_model,
        source_label=source_label,
    )

    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {out_path}"
        )

    mode = str(
        _manifest_cfg_value(
            payload,
            "MODE",
            "mode",
            default="tft_like",
        )
    )
    horizon = _infer_horizon(payload, x_np)
    dyn_names = list(
        _manifest_cfg_value(
            payload,
            "dynamic",
            default=[],
        )
        or []
    )
    fut_names = list(
        _manifest_cfg_value(
            payload,
            "future",
            default=[],
        )
        or []
    )

    seq_dims = (
        payload.get("artifacts", {})
        .get("sequences", {})
        .get("dims", {})
    )
    out_s_dim = int(
        _first_present(
            seq_dims,
            "output_subsidence_dim",
            default=1,
        )
    )
    out_g_dim = int(
        _first_present(
            seq_dims,
            "output_gwl_dim",
            default=1,
        )
    )
    n_samples = _infer_n_samples(x_np)
    y_dummy = _dummy_targets(
        n_samples=n_samples,
        horizon=horizon,
        out_s_dim=out_s_dim,
        out_g_dim=out_g_dim,
    )

    ds = make_tf_dataset(
        x_np,
        y_dummy,
        batch_size=batch_size,
        shuffle=False,
        mode=mode,
        forecast_horizon=horizon,
        check_npz_finite=check_npz_finite,
        check_finite=check_finite,
        dynamic_feature_names=dyn_names,
        future_feature_names=fut_names,
    )

    model_obj = _load_model(resolved_model)
    metadata = {
        "split": source_label,
        "source_inputs_npz": src_hint,
        "stage1_manifest": str(manifest_path),
        "model_path": str(resolved_model),
    }
    result = model_obj.export_physics_payload(
        ds,
        max_batches=max_batches,
        save_path=str(out_path),
        format="npz",
        overwrite=overwrite,
        metadata=metadata,
    )
    return out_path, result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="physics-payload-npz",
        description=(
            "Export a physics payload NPZ from a trained "
            "GeoPrior model and an input NPZ or Stage-1 split set."
        ),
    )
    p.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to a Stage-1 manifest.json.",
    )
    p.add_argument(
        "--stage1-dir",
        type=str,
        default=None,
        help=(
            "Stage-1 output directory containing manifest.json."
        ),
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=(
            "Results root used to auto-resolve "
            "<dataset>_<model>_stage1/manifest.json."
        ),
    )
    p.add_argument(
        "--city",
        type=str,
        default=None,
        help=(
            "Dataset or city label used for manifest "
            "auto-resolution."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name used for manifest auto-resolution.",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Path to a .keras model file or a run directory "
            "containing one. If omitted, the latest .keras model "
            "under the resolved Stage-1 tree is used."
        ),
    )
    p.add_argument(
        "--inputs-npz",
        type=str,
        default=None,
        help=(
            "Explicit inputs NPZ. If omitted, the command prefers "
            "<stage1_dir>/artifacts/full_inputs.npz and otherwise "
            "merges Stage-1 split NPZs in memory."
        ),
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help=(
            "Split inputs to merge when --inputs-npz is omitted. "
            "Default fallback: train val test."
        ),
    )
    p.add_argument(
        "--source-label",
        type=str,
        default=None,
        help=(
            "Metadata label stored in the payload, for example "
            "full_city_union or external_union."
        ),
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Explicit output NPZ path. If omitted, the file is "
            "written next to the resolved model."
        ),
    )
    p.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Output file name used when --output is omitted. "
            "Default: <dataset>_phys_payload_<label>.npz"
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size used to build the TF dataset.",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help=(
            "Optional export batch cap for debugging or partial "
            "payloads."
        ),
    )
    p.add_argument(
        "--no-check-npz-finite",
        action="store_true",
        help="Disable pre-build NPZ finite checks.",
    )
    p.add_argument(
        "--no-check-finite",
        action="store_true",
        help="Disable tf.data finite checks.",
    )
    p.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if the output already exists.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional config.py to install into nat.com/config.py "
            "before resolving defaults."
        ),
    )
    p.add_argument(
        "--config-root",
        type=str,
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
            "One-off config override persisted into config.json. "
            "May be repeated."
        ),
    )
    return p


def build_physics_payload_main(
    argv: list[str] | None = None,
) -> None:
    args = _build_parser().parse_args(argv)

    if args.config:
        installed = _install_user_config(
            args.config,
            config_root=args.config_root,
        )
        print(f"[Config] Using: {installed}")

    overrides = _parse_set_items(args.set_items)
    cfg = _persist_runtime_overrides(
        overrides,
        config_root=args.config_root,
    )

    city = args.city or cfg.get("CITY_NAME")
    model = args.model or cfg.get("MODEL_NAME")
    results_dir = (
        args.results_dir
        or cfg.get("RESULTS_DIR")
        or "results"
    )

    out, payload = build_physics_payload_npz(
        manifest=args.manifest,
        stage1_dir=args.stage1_dir,
        results_dir=results_dir,
        city=city,
        model=model,
        model_path=args.model_path,
        inputs_npz=args.inputs_npz,
        splits=(list(args.splits) if args.splits else None),
        output=args.output,
        output_name=args.output_name,
        source_label=args.source_label,
        batch_size=int(args.batch_size),
        max_batches=args.max_batches,
        overwrite=not args.no_overwrite,
        check_npz_finite=not args.no_check_npz_finite,
        check_finite=not args.no_check_finite,
    )

    print(f"Saved: {out}")
    keys = sorted(payload.keys())
    print(f"Keys: {keys}")
    for key in ("K", "Ss", "tau", "H"):
        if key in payload:
            print(f"{key}: {np.asarray(payload[key]).shape}")


def main(argv: list[str] | None = None) -> None:
    build_physics_payload_main(argv)


if __name__ == "__main__":
    main()
