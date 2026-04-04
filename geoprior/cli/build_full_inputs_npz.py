# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>
r"""CLI for building merged full-input NPZ artifacts."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from geoprior.utils.nat_utils import (
    ensure_config_json,
    get_config_paths,
)


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
    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def _load_npz_dict(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _save_npz(
    path: str | Path,
    data: dict[str, np.ndarray],
) -> None:
    np.savez_compressed(path, **data)


def _choose_output_path(
    manifest_path: Path,
    output: str | None,
    *,
    output_name: str,
) -> Path:
    if output:
        out = Path(output).expanduser().resolve()
        out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        return out

    artifacts_dir = manifest_path.parent / "artifacts"
    artifacts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    return artifacts_dir / output_name


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


def _extract_split_paths(
    manifest_payload: dict[str, Any],
    splits: list[str],
) -> dict[str, Path]:
    npz_art = manifest_payload.get("artifacts", {}).get(
        "numpy", {}
    )
    out: dict[str, Path] = {}
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
        out[split] = p
    return out


def _merge_inputs(
    split_arrays: dict[str, dict[str, np.ndarray]],
    *,
    strict_keys: bool = True,
) -> dict[str, np.ndarray]:
    if not split_arrays:
        raise ValueError("No split arrays were provided.")

    split_names = list(split_arrays)
    first = split_arrays[split_names[0]]
    first_keys = list(first)

    if strict_keys:
        ref = set(first_keys)
        for split in split_names[1:]:
            keys = set(split_arrays[split])
            if keys != ref:
                missing = sorted(ref - keys)
                extra = sorted(keys - ref)
                raise ValueError(
                    "Input-key mismatch across splits: "
                    f"split={split!r}, missing={missing}, "
                    f"extra={extra}"
                )
        keys = sorted(ref)
    else:
        keys = []
        seen: set[str] = set()
        for split in split_names:
            for key in split_arrays[split]:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)

    merged: dict[str, np.ndarray] = {}
    for key in keys:
        arrays: list[np.ndarray] = []
        for split in split_names:
            arr = split_arrays[split].get(key)
            if arr is None:
                if strict_keys:
                    raise KeyError(
                        f"Missing key {key!r} in split {split!r}."
                    )
                continue
            arrays.append(arr)

        if not arrays:
            continue

        if len(arrays) == 1:
            merged[key] = arrays[0]
        else:
            merged[key] = np.concatenate(
                arrays,
                axis=0,
            )

    return merged


def build_full_inputs_npz(
    *,
    manifest: str | None = None,
    stage1_dir: str | None = None,
    output: str | None = None,
    output_name: str = "full_inputs.npz",
    splits: list[str] | None = None,
    strict_keys: bool = True,
    results_dir: str | None = None,
    city: str | None = None,
    model: str | None = None,
) -> Path:
    split_list = splits or ["train", "val", "test"]
    manifest_path = _resolve_manifest_path(
        manifest=manifest,
        stage1_dir=stage1_dir,
        results_dir=results_dir,
        city=city,
        model=model,
    )

    with manifest_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    split_paths = _extract_split_paths(
        payload,
        split_list,
    )
    split_arrays = {
        split: _load_npz_dict(path)
        for split, path in split_paths.items()
    }
    merged = _merge_inputs(
        split_arrays,
        strict_keys=strict_keys,
    )
    out_path = _choose_output_path(
        manifest_path,
        output,
        output_name=output_name,
    )
    _save_npz(out_path, merged)
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="full-inputs-npz",
        description=(
            "Build a merged input NPZ from Stage-1 split input "
            "artifacts (for example train+val+test)."
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
            "<city>_<model>_stage1/manifest.json."
        ),
    )
    p.add_argument(
        "--city",
        type=str,
        default=None,
        help="City or dataset label used for manifest auto-resolution.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name used for manifest auto-resolution.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help=(
            "Input splits to concatenate in order. "
            "Example: --splits train val test"
        ),
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output NPZ path. If omitted, the file is written under "
            "<stage1_dir>/artifacts/."
        ),
    )
    p.add_argument(
        "--output-name",
        type=str,
        default="full_inputs.npz",
        help=(
            "Default file name used when --output is omitted."
        ),
    )
    p.add_argument(
        "--allow-missing-keys",
        action="store_true",
        help=(
            "Use the union of keys across splits instead of requiring "
            "all splits to share the same keys."
        ),
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


def build_full_inputs_main(
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

    city = args.city or cfg.get("CITY_NAME")
    model = args.model or cfg.get("MODEL_NAME")
    results_dir = (
        args.results_dir
        or cfg.get("RESULTS_DIR")
        or "results"
    )

    out = build_full_inputs_npz(
        manifest=args.manifest,
        stage1_dir=args.stage1_dir,
        output=args.output,
        output_name=args.output_name,
        splits=list(args.splits),
        strict_keys=not args.allow_missing_keys,
        results_dir=results_dir,
        city=city,
        model=model,
    )

    print(f"Saved: {out}")
    data = _load_npz_dict(out)
    for key, value in sorted(data.items()):
        print(f"{key}: {tuple(value.shape)}")


def main(argv: list[str] | None = None) -> None:
    build_full_inputs_main(argv)


if __name__ == "__main__":
    main()
