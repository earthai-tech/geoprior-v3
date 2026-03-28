# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""CLI wrapper for Stage-2 training.

This module makes Stage-2 safe to dispatch from ``geoprior.cli``.
The original training script body lives in ``stage2_legacy.py`` and
is executed only when ``run_stage2()`` is called.

Supported flows
---------------
- Use the existing ``nat.com/config.py`` as-is.
- Install a user-supplied config file before running.
- Apply one-off JSON-friendly overrides via ``--set KEY=VALUE``.
- Point Stage-2 at a specific Stage-1 manifest via
  ``--stage1-manifest``.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import runpy
import shutil
from pathlib import Path
from typing import Any

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


def _refresh_config_fields(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    out = dict(cfg)

    city = str(out.get("CITY_NAME", "")).strip().lower()
    variant = str(out.get("DATASET_VARIANT", "")).strip()

    big_t = out.get("BIG_FN_TEMPLATE")
    small_t = out.get("SMALL_FN_TEMPLATE")

    if city and variant and isinstance(big_t, str):
        out["BIG_FN"] = big_t.format(
            city=city,
            variant=variant,
        )

    if city and variant and isinstance(small_t, str):
        out["SMALL_FN"] = small_t.format(
            city=city,
            variant=variant,
        )

    return out


def _apply_config(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    return _refresh_config_fields(dict(cfg))


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
    cfg = _apply_config(cfg0)

    if overrides:
        cfg.update(overrides)
        cfg = _apply_config(cfg)

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


def _build_stage2_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stage2-train",
        description=(
            "Run GeoPrior Stage-2 training using "
            "nat.com/config.py and Stage-1 artifacts."
        ),
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional config.py to install into "
            "nat.com/config.py before running."
        ),
    )
    p.add_argument(
        "--config-root",
        type=str,
        default="nat.com",
        help="Config root directory.",
    )
    p.add_argument(
        "--city",
        type=str,
        default=None,
        help="Override CITY_NAME for this run.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override MODEL_NAME for this run.",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override DATA_DIR for this run.",
    )
    p.add_argument(
        "--stage1-manifest",
        type=str,
        default=None,
        help=(
            "Exact Stage-1 manifest.json to use. "
            "This is forwarded through the "
            "STAGE1_MANIFEST environment variable."
        ),
    )
    p.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra config override. Repeat as needed, "
            "for example --set EPOCHS=150."
        ),
    )
    return p


def _cli_overrides(
    args: argparse.Namespace,
) -> dict[str, Any]:
    out = _parse_set_items(args.sets)

    if args.city:
        out["CITY_NAME"] = str(args.city).strip().lower()
    if args.model:
        out["MODEL_NAME"] = str(args.model).strip()
    if args.data_dir:
        out["DATA_DIR"] = str(args.data_dir).strip()

    return out


def _legacy_module_name() -> str:
    pkg = __package__ or "geoprior.cli"
    return f"{pkg}._stage2"


def run_stage2(
    overrides: dict[str, object] | None = None,
    *,
    config_root: str = "nat.com",
    config_path: str | None = None,
    stage1_manifest: str | None = None,
) -> None:
    """Execute Stage-2 training pipeline."""
    if config_path:
        installed = _install_user_config(
            config_path,
            config_root=config_root,
        )
        print(f"[Config] Using: {installed}")

    cfg = _persist_runtime_overrides(
        overrides,
        config_root=config_root,
    )
    cfg = _apply_config(cfg)

    env_updates: dict[str, str] = {}
    if cfg.get("CITY_NAME"):
        env_updates["CITY"] = str(cfg["CITY_NAME"])
    if cfg.get("MODEL_NAME"):
        env_updates["MODEL_NAME_OVERRIDE"] = str(
            cfg["MODEL_NAME"]
        )
    if stage1_manifest:
        env_updates["STAGE1_MANIFEST"] = str(
            Path(stage1_manifest).expanduser().resolve()
        )

    old_env = {
        key: os.environ.get(key) for key in env_updates
    }

    os.environ.update(env_updates)
    try:
        runpy.run_module(
            _legacy_module_name(),
            run_name="__main__",
        )
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def stage2_main(
    argv: list[str] | None = None,
) -> None:
    parser = _build_stage2_parser()
    args = parser.parse_args(argv)

    overrides = _cli_overrides(args)
    run_stage2(
        overrides,
        config_root=args.config_root,
        config_path=args.config,
        stage1_manifest=args.stage1_manifest,
    )


def main(
    argv: list[str] | None = None,
) -> None:
    stage2_main(argv)


if __name__ == "__main__":
    stage2_main()
