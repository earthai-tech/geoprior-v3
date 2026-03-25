# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""CLI wrapper for Stage-4 inference.

This module makes Stage-4 consistent with the newer
``geoprior.cli`` dispatcher pattern while preserving the
existing inference implementation in ``stage4_legacy.py``.

Supported flows
---------------
- Use the existing ``nat.com/config.py`` as-is.
- Install a user-supplied config file before running.
- Apply one-off JSON-friendly overrides via ``--set KEY=VALUE``.
- Point Stage-4 at a specific Stage-1 manifest via
  ``--stage1-manifest``.
- Forward all other inference arguments to the original
  Stage-4 CLI unchanged.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import shutil
import sys
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


def _build_stage4_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stage4-infer",
        add_help=False,
        description=(
            "Run GeoPrior Stage-4 inference using "
            "nat.com/config.py and Stage-1 artifacts. "
            "Unknown arguments are forwarded to the "
            "legacy Stage-4 inference CLI."
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
            "for example --set CITY_NAME='nansha'."
        ),
    )
    p.add_argument(
        "-h",
        "--help",
        action="store_true",
        help=("Show the combined wrapper/legacy help."),
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
    return f"{pkg}.stage4_legacy"


def _print_help() -> None:
    parser = _build_stage4_parser()
    parser.print_help()
    print("")
    print("Forwarded legacy arguments include, for example:")
    print("  --stage1-dir PATH")
    print("  --manifest PATH")
    print("  --model-path PATH")
    print("  --dataset {test,val,train,custom}")
    print("  --inputs-npz PATH")
    print("  --targets-npz PATH")
    print("  --eval-losses")
    print("  --eval-physics")
    print("  --calibrator PATH")
    print("  --use-source-calibrator")
    print("  --fit-calibrator")
    print("  --cov-target FLOAT")
    print("  --batch-size INT")
    print("  --no-figs")
    print("  --include-gwl")


def run_stage4(
    argv: list[str] | None = None,
) -> None:
    parser = _build_stage4_parser()
    args, forwarded = parser.parse_known_args(argv)

    if args.help:
        _print_help()
        return

    if args.config:
        _install_user_config(
            args.config,
            config_root=args.config_root,
        )

    overrides = _cli_overrides(args)
    cfg = _persist_runtime_overrides(
        overrides,
        config_root=args.config_root,
    )

    city = str(cfg.get("CITY_NAME", "")).strip().lower()
    model = str(
        cfg.get("MODEL_NAME", "GeoPriorSubsNet")
    ).strip()

    if city:
        os.environ["CITY"] = city
    if model:
        os.environ["MODEL_NAME_OVERRIDE"] = model
    if args.stage1_manifest:
        os.environ["STAGE1_MANIFEST"] = str(
            args.stage1_manifest
        ).strip()

    mod = importlib.import_module(_legacy_module_name())
    fn = getattr(mod, "main", None)
    if fn is None:
        raise AttributeError(
            "Missing 'main' in stage4_legacy"
        )

    old = list(sys.argv)
    sys.argv = ["stage4-infer"] + list(forwarded)
    try:
        fn()
    finally:
        sys.argv = old


def stage4_main(
    argv: list[str] | None = None,
) -> None:
    run_stage4(argv)


def main(
    argv: list[str] | None = None,
) -> None:
    stage4_main(argv)


if __name__ == "__main__":
    stage4_main(sys.argv[1:])
