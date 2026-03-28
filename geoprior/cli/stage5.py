# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""CLI wrapper for Stage-5 cross-city transfer evaluation.

This wrapper aligns Stage-5 with the package CLI used by the
other stages while preserving the existing transfer
implementation in ``_stage5.py``.

Supported flows
---------------
- Use the existing ``nat.com/config.py`` as-is.
- Install a user-supplied config file before running.
- Apply one-off JSON-friendly overrides via ``--set KEY=VALUE``.
- Seed transfer defaults such as city pair, model name, and
  results directory from config without changing the legacy
  Stage-5 CLI behavior.
- Forward all original transfer arguments to the legacy
  implementation unchanged.
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


def _build_stage5_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stage5-transfer",
        add_help=False,
        description=(
            "Run GeoPrior Stage-5 cross-city transfer "
            "evaluation using nat.com/config.py. "
            "Unknown arguments are forwarded to the "
            "legacy Stage-5 transfer CLI."
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
        "--city-a",
        type=str,
        default=None,
        help=(
            "Default source city for transfer if not "
            "already supplied to the legacy CLI."
        ),
    )
    p.add_argument(
        "--city-b",
        type=str,
        default=None,
        help=(
            "Default target city for transfer if not "
            "already supplied to the legacy CLI."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Default model token passed as "
            "--model-name when omitted downstream."
        ),
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=(
            "Default results directory if not already "
            "supplied to the legacy CLI."
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
            "for example --set TRANSFER_CITY_A='nansha'."
        ),
    )
    p.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show the combined wrapper/legacy help.",
    )
    return p


def _cli_overrides(
    args: argparse.Namespace,
) -> dict[str, Any]:
    out = _parse_set_items(args.sets)

    if args.city_a:
        out["TRANSFER_CITY_A"] = (
            str(args.city_a).strip().lower()
        )
    if args.city_b:
        out["TRANSFER_CITY_B"] = (
            str(args.city_b).strip().lower()
        )
    if args.model:
        out["MODEL_NAME"] = str(args.model).strip()
    if args.results_dir:
        out["RESULTS_DIR"] = str(args.results_dir).strip()

    return out


def _legacy_module_name() -> str:
    pkg = __package__ or "geoprior.cli"
    return f"{pkg}._stage5"


def _print_help() -> None:
    parser = _build_stage5_parser()
    parser.print_help()
    print("")
    print("Forwarded legacy arguments include, for example:")
    print("  --city-a CITY")
    print("  --city-b CITY")
    print("  --results-dir PATH")
    print("  --splits {val,test}...")
    print("  --strategies {baseline,xfer,warm}...")
    print("  --calib-modes {none,source,target}...")
    print("  --rescale-modes {as_is,strict}...")
    print("  --model-name NAME")
    print("  --source-model {auto,tuned,trained}")
    print("  --source-load {auto,full,weights}")
    print("  --hps-mode {auto,tuned,trained}")
    print("  --prefer-artifact {keras,weights}")
    print("  --warm-split {train,val}")
    print("  --warm-samples INT")
    print("  --warm-frac FLOAT")
    print("  --warm-epochs INT")
    print("  --warm-lr FLOAT")
    print("  --continue-on-error")


def _has_flag(
    argv: list[str],
    *flags: str,
) -> bool:
    wanted = set(flags)
    for token in argv:
        if token in wanted:
            return True
        head = token.split("=", 1)[0]
        if head in wanted:
            return True
    return False


def _cfg_first(
    cfg: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    for key in keys:
        val = cfg.get(key)
        if val is not None and val != "":
            return val
    return default


def _seed_forwarded_args(
    forwarded: list[str],
    cfg: dict[str, Any],
) -> list[str]:
    out = list(forwarded)

    city_a = _cfg_first(
        cfg,
        "TRANSFER_CITY_A",
        "CITY_A",
        "SOURCE_CITY",
    )
    city_b = _cfg_first(
        cfg,
        "TRANSFER_CITY_B",
        "CITY_B",
        "TARGET_CITY",
    )
    model = _cfg_first(
        cfg,
        "MODEL_NAME",
        "TRANSFER_MODEL_NAME",
        default="GeoPriorSubsNet",
    )
    results_dir = _cfg_first(
        cfg,
        "RESULTS_DIR",
        default=os.getenv("RESULTS_DIR", "results"),
    )

    if city_a and not _has_flag(out, "--city-a"):
        out.extend(["--city-a", str(city_a).strip().lower()])
    if city_b and not _has_flag(out, "--city-b"):
        out.extend(["--city-b", str(city_b).strip().lower()])
    if model and not _has_flag(out, "--model-name"):
        out.extend(["--model-name", str(model).strip()])
    if results_dir and not _has_flag(out, "--results-dir"):
        out.extend(
            ["--results-dir", str(results_dir).strip()]
        )

    return out


def run_stage5(
    argv: list[str] | None = None,
) -> None:
    parser = _build_stage5_parser()
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

    forwarded = _seed_forwarded_args(
        list(forwarded),
        cfg,
    )

    mod = importlib.import_module(_legacy_module_name())
    fn = getattr(mod, "main", None)
    if fn is None:
        raise AttributeError(
            "Missing 'main' in stage5_legacy"
        )

    old = list(sys.argv)
    sys.argv = ["stage5-transfer"] + list(forwarded)
    try:
        fn()
    finally:
        sys.argv = old


def stage5_main(
    argv: list[str] | None = None,
) -> None:
    run_stage5(argv)


def main(
    argv: list[str] | None = None,
) -> None:
    stage5_main(argv)


if __name__ == "__main__":
    stage5_main(sys.argv[1:])
