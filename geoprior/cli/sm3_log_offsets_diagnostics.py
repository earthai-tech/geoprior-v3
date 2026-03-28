# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""CLI wrapper for SM3 log-offset diagnostics.

This wrapper integrates the standalone SM3 log-offset diagnostics
script into ``geoprior.cli`` so it can be dispatched from
``geoprior-run`` via the package command registry.

Supported flows
---------------
- Use the existing ``nat.com/config.py`` as-is.
- Install a user-supplied config file before running.
- Apply one-off JSON-friendly overrides via ``--set KEY=VALUE``.
- Seed ``--physics-npz``, ``--outdir``, ``--city``, and
  ``--model-name`` from config when omitted downstream.
- Forward all original SM3 legacy arguments unchanged.
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sm3-offset-diagnostics",
        add_help=False,
        description=(
            "Run SM3 log-offset diagnostics via geoprior-run. "
            "Unknown arguments are forwarded to the legacy SM3 CLI."
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
        "--physics-npz",
        type=str,
        default=None,
        help=(
            "Explicit physics payload path. If omitted, the wrapper "
            "tries to discover the latest matching payload."
        ),
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help=(
            "Output directory for CSVs and plots. Defaults to the "
            "payload directory when omitted."
        ),
    )
    p.add_argument(
        "--city",
        type=str,
        default=None,
        help="City tag for output files.",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model tag for output files.",
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=(
            "Search root used to auto-discover physics payloads when "
            "--physics-npz is not given."
        ),
    )
    p.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra config override. Repeat as needed, for example "
            "--set CITY_NAME='nansha'."
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

    if args.city:
        out["CITY_NAME"] = str(args.city).strip().lower()
    if args.model_name:
        out["MODEL_NAME"] = str(args.model_name).strip()
    if args.results_dir:
        out["RESULTS_DIR"] = str(args.results_dir).strip()
    if args.outdir:
        out["SM3_OFFSETS_OUTDIR"] = str(args.outdir).strip()
    if args.physics_npz:
        out["SM3_PHYSICS_NPZ"] = str(args.physics_npz).strip()

    return out


def _legacy_module_name() -> str:
    pkg = __package__ or "geoprior.cli"
    return f"{pkg}._sm3_log_offsets_diagnostics"


def _print_help() -> None:
    parser = _build_parser()
    parser.print_help()
    print("")
    print("Forwarded legacy arguments include, for example:")
    print("  --physics-npz PATH")
    print("  --outdir PATH")
    print("  --city CITY")
    print("  --model-name NAME")


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


def _iter_payload_candidates(
    results_dir: str,
    city: str | None,
) -> list[Path]:
    root = Path(results_dir).expanduser().resolve()
    if not root.exists():
        return []

    city_l = str(city or "").strip().lower()
    patterns: list[str] = []

    if city_l:
        patterns.extend(
            [
                f"{city_l}_tuned_phys_payload_run_val.npz",
                f"{city_l}_phys_payload_run_val.npz",
            ]
        )

    patterns.extend(
        [
            "*_tuned_phys_payload_run_val.npz",
            "*_phys_payload_run_val.npz",
            "*phys_payload*.npz",
        ]
    )

    hits: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            hits.append(path)

    hits.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return hits


def _discover_physics_npz(
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> str | None:
    explicit = args.physics_npz or _cfg_first(
        cfg,
        "SM3_PHYSICS_NPZ",
        default=None,
    )
    if explicit:
        p = Path(str(explicit)).expanduser().resolve()
        return str(p)

    city = args.city or _cfg_first(
        cfg,
        "CITY_NAME",
        default=None,
    )
    results_dir = args.results_dir or _cfg_first(
        cfg,
        "RESULTS_DIR",
        default=os.getenv("RESULTS_DIR", "results"),
    )
    candidates = _iter_payload_candidates(
        str(results_dir),
        None if city is None else str(city),
    )
    if not candidates:
        return None
    return str(candidates[0])


def _seed_forwarded_args(
    forwarded: list[str],
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    out = list(forwarded)

    physics_npz = _discover_physics_npz(cfg, args)
    city = args.city or _cfg_first(
        cfg,
        "CITY_NAME",
        default=None,
    )
    model_name = args.model_name or _cfg_first(
        cfg,
        "MODEL_NAME",
        default="GeoPriorSubsNet",
    )

    outdir = args.outdir or _cfg_first(
        cfg,
        "SM3_OFFSETS_OUTDIR",
        default=None,
    )
    if outdir is None and physics_npz:
        outdir = str(Path(physics_npz).resolve().parent)

    if physics_npz and not _has_flag(out, "--physics-npz"):
        out.extend(["--physics-npz", str(physics_npz)])
    if outdir and not _has_flag(out, "--outdir"):
        out.extend(["--outdir", str(outdir).strip()])
    if city and not _has_flag(out, "--city"):
        out.extend(["--city", str(city).strip().lower()])
    if model_name and not _has_flag(out, "--model-name"):
        out.extend(["--model-name", str(model_name).strip()])

    return out


def run_sm3_offsets(
    argv: list[str] | None = None,
) -> None:
    parser = _build_parser()
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
        args,
    )

    if not _has_flag(forwarded, "--physics-npz"):
        raise SystemExit(
            "Could not resolve --physics-npz automatically. "
            "Pass it explicitly or set RESULTS_DIR/CITY_NAME so the "
            "wrapper can discover a payload."
        )

    mod = importlib.import_module(_legacy_module_name())
    fn = getattr(mod, "main", None)
    if fn is None:
        raise AttributeError(
            "Missing 'main' in "
            "sm3_log_offsets_diagnostics_legacy"
        )

    old = list(sys.argv)
    sys.argv = ["sm3-offset-diagnostics"] + list(forwarded)
    try:
        fn()
    finally:
        sys.argv = old


def sm3_offsets_main(
    argv: list[str] | None = None,
) -> None:
    run_sm3_offsets(argv)


def main(
    argv: list[str] | None = None,
) -> None:
    sm3_offsets_main(argv)


if __name__ == "__main__":
    sm3_offsets_main(sys.argv[1:])
