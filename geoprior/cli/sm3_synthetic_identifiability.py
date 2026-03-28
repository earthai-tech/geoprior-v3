# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""CLI wrapper for SM3 synthetic identifiability.

This wrapper integrates the standalone SM3 identifiability script into
``geoprior.cli`` so it can be dispatched from ``geoprior-run`` via the
package command registry.

Supported flows
---------------
- Use the existing ``nat.com/config.py`` as-is.
- Install a user-supplied config file before running.
- Apply one-off JSON-friendly overrides via ``--set KEY=VALUE``.
- Seed a default ``--outdir`` and ``--ident-regime`` from config when
  the legacy script is called without them.
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
        prog="sm3-identifiability",
        add_help=False,
        description=(
            "Run SM3 synthetic identifiability via geoprior-run. "
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
        "--outdir",
        type=str,
        default=None,
        help=(
            "Default SM3 output directory if not already "
            "supplied to the legacy CLI."
        ),
    )
    p.add_argument(
        "--ident-regime",
        type=str,
        default=None,
        help=(
            "Default identifiability regime if omitted downstream."
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
            "--set IDENTIFIABILITY_REGIME='anchored'."
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

    if args.ident_regime:
        out["IDENTIFIABILITY_REGIME"] = str(
            args.ident_regime
        ).strip()
    if args.outdir:
        out["SM3_IDENT_OUTDIR"] = str(args.outdir).strip()

    return out


def _legacy_module_name() -> str:
    pkg = __package__ or "geoprior.cli"
    return f"{pkg}._sm3_synthetic_identifiability"


def _print_help() -> None:
    parser = _build_parser()
    parser.print_help()
    print("")
    print("Forwarded legacy arguments include, for example:")
    print("  --outdir PATH")
    print("  --n-realizations INT")
    print("  --n-years INT")
    print("  --time-steps INT")
    print("  --forecast-horizon INT")
    print("  --epochs INT")
    print("  --noise-std FLOAT")
    print("  --load-type {step,ramp}")
    print("  --identify {tau,k,both}")
    print("  --scenario {base,tau_only_derive_k}")
    print(
        "  --ident-regime "
        "{none,base,anchored,closure_locked,data_relaxed}"
    )


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


def _default_outdir(
    cfg: dict[str, Any],
) -> str:
    base = str(
        _cfg_first(
            cfg,
            "SM3_IDENT_OUTDIR",
            "SM3_OUTDIR",
            "RESULTS_DIR",
            default=os.getenv("RESULTS_DIR", "results"),
        )
    ).strip()

    city = str(cfg.get("CITY_NAME", "")).strip().lower()

    p = Path(base)
    name = p.name.lower()
    if name in {"sm3_identifiability", "sm3-identifiability"}:
        return str(p)
    if city:
        return str(p / "sm3_identifiability" / city)
    return str(p / "sm3_identifiability")


def _seed_forwarded_args(
    forwarded: list[str],
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    out = list(forwarded)

    outdir = args.outdir or _default_outdir(cfg)
    ident_regime = args.ident_regime or _cfg_first(
        cfg,
        "IDENTIFIABILITY_REGIME",
        "IDENT_REGIME",
        "GEOPRIOR_IDENTIFIABILITY_REGIME",
        default=None,
    )

    if outdir and not _has_flag(out, "--outdir"):
        out.extend(["--outdir", str(outdir).strip()])

    if ident_regime and not _has_flag(out, "--ident-regime"):
        out.extend(
            ["--ident-regime", str(ident_regime).strip()]
        )

    return out


def run_sm3_identifiability(
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

    mod = importlib.import_module(_legacy_module_name())
    fn = getattr(mod, "main", None)
    if fn is None:
        raise AttributeError(
            "Missing 'main' in "
            "sm3_synthetic_identifiability_legacy"
        )

    old = list(sys.argv)
    sys.argv = ["sm3-identifiability"] + list(forwarded)
    try:
        fn()
    finally:
        sys.argv = old


def sm3_identifiability_main(
    argv: list[str] | None = None,
) -> None:
    run_sm3_identifiability(argv)


def main(
    argv: list[str] | None = None,
) -> None:
    sm3_identifiability_main(argv)


if __name__ == "__main__":
    sm3_identifiability_main(sys.argv[1:])
