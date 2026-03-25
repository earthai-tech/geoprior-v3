# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""Shared CLI configuration helpers.

This module centralises the parser options and runtime helpers that
repeat across GeoPrior CLI commands. The goal is to keep command
modules small, consistent, and easy to maintain.

Scope
-----
This module is intentionally limited to:

- repeated parser arguments such as ``--config`` and ``--set``
- light argument aliases and normalisation
- config installation and runtime override persistence
- small path utilities used by many commands

It does **not** own command-specific business logic. Each command keeps
its own artifact resolution and domain-specific validation.

Examples
--------
Build a parser with shared arguments::

    import argparse
    from geoprior.cli._config import (
        add_city_arg,
        add_config_args,
        add_outdir_arg,
        add_results_dir_arg,
    )

    p = argparse.ArgumentParser()
    add_config_args(p)
    add_city_arg(p)
    add_results_dir_arg(p)
    add_outdir_arg(p)

Apply config installation and runtime overrides::

    cfg = bootstrap_runtime_config(
        args,
        field_map={
            "city": "CITY_NAME",
            "model": "MODEL_NAME",
            "results_dir": "RESULTS_DIR",
        },
    )

The returned ``cfg`` is the effective config dictionary after optional
config installation and any ``--set KEY=VALUE`` overrides.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from geoprior.utils.nat_utils import (
    ensure_config_json,
    get_config_paths,
    load_nat_config,
)

ConfigDict = dict[str, Any]
RefreshFn = Callable[[ConfigDict], ConfigDict]


def parse_override_value(raw: str) -> Any:
    """Parse a scalar or container value from ``--set``.

    Parameters
    ----------
    raw : str
        Raw string value from the CLI.

    Returns
    -------
    Any
        Parsed Python object when possible, otherwise the stripped
        string.

    Notes
    -----
    The parsing order is conservative:

    1. case-insensitive booleans and ``none``
    2. integer / float literals
    3. ``ast.literal_eval`` for lists, tuples, dicts, and quoted text
    4. fallback to the stripped input string
    """
    text = str(raw).strip()
    low = text.lower()

    if low == "none":
        return None
    if low == "true":
        return True
    if low == "false":
        return False

    try:
        if text.startswith("0") and text not in {"0", "0.0"}:
            raise ValueError
        return int(text)
    except Exception:
        pass

    try:
        return float(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def parse_set_items(
    items: list[str] | tuple[str, ...] | None,
) -> ConfigDict:
    """Parse repeated ``--set KEY=VALUE`` items."""
    out: ConfigDict = {}
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
        out[key] = parse_override_value(value)
    return out


def install_user_config(
    config_path: str,
    *,
    config_root: str = "nat.com",
) -> str:
    """Install a user ``config.py`` into the active config root."""
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


def persist_runtime_overrides(
    overrides: ConfigDict | None = None,
    *,
    config_root: str = "nat.com",
    refresh_fn: RefreshFn | None = None,
) -> ConfigDict:
    """Persist effective config to ``config.json``.

    Parameters
    ----------
    overrides : dict or None
        Optional config overrides to merge into the active config.
    config_root : str, default="nat.com"
        Config root directory.
    refresh_fn : callable or None
        Optional callback used to refresh derived fields after the
        overrides are applied.
    """
    cfg0, config_json = ensure_config_json(root=config_root)
    cfg = dict(cfg0)
    if refresh_fn is not None:
        cfg = refresh_fn(cfg)

    if overrides:
        cfg.update(overrides)
        if refresh_fn is not None:
            cfg = refresh_fn(cfg)

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


def args_to_config_overrides(
    args: argparse.Namespace,
    *,
    field_map: dict[str, str] | None = None,
) -> ConfigDict:
    """Map parsed argument fields to config keys.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI namespace.
    field_map : dict[str, str] or None
        Mapping from argument field name to config key.

    Returns
    -------
    dict
        Override dictionary combining ``--set`` items and selected
        explicit CLI fields.
    """
    out = parse_set_items(getattr(args, "sets", None))
    for field, key in (field_map or {}).items():
        value = getattr(args, field, None)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        out[key] = value
    return out


def bootstrap_runtime_config(
    args: argparse.Namespace,
    *,
    field_map: dict[str, str] | None = None,
    refresh_fn: RefreshFn | None = None,
) -> ConfigDict:
    """Install config, apply overrides, and return effective cfg."""
    config_root = getattr(args, "config_root", "nat.com")
    config_path = getattr(args, "config", None)

    if config_path:
        installed = install_user_config(
            config_path,
            config_root=config_root,
        )
        print(f"[Config] Using: {installed}")

    overrides = args_to_config_overrides(
        args,
        field_map=field_map,
    )
    persist_runtime_overrides(
        overrides,
        config_root=config_root,
        refresh_fn=refresh_fn,
    )
    cfg = load_nat_config(root=config_root)
    if refresh_fn is not None:
        cfg = refresh_fn(dict(cfg))
    return dict(cfg)


def ensure_outdir(
    outdir: str | Path,
) -> Path:
    """Create and return an output directory path."""
    path = Path(outdir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_latest_dir(
    root: str | Path,
    *,
    pattern: str = "*",
    must_contain: str | None = None,
) -> Path | None:
    """Return the newest matching directory under ``root``."""
    base = Path(root).expanduser().resolve()
    if not base.exists():
        return None

    cands = []
    for path in base.glob(pattern):
        if not path.is_dir():
            continue
        if must_contain is not None:
            marker = path / must_contain
            if not marker.exists():
                continue
        cands.append(path)

    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def add_config_args(
    parser: argparse.ArgumentParser,
    *,
    include_root: bool = True,
    include_set: bool = True,
) -> argparse.ArgumentParser:
    """Add shared config installation and override arguments."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional config.py to install into the active "
            "config root before running."
        ),
    )
    if include_root:
        parser.add_argument(
            "--config-root",
            type=str,
            default="nat.com",
            help="Config root directory.",
        )
    if include_set:
        parser.add_argument(
            "--set",
            dest="sets",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help=(
                "Extra config override. Repeat as needed, "
                "for example --set TIME_STEPS=6."
            ),
        )
    return parser


def add_city_arg(
    parser: argparse.ArgumentParser,
    *,
    dest: str = "city",
    default: str | None = None,
    required: bool = False,
    action: str | None = None,
    help: str | None = None,
) -> argparse.ArgumentParser:
    """Add one or repeated ``--city`` arguments."""
    kwargs: dict[str, object] = {
        "dest": dest,
        "type": str,
        "default": default,
        "required": required,
        "help": help
        or "Override CITY_NAME for this command.",
    }
    if action is not None:
        kwargs["action"] = action
    parser.add_argument("--city", **kwargs)
    return parser


def add_model_arg(
    parser: argparse.ArgumentParser,
    *,
    dest: str = "model",
    default: str | None = None,
    required: bool = False,
    help: str | None = None,
) -> argparse.ArgumentParser:
    """Add ``--model`` argument."""
    parser.add_argument(
        "--model",
        dest=dest,
        type=str,
        default=default,
        required=required,
        help=help or "Override MODEL_NAME for this command.",
    )
    return parser


def add_results_dir_arg(
    parser: argparse.ArgumentParser,
    *,
    dest: str = "results_dir",
    default: str | None = None,
) -> argparse.ArgumentParser:
    """Add results directory argument with a root alias."""
    parser.add_argument(
        "--results-dir",
        "--results-root",
        dest=dest,
        type=str,
        default=default,
        help=(
            "Results directory or results root. Both option names "
            "map to the same destination."
        ),
    )
    return parser


def add_manifest_arg(
    parser: argparse.ArgumentParser,
    *,
    dest: str = "manifest",
    option: str = "--manifest",
    help_text: str | None = None,
) -> argparse.ArgumentParser:
    """Add a manifest path argument."""
    parser.add_argument(
        option,
        dest=dest,
        type=str,
        default=None,
        help=help_text or "Explicit manifest path.",
    )
    return parser


def add_stage1_dir_arg(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add ``--stage1-dir`` argument."""
    parser.add_argument(
        "--stage1-dir",
        type=str,
        default=None,
        help="Stage-1 output directory.",
    )
    return parser


def add_outdir_arg(
    parser: argparse.ArgumentParser,
    *,
    dest: str = "outdir",
    default: str | None = None,
    required: bool = False,
    help: str | None = None,
) -> argparse.ArgumentParser:
    """Add output directory argument."""
    parser.add_argument(
        "--outdir",
        dest=dest,
        type=str,
        default=default,
        required=required,
        help=help or "Output directory.",
    )
    return parser


def add_output_format_arg(
    parser: argparse.ArgumentParser,
    *,
    choices: tuple[str, ...] = (
        "csv",
        "json",
        "npz",
        "parquet",
    ),
    default: str | None = None,
) -> argparse.ArgumentParser:
    """Add a reusable output format argument."""
    parser.add_argument(
        "--format",
        type=str,
        choices=choices,
        default=default,
        help="Preferred output format.",
    )
    return parser


def add_output_stem_arg(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = None,
) -> argparse.ArgumentParser:
    """Add an output stem argument for multi-file commands."""
    parser.add_argument(
        "--output-stem",
        type=str,
        default=default,
        help=(
            "Base file stem used when the command writes multiple "
            "related outputs."
        ),
    )
    return parser


def add_stage2_manifest_arg(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add ``--stage2-manifest`` argument."""
    parser.add_argument(
        "--stage2-manifest",
        type=str,
        default=None,
        help="Stage-2 manifest path.",
    )
    return parser


def add_split_arg(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = None,
    choices: tuple[str, ...] | None = None,
) -> argparse.ArgumentParser:
    """Add a reusable dataset split argument."""
    parser.add_argument(
        "--split",
        type=str,
        default=default,
        choices=choices,
        help="Dataset split name.",
    )
    return parser


def add_validation_csv_arg(
    parser: argparse.ArgumentParser,
    *,
    required: bool = False,
) -> argparse.ArgumentParser:
    """Add external validation CSV argument."""
    parser.add_argument(
        "--validation-csv",
        type=str,
        default=None,
        required=required,
        help="External validation CSV path.",
    )
    return parser
