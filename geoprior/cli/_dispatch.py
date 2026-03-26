# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import inspect
import runpy
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    """Shared command description for CLI dispatch."""

    package: str
    mod: str
    fn: str
    desc: str
    mode: str = "argv"  # argv | sysargv | module
    family: str | None = None
    public_name: str | None = None
    aliases: tuple[str, ...] = ()
    legacy_names: tuple[str, ...] = ()


def load_module(spec: CommandSpec):
    """Import a module for a command spec."""
    if spec.package:
        return importlib.import_module(
            f".{spec.mod}",
            package=spec.package,
        )
    return importlib.import_module(spec.mod)


def load_callable(spec: CommandSpec) -> Callable[..., None]:
    """Load the entry callable from a command module."""
    mod = load_module(spec)
    fn = getattr(mod, spec.fn, None)
    if fn is None:
        raise AttributeError(
            f"Missing {spec.fn!r} in {spec.package}.{spec.mod}"
        )
    return fn


def run_module(
    spec: CommandSpec,
    *,
    display_cmd: str,
    argv: list[str] | None,
) -> None:
    """Execute a module as __main__ with delegated argv."""
    old = list(sys.argv)
    sys.argv = [display_cmd]
    if argv:
        sys.argv += list(argv)

    mod_name = (
        f"{spec.package}.{spec.mod}"
        if spec.package
        else spec.mod
    )
    try:
        runpy.run_module(mod_name, run_name="__main__")
    finally:
        sys.argv = old


def call_entry(
    fn: Callable[..., None],
    *,
    argv: list[str] | None,
    display_cmd: str,
) -> None:
    """
    Call a command entrypoint.

    Preference order:
    1) fn(argv, prog=...)
    2) fn(argv)
    3) fn() with patched sys.argv
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None

    if sig is not None:
        params = sig.parameters
        if "prog" in params:
            fn(argv, prog=display_cmd)
            return

        positional = [
            p
            for p in params.values()
            if p.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        if positional:
            fn(argv)
            return

    old = list(sys.argv)
    sys.argv = [display_cmd]
    if argv:
        sys.argv += list(argv)
    try:
        fn()
    finally:
        sys.argv = old


def public_items(
    registry: dict[str, CommandSpec],
    *,
    family: str | None = None,
) -> list[tuple[str, CommandSpec]]:
    """Return public registry items filtered by family."""
    items = []
    for name, spec in registry.items():
        if spec.public_name is None:
            continue
        if family is not None and spec.family != family:
            continue
        items.append((spec.public_name, spec))
    items.sort(key=lambda x: x[0])
    return items


def alias_map(
    registry: dict[str, CommandSpec],
    *,
    family: str | None = None,
) -> dict[str, str]:
    """Build alias -> public-name mapping."""
    amap: dict[str, str] = {}
    for spec in registry.values():
        public_name = spec.public_name
        if public_name is None:
            continue
        if family is not None and spec.family != family:
            continue

        for alias in spec.aliases:
            amap[alias] = public_name
        for legacy in spec.legacy_names:
            amap[legacy] = public_name
    return amap


def print_help_table(
    title: str,
    items: Iterable[tuple[str, CommandSpec]],
    *,
    width: int = 30,
) -> None:
    """Print a compact help table."""
    shown = list(items)
    if not shown:
        return

    print(f"{title}:")
    for name, spec in shown:
        left = ("  " + name).ljust(width)
        print(f"{left}{spec.desc}")
    print("")
