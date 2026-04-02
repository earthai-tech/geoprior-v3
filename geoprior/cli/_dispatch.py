# SPDX-License-Identifier: Apache-2.0
r"""
Low-level dispatch helpers for GeoPrior command-line entry points.

This module provides the internal building blocks used by the GeoPrior
CLI frontends to resolve commands, import their implementation modules,
and execute the appropriate entry callable with a uniform interface.

It is intentionally small and generic. The higher-level command
registry, family routing, and user-facing help pages live elsewhere,
while this module focuses on the reusable mechanics required by all
CLI entry points.

Overview
--------
The dispatcher infrastructure revolves around a small command
description object, :class:`CommandSpec`, and a set of helper
functions that perform four main tasks:

1. Import a command module from a registry entry.
2. Load the entry callable exposed by that module.
3. Execute the callable or module with the correct delegated
   ``argv`` and program name.
4. Build compact command/alias listings for help output.

Design goals
------------
This module is designed to keep CLI execution:

- **registry-driven**
  so commands can be described declaratively rather than hard-coded,
- **lightweight**
  so frontends such as ``geoprior``, ``geoprior-run``,
  ``geoprior-build``, and ``geoprior-plot`` can share the same
  dispatch machinery,
- **backward-compatible**
  so legacy command names and aliases can still resolve cleanly,
- **implementation-agnostic**
  so a command may expose either a callable entrypoint or a
  module-style ``__main__`` execution path.

Execution model
---------------
A command is described by :class:`CommandSpec`, which records the
target package, module, callable name, public command name, family,
and accepted aliases.

Given such a specification, the helpers in this module support two
execution styles:

``call_entry``
    Load a Python callable and invoke it using the most compatible
    calling convention discovered from its signature.

``run_module``
    Execute a module as ``__main__`` after temporarily patching
    :data:`sys.argv`, which is useful for module-oriented CLI code.

The module also provides small helpers such as :func:`alias_map`,
:func:`public_items`, and :func:`print_help_table` to support
consistent help rendering across command families.

Notes
-----
This module is primarily internal and is not intended to contain
domain-specific workflow logic. It should remain focused on generic
dispatch behavior only.

See Also
--------
geoprior.cli.__main__
    Family-aware public CLI frontends built on top of this module.
geoprior.scripts.registry
    Registry definitions consumed by the dispatch layer.
"""

from __future__ import annotations

import importlib
import inspect
import runpy
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    package: str
    mod: str
    fn: str
    desc: str
    mode: str = "argv"  # argv | sysargv | module
    family: str | None = None
    public_name: str | None = None
    aliases: tuple[str, ...] = ()
    legacy_names: tuple[str, ...] = ()


CommandSpec.__doc__ = r"""
Immutable command description used by the CLI dispatch layer.

A :class:`CommandSpec` stores the minimal metadata needed to resolve
and execute a command from the registry-driven GeoPrior CLI system.

Each instance identifies

- the Python package containing the command,
- the module to import,
- the callable name to execute,
- a short human-readable description,
- the execution mode,
- the command family and public-facing names.

Attributes
----------
package : str
    Package path used for relative imports. If empty, ``mod`` is
    imported as an absolute module path.
mod : str
    Module name containing the command implementation.
fn : str
    Name of the callable entrypoint expected inside ``mod`` when the
    execution mode is callable-based.
desc : str
    Short help text shown in command listings.
mode : str, default="argv"
    Execution strategy used by the dispatcher.

    Supported values are typically:

    - ``"argv"`` for callable entrypoints that accept delegated
      argument lists,
    - ``"sysargv"`` for callables relying on patched
      :data:`sys.argv`,
    - ``"module"`` for modules executed through :mod:`runpy`.
family : str or None, default=None
    Optional command family such as ``"run"``, ``"build"``, or
    ``"plot"``.
public_name : str or None, default=None
    Canonical public command name exposed to users.
aliases : tuple of str, default=()
    Alternative spellings that resolve to ``public_name``.
legacy_names : tuple of str, default=()
    Backward-compatible historical command names.
"""


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
