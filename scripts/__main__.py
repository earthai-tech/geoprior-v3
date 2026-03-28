# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

from __future__ import annotations

import sys

from geoprior.cli._dispatch import (
    CommandSpec,
    alias_map,
    call_entry,
    load_callable,
    print_help_table,
    run_module,
)

from .registry import SCRIPT_COMMANDS, SCRIPT_GROUPS


def _legacy_registry() -> dict[str, CommandSpec]:
    items: dict[str, CommandSpec] = {}
    for legacy_name, spec in SCRIPT_COMMANDS.items():
        items[legacy_name] = CommandSpec(
            package="geoprior._scripts",
            mod=spec.mod,
            fn=spec.fn,
            desc=spec.desc,
            mode=spec.mode,
            family=spec.family,
            public_name=legacy_name,
            aliases=spec.aliases,
            legacy_names=(),
        )
    return items


_CMD = _legacy_registry()


def _print_help() -> None:
    print("Usage:")
    print("  python -m scripts <command> [args]")
    print("")

    for title, public_names in SCRIPT_GROUPS:
        items = [
            (
                next(
                    legacy
                    for legacy, spec in SCRIPT_COMMANDS.items()
                    if spec.public_name == name
                ),
                _CMD[
                    next(
                        legacy
                        for legacy, spec in SCRIPT_COMMANDS.items()
                        if spec.public_name == name
                    )
                ],
            )
            for name in public_names
            if any(
                spec.public_name == name
                for spec in SCRIPT_COMMANDS.values()
            )
        ]
        print_help_table(title, items)

    amap = alias_map(_CMD)
    if amap:
        print("Aliases:")
        for src in sorted(amap):
            print(f"  {src} -> {amap[src]}")
        print("")

    print("Tip:")
    print("  python -m scripts plot-physics-fields -h")
    print("")
    print("Modern entry points:")
    print("  geoprior plot physics-fields -h")
    print("  geoprior-build exposure -h")


def main(argv: list[str] | None = None) -> None:
    args = list(argv) if argv is not None else sys.argv[1:]

    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help()
        return

    amap = alias_map(_CMD)
    cmd = amap.get(args[0], args[0])
    rest = args[1:]
    spec = _CMD.get(cmd)

    if spec is None:
        print(f"[ERR] Unknown command: {cmd}")
        print("")
        _print_help()
        raise SystemExit(2)

    display_cmd = f"python -m scripts {cmd}"

    if spec.mode == "module":
        run_module(
            spec,
            display_cmd=display_cmd,
            argv=rest,
        )
        return

    fn = load_callable(spec)
    call_entry(
        fn,
        argv=rest,
        display_cmd=display_cmd,
    )


if __name__ == "__main__":
    main()
