# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class _CmdSpec:
    """CLI command specification."""

    mod: str
    fn: str
    desc: str
    mode: str = "argv"  # "argv" | "sysargv" | "module"
    family: str = "run"  # "run" | "build" | "plot"


_CMD: dict[str, _CmdSpec] = {
    # Run commands
    "init-config": _CmdSpec(
        "init_config",
        "main",
        "Create nat.com/config.py interactively.",
        mode="argv",
        family="run",
    ),
    "stage1-preprocess": _CmdSpec(
        "stage1",
        "stage1_main",
        "Stage-1 preprocessing and export.",
        family="run",
    ),
    "stage2-train": _CmdSpec(
        "stage2",
        "stage2_main",
        "Stage-2 training.",
        mode="argv",
        family="run",
    ),
    "stage3-tune": _CmdSpec(
        "stage3",
        "stage3_main",
        "Stage-3 hyperparameter tuning.",
        mode="argv",
        family="run",
    ),
    "stage4-infer": _CmdSpec(
        "stage4",
        "stage4_main",
        "Stage-4 inference.",
        mode="argv",
        family="run",
    ),
    "stage5-transfer": _CmdSpec(
        "stage5",
        "stage5_main",
        "Stage-5 transfer evaluation.",
        mode="argv",
        family="run",
    ),
    "sensitivity": _CmdSpec(
        "run_sensitivity",
        "sensitivity_main",
        "Physics sensitivity grid driver.",
        mode="argv",
        family="run",
    ),
    "sm3-identifiability": _CmdSpec(
        "sm3_synthetic_identifiability",
        "sm3_identifiability_main",
        "SM3 synthetic identifiability.",
        mode="argv",
        family="run",
    ),
    "sm3-offset-diagnostics": _CmdSpec(
        "sm3_log_offsets_diagnostics",
        "sm3_offsets_main",
        "SM3 log-offset diagnostics.",
        mode="argv",
        family="run",
    ),
    "sm3-suite": _CmdSpec(
        "run_sm3_suite",
        "sm3_suite_main",
        "Preset-driven SM3 multi-regime suite runner.",
        mode="argv",
        family="run",
    ),
    # Build commands
    "full-inputs-npz": _CmdSpec(
        "build_full_inputs_npz",
        "build_full_inputs_main",
        "Build merged full_inputs.npz from Stage-1 splits.",
        mode="argv",
        family="build",
    ),
    "physics-payload-npz": _CmdSpec(
        "build_physics_payload_npz",
        "build_physics_payload_main",
        "Build a physics payload NPZ from a model and inputs.",
        mode="argv",
        family="build",
    ),
    "external-validation-fullcity": _CmdSpec(
        "build_external_validation_fullcity",
        "build_external_validation_fullcity_main",
        "Build full-city external validation artifacts and metrics.",
        mode="argv",
        family="build",
    ),
    "sm3-collect-summaries": _CmdSpec(
        "build_sm3_collect_summaries",
        "build_sm3_collect_main",
        "Build one combined SM3 summary table for a suite.",
        mode="argv",
        family="build",
    ),
    "assign-boreholes": _CmdSpec(
        "build_assign_boreholes",
        "build_assign_boreholes_main",
        "Build nearest-city borehole assignment tables.",
        mode="argv",
        family="build",
    ),
    "add-zsurf-from-coords": _CmdSpec(
        "build_add_zsurf_from_coords",
        "build_add_zsurf_main",
        "Build z_surf-enriched harmonized datasets.",
        mode="argv",
        family="build",
    ),
    "external-validation-metrics": _CmdSpec(
        "build_external_validation_metrics",
        "build_external_validation_metrics_main",
        "Build external validation metrics from Stage-1 and payload artifacts.",
        mode="argv",
        family="build",
    ),
}


_ALIASES: dict[str, str] = {
    # bootstrap
    "init": "init-config",
    "bootstrap": "init-config",
    # stage 1
    "stage1": "stage1-preprocess",
    "s1": "stage1-preprocess",
    "preprocess": "stage1-preprocess",
    "prepare": "stage1-preprocess",
    # stage 2
    "stage2": "stage2-train",
    "s2": "stage2-train",
    "train": "stage2-train",
    "fit": "stage2-train",
    # stage 3
    "stage3": "stage3-tune",
    "s3": "stage3-tune",
    "tune": "stage3-tune",
    "tuning": "stage3-tune",
    "search": "stage3-tune",
    # stage 4
    "stage4": "stage4-infer",
    "s4": "stage4-infer",
    "infer": "stage4-infer",
    "inference": "stage4-infer",
    "predict": "stage4-infer",
    "forecast": "stage4-infer",
    # stage 5
    "stage5": "stage5-transfer",
    "s5": "stage5-transfer",
    "transfer": "stage5-transfer",
    "xfer": "stage5-transfer",
    # sensitivity
    "sens": "sensitivity",
    "lambda-sensitivity": "sensitivity",
    "run-sensitivity": "sensitivity",
    # SM3
    "identifiability": "sm3-identifiability",
    "ident": "sm3-identifiability",
    "indetifiability": "sm3-identifiability",
    "sm3-ident": "sm3-identifiability",
    "offset-diagnostics": "sm3-offset-diagnostics",
    "offsets": "sm3-offset-diagnostics",
    "sm3-offsets": "sm3-offset-diagnostics",
    "sm3-regimes": "sm3-suite",
    "sm3-preset": "sm3-suite",
    "sm3-batch": "sm3-suite",
    # build
    "build-full-inputs": "full-inputs-npz",
    "make-full-inputs": "full-inputs-npz",
    "full-inputs": "full-inputs-npz",
    "merge-inputs": "full-inputs-npz",
    "physics-payload": "physics-payload-npz",
    "payload-npz": "physics-payload-npz",
    "full-city-payload": "physics-payload-npz",
    "fullcity-payload": "physics-payload-npz",
    "export-physics-payload": "physics-payload-npz",
    "external-validation": "external-validation-fullcity",
    "fullcity-validation": "external-validation-fullcity",
    "validate-fullcity": "external-validation-fullcity",
    "ext-validation": "external-validation-fullcity",
    "sm3-summaries": "sm3-collect-summaries",
    "collect-summaries": "sm3-collect-summaries",
    "collect-sm3": "sm3-collect-summaries",
    "combined-sm3-summary": "sm3-collect-summaries",
    "classify-boreholes": "assign-boreholes",
    "borehole-city-assignment": "assign-boreholes",
    "boreholes-by-city": "assign-boreholes",
    "split-boreholes": "assign-boreholes",
    "add-zsurf": "add-zsurf-from-coords",
    "merge-zsurf": "add-zsurf-from-coords",
    "zsurf-from-coords": "add-zsurf-from-coords",
    "harmonized-zsurf": "add-zsurf-from-coords",
    "borehole-validation": "external-validation-metrics",
    "compute-external-validation": "external-validation-metrics",
}


_FAMILY_ALIASES: dict[str, str] = {
    "run": "run",
    "build": "build",
    "make": "build",
    "plot": "plot",
}


_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Pipeline",
        (
            "init-config",
            "stage1-preprocess",
            "stage2-train",
            "stage3-tune",
            "stage4-infer",
            "stage5-transfer",
            "sensitivity",
        ),
    ),
    (
        "Supplementary diagnostics",
        (
            "sm3-identifiability",
            "sm3-offset-diagnostics",
        ),
    ),
    (
        "Build commands",
        (
            "full-inputs-npz",
            "physics-payload-npz",
            "external-validation-fullcity",
            "sm3-collect-summaries",
            "assign-boreholes",
            "external-validation-metrics",
        ),
    ),
)


def _load_module(mod_name: str):
    """Import a CLI sibling module from ``geoprior.cli``."""
    pkg = __package__ or "geoprior.cli"
    return importlib.import_module(
        f".{mod_name}", package=pkg
    )


def _load_callable(spec: _CmdSpec) -> Callable[..., None]:
    """Load the target callable for ``argv``/``sysargv`` modes."""
    mod = _load_module(spec.mod)
    fn = getattr(mod, spec.fn, None)
    if fn is None:
        raise AttributeError(
            f"Missing {spec.fn!r} in geoprior.cli.{spec.mod}"
        )
    return fn


def _run_module(spec: _CmdSpec) -> None:
    """Execute a module for commands that run at import time."""
    _load_module(spec.mod)


def _auto_prog_name() -> str:
    """Return a best-effort program label."""
    argv0 = (sys.argv[0] or "").strip()
    if argv0.endswith("__main__.py"):
        return "python -m geoprior.cli"
    return argv0 or "geoprior"


def _entry_prog(family: str | None) -> str:
    """Return canonical console script name for a family."""
    if family == "run":
        return "geoprior-run"
    if family == "build":
        return "geoprior-build"
    if family == "plot":
        return "geoprior-plot"
    return "geoprior"


def _display_cmdline(
    prog: str,
    family: str,
    cmd: str,
    *,
    fixed_family: str | None,
) -> str:
    """Return display command line used in delegated sys.argv."""
    if fixed_family is None:
        return f"{prog} {family} {cmd}"
    return f"{prog} {cmd}"


def _call_with_sysargv(
    fn: Callable[[], None],
    display_cmd: str,
    argv: list[str] | None,
) -> None:
    """Call a zero-argument CLI function after patching ``sys.argv``."""
    old = list(sys.argv)
    sys.argv = [display_cmd]
    if argv:
        sys.argv += list(argv)
    try:
        fn()
    finally:
        sys.argv = old


def _family_cmds(family: str) -> tuple[str, ...]:
    """Return commands registered under a family."""
    return tuple(
        cmd
        for cmd, spec in _CMD.items()
        if spec.family == family
    )


def _print_group(
    title: str,
    cmds: tuple[str, ...],
    *,
    family: str | None = None,
) -> None:
    """Print a help group filtered by family."""
    shown = []
    for cmd in cmds:
        spec = _CMD.get(cmd)
        if spec is None:
            continue
        if family is not None and spec.family != family:
            continue
        shown.append((cmd, spec.desc))

    if not shown:
        return

    print(f"{title}:")
    width = 28
    for cmd, desc in shown:
        left = ("  " + cmd).ljust(width)
        print(f"{left}{desc}")
    print("")


def _print_aliases(*, family: str | None = None) -> None:
    """Print aliases filtered by family."""
    items = []
    for src in sorted(_ALIASES):
        target = _ALIASES[src]
        spec = _CMD.get(target)
        if spec is None:
            continue
        if family is not None and spec.family != family:
            continue
        items.append((src, target))

    if not items:
        return

    print("Aliases:")
    for src, target in items:
        print(f"  {src} -> {target}")
    print("")


def _print_help(
    *,
    fixed_family: str | None = None,
    prog: str | None = None,
) -> None:
    """Print help for root or a fixed family entry point."""
    prog_name = prog or _auto_prog_name()

    if fixed_family is None:
        print("Usage:")
        print(f"  {prog_name} run <command> [args]")
        print(f"  {prog_name} build <command> [args]")
        print(f"  {prog_name} plot <command> [args]")
        print("")
        print("Families:")
        print(
            "  run   Execute pipeline or diagnostic workflows."
        )
        print(
            "  build Materialize deterministic artifacts "
            "from existing outputs."
        )
        print("  make  Alias of build.")
        print("  plot  Plotting and figure commands.")
        print("")

        for title, cmds in _GROUPS:
            _print_group(title, cmds)

        _print_aliases()

        print("Examples:")
        print(f"  {prog_name} run stage1-preprocess")
        print(
            f"  {prog_name} run sensitivity --epochs 10 --gold"
        )
        print(
            f"  {prog_name} build full-inputs-npz "
            "--stage1-dir results/foo_stage1"
        )
        print(f"  {prog_name} make full-inputs")
        print(f"  {prog_name} plot <command>")
        return

    print("Usage:")
    print(f"  {prog_name} <command> [args]")
    print("")

    if fixed_family == "run":
        print("Run commands:")
        print("")
    elif fixed_family == "build":
        print("Build commands:")
        print("")
    elif fixed_family == "plot":
        print("Plot commands:")
        print("")

    for title, cmds in _GROUPS:
        _print_group(title, cmds, family=fixed_family)

    extras = tuple(
        sorted(
            cmd
            for cmd, spec in _CMD.items()
            if spec.family == fixed_family
            and all(cmd not in group for _, group in _GROUPS)
        )
    )
    if extras:
        _print_group("Other", extras, family=fixed_family)

    if fixed_family == "plot" and not _family_cmds("plot"):
        print("  No plot commands are registered yet.\n")

    _print_aliases(family=fixed_family)

    print("Examples:")
    if fixed_family == "run":
        print(f"  {prog_name} stage1-preprocess")
        print(f"  {prog_name} stage4-infer --help")
        print(f"  {prog_name} sensitivity --epochs 10 --gold")
    elif fixed_family == "build":
        print(
            f"  {prog_name} full-inputs-npz --stage1-dir results/foo_stage1"
        )
        print(
            f"  {prog_name} full-inputs --manifest path/to/manifest.json"
        )
    else:
        print(f"  {prog_name} <command>")


def _resolve_entry_command(
    args: list[str],
    *,
    fixed_family: str | None,
    prog: str,
) -> tuple[str, list[str], str]:
    """Resolve command token and target family from input args."""
    if fixed_family is None:
        family = _FAMILY_ALIASES.get(args[0])
        if family is None:
            print(
                "[ERR] Root entry point requires a command family "
                "first: run, build, make, or plot."
            )
            print("")
            _print_help(fixed_family=None, prog=prog)
            raise SystemExit(2)

        if len(args) == 1 or args[1] in {
            "-h",
            "--help",
            "help",
        }:
            _print_help(
                fixed_family=family, prog=f"{prog} {args[0]}"
            )
            raise SystemExit(0)

        cmd_token = args[1]
        rest = args[2:]
        cmd = _ALIASES.get(cmd_token, cmd_token)
        return cmd, rest, family

    repeated = _FAMILY_ALIASES.get(args[0])
    if repeated is not None:
        print(
            f"[ERR] {prog!r} already implies the {fixed_family!r} family."
        )
        print(
            f"[HINT] Use: {prog} <command> [args] "
            f"instead of {prog} {args[0]} <command>."
        )
        raise SystemExit(2)

    cmd = _ALIASES.get(args[0], args[0])
    rest = args[1:]
    return cmd, rest, fixed_family


def _dispatch(
    argv: list[str] | None = None,
    *,
    fixed_family: str | None = None,
    prog: str | None = None,
) -> None:
    """Dispatch GeoPrior commands with optional fixed family."""
    args = list(argv) if argv is not None else sys.argv[1:]
    prog_name = prog or _auto_prog_name()

    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help(fixed_family=fixed_family, prog=prog_name)
        return

    cmd, rest, family = _resolve_entry_command(
        args,
        fixed_family=fixed_family,
        prog=prog_name,
    )
    spec = _CMD.get(cmd)

    if spec is None:
        print(f"[ERR] Unknown command: {cmd}")
        print("")
        _print_help(fixed_family=family, prog=prog_name)
        raise SystemExit(2)

    if spec.family != family:
        other_prog = _entry_prog(spec.family)
        print(
            f"[ERR] Command {cmd!r} belongs to the {spec.family!r} family, "
            f"not {family!r}."
        )
        print(
            f"[HINT] Use {other_prog} {cmd} or "
            f"{_entry_prog(None)} {spec.family} {cmd}."
        )
        raise SystemExit(2)

    if spec.mode == "module":
        if rest:
            print(
                f"[ERR] Command {cmd!r} does not accept delegated "
                "CLI arguments yet."
            )
            print(
                "[HINT] Refactor the stage module to expose "
                "main(argv=None) if argument forwarding is needed."
            )
            raise SystemExit(2)
        _run_module(spec)
        return

    fn = _load_callable(spec)

    if spec.mode == "sysargv":
        _call_with_sysargv(
            fn,
            _display_cmdline(
                prog_name,
                family,
                cmd,
                fixed_family=fixed_family,
            ),
            rest,
        )
        return

    fn(rest)


def main(argv: list[str] | None = None) -> None:
    """Versatile root entry point requiring a family token."""
    _dispatch(argv, fixed_family=None)


def run_main(argv: list[str] | None = None) -> None:
    """Entry point bound to the run family only."""
    _dispatch(argv, fixed_family="run")


def build_main(argv: list[str] | None = None) -> None:
    """Entry point bound to the build family only."""
    _dispatch(argv, fixed_family="build")


def plot_main(argv: list[str] | None = None) -> None:
    """Entry point bound to the plot family only."""
    _dispatch(argv, fixed_family="plot")


if __name__ == "__main__":
    main()
