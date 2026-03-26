# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

from __future__ import annotations

import sys



from ._dispatch import (
    CommandSpec,
    alias_map,
    call_entry,
    load_callable,
    print_help_table,
    run_module,
)


def _scripts_as_cli() -> dict[str, CommandSpec]:
    from scripts.registry import SCRIPT_COMMANDS
    
    items: dict[str, CommandSpec] = {}

    for spec in SCRIPT_COMMANDS.values():
        public = spec.public_name
        if public is None:
            continue

        if public in items:
            raise ValueError(
                f"Duplicate script public name: {public}"
            )

        items[public] = CommandSpec(
            package="scripts",
            mod=spec.mod,
            fn=spec.fn,
            desc=spec.desc,
            mode=spec.mode,
            family=spec.family,
            public_name=public,
            aliases=spec.aliases,
            legacy_names=spec.legacy_names,
        )

    return items


_CMD: dict[str, CommandSpec] = {
    "init-config": CommandSpec(
        package="geoprior.cli",
        mod="init_config",
        fn="main",
        desc="Create nat.com/config.py interactively.",
        mode="argv",
        family="run",
        public_name="init-config",
        aliases=("init", "bootstrap"),
    ),
    "stage1-preprocess": CommandSpec(
        package="geoprior.cli",
        mod="stage1",
        fn="stage1_main",
        desc="Stage-1 preprocessing and export.",
        mode="argv",
        family="run",
        public_name="stage1-preprocess",
        aliases=(
            "stage1",
            "s1",
            "preprocess",
            "prepare",
        ),
    ),
    "stage2-train": CommandSpec(
        package="geoprior.cli",
        mod="stage2",
        fn="stage2_main",
        desc="Stage-2 training.",
        mode="argv",
        family="run",
        public_name="stage2-train",
        aliases=(
            "stage2",
            "s2",
            "train",
            "fit",
        ),
    ),
    "stage3-tune": CommandSpec(
        package="geoprior.cli",
        mod="stage3",
        fn="stage3_main",
        desc="Stage-3 hyperparameter tuning.",
        mode="argv",
        family="run",
        public_name="stage3-tune",
        aliases=(
            "stage3",
            "s3",
            "tune",
            "tuning",
            "search",
        ),
    ),
    "stage4-infer": CommandSpec(
        package="geoprior.cli",
        mod="stage4",
        fn="stage4_main",
        desc="Stage-4 inference.",
        mode="argv",
        family="run",
        public_name="stage4-infer",
        aliases=(
            "stage4",
            "s4",
            "infer",
            "inference",
            "predict",
            "forecast",
        ),
    ),
    "stage5-transfer": CommandSpec(
        package="geoprior.cli",
        mod="stage5",
        fn="stage5_main",
        desc="Stage-5 transfer evaluation.",
        mode="argv",
        family="run",
        public_name="stage5-transfer",
        aliases=(
            "stage5",
            "s5",
            "transfer",
            "xfer",
        ),
    ),
    "sensitivity": CommandSpec(
        package="geoprior.cli",
        mod="run_sensitivity",
        fn="sensitivity_main",
        desc="Physics sensitivity grid driver.",
        mode="argv",
        family="run",
        public_name="sensitivity",
        aliases=(
            "sens",
            "lambda-sensitivity",
            "run-sensitivity",
        ),
    ),
    "sm3-identifiability": CommandSpec(
        package="geoprior.cli",
        mod="sm3_synthetic_identifiability",
        fn="sm3_identifiability_main",
        desc="SM3 synthetic identifiability.",
        mode="argv",
        family="run",
        public_name="sm3-identifiability",
        aliases=(
            "identifiability",
            "ident",
            "indetifiability",
            "sm3-ident",
        ),
    ),
    "sm3-offset-diagnostics": CommandSpec(
        package="geoprior.cli",
        mod="sm3_log_offsets_diagnostics",
        fn="sm3_offsets_main",
        desc="SM3 log-offset diagnostics.",
        mode="argv",
        family="run",
        public_name="sm3-offset-diagnostics",
        aliases=(
            "offset-diagnostics",
            "offsets",
            "sm3-offsets",
        ),
    ),
    "sm3-suite": CommandSpec(
        package="geoprior.cli",
        mod="run_sm3_suite",
        fn="sm3_suite_main",
        desc="Preset-driven SM3 multi-regime suite runner.",
        mode="argv",
        family="run",
        public_name="sm3-suite",
        aliases=(
            "sm3-regimes",
            "sm3-preset",
            "sm3-batch",
        ),
    ),
    "full-inputs-npz": CommandSpec(
        package="geoprior.cli",
        mod="build_full_inputs_npz",
        fn="build_full_inputs_main",
        desc="Build merged full_inputs.npz from splits.",
        mode="argv",
        family="build",
        public_name="full-inputs-npz",
        aliases=(
            "build-full-inputs",
            "make-full-inputs",
            "full-inputs",
            "merge-inputs",
        ),
    ),
    "physics-payload-npz": CommandSpec(
        package="geoprior.cli",
        mod="build_physics_payload_npz",
        fn="build_physics_payload_main",
        desc="Build a physics payload NPZ.",
        mode="argv",
        family="build",
        public_name="physics-payload-npz",
        aliases=(
            "physics-payload",
            "payload-npz",
            "full-city-payload",
            "fullcity-payload",
            "export-physics-payload",
        ),
    ),
    "external-validation-fullcity": CommandSpec(
        package="geoprior.cli",
        mod="build_external_validation_fullcity",
        fn="build_external_validation_fullcity_main",
        desc="Build full-city validation artifacts.",
        mode="argv",
        family="build",
        public_name="external-validation-fullcity",
        aliases=(
            "external-validation",
            "fullcity-validation",
            "validate-fullcity",
            "ext-validation",
        ),
    ),
    "sm3-collect-summaries": CommandSpec(
        package="geoprior.cli",
        mod="build_sm3_collect_summaries",
        fn="build_sm3_collect_main",
        desc="Build one combined SM3 summary table.",
        mode="argv",
        family="build",
        public_name="sm3-collect-summaries",
        aliases=(
            "sm3-summaries",
            "collect-summaries",
            "collect-sm3",
            "combined-sm3-summary",
        ),
    ),
    "assign-boreholes": CommandSpec(
        package="geoprior.cli",
        mod="build_assign_boreholes",
        fn="build_assign_boreholes_main",
        desc="Build nearest-city borehole tables.",
        mode="argv",
        family="build",
        public_name="assign-boreholes",
        aliases=(
            "classify-boreholes",
            "borehole-city-assignment",
            "boreholes-by-city",
            "split-boreholes",
        ),
    ),
    "add-zsurf-from-coords": CommandSpec(
        package="geoprior.cli",
        mod="build_add_zsurf_from_coords",
        fn="build_add_zsurf_main",
        desc="Build z_surf-enriched datasets.",
        mode="argv",
        family="build",
        public_name="add-zsurf-from-coords",
        aliases=(
            "add-zsurf",
            "merge-zsurf",
            "zsurf-from-coords",
            "harmonized-zsurf",
        ),
    ),
    "external-validation-metrics": CommandSpec(
        package="geoprior.cli",
        mod="build_external_validation_metrics",
        fn="build_external_validation_metrics_main",
        desc="Build external validation metrics.",
        mode="argv",
        family="build",
        public_name="external-validation-metrics",
        aliases=(
            "borehole-validation",
            "compute-external-validation",
        ),
    ),
}

for _name, _spec in _scripts_as_cli().items():
    if _name in _CMD:
        raise ValueError(f"Duplicate public command: {_name}")
    _CMD[_name] = _spec


_FAMILY_ALIASES = {
    "run": "run",
    "build": "build",
    "make": "build",
    "plot": "plot",
}


_GROUPS = (
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
            "sm3-suite",
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
            "add-zsurf-from-coords",
            "external-validation-metrics",
            "brier-exceedance",
            "hotspots",
            "hotspots-summary",
            "extend-forecast",
            "update-ablation-records",
            "model-metrics",
            "ablation-table",
            "boundary",
            "exposure",
            "district-grid",
            "clusters-with-zones",
        ),
    ),
    (
        "Plot commands",
        (
            "driver-response",
            "core-ablation",
            "litho-parity",
            "uncertainty",
            "spatial-forecasts",
            "physics-sanity",
            "physics-maps",
            "physics-fields",
            "physics-profiles",
            "uncertainty-extras",
            "ablations-sensitivity",
            "physics-sensitivity",
            "sm3-identifiability",
            "sm3-bounds-ridge-summary",
            "sm3-log-offsets",
            "xfer-transferability",
            "xfer-impact",
            "transfer",
            "transfer-impact",
            "geo-cumulative",
            "hotspot-analytics",
            "external-validation",
        ),
    ),
)


def _auto_prog_name() -> str:
    argv0 = (sys.argv[0] or "").strip()
    if argv0.endswith("__main__.py"):
        return "python -m geoprior.cli"
    return argv0 or "geoprior"


def _entry_prog(family: str | None) -> str:
    if family == "run":
        return "geoprior-run"
    if family == "build":
        return "geoprior-build"
    if family == "plot":
        return "geoprior-plot"
    return "geoprior"


def _display_cmd(
    prog: str,
    family: str,
    cmd: str,
    *,
    fixed_family: str | None,
) -> str:
    if fixed_family is None:
        return f"{prog} {family} {cmd}"
    return f"{prog} {cmd}"


def _family_items(
    family: str,
) -> list[tuple[str, CommandSpec]]:
    items: list[tuple[str, CommandSpec]] = []

    for name, spec in _CMD.items():
        if spec.public_name != name:
            continue
        if spec.family != family:
            continue
        items.append((name, spec))

    items.sort(key=lambda x: x[0])
    return items


def _print_help(
    *,
    fixed_family: str | None = None,
    prog: str | None = None,
) -> None:
    prog_name = prog or _auto_prog_name()

    if fixed_family is None:
        print("Usage:")
        print(f"  {prog_name} run <command> [args]")
        print(f"  {prog_name} build <command> [args]")
        print(f"  {prog_name} plot <command> [args]")
        print("")
        print("Families:")
        print("  run   Execute model workflows.")
        print("  build Materialize artifacts.")
        print("  make  Alias of build.")
        print("  plot  Render figures and maps.")
        print("")

        for title, names in _GROUPS:
            items = [
                (name, _CMD[name])
                for name in names
                if name in _CMD
            ]
            print_help_table(title, items)

        amap = alias_map(_CMD)
        if amap:
            print("Aliases:")
            for src in sorted(amap):
                print(f"  {src} -> {amap[src]}")
            print("")

        print("Examples:")
        print(f"  {prog_name} plot physics-fields --help")
        print(f"  {prog_name} build exposure --help")
        print(f"  {prog_name} run stage1-preprocess")
        return

    print("Usage:")
    print(f"  {prog_name} <command> [args]")
    print("")
    print(f"{fixed_family.title()} commands:")
    print("")

    items = _family_items(fixed_family)
    print_help_table("Commands", items)

    amap = alias_map(_CMD, family=fixed_family)
    if amap:
        print("Aliases:")
        for src in sorted(amap):
            print(f"  {src} -> {amap[src]}")
        print("")

    print("Examples:")
    if fixed_family == "plot":
        print(f"  {prog_name} physics-fields --help")
    elif fixed_family == "build":
        print(f"  {prog_name} exposure --help")
    else:
        print(f"  {prog_name} stage1-preprocess")


def _resolve(
    args: list[str],
    *,
    fixed_family: str | None,
    prog: str,
) -> tuple[str, list[str], str]:
    amap = alias_map(_CMD)

    if fixed_family is None:
        family = _FAMILY_ALIASES.get(args[0])
        if family is None:
            _print_help(
                fixed_family=None,
                prog=prog,
            )
            raise SystemExit(2)

        if len(args) == 1:
            _print_help(
                fixed_family=family,
                prog=f"{prog} {args[0]}",
            )
            raise SystemExit(0)

        if args[1] in {"-h", "--help", "help"}:
            _print_help(
                fixed_family=family,
                prog=f"{prog} {args[0]}",
            )
            raise SystemExit(0)

        cmd = amap.get(args[1], args[1])
        return cmd, args[2:], family

    repeated = _FAMILY_ALIASES.get(args[0])
    if repeated is not None:
        print(
            f"[ERR] {prog!r} already implies "
            f"the {fixed_family!r} family."
        )
        raise SystemExit(2)

    cmd = amap.get(args[0], args[0])
    return cmd, args[1:], fixed_family


def _dispatch(
    argv: list[str] | None = None,
    *,
    fixed_family: str | None = None,
    prog: str | None = None,
) -> None:
    args = list(argv) if argv is not None else sys.argv[1:]
    prog_name = prog or _auto_prog_name()

    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help(
            fixed_family=fixed_family,
            prog=prog_name,
        )
        return

    cmd, rest, family = _resolve(
        args,
        fixed_family=fixed_family,
        prog=prog_name,
    )
    spec = _CMD.get(cmd)

    if spec is None:
        print(f"[ERR] Unknown command: {cmd}")
        print("")
        _print_help(
            fixed_family=family,
            prog=prog_name,
        )
        raise SystemExit(2)

    if spec.family != family:
        print(
            f"[ERR] Command {cmd!r} belongs to "
            f"{spec.family!r}, not {family!r}."
        )
        print(f"[HINT] Use {_entry_prog(spec.family)} {cmd}")
        raise SystemExit(2)

    display_cmd = _display_cmd(
        prog_name,
        family,
        cmd,
        fixed_family=fixed_family,
    )

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


def main(argv: list[str] | None = None) -> None:
    _dispatch(argv, fixed_family=None)


def run_main(argv: list[str] | None = None) -> None:
    _dispatch(argv, fixed_family="run")


def build_main(argv: list[str] | None = None) -> None:
    _dispatch(argv, fixed_family="build")


def plot_main(argv: list[str] | None = None) -> None:
    _dispatch(argv, fixed_family="plot")


if __name__ == "__main__":
    main()
