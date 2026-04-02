# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

from __future__ import annotations

import sys

from ..scripts.registry import SCRIPT_COMMANDS
from ._dispatch import (
    CommandSpec,
    alias_map,
    call_entry,
    load_callable,
    print_help_table,
    run_module,
)


def _scripts_as_cli() -> dict[str, CommandSpec]:
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
            package="geoprior.scripts",
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
    "identifiability": CommandSpec(
        package="geoprior.cli",
        mod="sm3_synthetic_identifiability",
        fn="sm3_identifiability_main",
        desc="SM3 synthetic identifiability.",
        mode="argv",
        family="run",
        public_name="identifiability",
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
    "batch-spatial-sampling": CommandSpec(
        package="geoprior.cli",
        mod="build_batch_spatial_sampling",
        fn="build_batch_spatial_sampling_main",
        desc="Build non-overlapping spatial sample batches.",
        mode="argv",
        family="build",
        public_name="batch-spatial-sampling",
        aliases=(
            "batch-sampling",
            "batch-spatial",
            "spatial-batches",
        ),
    ),
    "spatial-sampling": CommandSpec(
        package="geoprior.cli",
        mod="build_spatial_sampling",
        fn="build_spatial_sampling_main",
        desc="Build a stratified spatial sample table.",
        mode="argv",
        family="build",
        public_name="spatial-sampling",
        aliases=(
            "sample-spatial",
            "spatial-sample",
            "sampling",
        ),
    ),
    "spatial-roi": CommandSpec(
        package="geoprior.cli",
        mod="build_spatial_roi",
        fn="build_spatial_roi_main",
        desc="Build a spatial region-of-interest table.",
        mode="argv",
        family="build",
        public_name="spatial-roi",
        aliases=(
            "roi",
            "extract-roi",
            "roi-table",
        ),
    ),
    "spatial-clusters": CommandSpec(
        package="geoprior.cli",
        mod="build_spatial_clusters",
        fn="build_spatial_clusters_main",
        desc="Build a table with spatial cluster labels.",
        mode="argv",
        family="build",
        public_name="spatial-clusters",
        aliases=(
            "cluster-spatial",
            "cluster-regions",
            "clusters",
        ),
    ),
    "forecast-ready-sample": CommandSpec(
        package="geoprior.cli",
        mod="build_forecast_ready_sample",
        fn="build_forecast_ready_sample_main",
        desc="Build a compact forecast-ready panel sample.",
        mode="argv",
        family="build",
        public_name="forecast-ready-sample",
        aliases=(
            "forecast-sample",
            "panel-sample",
            "demo-panel",
            "ready-sample",
        ),
    ),
    "extract-zones": CommandSpec(
        package="geoprior.cli",
        mod="build_extract_zones",
        fn="build_extract_zones_main",
        desc="Build a threshold-based zone extraction table.",
        mode="argv",
        family="build",
        public_name="extract-zones",
        aliases=(
            "zones",
            "zones-from",
            "zone-extract",
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
            "forecast-ready-sample",
            "batch-spatial-sampling",
            "spatial-sampling",
            "spatial-roi",
            "spatial-clusters",
            "extract-zones",
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


main.__doc__ = r"""
Run the root GeoPrior command dispatcher.

This is the top-level CLI entry point behind the generic
``geoprior`` command. It routes user input to one of the three
public command families exposed by the project:

- ``run`` for staged model workflows,
- ``build`` for artifact materialization and table generation,
- ``plot`` for figure and map rendering.

The function itself is intentionally thin. It delegates all parsing,
family resolution, alias expansion, help rendering, and command
execution to the internal dispatcher while keeping a stable public
entry point for console scripts and programmatic invocation.

Conceptually, the root command supports calls of the form

.. code-block:: bash

   geoprior run <command> [args]
   geoprior build <command> [args]
   geoprior plot <command> [args]

which mirrors the stage-wise and artifact-aware workflow adopted by
the GeoPrior project for forecasting, diagnostics, and uncertainty
analysis :cite:p:`Kouadio2025XTFT,Limetal2021`.

Parameters
----------
argv : list of str or None, default=None
    Optional command-line tokens excluding the program name.
    When ``None``, the function reads arguments from
    :data:`sys.argv`.

Returns
-------
None
    This function is executed for its side effects. It prints help,
    dispatches a command, or raises :class:`SystemExit` on invalid
    user input.

Raises
------
SystemExit
    Raised when command resolution fails, when help is requested, or
    when a delegated subcommand exits.

Notes
-----
This function is the most user-facing CLI entry point in the module.
Use it when you want the full family-aware dispatcher behavior.

For family-specific wrappers that do not require the explicit
``run``, ``build``, or ``plot`` prefix, see :func:`run_main`,
:func:`build_main`, and :func:`plot_main`.

Examples
--------
Call from Python with an explicit token list:

>>> from geoprior.cli.__main__ import main
>>> main(["run", "stage1-preprocess"])  # doctest: +SKIP

Request top-level help:

>>> main(["--help"])  # doctest: +SKIP

Dispatch a plotting command:

>>> main(["plot", "physics-fields", "--help"])  # doctest: +SKIP

See Also
--------
run_main :
    Run-family wrapper used by the ``geoprior-run`` entry point.
build_main :
    Build-family wrapper used by the ``geoprior-build`` entry point.
plot_main :
    Plot-family wrapper used by the ``geoprior-plot`` entry point.
"""


run_main.__doc__ = r"""
Run the GeoPrior dispatcher in fixed ``run`` mode.

This wrapper exposes the workflow-oriented command family behind the
``geoprior-run`` console script. Unlike :func:`main`, it does not
expect a family token as the first positional argument. Instead, it
assumes that every command belongs to the ``run`` family and rejects
attempts to repeat the family prefix.

Typical commands dispatched through this entry point include stage
execution and synthetic or sensitivity workflows, such as
preprocessing, training, tuning, inference, transfer evaluation, and
selected supplementary diagnostics.

Supported usage therefore follows the shorter form

.. code-block:: bash

   geoprior-run <command> [args]

rather than

.. code-block:: bash

   geoprior run <command> [args]

Parameters
----------
argv : list of str or None, default=None
    Optional command-line tokens excluding the program name.
    When ``None``, the function reads arguments from
    :data:`sys.argv`.

Returns
-------
None
    This function dispatches a run-family command for its side
    effects.

Raises
------
SystemExit
    Raised when the requested command is unknown, belongs to a
    different family, or explicitly requests help.

Notes
-----
This wrapper is mainly intended for console-script integration and
family-specific convenience. It is especially useful in automation,
examples, and shell documentation where repeated family prefixes
would add noise.

Examples
--------
Run stage 1 preprocessing:

>>> from geoprior.cli.__main__ import run_main
>>> run_main(["stage1-preprocess"])  # doctest: +SKIP

Inspect the help of a training command:

>>> run_main(["stage2-train", "--help"])  # doctest: +SKIP

Request family-scoped help:

>>> run_main(["--help"])  # doctest: +SKIP

See Also
--------
main :
    Root family-aware dispatcher.
build_main :
    Build-family wrapper.
plot_main :
    Plot-family wrapper.
"""


build_main.__doc__ = r"""
Run the GeoPrior dispatcher in fixed ``build`` mode.

This wrapper exposes the artifact-building command family behind the
``geoprior-build`` console script. It assumes that the requested
subcommand already belongs to the ``build`` family and therefore uses
the compact invocation form

.. code-block:: bash

   geoprior-build <command> [args]

Typical commands in this family generate or transform reproducible
artifacts needed by downstream training, validation, interpretation,
or documentation workflows. Examples include merged NPZ payloads,
external-validation tables, spatial sampling products, hotspot
summaries, and other materialized intermediate datasets.

Parameters
----------
argv : list of str or None, default=None
    Optional command-line tokens excluding the program name.
    When ``None``, the function reads arguments from
    :data:`sys.argv`.

Returns
-------
None
    This function dispatches a build-family command for its side
    effects.

Raises
------
SystemExit
    Raised when the command is unknown, belongs to another family, or
    when help is requested.

Notes
-----
Use this wrapper when you want a stable programmatic entry point for
artifact generation without exposing the full root dispatcher. This
is the recommended choice for shell examples, gallery preparation,
and reproducible data-materialization scripts built around GeoPrior.

Examples
--------
Build a full input archive:

>>> from geoprior.cli.__main__ import build_main
>>> build_main(["full-inputs-npz", "--help"])  # doctest: +SKIP

Build a compact forecast-ready panel:

>>> build_main(["forecast-ready-sample"])  # doctest: +SKIP

Show the build-family help page:

>>> build_main(["--help"])  # doctest: +SKIP

See Also
--------
main :
    Root family-aware dispatcher.
run_main :
    Run-family wrapper.
plot_main :
    Plot-family wrapper.
"""


plot_main.__doc__ = r"""
Run the GeoPrior dispatcher in fixed ``plot`` mode.

This wrapper exposes the visualization command family behind the
``geoprior-plot`` console script. It dispatches plotting commands
without requiring the explicit ``plot`` family prefix and therefore
supports the compact form

.. code-block:: bash

   geoprior-plot <command> [args]

The plot family is used to render publication-style figures,
diagnostic graphics, uncertainty summaries, spatial forecast maps,
transfer panels, and other visual outputs derived from GeoPrior
artifacts. In practice, this family is central to the project's
interpretability and reporting workflow, where forecast accuracy,
uncertainty behavior, and physically informed diagnostics must be
read together :cite:p:`Kouadio2025XTFT`.

Parameters
----------
argv : list of str or None, default=None
    Optional command-line tokens excluding the program name.
    When ``None``, the function reads arguments from
    :data:`sys.argv`.

Returns
-------
None
    This function dispatches a plot-family command for its side
    effects.

Raises
------
SystemExit
    Raised when the command is unknown, belongs to another family, or
    when help is requested.

Notes
-----
This wrapper is useful when the calling context is already
visualization-specific, such as gallery scripts, reproducible figure
pipelines, or publication packaging code.

Examples
--------
Inspect help for a physics-field plot:

>>> from geoprior.cli.__main__ import plot_main
>>> plot_main(["physics-fields", "--help"])  # doctest: +SKIP

Render a transferability figure:

>>> plot_main(["xfer-transferability"])  # doctest: +SKIP

Show the plot-family help page:

>>> plot_main(["--help"])  # doctest: +SKIP

See Also
--------
main :
    Root family-aware dispatcher.
run_main :
    Run-family wrapper.
build_main :
    Build-family wrapper.
"""

if __name__ == "__main__":
    main()
