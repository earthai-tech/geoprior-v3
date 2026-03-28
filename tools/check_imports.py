from __future__ import annotations

import argparse
import fnmatch
import importlib
import pkgutil
import sys
import time
import traceback as tb_mod
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PACKAGES = ["geoprior", "scripts"]


@dataclass(slots=True)
class ModuleResult:
    """Store the result of one import check."""

    name: str
    ok: bool
    seconds: float
    error: str | None = None
    traceback: str | None = None
    skipped: bool = False


@dataclass(slots=True)
class PackageResult:
    """Store results for one package."""

    name: str
    discovered: int = 0
    checked: int = 0
    imported: int = 0
    failed: int = 0
    skipped: int = 0
    modules: list[ModuleResult] = field(default_factory=list)


def _format_seconds(seconds: float) -> str:
    """Format an elapsed time."""

    return f"{seconds:.3f}s"


def _find_repo_root(start: Path | None = None) -> Path:
    """Return the nearest project root."""

    here = (start or Path.cwd()).resolve()

    for root in (here, *here.parents):
        if (root / "pyproject.toml").exists():
            return root
        if (root / ".git").exists():
            return root

    return here


def _add_repo_root_to_syspath(repo_root: Path) -> None:
    """Prepend the repo root to ``sys.path``."""

    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _matches_any(
    name: str,
    patterns: list[str],
) -> bool:
    """Return ``True`` when a name matches a pattern."""

    return any(
        fnmatch.fnmatch(name, pattern) for pattern in patterns
    )


def _discover_modules(
    pkg_name: str,
) -> tuple[list[str], str | None]:
    """Discover a package and all its submodules."""

    try:
        pkg = importlib.import_module(pkg_name)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        return [], err

    names: list[str] = [pkg.__name__]

    if not hasattr(pkg, "__path__"):
        return names, None

    prefix = pkg.__name__ + "."
    seen = {pkg.__name__}

    for mod in pkgutil.walk_packages(
        pkg.__path__,
        prefix=prefix,
    ):
        if mod.name not in seen:
            names.append(mod.name)
            seen.add(mod.name)

    names.sort()
    return names, None


def _import_one(
    mod_name: str,
    *,
    show_traceback: bool,
) -> ModuleResult:
    """Import one module and record the outcome."""

    start = time.perf_counter()

    try:
        importlib.import_module(mod_name)
    except Exception as exc:
        seconds = time.perf_counter() - start
        trace = None

        if show_traceback:
            trace = tb_mod.format_exc()

        return ModuleResult(
            name=mod_name,
            ok=False,
            seconds=seconds,
            error=f"{type(exc).__name__}: {exc}",
            traceback=trace,
        )

    seconds = time.perf_counter() - start
    return ModuleResult(
        name=mod_name,
        ok=True,
        seconds=seconds,
    )


def check_package(
    pkg_name: str,
    *,
    exclude: list[str] | None = None,
    show_traceback: bool = False,
    fail_fast: bool = False,
) -> PackageResult:
    """Import all modules under one package."""

    patterns = exclude or []
    result = PackageResult(name=pkg_name)

    names, discover_error = _discover_modules(pkg_name)

    if discover_error is not None:
        result.discovered = 1
        result.checked = 1
        result.failed = 1
        result.modules.append(
            ModuleResult(
                name=pkg_name,
                ok=False,
                seconds=0.0,
                error=discover_error,
                traceback=None,
            )
        )
        return result

    result.discovered = len(names)

    for mod_name in names:
        if _matches_any(mod_name, patterns):
            result.skipped += 1
            result.modules.append(
                ModuleResult(
                    name=mod_name,
                    ok=True,
                    seconds=0.0,
                    skipped=True,
                )
            )
            continue

        mod_result = _import_one(
            mod_name,
            show_traceback=show_traceback,
        )
        result.modules.append(mod_result)
        result.checked += 1

        if mod_result.ok:
            result.imported += 1
        else:
            result.failed += 1
            if fail_fast:
                break

    return result


def _print_module_result(
    mod: ModuleResult,
    *,
    verbosity: int,
) -> None:
    """Print one module result."""

    if mod.skipped:
        if verbosity >= 2:
            print(f"  SKIP {mod.name}")
        return

    status = "OK" if mod.ok else "FAIL"
    timing = _format_seconds(mod.seconds)

    print(f"  {status:<4} {mod.name} [{timing}]")

    if not mod.ok and mod.error:
        print(f"       {mod.error}")

    if not mod.ok and mod.traceback:
        for line in mod.traceback.rstrip().splitlines():
            print(f"       {line}")


def _print_package_summary(
    pkg: PackageResult,
) -> None:
    """Print a per-package summary."""

    print(f"\n=== Checking package: {pkg.name} ===")
    print(f"Discovered: {pkg.discovered}")
    print(f"Checked:    {pkg.checked}")
    print(f"Imported:   {pkg.imported}")
    print(f"Failed:     {pkg.failed}")
    print(f"Skipped:    {pkg.skipped}")


def _print_overall_summary(
    results: list[PackageResult],
) -> None:
    """Print an overall summary."""

    discovered = sum(r.discovered for r in results)
    checked = sum(r.checked for r in results)
    imported = sum(r.imported for r in results)
    failed = sum(r.failed for r in results)
    skipped = sum(r.skipped for r in results)

    print("\n=== Overall summary ===")
    print(f"Discovered: {discovered}")
    print(f"Checked:    {checked}")
    print(f"Imported:   {imported}")
    print(f"Failed:     {failed}")
    print(f"Skipped:    {skipped}")


def _print_slowest(
    results: list[PackageResult],
    *,
    top_n: int,
) -> None:
    """Print the slowest successful imports."""

    imported = [
        mod
        for pkg in results
        for mod in pkg.modules
        if mod.ok and not mod.skipped
    ]

    if not imported:
        return

    slowest = sorted(
        imported,
        key=lambda mod: mod.seconds,
        reverse=True,
    )[:top_n]

    print(f"\n=== Slowest imports (top {top_n}) ===")
    for mod in slowest:
        print(
            f"  {mod.name:<45} {_format_seconds(mod.seconds)}"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Import-check all modules under one or "
            "more packages."
        )
    )

    parser.add_argument(
        "packages",
        nargs="*",
        default=DEFAULT_PACKAGES,
        help=(
            "Packages to check. Defaults to: geoprior scripts"
        ),
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Project root to prepend to sys.path. "
            "Defaults to the nearest pyproject.toml "
            "or .git root."
        ),
    )

    parser.add_argument(
        "-x",
        "--exclude",
        action="append",
        default=[],
        metavar="PATTERN",
        help=(
            "Glob pattern of module names to skip. "
            "Repeat as needed."
        ),
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed import.",
    )

    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Print full tracebacks for failures.",
    )

    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only summaries.",
    )

    parser.add_argument(
        "--slowest",
        type=int,
        default=0,
        metavar="N",
        help=("Print the N slowest successful imports."),
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=("Increase verbosity. Use -v, -vv, or -vvv."),
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        help=("Decrease verbosity. Use -q or -qq."),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run import checks for project packages."""

    parser = build_parser()
    args = parser.parse_args(argv)

    verbosity = max(0, args.verbose - args.quiet)
    show_traceback = args.traceback or verbosity >= 3

    repo_root = (
        args.repo_root.resolve()
        if args.repo_root is not None
        else _find_repo_root()
    )
    _add_repo_root_to_syspath(repo_root)

    if verbosity >= 1 and not args.summary_only:
        print(f"Repo root: {repo_root}")
        print(f"Packages:  {', '.join(args.packages)}")

        if args.exclude:
            print(
                "Exclude:   "
                + ", ".join(sorted(args.exclude))
            )

    results: list[PackageResult] = []

    for pkg_name in args.packages:
        pkg_result = check_package(
            pkg_name,
            exclude=args.exclude,
            show_traceback=show_traceback,
            fail_fast=args.fail_fast,
        )
        results.append(pkg_result)

        _print_package_summary(pkg_result)

        if not args.summary_only:
            for mod in pkg_result.modules:
                if verbosity >= 1 or not mod.ok:
                    _print_module_result(
                        mod,
                        verbosity=verbosity,
                    )

        if args.fail_fast and pkg_result.failed:
            break

    _print_overall_summary(results)

    if args.slowest > 0:
        _print_slowest(results, top_n=args.slowest)

    total_failed = sum(pkg.failed for pkg in results)
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
