# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""Preset-driven SM3 suite runner.

This command replaces the previous shell-only SM3 launchers with a
portable Python CLI that fits the GeoPrior run-family API.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ._presets import SM3_PRESETS, SM3_REGIMES, get_sm3_preset
from .config import (
    add_config_args,
    add_outdir_arg,
    add_results_dir_arg,
    bootstrap_runtime_config,
    find_latest_dir,
)


def _int01(raw: str | int | bool | None) -> int | None:
    """Return ``0`` or ``1`` from flexible user input."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return 1 if raw else 0

    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return 1
    if text in {"0", "false", "no", "off"}:
        return 0
    raise ValueError(
        f"Expected 0/1-style value, got: {raw!r}"
    )


def _parse_csv(raw: str | None) -> list[str]:
    """Split a comma-separated string into clean tokens."""
    if raw is None:
        return []
    out = []
    for token in str(raw).split(","):
        item = token.strip()
        if item:
            out.append(item)
    return out


def _resolve_regimes(args: argparse.Namespace) -> list[str]:
    """Resolve selected regimes with shell-compatible precedence."""
    if args.regime:
        items = list(args.regime)
    else:
        items = _parse_csv(args.regimes)

    if not items:
        ids = _parse_csv(args.regime_ids)
        if ids:
            items = []
            for raw in ids:
                if not raw.isdigit():
                    raise ValueError(
                        f"Non-integer regime id: {raw!r}"
                    )
                idx = int(raw)
                if idx < 1 or idx > len(SM3_REGIMES):
                    raise ValueError(
                        f"Regime id out of range: {idx}"
                    )
                items.append(SM3_REGIMES[idx - 1])

    if not items:
        items = list(SM3_REGIMES)

    out = []
    seen: set[str] = set()
    for item in items:
        key = str(item).strip()
        if key not in SM3_REGIMES:
            known = ", ".join(SM3_REGIMES)
            raise ValueError(
                f"Unknown regime: {key!r}. Known: {known}."
            )
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _gpu_count() -> int:
    """Return detected NVIDIA GPU count via ``nvidia-smi``."""
    if shutil.which("nvidia-smi") is None:
        return 0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return 0
    lines = [
        line for line in out.splitlines() if line.strip()
    ]
    return len(lines)


def _configure_runtime_env(
    device: str,
) -> tuple[str, dict[str, str]]:
    """Prepare environment variables for CPU or GPU execution."""
    env = dict(os.environ)
    gpu_count = _gpu_count()

    chosen = str(device).strip().lower()
    if chosen not in {"auto", "cpu", "gpu"}:
        raise ValueError(
            "--device must be one of: auto, cpu, gpu."
        )

    if chosen == "gpu" and gpu_count < 1:
        raise RuntimeError(
            "DEVICE=gpu requested, but no NVIDIA GPU was detected."
        )

    if chosen == "auto":
        chosen = "gpu" if gpu_count >= 1 else "cpu"

    if chosen == "gpu":
        env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        env.setdefault(
            "TF_GPU_ALLOCATOR", "cuda_malloc_async"
        )
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
        nthreads = int(
            env.get("NTHREADS", os.cpu_count() or 1)
        )
        env.setdefault("OMP_NUM_THREADS", str(nthreads))
        env.setdefault("MKL_NUM_THREADS", str(nthreads))
        env.setdefault("NUMEXPR_NUM_THREADS", str(nthreads))
        env.setdefault(
            "TF_NUM_INTRAOP_THREADS", str(nthreads)
        )
        env.setdefault("TF_NUM_INTEROP_THREADS", "2")

    return chosen, env


def _resolve_suite_root(
    args: argparse.Namespace,
    *,
    suite_prefix: str,
    results_root: Path,
) -> Path:
    """Resolve or create the suite root directory."""
    if args.suite_root:
        path = Path(args.suite_root).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    if args.resume_latest:
        latest = find_latest_dir(
            results_root,
            pattern=f"{suite_prefix}_*",
        )
        if latest is None:
            raise FileNotFoundError(
                "--resume-latest was requested but no existing "
                f"suite matched {suite_prefix}_* under {results_root}."
            )
        return latest

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = results_root / f"{suite_prefix}_{ts}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _infer_regime(folder_name: str) -> str:
    """Infer regime label from a suite run directory name."""
    match = re.search(
        r"sm3_(?:tau|both)_(.+?)_\d+$",
        str(folder_name),
    )
    if match:
        return match.group(1)
    return str(folder_name)


def _collect_summaries(
    suite_root: Path,
    *,
    out_csv: Path,
    out_json: Path,
    summary_name: str = "sm3_synth_summary.csv",
) -> None:
    """Collect per-regime summary CSV files into combined outputs."""
    rows = []
    for path in suite_root.rglob(summary_name):
        run_dir = path.parent
        regime = _infer_regime(run_dir.name)
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[skip] failed to read {path}: {exc}")
            continue

        if df.empty or "metric" not in df.columns:
            print(f"[skip] unexpected format: {path}")
            continue

        data = df.copy()
        data.insert(0, "regime", regime)
        data.insert(1, "run_dir", str(run_dir))
        rows.append(data)

    if not rows:
        raise RuntimeError(
            f"No {summary_name} files found under {suite_root}."
        )

    out = pd.concat(rows, ignore_index=True)
    sort_cols = [
        name
        for name in ("metric", "regime")
        if name in out.columns
    ]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(
            drop=True
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as stream:
        json.dump(out.to_dict("records"), stream, indent=2)

    print(f"[OK] wrote: {out_csv}")
    print(f"[OK] wrote: {out_json}")


def _run_and_tee(
    cmd: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
) -> None:
    """Run a subprocess while teeing stdout/stderr to a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[CMD]", " ".join(str(item) for item in cmd))

    with open(log_path, "a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()

    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def _parser() -> argparse.ArgumentParser:
    """Build suite runner parser."""
    ap = argparse.ArgumentParser(
        prog="sm3-suite",
        description=(
            "Run a preset-driven SM3 regime suite and collect "
            "combined summaries."
        ),
    )
    add_config_args(ap)
    add_results_dir_arg(ap)
    add_outdir_arg(
        ap,
        dest="suite_root",
        help="Explicit suite root. Enables resume into the same suite.",
    )

    ap.add_argument(
        "--preset",
        choices=tuple(sorted(SM3_PRESETS)),
        default="tau50",
        help="Named preset bundle to use as the base configuration.",
    )
    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default=None,
        help="Execution device selection.",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs per regime.",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size per regime.",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience.",
    )
    ap.add_argument(
        "--fast",
        type=int,
        choices=(0, 1),
        default=None,
        help="1 disables heavy diagnostics callbacks.",
    )
    ap.add_argument(
        "--disable-freeze",
        action="store_const",
        const=1,
        dest="fast",
        help="Alias for --fast 1.",
    )
    ap.add_argument(
        "--enable-freeze",
        action="store_const",
        const=0,
        dest="fast",
        help="Alias for --fast 0.",
    )
    ap.add_argument(
        "--nreal",
        "--n-realizations",
        dest="n_realizations",
        type=int,
        default=None,
        help="Number of realizations per regime.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed.",
    )

    ap.add_argument(
        "--regime",
        action="append",
        default=[],
        help="Run only this regime. Repeat as needed.",
    )
    ap.add_argument(
        "--regimes",
        type=str,
        default=None,
        help="Comma-separated regime names.",
    )
    ap.add_argument(
        "--regime-ids",
        type=str,
        default=None,
        help="Comma-separated 1-based regime ids.",
    )
    ap.add_argument(
        "--list-regimes",
        action="store_true",
        help="Print available regimes and exit.",
    )

    ap.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume the newest matching suite under results dir.",
    )
    ap.add_argument(
        "--start-realisation",
        type=int,
        default=None,
        help="1-based realization index forwarded to the SM3 script.",
    )
    ap.add_argument(
        "--skip-collect",
        action="store_true",
        help="Skip combined summary collection at the end.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved commands without executing them.",
    )

    ap.add_argument(
        "--identify", choices=("tau", "both"), default=None
    )
    ap.add_argument("--n-years", type=int, default=None)
    ap.add_argument("--time-steps", type=int, default=None)
    ap.add_argument(
        "--forecast-horizon", type=int, default=None
    )
    ap.add_argument("--val-tail", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--noise-std", type=float, default=None)
    ap.add_argument(
        "--load-type",
        choices=("step", "ramp"),
        default=None,
    )
    ap.add_argument("--tau-min", type=float, default=None)
    ap.add_argument("--tau-max", type=float, default=None)
    ap.add_argument(
        "--tau-spread-dex", type=float, default=None
    )
    ap.add_argument(
        "--Ss-spread-dex", type=float, default=None
    )
    ap.add_argument(
        "--K-spread-dex", type=float, default=None
    )
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--hd-factor", type=float, default=None)
    ap.add_argument(
        "--thickness-cap", type=float, default=None
    )
    ap.add_argument("--kappa-b", type=float, default=None)
    ap.add_argument("--gamma-w", type=float, default=None)
    ap.add_argument(
        "--scenario",
        choices=("base", "tau_only_derive_k"),
        default=None,
    )
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument(
        "--Lx-m", dest="Lx_m", type=float, default=None
    )
    ap.add_argument(
        "--h-right",
        dest="h_right",
        type=float,
        default=None,
    )
    return ap


def _resolved_params(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Resolve preset defaults plus explicit command overrides."""
    preset = get_sm3_preset(args.preset)
    params = preset.merged(
        identify=args.identify,
        n_realizations=args.n_realizations,
        n_years=args.n_years,
        time_steps=args.time_steps,
        forecast_horizon=args.forecast_horizon,
        val_tail=args.val_tail,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        patience=args.patience,
        noise_std=args.noise_std,
        load_type=args.load_type,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        tau_spread_dex=args.tau_spread_dex,
        Ss_spread_dex=args.Ss_spread_dex,
        K_spread_dex=args.K_spread_dex,
        alpha=args.alpha,
        hd_factor=args.hd_factor,
        thickness_cap=args.thickness_cap,
        kappa_b=args.kappa_b,
        gamma_w=args.gamma_w,
        scenario=args.scenario,
        nx=args.nx,
        Lx_m=args.Lx_m,
        h_right=args.h_right,
        device=args.device,
        fast=_int01(args.fast),
        seed=args.seed,
        start_realisation=args.start_realisation,
    )
    return params


def _sm3_script_path() -> Path:
    """Return the packaged SM3 identifiability script path."""
    here = Path(__file__).resolve().parent
    path = here / "sm3_synthetic_identifiability.py"
    if not path.exists():
        raise FileNotFoundError(
            f"SM3 identifiability script not found: {path}"
        )
    return path


def _build_run_cmd(
    script_path: Path,
    *,
    outdir: Path,
    regime: str,
    p: dict[str, Any],
) -> list[str]:
    """Build a subprocess command for one regime run."""
    cmd = [
        sys.executable,
        str(script_path),
        "--outdir",
        str(outdir),
        "--n-realizations",
        str(p["n_realizations"]),
        "--identify",
        str(p["identify"]),
        "--ident-regime",
        str(regime),
        "--scenario",
        str(p["scenario"]),
        "--n-years",
        str(p["n_years"]),
        "--time-steps",
        str(p["time_steps"]),
        "--forecast-horizon",
        str(p["forecast_horizon"]),
        "--val-tail",
        str(p["val_tail"]),
        "--epochs",
        str(p["epochs"]),
        "--batch",
        str(p["batch"]),
        "--patience",
        str(p["patience"]),
        "--lr",
        str(p["lr"]),
        "--noise-std",
        str(p["noise_std"]),
        "--load-type",
        str(p["load_type"]),
        "--seed",
        str(p["seed"]),
        "--tau-min",
        str(p["tau_min"]),
        "--tau-max",
        str(p["tau_max"]),
        "--tau-spread-dex",
        str(p["tau_spread_dex"]),
        "--Ss-spread-dex",
        str(p["Ss_spread_dex"]),
        "--alpha",
        str(p["alpha"]),
        "--hd-factor",
        str(p["hd_factor"]),
        "--thickness-cap",
        str(p["thickness_cap"]),
        "--kappa-b",
        str(p["kappa_b"]),
        "--gamma-w",
        str(p["gamma_w"]),
        "--nx",
        str(p["nx"]),
        "--Lx-m",
        str(p["Lx_m"]),
        "--h-right",
        str(p["h_right"]),
        "--start-realisation",
        str(p["start_realisation"]),
    ]
    if p.get("K_spread_dex") is not None:
        cmd.extend(["--K-spread-dex", str(p["K_spread_dex"])])
    if int(p.get("fast", 0)) == 1:
        cmd.append("--disable-freeze")
    return cmd


def sm3_suite_main(
    argv: list[str] | None = None,
) -> None:
    """Run a preset-driven SM3 suite."""
    parser = _parser()
    args = parser.parse_args(argv)

    if args.list_regimes:
        print("Available regimes:")
        for idx, name in enumerate(SM3_REGIMES, start=1):
            print(f"  {idx:>2}  {name}")
        return

    cfg = bootstrap_runtime_config(args)
    params = _resolved_params(args)
    regimes = _resolve_regimes(args)

    results_dir = (
        args.results_dir
        or cfg.get("RESULTS_DIR")
        or cfg.get("results_dir")
        or "results"
    )
    results_root = Path(results_dir).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    device, env = _configure_runtime_env(
        str(params["device"])
    )
    suite_root = _resolve_suite_root(
        args,
        suite_prefix=str(params["suite_prefix"]),
        results_root=results_root,
    )
    logdir = suite_root / "logs"
    combdir = suite_root / "combined"
    logdir.mkdir(parents=True, exist_ok=True)
    combdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"[SM3] suite root: {suite_root}")
    print(f"[SM3] preset: {args.preset}")
    print(f"[SM3] identify: {params['identify']}")
    print(f"[SM3] device: {device}")
    print(f"[SM3] regimes: {' '.join(regimes)}")
    print(
        f"[SM3] nreal={params['n_realizations']}  "
        f"epochs={params['epochs']}  batch={params['batch']}  "
        f"patience={params['patience']}  fast={params['fast']}"
    )
    print(
        f"[SM3] 1D domain: nx={params['nx']}  "
        f"Lx_m={params['Lx_m']}  h_right={params['h_right']}"
    )
    print("=" * 60)

    script_path = _sm3_script_path()
    ident = str(params["identify"])
    nreal = int(params["n_realizations"])

    for regime in regimes:
        outdir = suite_root / f"sm3_{ident}_{regime}_{nreal}"
        logfile = logdir / f"{regime}.log"
        cmd = _build_run_cmd(
            script_path,
            outdir=outdir,
            regime=regime,
            p=params,
        )

        print("=" * 60)
        print(f"[RUN] ident={ident}  regime={regime}")
        print(f"      outdir={outdir}")
        print(f"      log={logfile}")
        print("=" * 60)

        if args.dry_run:
            print("[DRY]", " ".join(cmd))
            continue

        _run_and_tee(cmd, env=env, log_path=logfile)

    if args.skip_collect or args.dry_run:
        return

    print("=" * 60)
    print("[COLLECT] building combined summary table...")
    print("=" * 60)
    _collect_summaries(
        suite_root,
        out_csv=combdir / "sm3_summary_combined.csv",
        out_json=combdir / "sm3_summary_combined.json",
    )
    print("Suite completed.")
    print(
        f"  Combined CSV:  {combdir / 'sm3_summary_combined.csv'}"
    )
    print(
        f"  Combined JSON: {combdir / 'sm3_summary_combined.json'}"
    )


def main(argv: list[str] | None = None) -> None:
    """Compatibility alias for the suite runner."""
    sm3_suite_main(argv)


if __name__ == "__main__":
    main()
