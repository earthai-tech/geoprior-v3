# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>
#
r"""
Supplement S6 • Extended ablations & sensitivities

Updated v3.2 behavior:
- Prefer ablation_record.updated*.jsonl when present.
- Dedupe (timestamp, city): keep best record (updated wins).
- Normalize pde_mode to buckets: none vs both
  (off/none -> none, else -> both).
- Optional explicit --input (csv/json/jsonl) and --models filtering.

Plot upgrades (v3.2+):
- Configurable colormap via --cmap.
- Multiple heatmap metrics in a single figure via --heatmap-metrics.
- Optional heatmap-only layout (no bar row) via --no-bars.
- Optional "density-like" views via --map-kind:
    * heatmap     : imshow (default; discrete grid)
    * smooth      : imshow with interpolation (same data, smoother look)
    * contour     : contourf on the pivot grid (requires rectangular grid)
    * tricontour  : tricontourf from scattered points (handles missing cells)
- Consistent lambda grid alignment across cities/metrics via --align-grid.
- Optional marker for a chosen (lambda_cons, lambda_prior) point.

- Optional Pareto trade-off view via --pareto:
    * scatter of (MAE vs Sharpness) colored by Coverage (one panel per city)
    * optional non-dominated front overlay
    * optional density overlay (hexbin) via --pareto-density
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

from . import config as cfg
from . import utils

_LOWER_IS_BETTER = {
    "mae",
    "rmse",
    "mse",
    "sharpness80",
    "epsilon_prior",
    "epsilon_cons",
}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="plot-ablations-sensitivity",
        description="Supplement S6: Ablations & sensitivities",
    )

    p.add_argument(
        "--root",
        type=str,
        default="results",
        help="Root to scan for ablation_record*.jsonl",
    )
    p.add_argument(
        "--input",
        type=str,
        default=None,
        action="append",
        help=(
            "Explicit ablation file/table (repeatable). "
            "Supports .jsonl/.json/.csv."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output dir override.",
    )

    p.add_argument("--font", type=int, default=cfg.PAPER_FONT)
    p.add_argument("--dpi", type=int, default=cfg.PAPER_DPI)

    # --- plot controls ------------------------------------------------
    p.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help=(
            "Colormap name for maps (e.g. viridis, magma, cividis). "
            "Used by heatmap/smooth/contour/tricontour."
        ),
    )
    p.add_argument(
        "--map-kind",
        type=str,
        default="heatmap",
        choices=[
            "heatmap",
            "smooth",
            "contour",
            "tricontour",
        ],
        help=(
            "Map rendering style. 'heatmap' uses imshow on the lambda "
            "grid. 'tricontour' gives a density-like smooth surface and "
            "handles missing cells."
        ),
    )
    p.add_argument(
        "--levels",
        type=int,
        default=12,
        help="Number of contour levels for contour/tricontour.",
    )
    p.add_argument(
        "--contour-lines",
        type=str,
        default="false",
        help="Overlay contour lines on contour/tricontour (true/false).",
    )

    p.add_argument(
        "--bar-metric",
        type=str,
        default="mae",
        choices=[
            "r2",
            "mae",
            "rmse",
            "mse",
            "coverage80",
            "sharpness80",
            "epsilon_prior",
            "epsilon_cons",
        ],
        help="Metric for bar plots (used when bars are enabled).",
    )
    p.add_argument(
        "--heatmap-metric",
        type=str,
        default="sharpness80",
        choices=[
            "mae",
            "rmse",
            "mse",
            "r2",
            "coverage80",
            "sharpness80",
            "epsilon_prior",
            "epsilon_cons",
        ],
        help="Single metric for heatmaps (legacy).",
    )
    p.add_argument(
        "--heatmap-metrics",
        type=str,
        default="",
        help=(
            "Comma list of map metrics to plot in one figure. "
            "Example: mae,coverage80,sharpness80. "
            "If provided, overrides --heatmap-metric."
        ),
    )
    p.add_argument(
        "--no-bars",
        type=str,
        default="false",
        help=(
            "Disable the bar-row entirely (true/false). "
            "Recommended when using --heatmap-metrics."
        ),
    )

    p.add_argument(
        "--align-grid",
        type=str,
        default="true",
        help=(
            "Align lambda grids (ticks/ordering) across cities/metrics "
            "(true/false). When true, missing cells are left blank."
        ),
    )

    p.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma list of model names to keep.",
    )

    p.add_argument(
        "--mute-values",
        type=str,
        default="false",
        help="Annotate values above bars (true/false).",
    )
    p.add_argument(
        "--no-colorbar",
        type=str,
        default="false",
        help="Hide colorbars (true/false).",
    )

    p.add_argument(
        "--mark-lambda-cons",
        type=float,
        default=None,
        help="Optional marker: lambda_cons value to mark on maps.",
    )
    p.add_argument(
        "--mark-lambda-prior",
        type=float,
        default=None,
        help="Optional marker: lambda_prior value to mark on maps.",
    )
    p.add_argument(
        "--no-best-marker",
        type=str,
        default="false",
        help="Disable automatic best-cell marker (true/false).",
    )

    p.add_argument(
        "--pareto",
        type=str,
        default="false",
        help=(
            "Plot Pareto scatter (trade-offs) instead of maps. "
            "Default: false."
        ),
    )
    p.add_argument(
        "--pareto-x",
        type=str,
        default="mae",
        choices=[
            "mae",
            "rmse",
            "mse",
            "r2",
            "coverage80",
            "sharpness80",
            "epsilon_prior",
            "epsilon_cons",
        ],
        help="X metric for Pareto scatter (default: mae).",
    )
    p.add_argument(
        "--pareto-y",
        type=str,
        default="sharpness80",
        choices=[
            "mae",
            "rmse",
            "mse",
            "r2",
            "coverage80",
            "sharpness80",
            "epsilon_prior",
            "epsilon_cons",
        ],
        help="Y metric for Pareto scatter (default: sharpness80).",
    )
    p.add_argument(
        "--pareto-color",
        type=str,
        default="coverage80",
        choices=[
            "coverage80",
            "sharpness80",
            "mae",
            "mse",
            "r2",
            "epsilon_prior",
            "epsilon_cons",
        ],
        help="Color metric for Pareto scatter (default: coverage80).",
    )
    p.add_argument(
        "--pareto-front",
        type=str,
        default="true",
        help="Overlay non-dominated front (true/false).",
    )
    p.add_argument(
        "--pareto-s",
        type=float,
        default=26.0,
        help="Marker size for Pareto scatter.",
    )
    p.add_argument(
        "--pareto-alpha",
        type=float,
        default=0.85,
        help="Marker alpha for Pareto scatter.",
    )

    p.add_argument(
        "--pareto-density",
        type=str,
        default="false",
        help=(
            "Add a density overlay (hexbin) behind the Pareto scatter "
            "(true/false). Useful when many runs overlap."
        ),
    )
    p.add_argument(
        "--pareto-density-gridsize",
        type=int,
        default=35,
        help="Hexbin gridsize for Pareto density overlay.",
    )
    p.add_argument(
        "--pareto-density-bins",
        type=str,
        default="log",
        choices=["log", "linear"],
        help="Hexbin bin scaling (log or linear).",
    )
    p.add_argument(
        "--pareto-density-cmap",
        type=str,
        default="Greys",
        help="Colormap for Pareto density overlay (hexbin).",
    )
    p.add_argument(
        "--pareto-density-alpha",
        type=float,
        default=0.25,
        help="Alpha for Pareto density overlay.",
    )
    p.add_argument(
        "--pareto-density-mincnt",
        type=int,
        default=1,
        help="Minimum count per hexagon to be shown (default: 1).",
    )

    p.add_argument(
        "--cities",
        type=str,
        default="",
        help="Comma list of cities (overrides city-a/b).",
    )
    p.add_argument(
        "--city-a",
        type=str,
        default="Nansha",
        help="First city name.",
    )
    p.add_argument(
        "--city-b",
        type=str,
        default="Zhongshan",
        help="Second city name.",
    )

    utils.add_plot_text_args(
        p,
        default_out="supp_fig_S6_ablations",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------
def _read_jsonl(fp: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with fp.open("r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            if isinstance(rec, dict):
                rows.append(rec)
    return rows


def _load_one_input(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()

    if suf == ".csv":
        df = pd.read_csv(path)
        if "_src" not in df.columns:
            df["_src"] = str(path)
        return df

    if suf == ".jsonl":
        rows = _read_jsonl(path)
        df = pd.DataFrame(rows)
        if "_src" not in df.columns:
            df["_src"] = str(path)
        return df

    if suf == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            obj = None

        if isinstance(obj, list):
            df = pd.DataFrame(obj)
            if "_src" not in df.columns:
                df["_src"] = str(path)
            return df

        if isinstance(obj, dict):
            df = pd.DataFrame([obj])
            if "_src" not in df.columns:
                df["_src"] = str(path)
            return df

    return pd.DataFrame([])


def _scan_records(root: Path) -> pd.DataFrame:
    files = utils.find_all(
        root,
        cfg.PATTERNS.get("ablation_record_jsonl", ()),
    )

    blocks: list[pd.DataFrame] = []
    for fp in files:
        df = _load_one_input(fp)
        if not df.empty:
            blocks.append(df)

    if not blocks:
        return pd.DataFrame([])
    return pd.concat(blocks, ignore_index=True)


def _load_records(args: argparse.Namespace) -> pd.DataFrame:
    if args.input:
        blocks: list[pd.DataFrame] = []
        for s in args.input:
            p = utils.as_path(s)
            if p.exists():
                df = _load_one_input(p)
                if not df.empty:
                    blocks.append(df)
        if not blocks:
            return pd.DataFrame([])
        return pd.concat(blocks, ignore_index=True)

    root = utils.as_path(args.root)
    return _scan_records(root)


# ---------------------------------------------------------------------
# Canonicalization / precedence
# ---------------------------------------------------------------------
def _canon_pde_mode(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in {"none", "off", "no", "0", "false"}:
        return "none"
    if s in {"both", "on", "1", "true"}:
        return "both"
    # treat any physics-on mode as "both" bucket
    if s in {"consolidation", "gw_flow"}:
        return "both"
    return s


def _canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    aliases = {
        "timestamp": ("timestamp", "ts"),
        "city": ("city", "City"),
        "model": ("model", "Model"),
        "pde_mode": ("pde_mode", "pde"),
        "lambda_prior": ("lambda_prior", "lambda_p"),
        "lambda_cons": ("lambda_cons", "lambda_c"),
        "coverage80": ("coverage80", "coverage8"),
        "sharpness80": ("sharpness80", "sharpness"),
    }
    utils.ensure_columns(df, aliases=aliases)

    if "city" in df.columns:
        df["city"] = (
            df["city"].astype(str).map(utils.canonical_city)
        )

    if "pde_mode" in df.columns:
        df["pde_mode"] = df["pde_mode"].map(_canon_pde_mode)
    else:
        df["pde_mode"] = "both"

    # stable bucket used by S6 layout:
    #   - "none": physics off
    #   - "both": any physics on mode
    df["pde_bucket"] = np.where(
        df["pde_mode"].astype(str).eq("none"),
        "none",
        "both",
    )

    num_cols = [
        "lambda_prior",
        "lambda_cons",
        "r2",
        "mae",
        "rmse",
        "mse",
        "coverage80",
        "sharpness80",
        "epsilon_prior",
        "epsilon_cons",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _needs_si_to_mm(row: pd.Series) -> bool:
    u = row.get("units", None)
    if isinstance(u, dict):
        uu = str(u.get("subs_metrics_unit", "")).lower()
        if uu == "mm":
            return False
        if uu in {"m", "meter", "metre"}:
            return True

    # heuristic: SI-scale metrics tend to be < 1 in meters
    mae = row.get("mae", np.nan)
    mse = row.get("mse", np.nan)
    shp = row.get("sharpness80", np.nan)

    ok = (
        np.isfinite(mae)
        and np.isfinite(mse)
        and np.isfinite(shp)
        and 0.0 < float(mae) < 1.0
        and 0.0 < float(mse) < 1.0
        and 0.0 < float(shp) < 1.0
    )
    return bool(ok)


def _harmonize_units(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "mae" not in df.columns or "mse" not in df.columns:
        return df

    mask = df.apply(_needs_si_to_mm, axis=1)
    if not mask.any():
        return df

    for c in ["mae", "rmse", "sharpness80"]:
        if c in df.columns:
            df.loc[mask, c] = df.loc[mask, c] * 1000.0

    if "mse" in df.columns:
        df.loc[mask, "mse"] = df.loc[mask, "mse"] * 1e6

    return df


def _record_score(row: pd.Series) -> int:
    s = 0
    src = str(row.get("_src", "")).lower()
    if "updated" in src:
        s += 100

    u = row.get("units", None)
    if isinstance(u, dict):
        uu = str(u.get("subs_metrics_unit", "")).lower()
        if uu == "mm":
            s += 50

    if isinstance(row.get("legacy", None), dict):
        s += 10
    if isinstance(row.get("metrics", None), dict):
        s += 10

    return int(s)


def _dedupe_prefer_best(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "timestamp" not in df.columns:
        return df
    if "city" not in df.columns:
        df["city"] = ""

    df = df.copy()
    df["_score"] = df.apply(_record_score, axis=1)

    df = df.sort_values(
        by=["timestamp", "city", "_score"],
        ascending=[True, True, False],
    )
    df = df.drop_duplicates(
        subset=["timestamp", "city"],
        keep="first",
    ).copy()

    return df.drop(columns=["_score"], errors="ignore")


def _filter_models(
    df: pd.DataFrame, models: str
) -> pd.DataFrame:
    s = str(models or "").strip()
    if not s or "model" not in df.columns:
        return df

    keep = [m.strip() for m in s.split(",") if m.strip()]
    if not keep:
        return df

    return df.loc[df["model"].astype(str).isin(keep)].copy()


def _resolve_cities(args: argparse.Namespace) -> list[str]:
    raw = str(args.cities or "").strip()
    if raw:
        parts = [
            p.strip() for p in raw.split(",") if p.strip()
        ]
        return [utils.canonical_city(p) for p in parts]

    a = utils.canonical_city(args.city_a)
    b = utils.canonical_city(args.city_b)
    return [a, b]


def _resolve_map_metrics(
    args: argparse.Namespace,
) -> list[str]:
    raw = str(args.heatmap_metrics or "").strip()
    if not raw:
        return [str(args.heatmap_metric)]
    mets = [m.strip() for m in raw.split(",") if m.strip()]
    # unique, stable order
    out: list[str] = []
    for m in mets:
        if m not in out:
            out.append(m)
    return out or [str(args.heatmap_metric)]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _coverage_target(metric: str) -> float | None:
    s = str(metric).strip().lower()
    if s.startswith("coverage"):
        tail = s[len("coverage") :]
        if tail.isdigit():
            return int(tail) / 100.0
    return None


def _metric_label(name: str) -> str:
    k = str(name).strip()
    if k in cfg.PHYS_LABELS:
        return cfg.PHYS_LABELS[k]

    # Coverage is special: target-based, not monotone.
    tgt = _coverage_target(k)
    if tgt is not None:
        p = int(round(100.0 * tgt))
        return f"Coverage ({p}% PI; target {tgt:.2f})"

    meta = cfg.PLOT_METRIC_META.get(k, None)
    if isinstance(meta, dict):
        txt = str(
            meta.get("title")
            or meta.get("ylabel")
            or k
        )
        unit = str(meta.get("unit", "") or "")
        return txt.format(unit=unit)

    kl = k.lower()
    if kl == "r2":
        return r"$R^2$ (↑)"
    return k


def _best_ij(
    arr: np.ndarray, *, metric: str
) -> tuple[int, int] | None:
    try:
        a = np.asarray(arr, dtype=float)
        if not np.isfinite(a).any():
            return None

        kl = str(metric).strip().lower()
        tgt = _coverage_target(kl)

        if tgt is not None:
            obj = np.abs(a - tgt)
            obj[~np.isfinite(a)] = np.nan
            idx = int(np.nanargmin(obj))
        elif kl in _LOWER_IS_BETTER:
            idx = int(np.nanargmin(a))
        else:
            idx = int(np.nanargmax(a))

        i, j = np.unravel_index(idx, a.shape)
        return (int(i), int(j))
    except Exception:
        return None
    
def _metric_fmt(metric: str) -> str:
    meta = cfg.PLOT_METRIC_META.get(str(metric), None)
    if isinstance(meta, dict):
        return str(meta.get("fmt", "{:.3g}"))
    return "{:.3g}"


def _axes_cleanup(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _bars_by_lambda(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    city: str,
    metric: str,
    color: str,
    annotate: bool,
    show_legend: bool,
    show_labels: bool,
    show_ticks: bool,
    show_title: bool,
) -> None:
    sub = df.loc[df["city"].astype(str).eq(str(city))].copy()
    if sub.empty or metric not in sub.columns:
        ax.set_axis_off()
        return

    keep = sub.loc[
        sub["pde_bucket"].isin(["none", "both"])
    ].copy()
    if keep.empty:
        ax.set_axis_off()
        return

    grp = (
        keep.groupby(
            ["pde_bucket", "lambda_prior"], dropna=False
        )[metric]
        .mean()
        .reset_index()
    )
    if grp.empty:
        ax.set_axis_off()
        return

    xs = sorted(grp["lambda_prior"].dropna().unique())
    if not xs:
        ax.set_axis_off()
        return

    modes = ["none", "both"]
    base = np.arange(len(xs), dtype=float)
    width = 0.8 / 2.0
    offset0 = 0.5

    fmt = _metric_fmt(metric)

    for i, m in enumerate(modes):
        y = [
            grp.loc[
                (grp["pde_bucket"] == m)
                & (grp["lambda_prior"] == x),
                metric,
            ].mean()
            for x in xs
        ]

        xloc = base + (i - offset0) * width
        ax.bar(
            xloc,
            y,
            width=width,
            label=m,
            color=color,
            alpha=0.35 if m == "none" else 0.95,
            edgecolor="white",
        )

        if annotate:
            for xi, yi in zip(xloc, y, strict=False):
                if pd.notna(yi):
                    ax.text(
                        xi,
                        yi,
                        fmt.format(float(yi)),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

    ax.set_xticks(base)

    if show_ticks:
        ax.set_xticklabels([f"{x:.2g}" for x in xs])
    else:
        ax.set_xticklabels([])

    if show_labels:
        ax.set_xlabel(r"$\lambda_{\mathrm{prior}}$")
        ax.set_ylabel(_metric_label(metric))
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    if show_title:
        ax.set_title(
            f"{city} — {_metric_label(metric)}",
            loc="left",
            pad=6,
            fontweight="bold",
        )

    if show_legend:
        ax.legend(title="physics", frameon=False)

    _axes_cleanup(ax)


def _lambda_grid(
    df: pd.DataFrame,
    *,
    cities: Iterable[str],
) -> tuple[list[float], list[float]]:
    sub = df.loc[
        df["city"].astype(str).isin([str(c) for c in cities])
    ].copy()
    sub = sub.loc[sub["pde_bucket"].eq("both")].copy()
    sub = sub.dropna(subset=["lambda_cons", "lambda_prior"])
    lc = sorted(
        set(float(x) for x in sub["lambda_cons"].unique())
    )
    lp = sorted(
        set(float(x) for x in sub["lambda_prior"].unique())
    )
    return lc, lp


def _pivot_grid(
    sub: pd.DataFrame,
    *,
    metric: str,
    lc_grid: list[float] | None,
    lp_grid: list[float] | None,
) -> pd.DataFrame:
    piv = sub.pivot_table(
        index="lambda_cons",
        columns="lambda_prior",
        values=metric,
        aggfunc="mean",
    )
    if piv.empty:
        return piv
    piv = piv.sort_index().sort_index(axis=1)

    if lc_grid is not None:
        piv = piv.reindex(index=lc_grid)
    if lp_grid is not None:
        piv = piv.reindex(columns=lp_grid)

    return piv


def _maybe_mark_point(
    ax: plt.Axes,
    *,
    lc_ticks: list[float],
    lp_ticks: list[float],
    mark_lc: float | None,
    mark_lp: float | None,
) -> None:
    if mark_lc is None or mark_lp is None:
        return
    if (mark_lc not in lc_ticks) or (mark_lp not in lp_ticks):
        return
    i = lc_ticks.index(mark_lc)
    j = lp_ticks.index(mark_lp)
    ax.scatter(
        [j],
        [i],
        marker="s",
        s=70,
        facecolors="none",
        edgecolors="white",
        linewidths=1.8,
    )


def _plot_map_on_grid(
    ax: plt.Axes,
    piv: pd.DataFrame,
    *,
    metric: str,
    cmap: str,
    map_kind: str,
    levels: int,
    contour_lines: bool,
    vmin: float | None,
    vmax: float | None,
) -> Any:
    """
    Returns an artist usable for colorbar (Image/ContourSet).
    """
    data = np.asarray(piv.values, dtype=float)

    # Mask NaNs so they render as "bad" color in imshow.
    mdata = np.ma.masked_invalid(data)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color=(0.85, 0.85, 0.85, 1.0))

    lp = list(piv.columns.astype(float))
    lc = list(piv.index.astype(float))

    if map_kind in {"heatmap", "smooth"}:
        interp = (
            "nearest" if map_kind == "heatmap" else "bicubic"
        )
        im = ax.imshow(
            mdata,
            aspect="auto",
            cmap=cm,
            interpolation=interp,
            vmin=vmin,
            vmax=vmax,
        )
        return im

    # contour/ tricontour want real coordinate grids
    X, Y = np.meshgrid(lp, lc)

    if map_kind == "contour":
        cs = ax.contourf(
            X,
            Y,
            mdata,
            levels=levels,
            cmap=cm,
            vmin=vmin,
            vmax=vmax,
        )
        if contour_lines:
            ax.contour(
                X,
                Y,
                mdata,
                levels=levels,
                colors="k",
                linewidths=0.4,
                alpha=0.35,
            )
        return cs

    raise ValueError(f"Unsupported map_kind: {map_kind}")


def _plot_map_tricontour(
    ax: plt.Axes,
    sub: pd.DataFrame,
    *,
    metric: str,
    cmap: str,
    levels: int,
    contour_lines: bool,
    vmin: float | None,
    vmax: float | None,
) -> Any:
    """
    Tricontour view from scattered (lambda_prior, lambda_cons) points.
    Handles missing lambda grid cells gracefully.
    """
    s = sub.dropna(
        subset=["lambda_cons", "lambda_prior", metric]
    ).copy()
    if s.empty:
        return None

    x = s["lambda_prior"].astype(float).to_numpy()
    y = s["lambda_cons"].astype(float).to_numpy()
    z = s[metric].astype(float).to_numpy()

    tri = mtri.Triangulation(x, y)
    cm = plt.get_cmap(cmap).copy()

    cs = ax.tricontourf(
        tri,
        z,
        levels=levels,
        cmap=cm,
        vmin=vmin,
        vmax=vmax,
    )
    if contour_lines:
        ax.tricontour(
            tri,
            z,
            levels=levels,
            colors="k",
            linewidths=0.4,
            alpha=0.35,
        )
    return cs


def _heatmap_one(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    city: str,
    metric: str,
    cmap: str,
    map_kind: str,
    levels: int,
    contour_lines: bool,
    lc_grid: list[float] | None,
    lp_grid: list[float] | None,
    vmin: float | None,
    vmax: float | None,
    show_labels: bool,
    show_ticks: bool,
    show_title: bool,
    mark_best: bool,
    mark_lc: float | None,
    mark_lp: float | None,
) -> Any | None:
    sub = df.loc[df["city"].astype(str).eq(str(city))].copy()
    sub = sub.loc[sub["pde_bucket"].eq("both")].copy()

    if sub.empty or metric not in sub.columns:
        ax.set_axis_off()
        return None

    sub = sub.dropna(subset=["lambda_cons", "lambda_prior"])
    if sub.empty:
        ax.set_axis_off()
        return None

    if map_kind == "tricontour":
        artist = _plot_map_tricontour(
            ax,
            sub,
            metric=metric,
            cmap=cmap,
            levels=levels,
            contour_lines=contour_lines,
            vmin=vmin,
            vmax=vmax,
        )
        if artist is None:
            ax.set_axis_off()
            return None

        # ticks: use aligned grids if available, else unique points
        lc_ticks = lc_grid or sorted(
            set(float(x) for x in sub["lambda_cons"].unique())
        )
        lp_ticks = lp_grid or sorted(
            set(
                float(x) for x in sub["lambda_prior"].unique()
            )
        )
        if show_ticks:
            ax.set_xticks(lp_ticks)
            ax.set_yticks(lc_ticks)
            ax.set_xticklabels([f"{x:.2g}" for x in lp_ticks])
            ax.set_yticklabels([f"{x:.2g}" for x in lc_ticks])
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if show_labels:
            ax.set_xlabel(r"$\lambda_{\mathrm{prior}}$")
            ax.set_ylabel(r"$\lambda_{\mathrm{cons}}$")
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")

        if show_title:
            ax.set_title(
                f"{city} — {_metric_label(metric)}",
                loc="left",
                pad=6,
                fontweight="bold",
            )

        # "best" marker: approximate by pivoting to the aligned grid
        if mark_best:
            piv = _pivot_grid(
                sub,
                metric=metric,
                lc_grid=lc_grid,
                lp_grid=lp_grid,
            )
            if not piv.empty:
                ij = _best_ij(piv.values, metric=metric)
                if ij is not None:
                    i, j = ij
                    ax.scatter(
                        [float(piv.columns[j])],
                        [float(piv.index[i])],
                        marker="o",
                        s=50,
                        facecolors="none",
                        edgecolors="white",
                        linewidths=1.5,
                    )

        if mark_lc is not None and mark_lp is not None:
            ax.scatter(
                [mark_lp],
                [mark_lc],
                marker="s",
                s=70,
                facecolors="none",
                edgecolors="white",
                linewidths=1.8,
            )

        _axes_cleanup(ax)
        return artist

    piv = _pivot_grid(
        sub,
        metric=metric,
        lc_grid=lc_grid,
        lp_grid=lp_grid,
    )
    if piv.empty:
        ax.set_axis_off()
        return None

    artist = _plot_map_on_grid(
        ax,
        piv,
        metric=metric,
        cmap=cmap,
        map_kind=map_kind,
        levels=levels,
        contour_lines=contour_lines,
        vmin=vmin,
        vmax=vmax,
    )

    if map_kind == "contour":
        xt = [float(c) for c in piv.columns]
        yt = [float(r) for r in piv.index]
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        if show_ticks:
            ax.set_xticklabels([f"{c:.2g}" for c in xt])
            ax.set_yticklabels([f"{r:.2g}" for r in yt])
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    else:
        ax.set_xticks(range(piv.shape[1]))
        ax.set_yticks(range(piv.shape[0]))
        if show_ticks:
            ax.set_xticklabels(
                [f"{c:.2g}" for c in piv.columns]
            )
            ax.set_yticklabels(
                [f"{r:.2g}" for r in piv.index]
            )
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    if show_labels:
        ax.set_xlabel(r"$\lambda_{\mathrm{prior}}$")
        ax.set_ylabel(r"$\lambda_{\mathrm{cons}}$")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    if show_title:
        ax.set_title(
            f"{city} — {_metric_label(metric)}",
            loc="left",
            pad=6,
            fontweight="bold",
        )

    if mark_best:
        ij = _best_ij(piv.values, metric=metric)
        if ij is not None:
            i, j = ij
            ax.scatter(
                [j],
                [i],
                marker="o",
                s=50,
                facecolors="none",
                edgecolors="white",
                linewidths=1.5,
            )

    if mark_lc is not None and mark_lp is not None:
        if map_kind == "contour":
            if (mark_lc in piv.index) and (
                mark_lp in piv.columns
            ):
                ax.scatter(
                    [float(mark_lp)],
                    [float(mark_lc)],
                    marker="s",
                    s=70,
                    facecolors="none",
                    edgecolors="white",
                    linewidths=1.8,
                )
        else:
            _maybe_mark_point(
                ax,
                lc_ticks=[float(x) for x in piv.index],
                lp_ticks=[float(x) for x in piv.columns],
                mark_lc=mark_lc,
                mark_lp=mark_lp,
            )

    _axes_cleanup(ax)
    return artist


def _resolve_out(*, out: str, out_dir: str | None) -> Path:
    if out_dir:
        base = Path(out_dir).expanduser()
        return (base / Path(out).expanduser()).resolve()
    return utils.resolve_fig_out(out)


def _compute_clim(
    df: pd.DataFrame,
    *,
    cities: list[str],
    metric: str,
) -> tuple[float | None, float | None]:
    """
    Compute a robust vmin/vmax for a metric across the selected cities
    (physics-on bucket only).
    """
    if metric not in df.columns:
        return None, None

    sub = df.loc[
        df["city"].astype(str).isin([str(c) for c in cities])
    ].copy()
    sub = sub.loc[sub["pde_bucket"].eq("both")].copy()
    v = pd.to_numeric(sub[metric], errors="coerce").dropna()
    if v.empty:
        return None, None

    # robust range (2-98 percentile) to avoid one-off extremes
    lo = float(np.nanpercentile(v.to_numpy(), 2))
    hi = float(np.nanpercentile(v.to_numpy(), 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None, None
    return lo, hi


def _to_min_obj(metric: str, v: np.ndarray) -> np.ndarray:
    """
    Map a metric to a minimization objective:
    - if lower is better -> keep as is
    - if higher is better -> negate
    """
    m = str(metric).lower()
    if m in _LOWER_IS_BETTER:
        return v
    return -v


def _pareto_front_mask(
    x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Non-dominated points for 2D minimization objectives.
    Returns a boolean mask of points on the Pareto front.
    """
    n = int(x.size)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # dominated if exists j with xj<=xi and yj<=yi and one strict
        dom = (
            (x <= x[i])
            & (y <= y[i])
            & ((x < x[i]) | (y < y[i]))
        )
        if np.any(dom):
            keep[i] = False
    return keep


def _pareto_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    city: str,
    x_metric: str,
    y_metric: str,
    c_metric: str,
    cmap: str,
    c_norm: Any | None,
    s: float,
    alpha: float,
    show_labels: bool,
    show_title: bool,
    mark_lc: float | None,
    mark_lp: float | None,
    show_front: bool,
    density: bool,
    density_gridsize: int,
    density_bins: str,
    density_cmap: str,
    density_alpha: float,
    density_mincnt: int,
) -> Any | None:
    sub = df.loc[df["city"].astype(str).eq(str(city))].copy()
    sub = sub.loc[sub["pde_bucket"].eq("both")].copy()

    need = [
        x_metric,
        y_metric,
        c_metric,
        "lambda_cons",
        "lambda_prior",
    ]
    for c in need:
        if c not in sub.columns:
            ax.set_axis_off()
            return None

    sub = sub.dropna(subset=need).copy()
    if sub.empty:
        ax.set_axis_off()
        return None

    x = pd.to_numeric(
        sub[x_metric], errors="coerce"
    ).to_numpy()
    y = pd.to_numeric(
        sub[y_metric], errors="coerce"
    ).to_numpy()
    c = pd.to_numeric(
        sub[c_metric], errors="coerce"
    ).to_numpy()

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    x, y, c = x[ok], y[ok], c[ok]
    sub = sub.loc[ok].copy()
    if x.size == 0:
        ax.set_axis_off()
        return None

    # Optional density overlay (hexbin) behind points.
    # NOTE: matplotlib's built-in `bins='log'` can become visually
    # invisible when most counts are 1 (log10(1)=0 and vmin=vmax=0).
    # We therefore apply a stable transform ourselves.
    if density and x.size >= 3:
        hb = ax.hexbin(
            x,
            y,
            gridsize=int(density_gridsize),
            mincnt=int(density_mincnt),
            bins=None,
            cmap=str(density_cmap),
            linewidths=0.0,
            alpha=float(density_alpha),
            zorder=0,
        )
        try:
            counts = np.asarray(hb.get_array(), dtype=float)
            if counts.size:
                if str(density_bins).lower() == "log":
                    vals = np.log10(counts + 1.0)
                    hb.set_array(vals)
                    vmax = float(np.nanmax(vals))
                    hb.set_clim(
                        0.0, vmax if vmax > 0 else 1.0
                    )
                else:
                    vmax = float(np.nanmax(counts))
                    hb.set_clim(
                        0.0, vmax if vmax > 0 else 1.0
                    )
        except Exception:
            pass

        # Rasterize to keep PDF size reasonable
        try:
            hb.set_rasterized(True)
        except Exception:
            pass

    sc = ax.scatter(
        x,
        y,
        c=c,
        cmap=cmap,
        norm=c_norm,
        s=float(s),
        alpha=float(alpha),
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )

    # Optional Pareto front (computed in objective space)
    if show_front and x.size >= 3:
        xo = _to_min_obj(x_metric, x)
        yo = _to_min_obj(y_metric, y)
        mask = _pareto_front_mask(xo, yo)
        if np.any(mask):
            xf = x[mask]
            yf = y[mask]
            order = np.argsort(xf)
            ax.plot(
                xf[order],
                yf[order],
                linewidth=1.2,
                alpha=0.9,
                zorder=4,
            )

    # Mark chosen (lambda_cons, lambda_prior) point if present
    if mark_lc is not None and mark_lp is not None:
        sel = sub.loc[
            (
                sub["lambda_cons"].astype(float)
                == float(mark_lc)
            )
            & (
                sub["lambda_prior"].astype(float)
                == float(mark_lp)
            )
        ]
        if not sel.empty:
            xv = float(np.nanmean(sel[x_metric].to_numpy()))
            yv = float(np.nanmean(sel[y_metric].to_numpy()))
            ax.scatter(
                [xv],
                [yv],
                marker="s",
                s=90,
                facecolors="none",
                edgecolors="black",
                linewidths=1.6,
                zorder=6,
            )

    if show_labels:
        ax.set_xlabel(_metric_label(x_metric))
        ax.set_ylabel(_metric_label(y_metric))
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    if show_title:
        ax.set_title(city, loc="left", fontweight="bold")

    _axes_cleanup(ax)
    return sc


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def plot_ablations_sensivity_main(
    argv: list[str] | None = None,
) -> None:
    args = _parse_args(argv)

    utils.set_paper_style(
        fontsize=int(args.font), dpi=int(args.dpi)
    )

    show_legend = utils.str_to_bool(
        args.show_legend, default=True
    )
    show_labels = utils.str_to_bool(
        args.show_labels, default=True
    )
    show_ticks = utils.str_to_bool(
        args.show_ticklabels, default=True
    )
    show_title = utils.str_to_bool(
        args.show_title, default=True
    )
    show_pan_t = utils.str_to_bool(
        args.show_panel_titles, default=True
    )

    annotate = not utils.str_to_bool(
        args.mute_values, default=False
    )
    show_cb = not utils.str_to_bool(
        args.no_colorbar, default=False
    )
    no_bars = utils.str_to_bool(args.no_bars, default=False)
    align_grid = utils.str_to_bool(
        args.align_grid, default=True
    )
    contour_lines = utils.str_to_bool(
        args.contour_lines, default=False
    )
    mark_best = not utils.str_to_bool(
        args.no_best_marker, default=False
    )

    df = _load_records(args)
    if df.empty:
        root = utils.as_path(args.root)
        raise SystemExit(
            "No ablation_record*.jsonl found under:\n"
            f"  {root.resolve()}\n"
            "Run ablations with the logger enabled first."
        )

    df = _canon_cols(df)
    df = _harmonize_units(df)
    df = _dedupe_prefer_best(df)
    df = _filter_models(df, args.models)

    cities = _resolve_cities(args)
    if not cities:
        raise SystemExit("No cities selected.")

    if len(cities) == 1:
        cities = [cities[0], cities[0]]
    city_a, city_b = cities[0], cities[1]

    out = _resolve_out(out=args.out, out_dir=args.out_dir)
    utils.ensure_dir(out.parent)

    tidy_csv = out.parent / "tableS6_ablations_used.csv"
    df.to_csv(tidy_csv, index=False)
    print(f"[OK] table -> {tidy_csv}")

    map_metrics = _resolve_map_metrics(args)

    # Auto-disable bars if multiple map metrics
    do_pareto = utils.str_to_bool(args.pareto, default=False)
    if do_pareto:
        x_metric = str(args.pareto_x)
        y_metric = str(args.pareto_y)
        c_metric = str(args.pareto_color)
        show_front = utils.str_to_bool(
            args.pareto_front, default=True
        )
        density_on = utils.str_to_bool(
            getattr(args, "pareto_density", "false"),
            default=False,
        )
        density_gridsize = int(
            getattr(args, "pareto_density_gridsize", 35)
        )
        density_bins = str(
            getattr(args, "pareto_density_bins", "log")
        )
        density_cmap = str(
            getattr(args, "pareto_density_cmap", "Greys")
        )
        density_alpha = float(
            getattr(args, "pareto_density_alpha", 0.25)
        )
        density_mincnt = int(
            getattr(args, "pareto_density_mincnt", 1)
        )

        # Shared color scale across cities for comparability
        cmin, cmax = _compute_clim(
            df, cities=[city_a, city_b], metric=c_metric
        )
        c_norm = None
        if cmin is not None and cmax is not None:
            c_norm = plt.Normalize(vmin=cmin, vmax=cmax)

        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(8.6, 3.8),
            dpi=int(args.dpi),
        )
        ax1, ax2 = axes[0], axes[1]

        sc1 = _pareto_panel(
            ax1,
            df,
            city=city_a,
            x_metric=x_metric,
            y_metric=y_metric,
            c_metric=c_metric,
            cmap=str(args.cmap),
            c_norm=c_norm,
            s=float(args.pareto_s),
            alpha=float(args.pareto_alpha),
            show_labels=True,
            show_title=True,
            mark_lc=args.mark_lambda_cons,
            mark_lp=args.mark_lambda_prior,
            show_front=show_front,
            density=density_on,
            density_gridsize=density_gridsize,
            density_bins=density_bins,
            density_cmap=density_cmap,
            density_alpha=density_alpha,
            density_mincnt=density_mincnt,
        )
        sc2 = _pareto_panel(
            ax2,
            df,
            city=city_b,
            x_metric=x_metric,
            y_metric=y_metric,
            c_metric=c_metric,
            cmap=str(args.cmap),
            c_norm=c_norm,
            s=float(args.pareto_s),
            alpha=float(args.pareto_alpha),
            show_labels=True,
            show_title=True,
            mark_lc=args.mark_lambda_cons,
            mark_lp=args.mark_lambda_prior,
            show_front=show_front,
            density=density_on,
            density_gridsize=density_gridsize,
            density_bins=density_bins,
            density_cmap=density_cmap,
            density_alpha=density_alpha,
            density_mincnt=density_mincnt,
        )

        if show_cb:
            sc = sc1 if sc1 is not None else sc2
            if sc is not None:
                cbar = fig.colorbar(
                    sc,
                    ax=[ax1, ax2],
                    orientation="vertical",
                    fraction=0.04,
                    pad=0.04,
                )
                cbar.set_label(_metric_label(c_metric))

        if show_title:
            default = "Supplement S6 • Pareto trade-offs"
            ttl = utils.resolve_title(
                default=default, title=args.title
            )
            fig.suptitle(ttl, fontsize=11, fontweight="bold")

        utils.save_figure(fig, out)
        return

    if len(map_metrics) > 1:
        no_bars = True

    # Build a shared lambda grid if requested
    lc_grid: list[float] | None = None
    lp_grid: list[float] | None = None
    if align_grid:
        lc_grid, lp_grid = _lambda_grid(
            df, cities=[city_a, city_b]
        )

    # -----------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------
    if not no_bars and len(map_metrics) == 1:
        # Legacy 2x2: bars (row1) + maps (row2)
        fig = plt.figure(figsize=(8.6, 7.0))
        gs = fig.add_gridspec(
            nrows=2,
            ncols=2,
            left=0.07,
            right=0.90 if (show_legend and show_cb) else 0.98,
            top=0.94,
            bottom=0.10,
            hspace=0.32,
            wspace=0.30,
        )

        ax11 = fig.add_subplot(gs[0, 0])
        _bars_by_lambda(
            ax11,
            df,
            city=city_a,
            metric=args.bar_metric,
            color=cfg.CITY_COLORS.get(city_a, "#1F78B4"),
            annotate=annotate,
            show_legend=show_legend,
            show_labels=show_labels,
            show_ticks=show_ticks,
            show_title=show_pan_t,
        )

        ax12 = fig.add_subplot(gs[0, 1])
        _bars_by_lambda(
            ax12,
            df,
            city=city_b,
            metric=args.bar_metric,
            color=cfg.CITY_COLORS.get(city_b, "#E31A1C"),
            annotate=annotate,
            show_legend=show_legend,
            show_labels=show_labels,
            show_ticks=show_ticks,
            show_title=show_pan_t,
        )

        metric = map_metrics[0]
        vmin, vmax = _compute_clim(
            df, cities=[city_a, city_b], metric=metric
        )

        ax21 = fig.add_subplot(gs[1, 0])
        im_a = _heatmap_one(
            ax21,
            df,
            city=city_a,
            metric=metric,
            cmap=args.cmap,
            map_kind=args.map_kind,
            levels=int(args.levels),
            contour_lines=contour_lines,
            lc_grid=lc_grid,
            lp_grid=lp_grid,
            vmin=vmin,
            vmax=vmax,
            show_labels=show_labels,
            show_ticks=show_ticks,
            show_title=show_pan_t,
            mark_best=mark_best,
            mark_lc=args.mark_lambda_cons,
            mark_lp=args.mark_lambda_prior,
        )

        ax22 = fig.add_subplot(gs[1, 1])
        im_b = _heatmap_one(
            ax22,
            df,
            city=city_b,
            metric=metric,
            cmap=args.cmap,
            map_kind=args.map_kind,
            levels=int(args.levels),
            contour_lines=contour_lines,
            lc_grid=lc_grid,
            lp_grid=lp_grid,
            vmin=vmin,
            vmax=vmax,
            show_labels=show_labels,
            show_ticks=show_ticks,
            show_title=show_pan_t,
            mark_best=mark_best,
            mark_lc=args.mark_lambda_cons,
            mark_lp=args.mark_lambda_prior,
        )

        if show_cb:
            cax = fig.add_axes([0.92, 0.12, 0.015, 0.30])
            im_for = im_a if im_a is not None else im_b
            if im_for is not None:
                fig.colorbar(
                    im_for,
                    cax=cax,
                    orientation="vertical",
                    label=_metric_label(metric),
                )
            else:
                cax.set_axis_off()

    else:
        # Heatmap-only multi-metric: 2 rows (cities) × M cols (metrics)
        M = max(1, len(map_metrics))
        fig = plt.figure(figsize=(3.2 * M, 5.8))
        # Allow room for per-column colorbars
        right = 0.98 if not show_cb else 0.92
        gs = fig.add_gridspec(
            nrows=2,
            ncols=M,
            left=0.07,
            right=right,
            top=0.93,
            bottom=0.11,
            hspace=0.20,
            wspace=0.25,
        )

        # Precompute per-metric color ranges and store artists for cbar
        artists: list[Any] = []
        top_axes: list[plt.Axes] = []
        bottom_axes: list[plt.Axes] = []

        for j, metric in enumerate(map_metrics):
            vmin, vmax = _compute_clim(
                df,
                cities=[city_a, city_b],
                metric=metric,
            )

            # City A (row 0)
            axA = fig.add_subplot(gs[0, j])
            top_axes.append(axA)
            artA = _heatmap_one(
                axA,
                df,
                city=city_a,
                metric=metric,
                cmap=args.cmap,
                map_kind=args.map_kind,
                levels=int(args.levels),
                contour_lines=contour_lines,
                lc_grid=lc_grid,
                lp_grid=lp_grid,
                vmin=vmin,
                vmax=vmax,
                show_labels=False,
                show_ticks=show_ticks,
                show_title=False,
                mark_best=mark_best,
                mark_lc=args.mark_lambda_cons,
                mark_lp=args.mark_lambda_prior,
            )

            # City B (row 1)
            axB = fig.add_subplot(gs[1, j])
            bottom_axes.append(axB)
            artB = _heatmap_one(
                axB,
                df,
                city=city_b,
                metric=metric,
                cmap=args.cmap,
                map_kind=args.map_kind,
                levels=int(args.levels),
                contour_lines=contour_lines,
                lc_grid=lc_grid,
                lp_grid=lp_grid,
                vmin=vmin,
                vmax=vmax,
                show_labels=(show_labels and j == 0),
                show_ticks=show_ticks,
                show_title=False,
                mark_best=mark_best,
                mark_lc=args.mark_lambda_cons,
                mark_lp=args.mark_lambda_prior,
            )

            # Column titles: metric names
            if show_pan_t:
                axA.set_title(
                    _metric_label(metric),
                    fontweight="bold",
                    pad=6,
                )

            # Row labels: cities on left-most column
            if j == 0 and show_pan_t:
                axA.text(
                    -0.20,
                    0.5,
                    city_a,
                    transform=axA.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontweight="bold",
                )
                axB.text(
                    -0.20,
                    0.5,
                    city_b,
                    transform=axB.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontweight="bold",
                )

            artists.append(artA if artA is not None else artB)

        # Colorbars: one per metric column (shared across the two cities)
        if show_cb:
            for j, metric in enumerate(map_metrics):
                art = artists[j] if j < len(artists) else None
                axA = (
                    top_axes[j] if j < len(top_axes) else None
                )
                axB = (
                    bottom_axes[j]
                    if j < len(bottom_axes)
                    else None
                )
                if art is None or axA is None or axB is None:
                    continue
                cbar = fig.colorbar(
                    art,
                    ax=[axA, axB],
                    orientation="horizontal",
                    fraction=0.06,
                    pad=0.10,
                )
                cbar.set_label(_metric_label(metric))

    # -----------------------------------------------------------------
    # Title + save
    # -----------------------------------------------------------------
    if show_title:
        default = "Supplement S6 • Extended ablations & sensitivities"
        ttl = utils.resolve_title(
            default=default, title=args.title
        )
        fig.suptitle(ttl, fontsize=11, fontweight="bold")

    utils.save_figure(fig, out)


def main(argv: list[str] | None = None) -> None:
    plot_ablations_sensivity_main(argv)


if __name__ == "__main__":
    main()
