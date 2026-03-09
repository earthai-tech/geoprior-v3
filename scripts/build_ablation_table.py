# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""scripts.build_ablation_table

Build tidy ablation/sensitivity tables from
ablation_records/ablation_record*.jsonl.

Design goals
------------
- Robust input: user can pass a results root, a city root,
  an ablation_records/ folder, or the ablation_record.jsonl
  file itself.
- Robust schema: tolerate older/newer column aliases.
- Robust units: normalize subsidence-distance metrics to mm.
- Robust paper export: compact TeX table with arrows (↑/↓)
  and a single error metric (RMSE or MSE).

Run
---
python -m scripts build-ablation-table --root results

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config as cfg
from . import utils

_LOWER_IS_BETTER = {
    "mae",
    "mse",
    "rmse",
    "sharpness80",
    "epsilon_prior",
    "epsilon_cons",
    "epsilon_gw",
}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _parse_args(
    argv: list[str] | None,
) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="build-ablation-table",
        description=(
            "Build ablation tables from ablation_record.jsonl."
        ),
    )

    p.add_argument(
        "--root",
        type=str,
        default="results",
        help=(
            "Root to scan (results/ or a city/job folder). "
            "Can also be an ablation_record*.jsonl file."
        ),
    )
    p.add_argument(
        "--input",
        type=str,
        default=None,
        action="append",
        help=(
            "Explicit inputs (repeatable). Supports "
            ".jsonl/.json/.csv. Overrides --root scan."
        ),
    )

    p.add_argument(
        "--cities",
        type=str,
        default="",
        help="Comma list of cities to keep (default: all).",
    )
    p.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma list of model names to keep.",
    )
    p.add_argument(
        "--sort-by",
        type=str,
        default="mae",
        help=(
            "Sort by this metric (default: mae). "
            "Use 'none' to keep input order."
        ),
    )
    p.add_argument(
        "--ascending",
        type=str,
        default="auto",
        help=(
            "Sort direction: true/false/auto. "
            "auto uses metric convention."
        ),
    )

    # Output
    p.add_argument(
        "--out",
        type=str,
        default="table_ablations",
        help=(
            "Output stem/path (scripts/out/ if relative). "
            "Suffix is stripped and treated as stem."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output dir override.",
    )
    p.add_argument(
        "--formats",
        type=str,
        default="csv,json",
        help=(
            "Comma list: csv,json,tex,txt. Default: csv,json"
        ),
    )

    # Table content
    p.add_argument(
        "--metrics",
        type=str,
        default="",
        help=(
            "Comma list of metric columns to keep. "
            "Default: keep all known metrics."
        ),
    )
    p.add_argument(
        "--keep-per-horizon",
        type=str,
        default="true",
        help=("Include per-horizon metrics (true/false)."),
    )
    p.add_argument(
        "--max-h",
        type=int,
        default=3,
        help="Max horizon columns to export (default 3).",
    )

    # Unit handling
    p.add_argument(
        "--metric-unit",
        type=str,
        default="mm",
        choices=["mm", "m"],
        help=(
            "Unit for subsidence distance metrics in output. "
            "Default: mm."
        ),
    )

    # Paper mode
    p.add_argument(
        "--for-paper",
        action="store_true",
        help="Produce compact paper-friendly table.",
    )
    p.add_argument(
        "--err-metric",
        type=str,
        default="rmse",
        choices=["rmse", "mse"],
        help=(
            "Paper table keeps only one error metric (rmse or mse)."
        ),
    )
    p.add_argument(
        "--keep-r2",
        action="store_true",
        help="In --for-paper mode, keep R2 column as well.",
    )
    p.add_argument(
        "--sideway",
        action="store_true",
        help="Write TeX as sidewaystable (landscape).",
    )
    p.add_argument(
        "--caption",
        type=str,
        default=(
            "Extended ablations and sensitivity analysis."
        ),
        help="TeX caption.",
    )
    p.add_argument(
        "--label",
        type=str,
        default="tab:ablations",
        help="TeX label.",
    )
    p.add_argument(
        "--best-per-city",
        action="store_true",
        help=(
            "Also export best row per city for --sort-by metric."
        ),
    )
    p.add_argument(
        "--stdout",
        action="store_true",
        help="Print final table to stdout.",
    )

    # -------------------------------------------------
    # Optional grouped outputs (Tables S6 / S7)
    # -------------------------------------------------
    p.add_argument(
        "--group-cols",
        type=str,
        default="",
        help=(
            "Extra grouped outputs. Comma list: s6,s7. "
            "s6: λ_cons×λ_prior grid. "
            "s7: toggle ablations summary."
        ),
    )
    p.add_argument(
        "--s6-metrics",
        type=str,
        default="",
        help=(
            "Comma list of metrics for Table S6 grids. "
            "Default: use --sort-by if present else mae."
        ),
    )
    p.add_argument(
        "--s6-agg",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Aggregation for duplicate grid points.",
    )

    p.add_argument(
        "--s6-tex-bold-best",
        type=str,
        default="true",
        help=(
            "In S6 TeX grids, bold the best cell "
            "(true/false)."
        ),
    )
    p.add_argument(
        "--s6-tex-gray-missing",
        type=str,
        default="true",
        help=(
            "In S6 TeX grids, gray missing/invalid "
            "cells (true/false)."
        ),
    )
    p.add_argument(
        "--s6-tex-gray-level",
        type=str,
        default="gray!15",
        help=("TeX color for missing cells (e.g., gray!15)."),
    )
    p.add_argument(
        "--s6-tex-one-file",
        type=str,
        default="false",
        help=(
            "Write one TeX file per (city,physics) with "
            "all metrics, instead of one per metric."
        ),
    )
    p.add_argument(
        "--s7-cols",
        type=str,
        default="",
        help=(
            "Comma list of grouping columns for Table S7. "
            "Default: a robust preset."
        ),
    )
    p.add_argument(
        "--s7-metrics",
        type=str,
        default="",
        help=(
            "Comma list of metrics for Table S7 summary. "
            "Default: paper metrics if --for-paper else all."
        ),
    )
    p.add_argument(
        "--s7-agg",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Aggregation for Table S7 groups.",
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


def _load_one(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()

    if suf == ".csv":
        df = pd.read_csv(path)
        df["_src"] = df.get("_src", str(path))
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
        elif isinstance(obj, dict):
            df = pd.DataFrame([obj])
        else:
            df = pd.DataFrame([])

        if not df.empty and "_src" not in df.columns:
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
        df = _load_one(fp)
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
                df = _load_one(p)
                if not df.empty:
                    blocks.append(df)
        if not blocks:
            return pd.DataFrame([])
        return pd.concat(blocks, ignore_index=True)

    root = utils.as_path(args.root)
    return _scan_records(root)


# ---------------------------------------------------------------------
# Canonicalization / flattening
# ---------------------------------------------------------------------
def _canon_pde_mode(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in {"none", "off", "no", "0", "false"}:
        return "none"
    if s in {"both", "on", "1", "true"}:
        return "both"
    if s in {"consolidation", "gw_flow"}:
        return "both"
    return s


def _expand_dict_col(
    df: pd.DataFrame,
    col: str,
    *,
    prefix: str,
    max_h: int,
    only_h: bool,
) -> pd.DataFrame:
    if col not in df.columns:
        return df

    def _keys(d: dict[str, Any]) -> list[str]:
        out: list[str] = []
        for k in d.keys():
            kk = str(k)
            if only_h:
                if not (
                    kk.startswith("H") and kk[1:].isdigit()
                ):
                    continue
                if int(kk[1:]) > int(max_h):
                    continue
            out.append(kk)
        if only_h:
            out.sort(key=lambda x: int(x[1:]))
        else:
            out.sort()
        return out

    keys: list[str] = []
    for v in df[col].to_list():
        if isinstance(v, dict):
            keys = _keys(v)
            if keys:
                break

    if not keys:
        return df

    for k in keys:
        name = f"{prefix}{k}"
        df[name] = df[col].apply(
            lambda d: (
                d.get(k) if isinstance(d, dict) else np.nan
            )
        )
        df[name] = pd.to_numeric(df[name], errors="coerce")

    return df


def _canon_cols(
    df: pd.DataFrame, *, max_h: int
) -> pd.DataFrame:
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

    df["pde_bucket"] = np.where(
        df["pde_mode"].astype(str).eq("none"),
        "none",
        "both",
    )

    # Flatten common nested blocks
    df = _expand_dict_col(
        df,
        "per_horizon_mae",
        prefix="per_horizon_mae.",
        max_h=max_h,
        only_h=True,
    )
    df = _expand_dict_col(
        df,
        "per_horizon_r2",
        prefix="per_horizon_r2.",
        max_h=max_h,
        only_h=True,
    )

    num_cols = [
        "lambda_prior",
        "lambda_cons",
        "lambda_gw",
        "lambda_smooth",
        "lambda_mv",
        "lambda_bounds",
        "lambda_q",
        "hd_factor",
        "r2",
        "mae",
        "mse",
        "rmse",
        "coverage80",
        "sharpness80",
        "epsilon_prior",
        "epsilon_cons",
        "epsilon_gw",
    ]

    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in list(df.columns):
        if c.startswith("per_horizon_"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ---------------------------------------------------------------------
# Units + derived metrics
# ---------------------------------------------------------------------
def _needs_m_to_mm(row: pd.Series) -> bool:
    u = row.get("units", None)
    if isinstance(u, dict):
        uu = str(u.get("subs_metrics_unit", "")).lower()
        if uu == "mm":
            return False
        if uu in {"m", "meter", "metre"}:
            return True

    # Heuristic fallback
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


def _scale_rows_to_mm(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    mask = df.apply(_needs_m_to_mm, axis=1)
    if not mask.any():
        return df

    dist_cols: list[str] = []
    mse_cols: list[str] = []
    for c in df.columns:
        s = str(c).lower()
        if s == "mse" or s.endswith(".mse"):
            mse_cols.append(c)
        if s == "mae" or s.endswith(".mae"):
            dist_cols.append(c)
        if s == "rmse" or s.endswith(".rmse"):
            dist_cols.append(c)
        if s == "sharpness80" or s.endswith(".sharpness80"):
            dist_cols.append(c)

    dist_cols = sorted(set(dist_cols))
    mse_cols = sorted(set(mse_cols))

    for c in dist_cols:
        df.loc[mask, c] = df.loc[mask, c] * 1000.0
    for c in mse_cols:
        df.loc[mask, c] = df.loc[mask, c] * 1e6

    return df


def _to_unit(df: pd.DataFrame, *, unit: str) -> pd.DataFrame:
    """Convert mm->m if requested. Default table unit is mm."""
    unit2 = str(unit).strip().lower()
    if unit2 not in {"mm", "m"}:
        return df
    if unit2 == "mm":
        return df

    out = df.copy()

    dist_cols: list[str] = []
    mse_cols: list[str] = []
    for c in out.columns:
        s = str(c).lower()
        if s == "mse" or s.endswith(".mse"):
            mse_cols.append(c)
        if s == "mae" or s.endswith(".mae"):
            dist_cols.append(c)
        if s == "rmse" or s.endswith(".rmse"):
            dist_cols.append(c)
        if s == "sharpness80" or s.endswith(".sharpness80"):
            dist_cols.append(c)

    for c in dist_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce") * 1e-3
    for c in mse_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce") * 1e-6

    return out


def _ensure_mse_rmse(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    mse = out["mse"] if "mse" in out.columns else None
    rmse = out["rmse"] if "rmse" in out.columns else None

    if "rmse" not in out.columns and mse is not None:
        out["rmse"] = np.sqrt(
            np.maximum(0.0, mse.to_numpy(float))
        )

    if "mse" not in out.columns and rmse is not None:
        out["mse"] = rmse.to_numpy(float) ** 2

    return out


# ---------------------------------------------------------------------
# Dedupe / filters
# ---------------------------------------------------------------------
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

    if isinstance(row.get("metrics", None), dict):
        s += 10
    return int(s)


def _dedupe_prefer_best(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "timestamp" not in df.columns:
        return df
    if "city" not in df.columns:
        df = df.copy()
        df["city"] = ""

    x = df.copy()
    x["_score"] = x.apply(_record_score, axis=1)
    x = x.sort_values(
        by=["timestamp", "city", "_score"],
        ascending=[True, True, False],
    )
    x = x.drop_duplicates(
        subset=["timestamp", "city"],
        keep="first",
    ).copy()
    return x.drop(columns=["_score"], errors="ignore")


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


def _filter_cities(
    df: pd.DataFrame, cities: str
) -> pd.DataFrame:
    raw = str(cities or "").strip()
    if not raw or "city" not in df.columns:
        return df

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    keep = [utils.canonical_city(p) for p in parts]
    if not keep:
        return df

    return df.loc[df["city"].astype(str).isin(keep)].copy()


def _sort_df(
    df: pd.DataFrame,
    *,
    metric: str,
    ascending: str,
) -> pd.DataFrame:
    m = str(metric or "").strip().lower()
    if not m or m == "none":
        return df
    if m not in df.columns:
        return df

    asc_raw = str(ascending or "auto").strip().lower()
    if asc_raw in {"true", "1", "yes"}:
        asc = True
    elif asc_raw in {"false", "0", "no"}:
        asc = False
    else:
        asc = m in _LOWER_IS_BETTER

    return df.sort_values(by=[m], ascending=asc)


# ---------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------
def _default_metric_cols(df: pd.DataFrame) -> list[str]:
    base = [
        "mae",
        "rmse",
        "mse",
        "r2",
        "coverage80",
        "sharpness80",
        "epsilon_prior",
        "epsilon_cons",
        "epsilon_gw",
    ]

    out: list[str] = [c for c in base if c in df.columns]
    for c in df.columns:
        if str(c).startswith("per_horizon_mae."):
            out.append(str(c))
        if str(c).startswith("per_horizon_r2."):
            out.append(str(c))
    return out


def _pick_metrics(df: pd.DataFrame, spec: str) -> list[str]:
    raw = str(spec or "").strip()
    if not raw:
        return _default_metric_cols(df)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [p for p in parts if p in df.columns]


def _param_cols(df: pd.DataFrame) -> list[str]:
    base = [
        "timestamp",
        "city",
        "model",
        "pde_mode",
        "use_effective_h",
        "kappa_mode",
        "hd_factor",
        "lambda_cons",
        "lambda_gw",
        "lambda_prior",
        "lambda_smooth",
        "lambda_mv",
        "lambda_bounds",
        "lambda_q",
    ]
    return [c for c in base if c in df.columns]


def _paper_cols(
    df: pd.DataFrame,
    *,
    err_metric: str,
    keep_r2: bool,
) -> tuple[list[str], list[str]]:
    params = [
        "city",
        "pde_bucket",
        "lambda_cons",
        "lambda_prior",
        "lambda_gw",
        "lambda_smooth",
        "lambda_bounds",
        "lambda_mv",
        "lambda_q",
        "use_effective_h",
        "kappa_mode",
        "hd_factor",
    ]
    params = [c for c in params if c in df.columns]

    mets = [
        "mae",
        str(err_metric),
        "coverage80",
        "sharpness80",
    ]
    if keep_r2:
        mets.append("r2")
    mets = [c for c in mets if c in df.columns]
    return params, mets


# ---------------------------------------------------------------------
# TeX
# ---------------------------------------------------------------------
def _tex_escape(s: Any) -> str:
    t = str(s)
    t = t.replace("\\", r"\textbackslash{}")
    t = t.replace("_", r"\_")
    t = t.replace("%", r"\%")
    t = t.replace("&", r"\&")
    return t


def _fmt_bool(x: Any) -> str:
    if isinstance(x, bool):
        return "Yes" if x else "No"
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return "Yes"
    if s in {"0", "false", "no", "n", "off"}:
        return "No"
    return _tex_escape(x)


def _tex_header(
    col: str,
    *,
    unit: str,
) -> str:
    c = str(col)
    if c == "city":
        return "City"
    if c == "model":
        return "Model"
    if c in {"pde_bucket", "pde_mode"}:
        return "Physics"
    if c == "use_effective_h":
        return r"Use $H_{\mathrm{eff}}$"
    if c == "kappa_mode":
        return r"$\kappa$ mode"
    if c == "hd_factor":
        return r"$H_d/H_{\mathrm{eff}}$"

    lam = {
        "lambda_cons": r"$\lambda_{\mathrm{cons}}$",
        "lambda_gw": r"$\lambda_{\mathrm{gw}}$",
        "lambda_prior": r"$\lambda_{\mathrm{prior}}$",
        "lambda_smooth": r"$\lambda_{\mathrm{smooth}}$",
        "lambda_bounds": r"$\lambda_{\mathrm{bounds}}$",
        "lambda_mv": r"$\lambda_{\mathrm{mv}}$",
        "lambda_q": r"$\lambda_{q}$",
    }
    if c in lam:
        return lam[c]

    meta = cfg.PLOT_METRIC_META.get(c, None)
    if isinstance(meta, dict):
        ttl = str(meta.get("title", c))
        u = str(meta.get("unit", "") or "")
        if u:
            # Override unit if caller requests meters.
            if unit == "m" and u in {"mm", "mm²"}:
                u = "m" if u == "mm" else "m$^2$"
            ttl = ttl.format(unit=u)
        return _tex_escape(ttl)

    if c in {"epsilon_prior", "epsilon_cons", "epsilon_gw"}:
        suf = "(↓)"
        if c == "epsilon_prior":
            return r"$\epsilon_{\mathrm{prior}}$ " + suf
        if c == "epsilon_cons":
            return r"$\epsilon_{\mathrm{cons}}$ " + suf
        return r"$\epsilon_{\mathrm{gw}}$ " + suf

    if c.startswith("per_horizon_mae."):
        h = c.split(".", 1)[1]
        return rf"MAE$_{{{_tex_escape(h)}}}$"
    if c.startswith("per_horizon_r2."):
        h = c.split(".", 1)[1]
        return rf"$R^2_{{{_tex_escape(h)}}}$"

    return _tex_escape(c)


def _tex_fmt_val(
    v: Any,
    *,
    col: str,
) -> str:
    if v is None:
        return "--"
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return "--"

    if col == "use_effective_h":
        return _fmt_bool(v)
    if col in {"pde_bucket", "pde_mode"}:
        s = str(v).strip().lower()
        return "On" if s != "none" else "Off"

    meta = cfg.PLOT_METRIC_META.get(str(col), None)
    if isinstance(meta, dict):
        fmt = str(meta.get("fmt", "{:.3g}"))
        try:
            return fmt.format(float(v))
        except Exception:
            return "--"

    try:
        if isinstance(v, int | np.integer):
            return str(int(v))
        if isinstance(v, float | np.floating):
            return f"{float(v):.3g}"
    except Exception:
        pass

    return _tex_escape(v)


def _to_tex_table(
    df: pd.DataFrame,
    *,
    cols: list[str],
    unit: str,
    sideway: bool,
    caption: str,
    label: str,
) -> str:
    # Alignment: first 2 columns left, rest right.
    aligns: list[str] = []
    for i, _c in enumerate(cols):
        if i < 2:
            aligns.append("l")
        else:
            aligns.append("r")
    spec = "".join(aligns)

    head = " & ".join(_tex_header(c, unit=unit) for c in cols)

    lines: list[str] = []
    env = "sidewaystable" if sideway else "table"
    lines.append(rf"\\begin{{{env}}}[t]")
    lines.append(r"\\centering")
    lines.append(r"\\small")
    lines.append(r"\\setlength{\\tabcolsep}{4pt}")
    lines.append(r"\\renewcommand{\\arraystretch}{1.15}")
    lines.append(r"\\begin{tabular}{" + spec + r"}")
    lines.append(r"\\toprule")
    lines.append(head + r" \\")
    lines.append(r"\\midrule")

    for _, row in df.iterrows():
        vals = [
            _tex_fmt_val(row.get(c, None), col=c)
            for c in cols
        ]
        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\\bottomrule")
    lines.append(r"\\end{tabular}")
    lines.append(rf"\\caption{{{_tex_escape(caption)}}}")
    lines.append(rf"\\label{{{_tex_escape(label)}}}")
    lines.append(rf"\\end{{{env}}}")
    lines.append("")
    return "\n".join(lines)


def _resolve_out(out: str, out_dir: str | None) -> Path:
    p = Path(out).expanduser()
    if p.suffix:
        p = p.with_suffix("")
    if out_dir:
        base = Path(out_dir).expanduser()
        return (base / p).resolve()
    if not p.is_absolute():
        p = cfg.OUT_DIR / p
    return p


def _best_by_city(
    df: pd.DataFrame,
    *,
    metric: str,
    ascending: str,
) -> pd.DataFrame:
    if df.empty or "city" not in df.columns:
        return df
    x = _sort_df(df, metric=metric, ascending=ascending)
    return x.groupby("city", sort=False).head(1).copy()


# ---------------------------------------------------------------------
# Grouped outputs (S6 / S7)
# ---------------------------------------------------------------------
def _split_csv(raw: Any) -> list[str]:
    s = str(raw or "").strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def _agg_fn(name: str):
    k = str(name or "mean").strip().lower()
    if k == "median":
        return "median"
    return "mean"


def _s6_metrics(args: argparse.Namespace) -> list[str]:
    ms = _split_csv(args.s6_metrics)
    if ms:
        return ms
    s = str(args.sort_by or "").strip().lower()
    return [s] if s else ["mae"]


def _s7_group_cols(
    df: pd.DataFrame, args: argparse.Namespace
) -> list[str]:
    cols = _split_csv(args.s7_cols)
    if cols:
        return [c for c in cols if c in df.columns]

    preset = [
        "city",
        "pde_bucket",
        "use_effective_h",
        "kappa_mode",
        "hd_factor",
        "smooth_on",
        "bounds_on",
        "mv_on",
        "q_on",
    ]
    return [c for c in preset if c in df.columns]


def _s7_metrics(
    df: pd.DataFrame, args: argparse.Namespace
) -> list[str]:
    ms = _split_csv(args.s7_metrics)
    if ms:
        return [m for m in ms if m in df.columns]

    if args.for_paper:
        mets = [
            "mae",
            str(args.err_metric),
            "coverage80",
            "sharpness80",
        ]
        if args.keep_r2:
            mets.append("r2")
        return [m for m in mets if m in df.columns]

    return _default_metric_cols(df)


def _add_toggle_flags(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    def _on(col: str, out: str) -> None:
        if col not in x.columns:
            return
        v = pd.to_numeric(x[col], errors="coerce")
        x[out] = np.where(v > 0.0, "on", "off")

    _on("lambda_smooth", "smooth_on")
    _on("lambda_bounds", "bounds_on")
    _on("lambda_mv", "mv_on")
    _on("lambda_q", "q_on")
    return x


def _metric_lower_is_better(metric: str) -> bool:
    m = str(metric or "").strip().lower()
    if not m:
        return True
    if m in _LOWER_IS_BETTER:
        return True

    meta = cfg.PLOT_METRIC_META.get(m, None)
    if isinstance(meta, dict):
        ttl = str(meta.get("title", ""))
        if "↓" in ttl:
            return True
        if "↑" in ttl:
            return False

    if "r2" in m or "coverage" in m:
        return False
    return True


def _best_mask(
    piv: pd.DataFrame, *, metric: str
) -> pd.DataFrame:
    if piv.empty:
        return pd.DataFrame(
            False, index=piv.index, columns=piv.columns
        )

    arr = piv.to_numpy(dtype=float, copy=True)
    ok = np.isfinite(arr)
    if not ok.any():
        return pd.DataFrame(
            False, index=piv.index, columns=piv.columns
        )

    if _metric_lower_is_better(metric):
        best = np.nanmin(arr)
    else:
        best = np.nanmax(arr)

    mask = ok & np.isclose(arr, best, rtol=1e-9, atol=1e-12)
    return pd.DataFrame(
        mask, index=piv.index, columns=piv.columns
    )


def _tex_pivot(
    piv: pd.DataFrame,
    *,
    metric: str,
    unit: str,
    sideway: bool,
    caption: str,
    label: str,
    bold_best: bool,
    gray_missing: bool,
    gray_level: str,
) -> str:
    cols = ["$\\lambda_{\\mathrm{cons}}$"]
    for c in piv.columns:
        try:
            c2 = float(c)
            cols.append(
                f"$\\lambda_{{\\mathrm{{prior}}}}={c2:g}$"
            )
        except Exception:
            cols.append(
                f"$\\lambda_{{\\mathrm{{prior}}}}={_tex_escape(c)}$"
            )

    spec = "l" + "r" * int(len(piv.columns))
    env = "sidewaystable" if sideway else "table"

    meta = cfg.PLOT_METRIC_META.get(metric, {})
    fmt = str(meta.get("fmt", "{:.3g}"))

    best = (
        _best_mask(piv, metric=metric) if bold_best else None
    )

    lines: list[str] = []
    if gray_missing:
        lines.append(
            "% Requires: \\usepackage[table]{xcolor}"
        )
    lines.append("% Requires: \\usepackage{booktabs}")

    lines.append(rf"\\begin{{{env}}}[t]")
    lines.append(r"\\centering")
    lines.append(r"\\small")
    lines.append(r"\\setlength{\\tabcolsep}{4pt}")
    lines.append(r"\\renewcommand{\\arraystretch}{1.15}")
    lines.append(r"\\begin{tabular}{" + spec + r"}")
    lines.append(r"\\toprule")
    lines.append(" & ".join(cols) + r" \\")
    lines.append(r"\\midrule")

    gl = str(gray_level or "gray!15").strip() or "gray!15"

    for lc, row in piv.iterrows():
        vals: list[str] = []
        try:
            vals.append(f"{float(lc):g}")
        except Exception:
            vals.append("--")

        for c in piv.columns:
            v = row.get(c, np.nan)
            if v is None or (
                isinstance(v, float) and not np.isfinite(v)
            ):
                miss = "--"
                if gray_missing:
                    miss = rf"\\cellcolor{{{gl}}}--"
                vals.append(miss)
                continue

            try:
                s = fmt.format(float(v))
            except Exception:
                s = "--"

            if best is not None and bool(best.loc[lc, c]):
                s = rf"\\textbf{{{s}}}"
            vals.append(s)

        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\\bottomrule")
    lines.append(r"\\end{tabular}")
    lines.append(rf"\\caption{{{_tex_escape(caption)}}}")
    lines.append(rf"\\label{{{_tex_escape(label)}}}")
    lines.append(rf"\\end{{{env}}}")
    lines.append("")
    return "\n".join(lines)


def _write_table(
    df: pd.DataFrame,
    *,
    out: Path,
    fmts: list[str],
) -> None:
    if "csv" in fmts:
        p = out.with_suffix(".csv")
        df.to_csv(p, index=False)
        print(f"[OK] csv -> {p}")

    if "json" in fmts:
        p = out.with_suffix(".json")
        p.write_text(
            json.dumps(
                df.to_dict(orient="records"), indent=2
            ),
            encoding="utf-8",
        )
        print(f"[OK] json -> {p}")

    if "txt" in fmts:
        p = out.with_suffix(".txt")
        p.write_text(
            df.to_string(index=False), encoding="utf-8"
        )
        print(f"[OK] txt -> {p}")


def _write_tex(
    text: str, *, out: Path, fmts: list[str]
) -> None:
    if "tex" not in fmts:
        return
    p = out.with_suffix(".tex")
    p.write_text(text, encoding="utf-8")
    print(f"[OK] tex -> {p}")


def _build_table_s6(
    df: pd.DataFrame,
    *,
    args: argparse.Namespace,
    out: Path,
    fmts: list[str],
) -> None:
    if "lambda_cons" not in df.columns:
        return
    if "lambda_prior" not in df.columns:
        return

    agg = _agg_fn(args.s6_agg)
    unit = str(args.metric_unit)

    metrics = [
        m for m in _s6_metrics(args) if m in df.columns
    ]
    if not metrics:
        return

    bold_best = utils.str_to_bool(
        getattr(args, "s6_tex_bold_best", "true"),
        default=True,
    )
    gray_missing = utils.str_to_bool(
        getattr(args, "s6_tex_gray_missing", "true"),
        default=True,
    )
    gray_level = (
        str(
            getattr(args, "s6_tex_gray_level", "gray!15")
            or "gray!15"
        ).strip()
        or "gray!15"
    )
    one_file = utils.str_to_bool(
        getattr(args, "s6_tex_one_file", "false"),
        default=False,
    )

    cities = sorted(set(df.get("city", [])))
    buckets = sorted(set(df.get("pde_bucket", [])))

    for city in cities:
        for pb in buckets:
            sub = df.copy()
            if "city" in sub.columns:
                sub = sub.loc[sub["city"].eq(city)]
            if "pde_bucket" in sub.columns:
                sub = sub.loc[sub["pde_bucket"].eq(pb)]

            if sub.empty:
                continue

            lc = pd.to_numeric(
                sub["lambda_cons"], errors="coerce"
            )
            lp = pd.to_numeric(
                sub["lambda_prior"], errors="coerce"
            )
            lc_vals = [
                float(x) for x in sorted(set(lc.dropna()))
            ]
            lp_vals = [
                float(x) for x in sorted(set(lp.dropna()))
            ]
            if not lc_vals or not lp_vals:
                continue

            tex_blocks: list[str] = []

            for m in metrics:
                piv = sub.pivot_table(
                    index="lambda_cons",
                    columns="lambda_prior",
                    values=m,
                    aggfunc=agg,
                )

                piv = piv.reindex(
                    index=lc_vals, columns=lp_vals
                )

                p2 = out.with_name(
                    out.name + f"__S6__{m}__{city}__{pb}"
                )

                flat = piv.reset_index()
                _write_table(flat, out=p2, fmts=fmts)

                if "tex" in fmts:
                    ttl = _tex_header(m, unit=unit)
                    cap = (
                        f"Table S6 ({city}, physics={pb}): "
                        f"$\\lambda_\\mathrm{{cons}}$"
                        f"$\\times$"
                        f"$\\lambda_\\mathrm{{prior}}$ "
                        f"grid for {ttl}."
                    )
                    lab = f"tab:S6_{city}_{pb}_{m}"
                    tex = _tex_pivot(
                        piv,
                        metric=m,
                        unit=unit,
                        sideway=bool(args.sideway),
                        caption=cap,
                        label=lab,
                        bold_best=bold_best,
                        gray_missing=gray_missing,
                        gray_level=gray_level,
                    )

                    if one_file:
                        tex_blocks.append(tex)
                    else:
                        _write_tex(tex, out=p2, fmts=fmts)

            if one_file and tex_blocks and "tex" in fmts:
                p3 = out.with_name(
                    out.name + f"__S6__{city}__{pb}"
                )
                _write_tex(
                    "\n".join(tex_blocks),
                    out=p3,
                    fmts=fmts,
                )


def _build_table_s7(
    df: pd.DataFrame,
    *,
    args: argparse.Namespace,
    out: Path,
    fmts: list[str],
) -> None:
    x = _add_toggle_flags(df)

    grp_cols = _s7_group_cols(x, args)
    mets = _s7_metrics(x, args)
    if not grp_cols or not mets:
        return

    agg = _agg_fn(args.s7_agg)

    keep = grp_cols + mets
    sub = x.loc[:, [c for c in keep if c in x.columns]].copy()
    if sub.empty:
        return

    gb = sub.groupby(grp_cols, dropna=False)
    out_df = gb[mets].agg(agg).reset_index()
    out_df["n"] = gb.size().to_numpy(int)

    # Paper table: n first, then metrics.
    cols = grp_cols + ["n"] + mets
    out_df = out_df.loc[:, cols]

    p2 = out.with_name(out.name + "__S7")
    _write_table(out_df, out=p2, fmts=fmts)

    if "tex" in fmts:
        cap = "Table S7: ablation toggles summary (group means)."
        lab = "tab:S7_ablations"
        tex = _to_tex_table(
            out_df,
            cols=list(out_df.columns),
            unit=str(args.metric_unit),
            sideway=bool(args.sideway),
            caption=cap,
            label=lab,
        )
        _write_tex(tex, out=p2, fmts=fmts)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def build_ablation_table_main(
    argv: list[str] | None = None,
) -> None:
    args = _parse_args(argv)
    utils.ensure_script_dirs()

    df = _load_records(args)
    if df.empty:
        root = utils.as_path(args.root)
        raise SystemExit(
            "No ablation_record*.jsonl found under:\n"
            f"  {root.resolve()}\n"
            "Run sensitivity/ablations first."
        )

    df = _canon_cols(df, max_h=int(args.max_h))
    df = _scale_rows_to_mm(df)
    df = _ensure_mse_rmse(df)
    df = _dedupe_prefer_best(df)
    df = _filter_models(df, args.models)
    df = _filter_cities(df, args.cities)

    keep_ph = utils.str_to_bool(
        args.keep_per_horizon, default=True
    )
    if not keep_ph:
        drop = [
            c
            for c in df.columns
            if str(c).startswith("per_horizon_")
        ]
        df = df.drop(columns=drop, errors="ignore")

    # Convert to requested unit after derived-metric completion.
    df = _to_unit(df, unit=args.metric_unit)

    df = _sort_df(
        df,
        metric=args.sort_by,
        ascending=args.ascending,
    )

    out = _resolve_out(args.out, args.out_dir)
    utils.ensure_dir(out.parent)

    if args.for_paper:
        params, mets = _paper_cols(
            df,
            err_metric=args.err_metric,
            keep_r2=bool(args.keep_r2),
        )
        cols = params + mets
    else:
        cols = _param_cols(df) + _pick_metrics(
            df, args.metrics
        )
        cols = [c for c in cols if c in df.columns]

    tab = df.loc[:, cols].copy()

    fmts = [
        s.strip().lower()
        for s in str(args.formats).split(",")
    ]
    fmts = [f for f in fmts if f]
    if not fmts:
        fmts = ["csv", "json"]

    if "csv" in fmts:
        p = out.with_suffix(".csv")
        tab.to_csv(p, index=False)
        print(f"[OK] csv -> {p}")

    if "json" in fmts:
        p = out.with_suffix(".json")
        p.write_text(
            json.dumps(
                tab.to_dict(orient="records"),
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[OK] json -> {p}")

    if "txt" in fmts:
        p = out.with_suffix(".txt")
        p.write_text(
            tab.to_string(index=False), encoding="utf-8"
        )
        print(f"[OK] txt -> {p}")

    if "tex" in fmts:
        if not args.for_paper:
            print(
                "[Warn] --formats tex is recommended with --for-paper "
                "(compact columns)."
            )
        tex = _to_tex_table(
            tab,
            cols=list(tab.columns),
            unit=str(args.metric_unit),
            sideway=bool(args.sideway),
            caption=str(args.caption),
            label=str(args.label),
        )
        p = out.with_suffix(".tex")
        p.write_text(tex, encoding="utf-8")
        print(f"[OK] tex -> {p}")

    if args.best_per_city:
        best = _best_by_city(
            tab,
            metric=str(args.sort_by),
            ascending=str(args.ascending),
        )
        p = out.with_name(
            out.name + "__best_per_city"
        ).with_suffix(".csv")
        best.to_csv(p, index=False)
        print(f"[OK] best/city -> {p}")

    # -------------------------------------------------
    # Optional grouped tables (S6 / S7)
    # -------------------------------------------------
    groups = [g.lower() for g in _split_csv(args.group_cols)]
    if groups:
        if "all" in groups:
            groups = ["s6", "s7"]

        if "s6" in groups:
            _build_table_s6(
                df,
                args=args,
                out=out,
                fmts=fmts,
            )

        if "s7" in groups:
            _build_table_s7(
                df,
                args=args,
                out=out,
                fmts=fmts,
            )

    if args.stdout:
        print("\n" + tab.to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    build_ablation_table_main(argv)


if __name__ == "__main__":
    main()
