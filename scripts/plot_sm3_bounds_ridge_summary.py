# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>
#
r"""
SM3 • Bounds vs ridge summary (paper-style)

Summarize two failure modes in SM3 synthetic runs:
- "clipped to bounds" (inferred from observed extrema)
- "ridge non-identifiability" (ridge_resid_q50 > threshold)

Panels
------
(a) Bound hits (counts + %)
(b) Ridge residual distribution (with threshold)
(c) 2×2 matrix: clipped vs ridge (counts + %)
(d) Category fractions (overall or by lithology)

Outputs
-------
- Figure: scripts/figs/<out>.png and <out>.svg
- JSON summary: scripts/out/<out-json>
- CSV category table: scripts/out/<out-csv>

Notes
-----
Bounds are inferred from run extrema:
  K_min/max   = min/max(K_est_med_mps)
  tau_min/max = min/max(tau_est_med_sec)
  Hd_min/max  = min/max(Hd_est_med)

Clipped modes:
  - primary: K@max OR tau@min OR Hd@max
  - any:     any side of (K, tau, Hd)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

from . import config as cfg
from . import utils


# -------------------------------------------------------------------
# Core helpers
# -------------------------------------------------------------------
def _require_cols(
    df: pd.DataFrame,
    cols: List[str],
    *,
    ctx: str,
) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(
            f"Missing cols ({ctx}): {', '.join(miss)}"
        )


def _sci_tex(x: float, sig: int = 3) -> str:
    if x is None:
        return r"\mathrm{NA}"
    x = float(x)
    if not np.isfinite(x):
        return r"\mathrm{nan}"
    if x == 0.0:
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / (10.0 ** exp)
    mant_s = f"{mant:.{sig}g}"
    return rf"{mant_s}\times 10^{{{exp}}}"

def _paper_bound_ticks() -> List[str]:
    return [
        r"$K_{\max}$",
        r"$K_{\min}$",
        r"$\tau_{\min}$",
        r"$\tau_{\max}$",
        r"$H_{d,\max}$",
        r"$H_{d,\min}$",
    ]


def _isclose(
    a: np.ndarray,
    b: float,
    *,
    rtol: float,
) -> np.ndarray:
    a = np.asarray(a, float)
    return np.isclose(a, float(b), rtol=float(rtol), atol=0.0)


@dataclass(frozen=True)
class BoundInfo:
    K_min: float
    K_max: float
    tau_min: float
    tau_max: float
    Hd_min: float
    Hd_max: float


def infer_bounds(df: pd.DataFrame) -> BoundInfo:
    K = df["K_est_med_mps"].to_numpy(float)
    tau = df["tau_est_med_sec"].to_numpy(float)
    Hd = df["Hd_est_med"].to_numpy(float)

    return BoundInfo(
        K_min=float(np.nanmin(K)),
        K_max=float(np.nanmax(K)),
        tau_min=float(np.nanmin(tau)),
        tau_max=float(np.nanmax(tau)),
        Hd_min=float(np.nanmin(Hd)),
        Hd_max=float(np.nanmax(Hd)),
    )


def compute_flags(
    df: pd.DataFrame,
    bounds: BoundInfo,
    *,
    rtol: float,
    ridge_thr: float,
) -> Dict[str, np.ndarray]:
    K = df["K_est_med_mps"].to_numpy(float)
    tau = df["tau_est_med_sec"].to_numpy(float)
    Hd = df["Hd_est_med"].to_numpy(float)
    rr = df["ridge_resid_q50"].to_numpy(float)

    K_hi = _isclose(K, bounds.K_max, rtol=rtol)
    K_lo = _isclose(K, bounds.K_min, rtol=rtol)

    tau_lo = _isclose(tau, bounds.tau_min, rtol=rtol)
    tau_hi = _isclose(tau, bounds.tau_max, rtol=rtol)

    Hd_hi = _isclose(Hd, bounds.Hd_max, rtol=rtol)
    Hd_lo = _isclose(Hd, bounds.Hd_min, rtol=rtol)

    clipped_primary = K_hi | tau_lo | Hd_hi
    clipped_any = (
        K_hi | K_lo | tau_lo | tau_hi | Hd_hi | Hd_lo
    )

    ridge_strong = np.asarray(rr, float) > float(ridge_thr)

    return {
        "K_clip_hi": K_hi,
        "K_clip_lo": K_lo,
        "tau_clip_lo": tau_lo,
        "tau_clip_hi": tau_hi,
        "Hd_clip_hi": Hd_hi,
        "Hd_clip_lo": Hd_lo,
        "clipped_primary": clipped_primary,
        "clipped_any": clipped_any,
        "ridge_strong": ridge_strong,
        "ridge_resid_q50": np.asarray(rr, float),
    }


def summarize_counts(
    flags: Dict[str, np.ndarray],
    *,
    use: str,
) -> Dict[str, float]:
    if use not in ("primary", "any"):
        raise ValueError("use must be 'primary' or 'any'")

    clipped = (
        flags["clipped_primary"]
        if use == "primary"
        else flags["clipped_any"]
    )
    ridge = flags["ridge_strong"]
    n = int(clipped.size)

    both = int((clipped & ridge).sum())
    clip_only = int((clipped & ~ridge).sum())
    ridge_only = int((~clipped & ridge).sum())
    neither = int((~clipped & ~ridge).sum())

    def frac(k: int) -> float:
        return float(k) / float(n) if n else float("nan")

    return {
        "n": n,
        "clipped": int(clipped.sum()),
        "ridge_strong": int(ridge.sum()),
        "both": both,
        "clipped_only": clip_only,
        "ridge_only": ridge_only,
        "neither": neither,
        "clipped_frac": frac(int(clipped.sum())),
        "ridge_strong_frac": frac(int(ridge.sum())),
        "both_frac": frac(both),
        "clipped_only_frac": frac(clip_only),
        "ridge_only_frac": frac(ridge_only),
        "neither_frac": frac(neither),
    }


def build_category_table(
    df: pd.DataFrame,
    flags: Dict[str, np.ndarray],
) -> pd.DataFrame:
    lith = None
    if "lith_idx" in df.columns:
        lith = df["lith_idx"].to_numpy(int)

    lith_names = {
        0: "Fine",
        1: "Mixed",
        2: "Coarse",
        3: "Rock",
    }

    cats = [
        ("both", "Clipped+Ridge"),
        ("clipped_only", "Clipped only"),
        ("ridge_only", "Ridge only"),
        ("neither", "Neither"),
    ]

    rows: List[dict] = []

    for use in ("primary", "any"):
        clipped = (
            flags["clipped_primary"]
            if use == "primary"
            else flags["clipped_any"]
        )
        ridge = flags["ridge_strong"]

        masks = {
            "both": clipped & ridge,
            "clipped_only": clipped & ~ridge,
            "ridge_only": ~clipped & ridge,
            "neither": ~clipped & ~ridge,
        }

        # overall
        n_all = int(len(df))
        for key, lab in cats:
            c = int(masks[key].sum())
            f = float(c) / float(n_all) if n_all else float("nan")
            rows.append(
                {
                    "use": use,
                    "group": "overall",
                    "lith_idx": -1,
                    "lithology": "overall",
                    "category": lab,
                    "count": c,
                    "denom": n_all,
                    "frac": f,
                }
            )

        # by lithology
        if lith is None:
            continue

        for li in sorted(set(lith.tolist())):
            mm = lith == li
            denom = int(mm.sum())

            for key, lab in cats:
                c = int((masks[key] & mm).sum())
                f = float(c) / float(denom) if denom else 0.0
                rows.append(
                    {
                        "use": use,
                        "group": "lithology",
                        "lith_idx": int(li),
                        "lithology": lith_names.get(li, f"L{li}"),
                        "category": lab,
                        "count": c,
                        "denom": denom,
                        "frac": f,
                    }
                )

    return pd.DataFrame(rows)


def _beautify(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.6)


def _panel_label(
    ax: plt.Axes,
    lab: str,
    enabled: bool,
) -> None:
    if not enabled:
        return
    ax.text(
        -0.14,
        1.05,
        lab,
        transform=ax.transAxes,
        fontweight="bold",
        va="bottom",
    )


def plot_sm3_bounds_ridge_summary(
    df: pd.DataFrame,
    *,
    flags: Dict[str, np.ndarray],
    bounds: BoundInfo,
    ridge_thr: float,
    use: str,
    out: str,
    out_json: str,
    out_csv: str,
    dpi: int,
    font: int,
    show_legend: bool,
    show_labels: bool,
    show_ticklabels: bool,
    show_title: bool,
    show_panel_titles: bool,
    show_panel_labels: bool,
    paper_format: bool,
    title: Optional[str],
) -> None:
    utils.ensure_script_dirs()
    utils.set_paper_style(fontsize=int(font), dpi=int(dpi))

    clipped = (
        flags["clipped_primary"]
        if use == "primary"
        else flags["clipped_any"]
    )
    ridge = flags["ridge_strong"]
    rr = flags["ridge_resid_q50"]

    n = int(len(df))
    both = clipped & ridge
    clip_only = clipped & ~ridge
    ridge_only = ~clipped & ridge
    neither = ~clipped & ~ridge

    # -------------------------
    # Figure canvas
    # -------------------------
    fig = plt.figure(
        figsize=(7.2, 4.2),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(2, 2)

    # -------------------------
    # (a) Bound hits
    # -------------------------
    axA = fig.add_subplot(gs[0, 0])
    _beautify(axA)
    _panel_label(axA, "a", show_panel_labels)

    labels = ["K@max", "K@min", "τ@min", "τ@max", "Hd@max", "Hd@min"]
    if paper_format:
        labels = _paper_bound_ticks()
        rot, ha = 0, "center"
    else:
        labels = ["K@max","K@min","τ@min","τ@max","Hd@max","Hd@min"]
        rot, ha = 30, "right"

    counts = np.array(
        [
            int(flags["K_clip_hi"].sum()),
            int(flags["K_clip_lo"].sum()),
            int(flags["tau_clip_lo"].sum()),
            int(flags["tau_clip_hi"].sum()),
            int(flags["Hd_clip_hi"].sum()),
            int(flags["Hd_clip_lo"].sum()),
        ],
        dtype=int,
    )

    x = np.arange(len(labels))
    axA.bar(x, counts)
    axA.set_xticks(x)

    if show_ticklabels:
        # axA.set_xticklabels(labels, rotation=30, ha="right")
        axA.set_xticklabels(labels, rotation=rot, ha=ha)
    else:
        axA.set_xticklabels([])

    if show_labels:
        axA.set_ylabel("Count")

    if show_panel_titles:
        axA.set_title("Bound hits inferred from runs", pad=2)

    for i, c in enumerate(counts.tolist()):
        pct = (100.0 * float(c) / float(n)) if n else float("nan")
        axA.text(
            i,
            c + 0.5,
            f"{c}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
        )

    if show_labels:
        if paper_format:
            k_tex = _sci_tex(bounds.K_max, sig=3)
            t_tex = _sci_tex(bounds.tau_min, sig=3)
            h_tex = _sci_tex(bounds.Hd_max, sig=3)
            msg = (
                rf"$K_{{\max}}={k_tex}\,\mathrm{{m\,s^{{-1}}}}$"
                "\n"
                rf"$\tau_{{\min}}={t_tex}\,\mathrm{{s}}$"
                "\n"
                rf"$H_{{d,\max}}={h_tex}$"
            )
        else:
            msg = (
                f"K_max={bounds.K_max:.3e} m/s\n"
                f"τ_min={bounds.tau_min:.2f} s\n"
                f"Hd_max={bounds.Hd_max:g}"
            )
        axA.text(
            0.02, 0.02, msg,
            transform=axA.transAxes,
            va="bottom", ha="left",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.85,
                linewidth=0.0,
            ),
        )
        axA.set_ylim(0, float(counts.max()) * 1.25 + 1.0)

    # -------------------------
    # (b) Ridge distribution
    # -------------------------
    axB = fig.add_subplot(gs[0, 1])
    _beautify(axB)
    _panel_label(axB, "b", show_panel_labels)

    rr_f = np.asarray(rr, float)
    rr_f = rr_f[np.isfinite(rr_f)]

    axB.hist(rr_f, bins=18)
    axB.axvline(float(ridge_thr), linestyle="--", linewidth=0.9)

    if show_labels:
        # axB.set_xlabel("ridge_resid_q50 (decades)")
        if paper_format:
            axB.set_xlabel("Median ridge residual (decades)")
        else:
            axB.set_xlabel("ridge_resid_q50 (decades)")

        axB.set_ylabel("Count")

    if show_panel_titles:
        axB.set_title("Ridge non-identifiability", pad=2)

    if rr_f.size:
        frac_r = float((rr_f > float(ridge_thr)).sum()) / float(rr_f.size)
    else:
        frac_r = float("nan")

    if show_labels:
        axB.text(
            0.03,
            0.97,
            f"Strong ridge (> {ridge_thr:g}) = {100*frac_r:.1f}%",
            transform=axB.transAxes,
            va="top",
            ha="left",
             bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.85,
                linewidth=0.0,
            ),
         )
    
    # -------------------------
    # (c) 2×2 matrix
    # -------------------------
    axC = fig.add_subplot(gs[1, 0])
    _beautify(axC)
    _panel_label(axC, "c", show_panel_labels)

    mat = np.array(
        [
            [int((~clipped & ~ridge).sum()), int((~clipped & ridge).sum())],
            [int((clipped & ~ridge).sum()), int((clipped & ridge).sum())],
        ],
        dtype=int,
    )

    axC.imshow(mat, aspect="auto")
    axC.set_xticks([0, 1])
    axC.set_yticks([0, 1])

    if show_ticklabels:
        axC.set_xticklabels(["No ridge", "Strong ridge"])
        axC.set_yticklabels(["Not clipped", "Clipped"])
    else:
        axC.set_xticklabels([])
        axC.set_yticklabels([])

    if show_panel_titles:
        axC.set_title(f"Clipping vs ridge ({use})", pad=2)

    for (i, j), v in np.ndenumerate(mat):
        pct = (100.0 * float(v) / float(n)) if n else float("nan")
        axC.text(j, i, f"{v}\n({pct:.1f}%)", ha="center", va="center")

    # -------------------------
    # (d) Fractions overall/by lith
    # -------------------------
    axD = fig.add_subplot(gs[1, 1])
    _beautify(axD)
    _panel_label(axD, "d", show_panel_labels)

    cats = ["Clipped+Ridge", "Clipped only", "Ridge only", "Neither"]
    fracs = np.array(
        [
            float(both.sum()) / float(n) if n else float("nan"),
            float(clip_only.sum()) / float(n) if n else float("nan"),
            float(ridge_only.sum()) / float(n) if n else float("nan"),
            float(neither.sum()) / float(n) if n else float("nan"),
        ],
        dtype=float,
    )

    if "lith_idx" not in df.columns:
        axD.bar(np.arange(4), fracs)
        axD.set_xticks(np.arange(4))
        if show_ticklabels:
            if paper_format:
                axD.set_xticklabels(
                    ["Clipped\n+ ridge","Clipped\nonly",
                     "Ridge\nonly","Neither"]
                )
            else:
                axD.set_xticklabels(cats, rotation=25, ha="right")

        else:
            axD.set_xticklabels([])
        axD.set_ylim(0, 1)

        if show_labels:
            axD.set_ylabel("Fraction")

        if show_panel_titles:
            axD.set_title("Category fractions (overall)", pad=2)
    else:
        lith = df["lith_idx"].to_numpy(int)
        lith_names = {0: "Fine", 1: "Mixed", 2: "Coarse", 3: "Rock"}
        order = [0, 1, 2, 3]
        x = np.arange(len(order))
        bottoms = np.zeros_like(x, float)

        masks = [both, clip_only, ridge_only, neither]
        for lab, m in zip(cats, masks):
            vals: List[float] = []
            for li in order:
                mm = lith == li
                denom = float(mm.sum())
                if denom:
                    vals.append(float((m & mm).sum()) / denom)
                else:
                    vals.append(0.0)

            v = np.asarray(vals, float)
            axD.bar(x, v, bottom=bottoms, label=lab)
            bottoms = bottoms + v

        axD.set_xticks(x)
        if show_ticklabels:
            axD.set_xticklabels([lith_names[i] for i in order])
        else:
            axD.set_xticklabels([])
        axD.set_ylim(0, 1)

        if show_labels:
            axD.set_ylabel("Fraction within lithology")

        if show_panel_titles:
            axD.set_title("Failure modes by lithology", pad=2)

        if show_legend:
            axD.legend(
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
            )

    if show_title:
        ttl = utils.resolve_title(
            default="SM3 • Bounds vs ridge summary",
            title=title,
        )
        fig.suptitle(ttl, x=0.02, ha="left")

    utils.save_figure(fig, out, dpi = int(dpi))

    # -------------------------
    # Exports: JSON + CSV
    # -------------------------
    summ_primary = summarize_counts(flags, use="primary")
    summ_any = summarize_counts(flags, use="any")
    cat_df = build_category_table(df, flags)

    out_csv_p = utils.resolve_out_out(out_csv)
    out_json_p = utils.resolve_out_out(out_json)

    cat_df.to_csv(out_csv_p, index=False)

    payload = {
        "csv": str(Path(df.attrs.get("csv_path", "")).resolve())
        if df.attrs.get("csv_path")
        else "",
        "bounds_inferred": {
            "K_min": bounds.K_min,
            "K_max": bounds.K_max,
            "tau_min": bounds.tau_min,
            "tau_max": bounds.tau_max,
            "Hd_min": bounds.Hd_min,
            "Hd_max": bounds.Hd_max,
        },
        "ridge_thr": float(ridge_thr),
        "summary_primary": summ_primary,
        "summary_any": summ_any,
        "category_csv": str(out_csv_p),
    }

    out_json_p.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"[OK] wrote {out_csv_p}")
    print(f"[OK] wrote {out_json_p}")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def plot_sm3_bounds_ridge_summary_main(
    argv: Optional[List[str]] = None,
) -> None:
    ap = argparse.ArgumentParser(
        prog="plot-sm3-bounds-ridge-summary",
        description="SM3: bounds vs ridge summary plot.",
    )

    ap.add_argument(
        "--csv",
        type=str,
        default="results/sm3_synth_1d/sm3_synth_runs.csv",
        help="Input SM3 runs CSV.",
    )
    ap.add_argument(
        "--only-identify",
        type=str,
        default=None,
        help="Filter identify column (e.g., both/tau).",
    )
    ap.add_argument(
        "--nx-min",
        type=int,
        default=None,
        help="Filter rows with nx >= nx-min.",
    )

    ap.add_argument(
        "--use",
        type=str,
        default="any",
        choices=["any", "primary"],
        help="Which clipping definition to use in panel (c).",
    )
    ap.add_argument("--ridge-thr", type=float, default=2.0)
    ap.add_argument("--rtol", type=float, default=1e-6)

    ap.add_argument("--dpi", type=int, default=cfg.PAPER_DPI)
    ap.add_argument("--font", type=int, default=cfg.PAPER_FONT)

    ap.add_argument(
        "--paper-format",
        action="store_true",
        help="Paper-friendly labels (no @, no raw underscores).",
    )

    ap.add_argument(
        "--out-json",
        type=str,
        default="sm3-clip-vs-ridge-summary.json",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="sm3-clip-vs-ridge-categories.csv",
    )

    ap.add_argument(
        "--show-panel-labels",
        type=str,
        default="true",
        help="Show panel letters a–d (true/false).",
    )

    utils.add_plot_text_args(ap, default_out="sm3-clip-vs-ridge")

    args = ap.parse_args(argv)

    df = pd.read_csv(Path(args.csv).expanduser())
    df.attrs["csv_path"] = str(Path(args.csv).expanduser())

    need = [
        "K_est_med_mps",
        "tau_est_med_sec",
        "Hd_est_med",
        "ridge_resid_q50",
    ]
    _require_cols(df, need, ctx="summary")

    # strict filtering (optional)
    if args.only_identify is not None:
        if "identify" not in df.columns:
            raise KeyError(
                "--only-identify set but missing identify."
            )
        want = str(args.only_identify).strip().lower()
        got = df["identify"].astype(str).str.strip().str.lower()
        df = df.loc[got == want].copy()

    if args.nx_min is not None:
        if "nx" not in df.columns:
            raise KeyError("--nx-min set but missing nx.")
        nxv = df["nx"].astype(int)
        df = df.loc[nxv >= int(args.nx_min)].copy()

    if df.empty:
        raise ValueError("No rows left after filtering.")

    bounds = infer_bounds(df)
    flags = compute_flags(
        df,
        bounds,
        rtol=float(args.rtol),
        ridge_thr=float(args.ridge_thr),
    )

    show_legend = utils.str_to_bool(args.show_legend, default=True)
    show_labels = utils.str_to_bool(args.show_labels, default=True)
    show_ticks = utils.str_to_bool(args.show_ticklabels, default=True)
    show_title = utils.str_to_bool(args.show_title, default=True)
    show_pt = utils.str_to_bool(args.show_panel_titles, default=True)
    show_pl = utils.str_to_bool(
        args.show_panel_labels,
        default=True,
    )

    plot_sm3_bounds_ridge_summary(
        df,
        flags=flags,
        bounds=bounds,
        ridge_thr=float(args.ridge_thr),
        use=str(args.use),
        out=str(args.out),
        out_json=str(args.out_json),
        out_csv=str(args.out_csv),
        dpi=int(args.dpi),
        font=int(args.font),
        show_legend=show_legend,
        show_labels=show_labels,
        show_ticklabels=show_ticks,
        show_title=show_title,
        show_panel_titles=show_pt,
        show_panel_labels=show_pl,
        paper_format=bool(args.paper_format),
        title=args.title,
    )


def main(argv: Optional[List[str]] = None) -> None:
    plot_sm3_bounds_ridge_summary_main(argv)


if __name__ == "__main__":
    main()
