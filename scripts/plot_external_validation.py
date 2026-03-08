# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""
plot_external_validation

Supplementary figure:
Zhongshan point-support external validation of inferred
effective fields.

Panels
------
a) Zhongshan H_eff field with borehole / pumping-test
   locations (SW1--SW5) and matched pixels.
b) Borehole-derived compressible thickness vs model H_eff.
c) Late-step specific capacity vs model effective K.

Inputs
------
- site-level validation table (.csv or .xlsx)
- full_inputs.npz
- physics_payload_fullcity.npz
- optional coord scaler (.joblib)

Outputs
-------
- <out>.png
- <out>.svg
- optional summary JSON
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

from . import config as cfg
from . import utils


ArrayDict = Dict[str, np.ndarray]


def load_npz_dict(path: str) -> ArrayDict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def read_table(path: str) -> pd.DataFrame:
    p = Path(path).expanduser()
    suf = p.suffix.lower()

    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(p)

    return pd.read_csv(p)

def load_boundary_xy(
    path: Optional[str],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load an optional boundary polyline/polygon.

    Preferred behaviour:
    - if nat.com.utils exposes load_boundary_xy(), use it
    - otherwise fall back to CSV/XLSX with x/y-like columns

    Accepted fallback column names:
    x, x_m, coord_x
    y, y_m, coord_y

    NaN rows are preserved so multi-part boundaries can be drawn
    with line breaks.
    """
    if not path:
        return None

    if hasattr(utils, "load_boundary_xy"):
        try:
            out = utils.load_boundary_xy(path)
            if isinstance(out, tuple) and len(out) == 2:
                bx = np.asarray(out[0], dtype=float)
                by = np.asarray(out[1], dtype=float)
                if bx.ndim == 1 and by.ndim == 1 and bx.size == by.size:
                    return bx, by
        except Exception:
            pass

    df = read_table(path)

    x_col = next(
        (
            c
            for c in ("x", "x_m", "coord_x")
            if c in df.columns
        ),
        None,
    )
    y_col = next(
        (
            c
            for c in ("y", "y_m", "coord_y")
            if c in df.columns
        ),
        None,
    )

    if x_col is None or y_col is None:
        raise KeyError(
            "Boundary file must contain one of "
            "{'x','x_m','coord_x'} and one of "
            "{'y','y_m','coord_y'}."
        )

    bx = pd.to_numeric(
        df[x_col], errors="coerce"
    ).to_numpy(dtype=float)
    by = pd.to_numeric(
        df[y_col], errors="coerce"
    ).to_numpy(dtype=float)

    ok = np.isfinite(bx) & np.isfinite(by)
    if not np.any(ok):
        raise ValueError(
            "Boundary file does not contain any finite x/y pairs."
        )

    return bx, by


def apply_no_offset(
    ax,
    *,
    x: bool = True,
    y: bool = True,
) -> None:
    """
    Disable Matplotlib offset/scientific-offset text on linear axes.
    """
    if x and ax.get_xscale() == "linear":
        fmt_x = ScalarFormatter(useOffset=False)
        fmt_x.set_scientific(False)
        ax.xaxis.set_major_formatter(fmt_x)

    if y and ax.get_yscale() == "linear":
        fmt_y = ScalarFormatter(useOffset=False)
        fmt_y.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt_y)
        
def inverse_txy(
    coords_bh3: np.ndarray,
    coord_scaler_path: Optional[str],
) -> np.ndarray:
    first = np.asarray(coords_bh3[:, 0, :], dtype=float)

    if not coord_scaler_path:
        return first

    scaler = joblib.load(coord_scaler_path)
    return scaler.inverse_transform(first)


def reduce_horizon(
    arr: np.ndarray,
    n_seq: int,
    horizon: int,
    how: str,
) -> np.ndarray:
    x = np.asarray(arr, dtype=float)

    if x.size == n_seq:
        x = x.reshape(n_seq, 1)
    elif x.size == n_seq * horizon:
        x = x.reshape(n_seq, horizon)
    else:
        raise ValueError(
            "Cannot align array with sequences. "
            f"size={x.size}, n_seq={n_seq}, horizon={horizon}"
        )

    how = str(how).strip().lower()
    if how == "first":
        return x[:, 0]
    if how == "mean":
        return np.mean(x, axis=1)
    if how == "median":
        return np.median(x, axis=1)

    raise ValueError(
        "horizon reducer must be one of "
        "{'first','mean','median'}"
    )


def build_pixel_table(
    inputs_npz: str,
    payload_npz: str,
    coord_scaler_path: Optional[str],
    horizon_reducer: str,
    site_reducer: str,
) -> pd.DataFrame:
    x_np = load_npz_dict(inputs_npz)
    payload = load_npz_dict(payload_npz)

    coords = np.asarray(x_np["coords"], dtype=float)
    n_seq, horizon, _ = coords.shape

    txy = inverse_txy(coords, coord_scaler_path)
    x = txy[:, 1]
    y = txy[:, 2]

    h_eff = reduce_horizon(
        x_np["H_field"],
        n_seq=n_seq,
        horizon=horizon,
        how=horizon_reducer,
    )
    k_vals = reduce_horizon(
        payload["K"],
        n_seq=n_seq,
        horizon=horizon,
        how=horizon_reducer,
    )
    hd_vals = reduce_horizon(
        payload["Hd"],
        n_seq=n_seq,
        horizon=horizon,
        how=horizon_reducer,
    )

    df = pd.DataFrame(
        {
            "seq_idx": np.arange(n_seq, dtype=int),
            "x": x,
            "y": y,
            "H_eff_input_m": h_eff,
            "K_mps": k_vals,
            "Hd_m": hd_vals,
        }
    )

    agg_fun = (
        "mean"
        if str(site_reducer).lower() == "mean"
        else "median"
    )

    pix = (
        df.groupby(["x", "y"], as_index=False)[
            ["H_eff_input_m", "K_mps", "Hd_m"]
        ]
        .agg(agg_fun)
        .reset_index(drop=True)
    )
    pix["pixel_idx"] = np.arange(len(pix), dtype=int)
    return pix


def grid_mean(
    x: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    *,
    res: int,
    extent: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    xmin, xmax, ymin, ymax = extent
    xb = np.linspace(xmin, xmax, int(res) + 1)
    yb = np.linspace(ymin, ymax, int(res) + 1)

    sumv, _, _ = np.histogram2d(
        x,
        y,
        bins=[xb, yb],
        weights=v,
    )
    cnts, _, _ = np.histogram2d(
        x,
        y,
        bins=[xb, yb],
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        z = sumv / cnts

    z[cnts == 0] = np.nan
    return (z.T, extent)


def field_extent(
    pixels: pd.DataFrame,
    pad_frac: float = 0.02,
) -> Tuple[float, float, float, float]:
    x = pixels["x"].to_numpy(dtype=float)
    y = pixels["y"].to_numpy(dtype=float)

    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    ymin = float(np.nanmin(y))
    ymax = float(np.nanmax(y))

    dx = max((xmax - xmin) * float(pad_frac), 1.0)
    dy = max((ymax - ymin) * float(pad_frac), 1.0)

    return (
        xmin - dx,
        xmax + dx,
        ymin - dy,
        ymax + dy,
    )

def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size != y.size or x.size < 2:
        return float("nan")

    # Only return NaN if one vector is truly constant
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")

    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()

    xs = xr - xr.mean()
    ys = yr - yr.mean()
    den = math.sqrt(float(np.sum(xs * xs) * np.sum(ys * ys)))

    if den <= 0.0:
        return float("nan")

    return float(np.sum(xs * ys) / den)


def mae(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return float(np.mean(np.abs(x[mask] - y[mask])))


def median_bias(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    return float(np.median(pred[mask] - obs[mask]))


def annotate_sites(
    ax,
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    label_col: str = "well_id",
    fontsize: int = 8,
) -> None:
    for _, row in df.iterrows():
        ax.text(
            float(row[x_col]),
            float(row[y_col]),
            str(row[label_col]),
            ha="left",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
        )


def plot_external_validation(
    *,
    site_csv: str,
    full_inputs_npz: str,
    full_payload_npz: str,
    coord_scaler: Optional[str],
    out: str,
    out_json: Optional[str],
    horizon_reducer: str,
    site_reducer: str,
    grid_res: int,
    dpi: int,
    font: int,
    city: str,
    show_legend: bool,
    show_labels: bool,
    show_ticklabels: bool,
    show_title: bool,
    show_panel_titles: bool,
    title: Optional[str],
    boundary: Optional[str],
    paper_format: bool,
    paper_no_offset: bool,
) -> None:
    utils.ensure_script_dirs()
    utils.set_paper_style(
        fontsize=int(font),
        dpi=int(dpi),
    )

    site_df = read_table(site_csv).copy()
    pix = build_pixel_table(
        full_inputs_npz,
        full_payload_npz,
        coord_scaler,
        horizon_reducer,
        site_reducer,
    )

    boundary_xy = load_boundary_xy(boundary)

    if paper_format and title is None:
        show_title = False

    map_site_size = 22 if paper_format else 28
    map_match_size = 38 if paper_format else 45
    map_linewidth = 0.55 if paper_format else 0.6
    map_edgewidth = 0.7 if paper_format else 0.8
    label_font = max(7, int(font) - 1)

    fig_w = 8.8 if paper_format else 9.2
    fig_h = 3.0 if paper_format else 3.3
    
    needed = {
        "well_id",
        "x",
        "y",
        "matched_pixel_x",
        "matched_pixel_y",
        "model_H_eff_m",
        "model_K_mps",
        "approx_compressible_thickness_m",
        "step3_specific_capacity_Lps_per_m",
        "match_distance_m",
    }
    miss = [c for c in needed if c not in site_df.columns]
    if miss:
        raise KeyError(
            "Site table missing columns: "
            f"{miss}"
        )

    ext = field_extent(pix)
    z_h, ext2 = grid_mean(
        pix["x"].to_numpy(float),
        pix["y"].to_numpy(float),
        pix["H_eff_input_m"].to_numpy(float),
        res=int(grid_res),
        extent=ext,
    )

    obs_h = site_df[
        "approx_compressible_thickness_m"
    ].to_numpy(dtype=float)
    mod_h = site_df["model_H_eff_m"].to_numpy(dtype=float)
    prod = site_df[
        "step3_specific_capacity_Lps_per_m"
    ].to_numpy(dtype=float)
    mod_k = site_df["model_K_mps"].to_numpy(dtype=float)

    rho_h = spearman_rho(obs_h, mod_h)
    mae_h = mae(obs_h, mod_h)
    bias_h = median_bias(obs_h, mod_h)
    rho_k = spearman_rho(prod, mod_k)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(fig_w, fig_h),
        constrained_layout=False,
    )
    plt.subplots_adjust(
        left=0.06,
        right=0.93 if show_legend else 0.98,
        top=0.84 if paper_format else 0.86,
        bottom=0.16,
        wspace=0.28 if paper_format else 0.33,
    )

    city_name = utils.canonical_city(city)
    city_color = cfg.CITY_COLORS.get(city_name, "#333333")

    # -------------------------------------------------
    # a) map
    # -------------------------------------------------
    ax = axes[0]
    im = ax.imshow(
        z_h,
        extent=ext2,
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        aspect="equal",
    )

    if boundary_xy is not None:
        bx, by = boundary_xy
        ax.plot(
            bx,
            by,
            color="black",
            linewidth=0.7 if paper_format else 0.8,
            alpha=0.9,
            zorder=2,
        )
        
    ax.scatter(
        site_df["x"].to_numpy(float),
        site_df["y"].to_numpy(float),
        marker="o",
        s=map_site_size,
        facecolors="white",
        edgecolors="black",
        linewidths=map_edgewidth,
        zorder=3,
    )

    ax.scatter(
        site_df["matched_pixel_x"].to_numpy(float),
        site_df["matched_pixel_y"].to_numpy(float),
        marker="+",
        s=map_match_size,
        color="black",
        linewidths=0.9,
        zorder=3,
    )
    
    
    for _, row in site_df.iterrows():
        ax.plot(
            [
                float(row["x"]),
                float(row["matched_pixel_x"]),
            ],
            [
                float(row["y"]),
                float(row["matched_pixel_y"]),
            ],
            color="#666666",
            linewidth=map_linewidth,
            alpha=0.8,
            zorder=2,
        )
        
    annotate_sites(
        ax,
        site_df,
        x_col="x",
        y_col="y",
        fontsize=label_font,
    )

    if show_panel_titles:
        ax.set_title(
            r"(a) $H_{\mathrm{eff}}$ field and sites",
            loc="left",
            fontweight="bold",
            pad=2.0 if paper_format else None,
        )
        
    if show_labels:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    if not show_ticklabels:
        ax.set_xticks([])
        ax.set_yticks([])

    if paper_no_offset:
        apply_no_offset(ax, x=True, y=True)
        
    if show_legend:
        cb = fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
        if show_labels:
            cb.set_label(
                r"$H_{\mathrm{eff}}$ (m)"
            )

    # -------------------------------------------------
    # b) borehole thickness vs model H_eff
    # -------------------------------------------------
    ax = axes[1]

    ax.scatter(
        obs_h,
        mod_h,
        s=36,
        marker="o",
        color=city_color,
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )

    for _, row in site_df.iterrows():
        ax.text(
            float(row["approx_compressible_thickness_m"]),
            float(row["model_H_eff_m"]),
            str(row["well_id"]),
            ha="left",
            va="bottom",
            fontsize=label_font,
            # fontsize=max(7, int(font) - 1),
        )

    vmin = float(
        min(np.nanmin(obs_h), np.nanmin(mod_h), 0.0)
    )
    vmax = float(
        max(np.nanmax(obs_h), np.nanmax(mod_h))
    )
    pad = 0.05 * max(vmax - vmin, 1.0)
    lo = vmin - pad
    hi = vmax + pad

    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        color="#444444",
        linewidth=0.8,
        zorder=1,
    )
    ax.axhline(
        30.0,
        linestyle=":",
        color="#444444",
        linewidth=0.9,
        zorder=1,
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    txt = (
        rf"$\rho={rho_h:.2f}$"
        "\n"
        rf"MAE$={mae_h:.2f}$ m"
        "\n"
        rf"bias$={bias_h:.2f}$ m"
    )
    ax.text(
        0.97,
        0.03,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            alpha=0.9,
            linewidth=0.0,
        ),
    )

    if show_panel_titles:
        ax.set_title(
            r"(b) Borehole thickness vs model "
            r"$H_{\mathrm{eff}}$",
            loc="left",
            fontweight="bold",
            pad=2.0 if paper_format else None,
        )

    if show_labels:
        ax.set_xlabel(
            "Borehole thickness (m)"
        )
        ax.set_ylabel(
            r"Model $H_{\mathrm{eff}}$ (m)"
        )

    if paper_no_offset:
        apply_no_offset(ax, x=True, y=True)
        
    # -------------------------------------------------
    # c) late-step specific capacity vs model K
    # -------------------------------------------------
    ax = axes[2]

    ax.scatter(
        prod,
        mod_k,
        s=36,
        marker="o",
        color=city_color,
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )

    for _, row in site_df.iterrows():
        ax.text(
            float(row["step3_specific_capacity_Lps_per_m"]),
            float(row["model_K_mps"]),
            str(row["well_id"]),
            ha="left",
            va="bottom",
            # fontsize=max(7, int(font) - 1),
            fontsize=label_font,
        )

    ax.set_yscale("log")

    txt = rf"$\rho={rho_k:.2f}$"
    ax.text(
        0.97,
        0.03,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            alpha=0.9,
            linewidth=0.0,
        ),
    )

    if show_panel_titles:
        ax.set_title(
            r"(c) Specific capacity vs model $K$",
            loc="left",
            fontweight="bold",
            pad=2.0 if paper_format else None,
        )

    if show_labels:
        ax.set_xlabel(
            r"Late-step specific capacity "
            r"(L s$^{-1}$ m$^{-1}$)"
        )
        ax.set_ylabel(
            r"Model $K$ (m s$^{-1}$)"
        )

    if paper_no_offset:
        apply_no_offset(ax, x=True, y=False)
        
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        
    if show_title:
        ttl = utils.resolve_title(
            default=(
                "Zhongshan point-support external "
                "validation of inferred effective fields"
            ),
            title=title,
        )
        fig.suptitle(ttl, x=0.02, ha="left")

    utils.save_figure(
        fig,
        out,
        dpi=int(dpi),
    )

    if out_json:
        p = utils.resolve_out_out(out_json)
        if p.suffix.lower() != ".json":
            p = p.with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)

        rec = {
            "city": city_name,
            "n_sites": int(len(site_df)),
            "borehole_vs_H_eff": {
                "spearman_rho": rho_h,
                "mae_m": mae_h,
                "median_bias_m": bias_h,
            },
            "pumping_vs_K": {
                "spearman_rho": rho_k,
            },
            "match_distance_m": {
                "min": float(
                    site_df["match_distance_m"].min()
                ),
                "median": float(
                    site_df["match_distance_m"].median()
                ),
                "max": float(
                    site_df["match_distance_m"].max()
                ),
            },
            "plot_options": {
                "boundary": (
                    str(Path(boundary).resolve())
                    if boundary
                    else None
                ),
                "paper_format": bool(paper_format),
                "paper_no_offset": bool(paper_no_offset),
            },
            "files": {
                "site_csv": str(Path(site_csv).resolve()),
                "full_inputs_npz": str(
                    Path(full_inputs_npz).resolve()
                ),
                "full_payload_npz": str(
                    Path(full_payload_npz).resolve()
                ),
                "coord_scaler": (
                    str(Path(coord_scaler).resolve())
                    if coord_scaler
                    else None
                ),
            },
        }
        p.write_text(
            json.dumps(rec, indent=2),
            encoding="utf-8",
        )
        print(f"[OK] wrote {p}")


def _add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--site-csv",
        type=str,
        required=True,
        help="Site-level validation CSV/XLSX.",
    )
    ap.add_argument(
        "--full-inputs-npz",
        type=str,
        required=True,
    )
    ap.add_argument(
        "--full-payload-npz",
        type=str,
        required=True,
    )
    ap.add_argument(
        "--coord-scaler",
        type=str,
        default=None,
    )
    ap.add_argument(
        "--city",
        type=str,
        default="Zhongshan",
    )
    ap.add_argument(
        "--horizon-reducer",
        type=str,
        default="mean",
        choices=["first", "mean", "median"],
    )
    ap.add_argument(
        "--boundary",
        type=str,
        default=None,
        help=(
            "Optional boundary file for panel (a). "
            "Uses utils.load_boundary_xy() if available; "
            "otherwise expects CSV/XLSX with x/y-like columns."
        ),
    )
    ap.add_argument(
        "--paper-format",
        action="store_true",
        help="Use a tighter paper-friendly layout.",
    )
    ap.add_argument(
        "--paper-no-offset",
        action="store_true",
        help=(
            "Disable Matplotlib axis offset / scientific-offset text "
            "on linear axes."
        ),
    )
    ap.add_argument(
        "--site-reducer",
        type=str,
        default="median",
        choices=["mean", "median"],
    )
    ap.add_argument(
        "--grid-res",
        type=int,
        default=260,
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=cfg.PAPER_DPI,
    )
    ap.add_argument(
        "--font",
        type=int,
        default=cfg.PAPER_FONT,
    )

    utils.add_plot_text_args(
        ap,
        default_out="supp_external_validation",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=None,
    )


def plot_external_validation_main(
    argv: Optional[List[str]] = None,
) -> None:
    ap = argparse.ArgumentParser(
        prog="plot-external-validation",
        description=(
            "Supplementary figure for Zhongshan "
            "point-support external validation."
        ),
    )
    _add_args(ap)
    args = ap.parse_args(argv)

    show_legend = utils.str_to_bool(
        args.show_legend,
        default=True,
    )
    show_labels = utils.str_to_bool(
        args.show_labels,
        default=True,
    )
    show_ticklabels = utils.str_to_bool(
        args.show_ticklabels,
        default=True,
    )
    show_title = utils.str_to_bool(
        args.show_title,
        default=True,
    )
    show_panel_titles = utils.str_to_bool(
        args.show_panel_titles,
        default=True,
    )

    plot_external_validation(
        site_csv=str(args.site_csv),
        full_inputs_npz=str(args.full_inputs_npz),
        full_payload_npz=str(args.full_payload_npz),
        coord_scaler=args.coord_scaler,
        out=str(args.out),
        out_json=args.out_json,
        horizon_reducer=str(args.horizon_reducer),
        site_reducer=str(args.site_reducer),
        grid_res=int(args.grid_res),
        dpi=int(args.dpi),
        font=int(args.font),
        city=str(args.city),
        show_legend=show_legend,
        show_labels=show_labels,
        show_ticklabels=show_ticklabels,
        show_title=show_title,
        show_panel_titles=show_panel_titles,
        title=args.title,
        boundary=args.boundary,
        paper_format=bool(args.paper_format),
        paper_no_offset=bool(args.paper_no_offset),
    )


def main(argv: Optional[List[str]] = None) -> None:
    plot_external_validation_main(argv)


if __name__ == "__main__":
    main()
    
# Daniel@daniel03 MINGW64 /d/projects/geoprior-v3 (develop)
# $ python -m scripts plot-external-validation   --site-csv D:/projects/geoprior-v3/nat.com/boreholes_zhongshan_with_model.csv   --full-inputs-npz E:/nature/results/zhongshan_GeoPriorSubsNet_stage1/external_validation_fullcity/full_inputs.npz   --full-payload-npz E:/nature/results/zhongshan_GeoPriorSubsNet_stage1/external_validation_fullcity/physics_payload_fullcity.npz   --coord-scaler E:/nature/results/zhongshan_GeoPriorSubsNet_stage1/artifacts/zhongshan_coord_scaler.joblib   --city Zhongshan    --out-json supp_zhongshan_external_validation.json   --paper-format --paper-no-offset --out "scripts/out/FigS06"
