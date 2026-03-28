# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""Build external validation metrics from Stage-1 inputs and a physics payload.

This command computes borehole / pumping validation metrics by matching
site coordinates from an external validation table to the nearest pixel
in a Stage-1 input grid. It then joins model-derived fields from a saved
physics payload and writes both a site-level table and a metrics JSON.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import (
    add_city_arg,
    add_config_args,
    add_manifest_arg,
    add_model_arg,
    add_outdir_arg,
    add_results_dir_arg,
    add_split_arg,
    add_stage1_dir_arg,
    add_stage2_manifest_arg,
    add_validation_csv_arg,
    bootstrap_runtime_config,
    ensure_outdir,
    find_latest_dir,
)

ArrayDict = dict[str, np.ndarray]


@dataclass
class MatchResult:
    """Nearest matched model pixel for one validation site."""

    well_id: str
    match_mode: str
    site_x: float
    site_y: float
    pixel_x: float
    pixel_y: float
    distance_m: float
    pixel_idx: int


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file into a dictionary."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_npz_dict(path: str | Path) -> ArrayDict:
    """Load all arrays from a compressed NPZ file."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def pick_existing(*paths: str | Path | None) -> Path | None:
    """Return the first existing path from a candidate list."""
    for path in paths:
        if path is None:
            continue
        p = Path(path).expanduser().resolve()
        if p.exists():
            return p
    return None


def _norm_stem(text: str) -> str:
    return (
        str(text).strip().replace("-", "_").replace(" ", "_")
    )


def infer_stage1_dir(
    *,
    stage1_dir: str | None,
    manifest_path: str | None,
) -> Path | None:
    """Infer Stage-1 directory from explicit hints."""
    if stage1_dir:
        p = Path(stage1_dir).expanduser().resolve()
        return p if p.exists() else None
    if manifest_path:
        p = Path(manifest_path).expanduser().resolve()
        if p.exists():
            return p.parent
    return None


def infer_stage1_dir_from_layout(
    *,
    results_dir: str | None,
    city: str | None,
    model: str | None,
) -> Path | None:
    """Infer Stage-1 directory from a standard results layout."""
    if not results_dir:
        return None

    root = Path(results_dir).expanduser().resolve()
    if not root.exists():
        return None

    if city and model:
        cand = root / f"{city}_{model}_stage1"
        if cand.exists():
            return cand

    matches = sorted(root.glob("*_stage1"))
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_stage1_manifest(
    *,
    manifest: str | None,
    stage1_dir: str | None,
    results_dir: str | None,
    city: str | None,
    model: str | None,
) -> Path:
    """Resolve Stage-1 manifest path from CLI and layout hints."""
    direct = pick_existing(manifest)
    if direct is not None:
        return direct

    s1 = infer_stage1_dir(
        stage1_dir=stage1_dir, manifest_path=manifest
    )
    if s1 is None:
        s1 = infer_stage1_dir_from_layout(
            results_dir=results_dir,
            city=city,
            model=model,
        )
    if s1 is not None:
        cand = s1 / "manifest.json"
        if cand.exists():
            return cand

    raise FileNotFoundError(
        "Could not resolve Stage-1 manifest. Pass --stage1-manifest "
        "or --stage1-dir explicitly."
    )


def resolve_inputs_npz(
    stage1_manifest: dict[str, Any],
    split: str,
    override: str | None,
) -> Path:
    """Resolve the inputs NPZ used for validation matching."""
    direct = pick_existing(override)
    if direct is not None:
        return direct

    npz_art = (stage1_manifest.get("artifacts") or {}).get(
        "numpy", {}
    )
    path = npz_art.get(f"{split}_inputs_npz")
    if isinstance(path, str):
        existing = pick_existing(path)
        if existing is not None:
            return existing

    raise FileNotFoundError(
        f"Could not resolve {split}_inputs_npz. Pass --inputs-npz explicitly."
    )


def resolve_coord_scaler_path(
    stage1_manifest: dict[str, Any],
    override: str | None,
) -> Path | None:
    """Resolve optional coordinate scaler path."""
    direct = pick_existing(override)
    if direct is not None:
        return direct

    enc = (stage1_manifest.get("artifacts") or {}).get(
        "encoders", {}
    )
    path = enc.get("coord_scaler")
    if isinstance(path, str):
        return pick_existing(path)
    return None


def resolve_stage2_manifest(
    *,
    override: str | None,
    stage1_dir: Path | None,
) -> Path | None:
    """Resolve optional Stage-2 manifest."""
    direct = pick_existing(override)
    if direct is not None:
        return direct

    if stage1_dir is None or not stage1_dir.exists():
        return None

    latest = find_latest_dir(
        stage1_dir,
        pattern="train_*",
        must_contain="manifest.json",
    )
    if latest is None:
        return None
    cand = latest / "manifest.json"
    return cand if cand.exists() else None


def resolve_payload_path(
    override: str | None,
    stage2_manifest: dict[str, Any] | None,
) -> Path:
    """Resolve saved physics payload path."""
    direct = pick_existing(override)
    if direct is not None:
        return direct

    if not stage2_manifest:
        raise FileNotFoundError(
            "No payload path was found. Pass --physics-payload explicitly."
        )

    paths = stage2_manifest.get("paths", {}) or {}
    run_dir = paths.get("run_dir")
    if isinstance(run_dir, str):
        run_path = Path(run_dir).expanduser().resolve()
        if run_path.is_dir():
            cands = sorted(
                p
                for p in run_path.iterdir()
                if p.is_file()
                and p.suffix.lower() == ".npz"
                and "phys_payload" in p.name.lower()
            )
            if cands:
                return cands[0]

    raise FileNotFoundError(
        "Could not resolve a physics payload. Pass --physics-payload explicitly."
    )


def resolve_validation_csv(
    args: argparse.Namespace,
    cfg: dict[str, Any],
) -> Path:
    """Resolve validation CSV path from CLI or config."""
    direct = pick_existing(args.validation_csv)
    if direct is not None:
        return direct

    for key in (
        "EXTERNAL_VALIDATION_CSV",
        "VALIDATION_CSV",
        "BOREHOLE_PUMPING_VALIDATION_CSV",
    ):
        value = cfg.get(key)
        if isinstance(value, str):
            existing = pick_existing(value)
            if existing is not None:
                return existing

    raise FileNotFoundError(
        "Could not resolve validation CSV. Pass --validation-csv explicitly."
    )


def resolve_outdir(
    args: argparse.Namespace,
    *,
    stage1_dir: Path,
    stage2_manifest_path: Path | None,
    split: str,
) -> Path:
    """Resolve output directory for metrics artifacts."""
    if args.outdir:
        return ensure_outdir(args.outdir)

    base = None
    if (
        stage2_manifest_path is not None
        and stage2_manifest_path.exists()
    ):
        stage2 = read_json(stage2_manifest_path)
        run_dir = (stage2.get("paths") or {}).get("run_dir")
        if isinstance(run_dir, str):
            cand = Path(run_dir).expanduser().resolve()
            if cand.exists():
                base = cand

    if base is None:
        base = stage1_dir

    return ensure_outdir(
        base / f"external_validation_{_norm_stem(split)}"
    )


def inverse_txy(
    coords_bh3: np.ndarray,
    coord_scaler,
) -> np.ndarray:
    """Inverse-transform BH3 coordinates to physical space."""
    first = np.asarray(coords_bh3[:, 0, :], dtype=float)
    if coord_scaler is None:
        return first
    return coord_scaler.inverse_transform(first)


def reduce_horizon(
    arr: np.ndarray,
    n_seq: int,
    horizon: int,
    how: str,
) -> np.ndarray:
    """Reduce a sequence or horizon-aligned array to one value per site."""
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
        "horizon reducer must be one of {'first','mean','median'}"
    )


def build_pixel_table(
    inputs_npz: str | Path,
    payload_npz: str | Path,
    coord_scaler,
    horizon_reducer: str,
    site_reducer: str,
) -> pd.DataFrame:
    """Aggregate model pixels to one row per unique site location."""
    x_np = load_npz_dict(inputs_npz)
    payload = load_npz_dict(payload_npz)

    coords = np.asarray(x_np["coords"], dtype=float)
    n_seq, horizon, _ = coords.shape

    txy = inverse_txy(coords, coord_scaler)
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

    h_payload = None
    if "H" in payload:
        h_payload = reduce_horizon(
            payload["H"],
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
    if h_payload is not None:
        df["H_payload_m"] = h_payload

    group_cols = ["x", "y"]
    num_cols = ["H_eff_input_m", "K_mps", "Hd_m"]
    if h_payload is not None:
        num_cols.append("H_payload_m")

    site_reducer = str(site_reducer).strip().lower()
    if site_reducer not in {"mean", "median"}:
        raise ValueError(
            "site reducer must be one of {'mean','median'}"
        )

    agg_fun = np.mean if site_reducer == "mean" else np.median
    pix = (
        df.groupby(group_cols, as_index=False)[num_cols]
        .agg(agg_fun)
        .reset_index(drop=True)
    )
    pix["pixel_idx"] = np.arange(len(pix), dtype=int)
    return pix


def nearest_match(
    pixels: pd.DataFrame,
    sx: float,
    sy: float,
    well_id: str,
) -> MatchResult:
    """Return the nearest pixel using direct and swapped XY checks."""
    px = pixels["x"].to_numpy(dtype=float)
    py = pixels["y"].to_numpy(dtype=float)

    d_dir = np.sqrt((px - sx) ** 2 + (py - sy) ** 2)
    d_swp = np.sqrt((px - sy) ** 2 + (py - sx) ** 2)

    i_dir = int(np.argmin(d_dir))
    i_swp = int(np.argmin(d_swp))

    if float(d_swp[i_swp]) < float(d_dir[i_dir]):
        idx = i_swp
        mode = "swapped_xy"
        dist = float(d_swp[idx])
    else:
        idx = i_dir
        mode = "direct_xy"
        dist = float(d_dir[idx])

    row = pixels.iloc[idx]
    return MatchResult(
        well_id=str(well_id),
        match_mode=mode,
        site_x=float(sx),
        site_y=float(sy),
        pixel_x=float(row["x"]),
        pixel_y=float(row["y"]),
        distance_m=dist,
        pixel_idx=int(row["pixel_idx"]),
    )


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation with finite-value checks."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size != y.size or x.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
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
    """Compute mean absolute error."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(x[mask] - y[mask])))


def median_bias(obs: np.ndarray, pred: np.ndarray) -> float:
    """Compute median prediction bias."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    if not np.any(mask):
        return float("nan")
    return float(np.median(pred[mask] - obs[mask]))


def validate_site_matches(
    site_df: pd.DataFrame,
    *,
    max_distance_m: float = 5000.0,
    min_unique_pixels: int = 3,
) -> None:
    """Sanity-check site-to-pixel matches."""
    n_unique = int(site_df["pixel_idx"].nunique())
    max_dist = float(site_df["match_distance_m"].max())

    if n_unique < min_unique_pixels:
        raise RuntimeError(
            "Too few unique matched pixels for site validation "
            f"(unique={n_unique}). This usually means CRS/order mismatch."
        )

    if max_dist > max_distance_m:
        raise RuntimeError(
            "Site-to-pixel matching distance is too large "
            f"(max={max_dist:.1f} m > {max_distance_m:.1f} m). "
            "Coordinates are likely not in the same CRS/grid as Stage-1."
        )


def compute_metrics(
    *,
    validation_csv: str | Path,
    stage1_manifest_path: str | Path,
    outdir: str | Path,
    split: str = "test",
    inputs_npz: str | Path | None = None,
    physics_payload: str | Path | None = None,
    coord_scaler: str | Path | None = None,
    stage2_manifest_path: str | Path | None = None,
    productivity_col: str = "step3_specific_capacity_Lps_per_m",
    horizon_reducer: str = "mean",
    site_reducer: str = "median",
    max_distance_m: float = 5000.0,
    min_unique_pixels: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute site-level validation joins and headline metrics."""
    outdir = ensure_outdir(outdir)

    stage1 = read_json(stage1_manifest_path)
    stage2 = (
        read_json(stage2_manifest_path)
        if stage2_manifest_path is not None
        else None
    )

    inputs_npz_path = resolve_inputs_npz(
        stage1,
        split=split,
        override=str(inputs_npz)
        if inputs_npz is not None
        else None,
    )
    payload_npz_path = resolve_payload_path(
        str(physics_payload)
        if physics_payload is not None
        else None,
        stage2_manifest=stage2,
    )
    coord_scaler_path = resolve_coord_scaler_path(
        stage1,
        override=str(coord_scaler)
        if coord_scaler is not None
        else None,
    )
    coord_scaler_obj = (
        joblib.load(coord_scaler_path)
        if coord_scaler_path is not None
        else None
    )

    pixels = build_pixel_table(
        inputs_npz=inputs_npz_path,
        payload_npz=payload_npz_path,
        coord_scaler=coord_scaler_obj,
        horizon_reducer=horizon_reducer,
        site_reducer=site_reducer,
    )

    sites = pd.read_csv(validation_csv)
    rows = []
    for _, rec in sites.iterrows():
        m = nearest_match(
            pixels=pixels,
            sx=float(rec["x"]),
            sy=float(rec["y"]),
            well_id=str(rec["well_id"]),
        )
        pix = pixels.loc[
            pixels["pixel_idx"] == m.pixel_idx
        ].iloc[0]

        out = dict(rec)
        out.update(
            {
                "match_mode": m.match_mode,
                "matched_pixel_x": m.pixel_x,
                "matched_pixel_y": m.pixel_y,
                "match_distance_m": m.distance_m,
                "pixel_idx": m.pixel_idx,
                "model_H_eff_m": float(pix["H_eff_input_m"]),
                "model_K_mps": float(pix["K_mps"]),
                "model_Hd_m": float(pix["Hd_m"]),
            }
        )
        if "H_payload_m" in pix.index:
            out["payload_H_m"] = float(pix["H_payload_m"])
        rows.append(out)

    site_df = pd.DataFrame(rows)
    validate_site_matches(
        site_df,
        max_distance_m=max_distance_m,
        min_unique_pixels=min_unique_pixels,
    )

    obs_h = site_df[
        "approx_compressible_thickness_m"
    ].to_numpy(dtype=float)
    mod_h = site_df["model_H_eff_m"].to_numpy(dtype=float)

    prod = site_df[productivity_col].to_numpy(dtype=float)
    mod_k = site_df["model_K_mps"].to_numpy(dtype=float)

    metrics = {
        "n_sites": int(len(site_df)),
        "split_used": split,
        "borehole_vs_H_eff": {
            "spearman_rho": spearman_rho(obs_h, mod_h),
            "mae_m": mae(obs_h, mod_h),
            "median_bias_m": median_bias(obs_h, mod_h),
        },
        "pumping_vs_K": {
            "productivity_column": productivity_col,
            "spearman_rho": spearman_rho(prod, mod_k),
        },
        "files": {
            "validation_csv": str(
                Path(validation_csv).expanduser().resolve()
            ),
            "stage1_manifest": str(
                Path(stage1_manifest_path)
                .expanduser()
                .resolve()
            ),
            "stage2_manifest": (
                str(
                    Path(stage2_manifest_path)
                    .expanduser()
                    .resolve()
                )
                if stage2_manifest_path is not None
                else None
            ),
            "inputs_npz": str(inputs_npz_path),
            "physics_payload": str(payload_npz_path),
        },
        "reducers": {
            "horizon_reducer": horizon_reducer,
            "site_reducer": site_reducer,
        },
        "matching": {
            "max_distance_m": max_distance_m,
            "min_unique_pixels": min_unique_pixels,
        },
    }

    site_csv = outdir / "site_level_external_validation.csv"
    metrics_json = outdir / "external_validation_metrics.json"

    site_df.to_csv(site_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return site_df, metrics


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    p = argparse.ArgumentParser(
        description=(
            "Compute borehole / pumping validation metrics from Stage-1 "
            "inputs plus a saved physics payload."
        )
    )
    add_config_args(p)
    add_city_arg(p)
    add_model_arg(p)
    add_results_dir_arg(p)
    add_stage1_dir_arg(p)
    add_manifest_arg(
        p,
        dest="stage1_manifest",
        option="--stage1-manifest",
        help_text="Stage-1 manifest path.",
    )
    add_stage2_manifest_arg(p)
    add_outdir_arg(p)
    add_split_arg(
        p,
        default="test",
        choices=("train", "val", "test"),
    )
    add_validation_csv_arg(p)

    p.add_argument(
        "--inputs-npz",
        type=str,
        default=None,
        help="Override Stage-1 inputs NPZ path.",
    )
    p.add_argument(
        "--physics-payload",
        type=str,
        default=None,
        help="Saved physics payload NPZ path.",
    )
    p.add_argument(
        "--coord-scaler",
        type=str,
        default=None,
        help="Coordinate scaler joblib path.",
    )
    p.add_argument(
        "--productivity-col",
        type=str,
        default="step3_specific_capacity_Lps_per_m",
        help="Productivity column used against model K.",
    )
    p.add_argument(
        "--horizon-reducer",
        type=str,
        default="mean",
        choices=["first", "mean", "median"],
        help="Reducer used across forecast horizon.",
    )
    p.add_argument(
        "--site-reducer",
        type=str,
        default="median",
        choices=["mean", "median"],
        help="Reducer used across repeated pixel visits.",
    )
    p.add_argument(
        "--max-distance-m",
        type=float,
        default=5000.0,
        help="Maximum acceptable site-to-pixel distance.",
    )
    p.add_argument(
        "--min-unique-pixels",
        type=int,
        default=3,
        help="Minimum number of unique matched pixels.",
    )
    p.add_argument(
        "--print-site-table",
        action="store_true",
        help="Print the full per-site table after writing outputs.",
    )
    return p


def build_external_validation_metrics_main(
    argv: list[str] | None = None,
) -> None:
    """CLI entry point for external validation metrics build."""
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = bootstrap_runtime_config(
        args,
        field_map={
            "city": "CITY_NAME",
            "model": "MODEL_NAME",
            "results_dir": "RESULTS_DIR",
        },
    )

    city = args.city or cfg.get("CITY_NAME")
    model = args.model or cfg.get("MODEL_NAME")
    results_dir = args.results_dir or cfg.get("RESULTS_DIR")

    stage1_manifest_path = resolve_stage1_manifest(
        manifest=args.stage1_manifest,
        stage1_dir=args.stage1_dir,
        results_dir=results_dir,
        city=city,
        model=model,
    )
    stage1_dir = stage1_manifest_path.parent

    stage2_manifest_path = resolve_stage2_manifest(
        override=args.stage2_manifest,
        stage1_dir=stage1_dir,
    )
    outdir = resolve_outdir(
        args,
        stage1_dir=stage1_dir,
        stage2_manifest_path=stage2_manifest_path,
        split=args.split,
    )
    validation_csv = resolve_validation_csv(args, cfg)

    site_df, metrics = compute_metrics(
        validation_csv=validation_csv,
        stage1_manifest_path=stage1_manifest_path,
        outdir=outdir,
        split=args.split,
        inputs_npz=args.inputs_npz,
        physics_payload=args.physics_payload,
        coord_scaler=args.coord_scaler,
        stage2_manifest_path=stage2_manifest_path,
        productivity_col=args.productivity_col,
        horizon_reducer=args.horizon_reducer,
        site_reducer=args.site_reducer,
        max_distance_m=args.max_distance_m,
        min_unique_pixels=args.min_unique_pixels,
    )

    print(
        "[OK] wrote:",
        str(outdir / "site_level_external_validation.csv"),
    )
    print(
        "[OK] wrote:",
        str(outdir / "external_validation_metrics.json"),
    )
    if args.print_site_table:
        print("\nPer-site summary\n")
        print(site_df.to_string(index=False))
    print("\nHeadline metrics\n")
    print(json.dumps(metrics, indent=2))


main = build_external_validation_metrics_main


if __name__ == "__main__":
    build_external_validation_metrics_main()
