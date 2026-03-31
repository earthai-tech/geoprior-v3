# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>
#
# Utilities to:
# - enforce paper style (matplotlib rcParams)
# - auto-detect artifacts from a "src" (file or folder)
# - load JSON/CSV robustly (schema checks)
# - harmonize GeoPrior JSON units (raw vs interpretable variants)

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg

_TRUE = {"1", "true", "yes", "y", "t", "on"}
_FALSE = {"0", "false", "no", "n", "f", "off"}


def _as_pathlike_str(
    value: str | os.PathLike[str] | Any,
) -> str:
    """Return a filesystem-like value as a string."""
    return os.fspath(value)


def _candidate_raster_from_stem(
    stem: str | os.PathLike[str],
    *,
    raster_exts: tuple[str, ...] = (
        "png",
        "jpg",
        "jpeg",
        "tif",
        "tiff",
        "webp",
    ),
) -> list[Path]:
    """
    Build raster candidates from an output stem or file path.

    Parameters
    ----------
    stem : path-like
        Output stem or filename. If a suffix is already present,
        it is tested first. If the suffix is not a supported raster
        type, the suffix is stripped and raster candidates are built
        from the stem.
    raster_exts : tuple of str, default=("png", "jpg", ...)
        Raster extensions to test in order.

    Returns
    -------
    list of Path
        Candidate raster paths in priority order.
    """
    p = Path(_as_pathlike_str(stem)).expanduser()
    out: list[Path] = []

    if p.suffix:
        ext = p.suffix.lower().lstrip(".")
        if ext in raster_exts:
            out.append(p)
            return out
        p = p.with_suffix("")

    for ext in raster_exts:
        out.append(p.with_suffix(f".{ext}"))

    return out


def resolve_gallery_raster_path(
    *,
    out_paths: Any | None = None,
    out_base: str | os.PathLike[str] | None = None,
    raster_exts: tuple[str, ...] = (
        "png",
        "jpg",
        "jpeg",
        "tif",
        "tiff",
        "webp",
    ),
    must_exist: bool = True,
) -> Path:
    """
    Resolve a saved raster image for gallery display.

    This helper supports the two common lesson patterns:

    1. a plotting backend returns ``out_paths``
    2. a plotting backend writes from an ``out_base`` stem

    Parameters
    ----------
    out_paths : Any or None, default=None
        Collection of written paths returned by a backend. This may
        be a list/tuple/set of paths or strings, or a mapping whose
        values are paths.
    out_base : path-like or None, default=None
        Output stem or filename used by the backend.
    raster_exts : tuple of str, default=("png", "jpg", ...)
        Raster extensions accepted for gallery display, in priority
        order.
    must_exist : bool, default=True
        If ``True``, raise an error when no existing raster file is
        found.

    Returns
    -------
    Path
        Resolved raster path.

    Raises
    ------
    RuntimeError
        If no usable raster path can be resolved.
    FileNotFoundError
        If a candidate path was resolved but does not exist and
        ``must_exist=True``.

    Notes
    -----
    Sphinx-Gallery reliably displays raster images when they are
    reloaded into a matplotlib figure. This helper resolves the
    raster file that should be re-opened.
    """
    candidates: list[Path] = []

    if out_paths is not None:
        values: list[Any]
        if isinstance(out_paths, dict):
            values = list(out_paths.values())
        else:
            try:
                values = list(out_paths)
            except TypeError:
                values = [out_paths]

        for item in values:
            p = Path(_as_pathlike_str(item)).expanduser()
            ext = p.suffix.lower().lstrip(".")
            if ext in raster_exts:
                candidates.append(p)

    if not candidates and out_base is not None:
        candidates.extend(
            _candidate_raster_from_stem(
                out_base,
                raster_exts=raster_exts,
            )
        )

    for p in candidates:
        if p.exists():
            return p.resolve()

    if candidates and must_exist:
        tried = "\n".join(f" - {p}" for p in candidates)
        raise FileNotFoundError(
            "Could not locate a saved raster image for gallery "
            f"display. Tried:\n{tried}"
        )

    raise RuntimeError(
        "Could not resolve a gallery raster path. Provide either "
        "`out_paths` or `out_base`."
    )


def show_gallery_saved_figure(
    *,
    out_paths: Any | None = None,
    out_base: str | os.PathLike[str] | None = None,
    image_path: str | os.PathLike[str] | None = None,
    figsize: tuple[float, float] = (8.4, 5.4),
    hide_axes: bool = True,
    title: str | None = None,
):
    """
    Reload a saved raster figure and display it in matplotlib.

    Parameters
    ----------
    out_paths : Any or None, default=None
        Written paths returned by a plotting backend.
    out_base : path-like or None, default=None
        Output stem or filename used by the backend.
    image_path : path-like or None, default=None
        Explicit image path. When provided, this takes precedence
        over ``out_paths`` and ``out_base``.
    figsize : tuple of float, default=(8.4, 5.4)
        Figure size passed to ``plt.subplots``.
    hide_axes : bool, default=True
        If ``True``, hide axis decorations.
    title : str or None, default=None
        Optional title for the display figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure.
    ax : matplotlib.axes.Axes
        Created axis.
    image_path : pathlib.Path
        Resolved image path that was displayed.

    Notes
    -----
    Imports are done locally so lesson pages can reuse this helper
    without adding top-level matplotlib image imports everywhere.
    """
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    if image_path is None:
        image_path = resolve_gallery_raster_path(
            out_paths=out_paths,
            out_base=out_base,
        )

    p = Path(_as_pathlike_str(image_path)).expanduser()
    if not p.exists():
        raise FileNotFoundError(
            f"Gallery image does not exist: {p}"
        )

    img = mpimg.imread(str(p))
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)

    if hide_axes:
        ax.axis("off")

    if title:
        ax.set_title(title)

    return fig, ax, p.resolve()


def str_to_bool(x: object, *, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return default


def add_city_flags(ap, *, default_both: bool = True) -> None:
    ap.add_argument(
        "-ns",
        "--ns",
        "--nansha",
        dest="use_ns",
        action="store_true",
        help="Select Nansha.",
    )
    ap.add_argument(
        "-zh",
        "--zh",
        "--zhongshan",
        dest="use_zh",
        action="store_true",
        help="Select Zhongshan.",
    )
    ap.add_argument(
        "--cities",
        type=str,
        default="ns,zh" if default_both else "",
        help="Comma list: ns,zh (default ns,zh).",
    )


def add_plot_text_args(
    ap,
    *,
    default_out: str,
) -> None:
    """
    Common plot args for Nature workflows.

    Use cases:
    - Hide text for Illustrator editing.
    - Keep titles/labels in SVG as editable text.

    Conventions:
    - "legend" includes colorbar (if present).
    - "labels" means axis labels and cbar label.
    - ticklabels are controlled separately.
    """
    ap.add_argument(
        "--out",
        "-o",
        type=str,
        default=default_out,
        help="Output stem/path (scripts/figs/ if rel).",
    )
    ap.add_argument(
        "--show-legend",
        type=str,
        default="true",
        help="Show legend/colorbar (true/false).",
    )
    ap.add_argument(
        "--show-labels",
        type=str,
        default="true",
        help="Show axis labels (true/false).",
    )
    ap.add_argument(
        "--show-ticklabels",
        type=str,
        default="true",
        help="Show tick labels (true/false).",
    )
    ap.add_argument(
        "--show-title",
        type=str,
        default="true",
        help="Show suptitle (true/false).",
    )
    ap.add_argument(
        "--show-panel-titles",
        type=str,
        default="true",
        help="Show per-row panel titles (true/false).",
    )
    ap.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override suptitle text.",
    )


def add_render_args(
    ap,
    *,
    default: str = "heatmap",
) -> None:
    """
    Common render args for 2D sensitivity plots.

    render:
      - heatmap: pivot->imshow (discrete grid)
      - tricontour: smooth contourf on scattered points
      - pcolormesh: grid-aware shading using real coords
    """
    ap.add_argument(
        "--render",
        type=str,
        default=default,
        choices=[
            "heatmap",
            "tricontour",
            "pcolormesh",
        ],
        help="Render style (heatmap/tricontour/pcolormesh).",
    )
    ap.add_argument(
        "--levels",
        type=int,
        default=14,
        help="Levels for tricontour render.",
    )
    ap.add_argument(
        "--clip",
        type=str,
        default="2,98",
        help="Color scale clip percentiles (lo,hi).",
    )
    ap.add_argument(
        "--agg",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Aggregation for duplicate (λc, λp).",
    )
    ap.add_argument(
        "--show-points",
        type=str,
        default="true",
        help="Overlay sampled points (true/false).",
    )

    ap.add_argument(
        "--trend-arrow",
        type=str,
        default="false",
        help="Overlay a trend arrow (true/false).",
    )
    ap.add_argument(
        "--trend-arrow-len",
        type=float,
        default=0.22,
        help="Arrow length in axes fraction.",
    )
    ap.add_argument(
        "--trend-arrow-pos",
        type=str,
        default="0.78,0.14",
        help="Arrow anchor in axes frac: x,y",
    )
    ap.add_argument(
        "--trend-arrow-min-n",
        type=int,
        default=4,
        help="Min points needed to fit trend.",
    )


def find_all(
    src: Any,
    patterns: Sequence[str],
    *,
    must_exist: bool = False,
) -> list[Path]:
    """
    Find all files matching any pattern under src.

    - If src is a file:
        return [src] if it matches, else [].
    - If src is a directory:
        search recursively (rglob) for all patterns.

    Returned list is sorted by mtime (newest first),
    with duplicates removed.
    """
    p = as_path(src)

    if p.is_file():
        name = p.name
        ok = any(_glob_match(name, pat) for pat in patterns)
        if ok:
            return [p]
        if must_exist:
            raise FileNotFoundError(str(p))
        return []

    if not p.exists():
        if must_exist:
            raise FileNotFoundError(str(p))
        return []

    out: dict[str, Path] = {}
    for pat in patterns:
        for fp in p.rglob(pat):
            if fp.exists():
                out[str(fp.resolve())] = fp

    files = list(out.values())
    files.sort(
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    return files


def resolve_title(
    *,
    default: str,
    title: str | None,
) -> str:
    if title is None:
        return default
    t = str(title).strip()
    return default if not t else t


def resolve_cities(args) -> list[str]:
    picked: list[str] = []
    if getattr(args, "use_ns", False):
        picked.append("Nansha")
    if getattr(args, "use_zh", False):
        picked.append("Zhongshan")

    if picked:
        return picked

    raw = str(getattr(args, "cities", "") or "")
    parts = [p.strip().lower() for p in raw.split(",")]
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        if p in {"ns", "nansha"}:
            out.append("Nansha")
        elif p in {"zh", "zhongshan"}:
            out.append("Zhongshan")
        else:
            out.append(p.title())
    return out


def resolve_fig_out(out: str) -> Path:
    """
    Resolve figure output with user-path semantics.

    Bare relative names go to scripts/figs/.
    Relative paths with an explicit parent are kept.
    """
    return cfg.resolve_user_artifact_path(
        out,
        kind="fig",
    )


def _norm_fig_formats(
    formats: Sequence[str] | None,
) -> tuple[str, ...]:
    if not formats:
        return ("png", "svg", "pdf", "eps")

    out: list[str] = []
    for f in formats:
        s = str(f).strip().lower().lstrip(".")
        if not s:
            continue
        if s not in out:
            out.append(s)
    return tuple(out)


def resolve_fig_stem(out: Any) -> Path:
    """
    Resolve an output stem under scripts/figs/ if relative.

    If user passes a filename with suffix (e.g. foo.png),
    we treat it as a stem and strip the suffix.
    """
    p = resolve_fig_out(str(out))
    if p.suffix:
        p = p.with_suffix("")
    return p


def save_figure(
    fig: Any,
    out: Any,
    *,
    formats: Sequence[str] | None = None,
    dpi: int | None = None,
    bbox_inches: str = "tight",
    pad_inches: float = 0.02,
    transparent: bool = False,
    close: bool = True,
    verbose: bool = True,
    strict: bool = False,
) -> dict[str, Path]:
    """
    Save a Matplotlib figure to multiple formats.

    Parameters
    ----------
    fig:
        Matplotlib Figure.
    out:
        Output stem or path. If relative, saved under scripts/figs/.
        If has suffix, suffix is stripped and treated as stem.
    formats:
        Iterable of extensions (png/svg/pdf/eps). Default exports all 4.
    dpi:
        DPI for raster formats (png). If None, uses current rcParams.
    bbox_inches, pad_inches, transparent:
        Passed through to fig.savefig.
    close:
        Close the figure after saving.
    verbose:
        Print a compact "[OK]" line with what was written.
    strict:
        If True, any save failure raises immediately.
        If False, failures are warned and other formats continue.

    Returns
    -------
    written:
        Mapping ext -> output Path for successfully written files.
    """
    fmts = _norm_fig_formats(formats)
    stem = resolve_fig_stem(out)
    ensure_dir(stem.parent)

    raster = {"png", "jpg", "jpeg", "tif", "tiff"}

    written: dict[str, Path] = {}
    failed: dict[str, str] = {}

    for ext in fmts:
        p = stem.with_suffix("." + ext)

        kw: dict[str, Any] = dict(
            bbox_inches=bbox_inches,
            pad_inches=float(pad_inches),
            transparent=bool(transparent),
        )
        if ext in raster and dpi is not None:
            kw["dpi"] = int(dpi)

        try:
            fig.savefig(str(p), **kw)
            written[ext] = p
        except Exception as e:
            if strict:
                raise
            failed[ext] = repr(e)

    if close:
        plt.close(fig)

    if verbose:
        ok_exts = ",".join(sorted(written.keys()))
        msg = f"[OK] wrote {stem} ({ok_exts})"
        print(msg)
        if failed:
            bad = ", ".join(sorted(failed.keys()))
            print(f"[WARN] failed formats: {bad}")

    return written


def ensure_columns(
    df: pd.DataFrame,
    *,
    aliases: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    """
    Ensure canonical columns exist by copying from
    the first available alias.

    Returns:
      mapping canonical -> source column used
    """
    used: dict[str, str] = {}

    for canon, alts in (aliases or {}).items():
        if canon in df.columns:
            used[canon] = canon
            continue

        found = None
        for a in alts:
            if a in df.columns:
                found = a
                break

        if found is not None:
            df[canon] = df[found]
            used[canon] = found

    return used


def load_dataset_any(
    src: Path,
    *,
    file: str | None = None,
    ns_file: str = "nansha_dataset.final.ready.csv",
    zh_file: str = "zhongshan_dataset.final.ready.csv",
) -> pd.DataFrame:
    """
    Load a combined dataset if:
      - src is a CSV file, or
      - src is a dir and --file is provided.

    Otherwise load ns_file + zh_file from src dir and
    concatenate.
    """
    src = Path(src).expanduser()

    if src.is_file():
        return pd.read_csv(src)

    if file:
        fp = (src / file).expanduser()
        if not fp.exists():
            raise FileNotFoundError(str(fp))
        return pd.read_csv(fp)

    ns_fp = (src / ns_file).expanduser()
    zh_fp = (src / zh_file).expanduser()

    if not ns_fp.exists():
        raise FileNotFoundError(str(ns_fp))
    if not zh_fp.exists():
        raise FileNotFoundError(str(zh_fp))

    ns = pd.read_csv(ns_fp)
    zh = pd.read_csv(zh_fp)

    if "city" not in ns.columns:
        ns["city"] = "Nansha"
    if "city" not in zh.columns:
        zh["city"] = "Zhongshan"

    return pd.concat([ns, zh], ignore_index=True)


def filter_year(
    df: pd.DataFrame,
    year: str,
) -> pd.DataFrame:
    if year == "all":
        return df
    y = int(year)
    if "year" not in df.columns:
        return df
    return df.loc[df["year"] == y].copy()


def sample_df(
    df: pd.DataFrame,
    *,
    sample_frac: float | None,
    sample_n: int | None,
    seed: int = 42,
) -> pd.DataFrame:
    if sample_n is not None:
        n = min(int(sample_n), len(df))
        return df.sample(n=n, random_state=seed)

    if sample_frac is not None:
        f = float(sample_frac)
        f = max(0.0, min(1.0, f))
        if f < 1.0:
            return df.sample(frac=f, random_state=seed)

    return df


@dataclass(frozen=True)
class SpatialSupport:
    """
    Lightweight container for synthetic spatial support.
    """

    city: str
    sample_idx: np.ndarray
    coord_x: np.ndarray
    coord_y: np.ndarray
    x_norm: np.ndarray
    y_norm: np.ndarray
    extent: tuple[float, float, float, float]

    def to_frame(self) -> pd.DataFrame:
        """
        Return support as a compact DataFrame.
        """
        return pd.DataFrame(
            {
                "sample_idx": self.sample_idx.astype(int),
                "coord_x": self.coord_x.astype(float),
                "coord_y": self.coord_y.astype(float),
                "x_norm": self.x_norm.astype(float),
                "y_norm": self.y_norm.astype(float),
                "city": [self.city] * self.sample_idx.size,
            }
        )


@dataclass(frozen=True)
class SpatialSupportSpec:
    """
    Configuration for a synthetic city support.
    """

    city: str = "City"
    center_x: float = 0.0
    center_y: float = 0.0
    span_x: float = 1_000.0
    span_y: float = 800.0
    nx: int = 60
    ny: int = 44
    jitter_x: float = 0.0
    jitter_y: float = 0.0
    footprint: str = "ellipse"
    keep_frac: float = 1.0
    seed: int = 0


def _normalize01(x: np.ndarray) -> np.ndarray:
    """
    Normalize values to [0, 1].

    Degenerate arrays are mapped to zeros.
    """
    x = np.asarray(x, dtype=float)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.zeros_like(x, dtype=float)
    if hi <= lo:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def _footprint_mask(
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    *,
    footprint: str,
) -> np.ndarray:
    """
    Return a synthetic city-like footprint mask.

    Supported footprints
    --------------------
    - "ellipse"
    - "rectangle"
    - "nansha_like"
    - "zhongshan_like"
    - "corridor"
    """
    fp = str(footprint).strip().lower()

    if fp == "rectangle":
        return np.ones_like(x_norm, dtype=bool)

    if fp == "ellipse":
        return ((x_norm - 0.50) / 0.44) ** 2 + (
            (y_norm - 0.50) / 0.34
        ) ** 2 <= 1.0

    e1 = ((x_norm - 0.50) / 0.44) ** 2 + (
        (y_norm - 0.48) / 0.34
    ) ** 2 <= 1.0
    e2 = ((x_norm - 0.70) / 0.22) ** 2 + (
        (y_norm - 0.58) / 0.18
    ) ** 2 <= 1.0
    cut = ((x_norm - 0.16) / 0.12) ** 2 + (
        (y_norm - 0.74) / 0.14
    ) ** 2 <= 1.0

    if fp == "nansha_like":
        band = (
            (x_norm > 0.10)
            & (x_norm < 0.92)
            & (y_norm > 0.24)
            & (y_norm < 0.82)
        )
        return (e1 | e2 | band) & (~cut)

    if fp == "zhongshan_like":
        e3 = ((x_norm - 0.34) / 0.18) ** 2 + (
            (y_norm - 0.30) / 0.15
        ) ** 2 <= 1.0
        corridor = (
            (x_norm > 0.22)
            & (x_norm < 0.88)
            & (y_norm > 0.18)
            & (y_norm < 0.72)
        )
        return (e1 | e3 | corridor) & (~cut)

    if fp == "corridor":
        return (
            (x_norm > 0.18)
            & (x_norm < 0.88)
            & (y_norm > 0.24)
            & (y_norm < 0.72)
        )

    raise ValueError(
        "Unknown footprint. Use one of: "
        "ellipse, rectangle, nansha_like, "
        "zhongshan_like, corridor."
    )


def make_spatial_support(
    spec: SpatialSupportSpec,
) -> SpatialSupport:
    """
    Build a synthetic spatial support cloud.

    Parameters
    ----------
    spec : SpatialSupportSpec
        Support configuration.

    Returns
    -------
    SpatialSupport
        Support container with coordinates, normalized
        coordinates, and sample indices.

    Notes
    -----
    This helper is intended for gallery pages and tests.
    It produces a stable, reproducible spatial support
    that looks more like a compact urban footprint than
    a plain rectangular mesh.
    """
    rng = np.random.default_rng(int(spec.seed))

    xs = np.linspace(
        spec.center_x - spec.span_x,
        spec.center_x + spec.span_x,
        int(spec.nx),
    )
    ys = np.linspace(
        spec.center_y - spec.span_y,
        spec.center_y + spec.span_y,
        int(spec.ny),
    )

    X, Y = np.meshgrid(xs, ys)

    if float(spec.jitter_x) > 0.0:
        X = X + rng.normal(
            0.0,
            float(spec.jitter_x),
            size=X.shape,
        )
    if float(spec.jitter_y) > 0.0:
        Y = Y + rng.normal(
            0.0,
            float(spec.jitter_y),
            size=Y.shape,
        )

    x_flat = X.ravel().astype(float)
    y_flat = Y.ravel().astype(float)

    x_norm = _normalize01(x_flat)
    y_norm = _normalize01(y_flat)

    keep = _footprint_mask(
        x_norm,
        y_norm,
        footprint=spec.footprint,
    )

    if float(spec.keep_frac) < 1.0:
        frac = max(0.0, min(1.0, float(spec.keep_frac)))
        idx_keep = np.flatnonzero(keep)
        n_keep = int(np.floor(frac * idx_keep.size))
        n_keep = max(1, n_keep)
        picked = rng.choice(
            idx_keep,
            size=n_keep,
            replace=False,
        )
        picked.sort()

        keep2 = np.zeros_like(keep, dtype=bool)
        keep2[picked] = True
        keep = keep2

    x_flat = x_flat[keep]
    y_flat = y_flat[keep]
    x_norm = x_norm[keep]
    y_norm = y_norm[keep]

    sample_idx = np.arange(x_flat.size, dtype=int)

    extent = (
        float(np.nanmin(x_flat)),
        float(np.nanmax(x_flat)),
        float(np.nanmin(y_flat)),
        float(np.nanmax(y_flat)),
    )

    return SpatialSupport(
        city=str(spec.city),
        sample_idx=sample_idx,
        coord_x=x_flat,
        coord_y=y_flat,
        x_norm=x_norm,
        y_norm=y_norm,
        extent=extent,
    )


def make_spatial_field(
    support: SpatialSupport,
    *,
    amplitude: float = 1.0,
    drift_x: float = 0.0,
    drift_y: float = 0.0,
    phase: float = 0.0,
    hotspot_weight: float = 0.92,
    secondary_weight: float = 0.54,
    ridge_weight: float = 0.16,
    wave_weight: float = 0.10,
    local_weight: float = 0.0,
) -> np.ndarray:
    """
    Build a smooth synthetic spatial field.

    The field combines:
    - one main hotspot,
    - one secondary lobe,
    - one ridge,
    - one wave component,
    - one linear drift.

    Returns
    -------
    np.ndarray
        Field values on the support points.
    """
    xn = support.x_norm
    yn = support.y_norm

    g1 = np.exp(
        -(
            ((xn - 0.66) ** 2) / 0.020
            + ((yn - 0.42) ** 2) / 0.032
        )
    )
    g2 = np.exp(
        -(
            ((xn - 0.38) ** 2) / 0.042
            + ((yn - 0.64) ** 2) / 0.022
        )
    )
    ridge = np.exp(-((yn - (0.30 + 0.24 * xn)) ** 2) / 0.020)
    wave = np.sin(2.4 * np.pi * xn + float(phase)) * np.cos(
        1.7 * np.pi * yn
    )
    trend = float(drift_x) * xn + float(drift_y) * yn
    local = np.sin(4.8 * xn) + np.cos(3.7 * yn)

    return (
        float(amplitude)
        * (
            float(hotspot_weight) * g1
            + float(secondary_weight) * g2
            + float(ridge_weight) * ridge
            + float(wave_weight) * wave
        )
        + trend
        + float(local_weight) * local
    )


def make_spatial_scale(
    support: SpatialSupport,
    *,
    base: float = 0.30,
    x_weight: float = 0.08,
    hotspot_weight: float = 0.05,
    step_weight: float = 0.0,
    step: int = 0,
    floor: float = 1e-6,
) -> np.ndarray:
    """
    Build a positive spatial spread / uncertainty field.

    Parameters
    ----------
    support : SpatialSupport
        Support container.
    base : float, default=0.30
        Base scale level.
    x_weight : float, default=0.08
        East-west spread increase.
    hotspot_weight : float, default=0.05
        Extra spread near the main hotspot.
    step_weight : float, default=0.0
        Optional horizon-dependent increase.
    step : int, default=0
        Horizon index used with `step_weight`.
    floor : float, default=1e-6
        Minimum allowed scale.

    Returns
    -------
    np.ndarray
        Positive spread field.
    """
    xn = support.x_norm
    yn = support.y_norm

    hotspot = np.exp(
        -(
            ((xn - 0.66) ** 2) / 0.022
            + ((yn - 0.42) ** 2) / 0.035
        )
    )

    scale = (
        float(base)
        + float(x_weight) * xn
        + float(hotspot_weight) * hotspot
        + float(step_weight) * float(step)
    )

    return np.clip(scale, float(floor), None)


def evolve_spatial_field(
    support: SpatialSupport,
    *,
    base_field: np.ndarray,
    step: int,
    growth: float = 1.0,
    phase_step: float = 0.18,
    drift_scale: float = 0.65,
    drift_x: float = 0.34,
    drift_y: float = 0.20,
    wave_amp: float = 0.70,
) -> np.ndarray:
    """
    Evolve a base field across horizons.

    This keeps the same geography but lets:
    - the mean intensity grow,
    - the wave phase shift,
    - the field drift slightly.

    Returns
    -------
    np.ndarray
        Evolved field for the requested step.
    """
    xn = support.x_norm
    yn = support.y_norm

    i = int(step)
    wave = float(wave_amp) * np.sin(
        2.0 * np.pi * (xn + float(phase_step) * i)
    )
    drift = (
        float(drift_scale)
        * i
        * (float(drift_x) * xn + float(drift_y) * yn)
    )

    return (
        float(growth)
        * np.asarray(
            base_field,
            dtype=float,
        )
        + wave
        + drift
    )


def quantile_columns_from_mean_scale(
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    prefix: str = "subsidence",
) -> dict[str, np.ndarray]:
    """
    Convert mean + scale fields into quantile columns.

    Parameters
    ----------
    mean : np.ndarray
        Central field.
    scale : np.ndarray
        Positive spread field.
    quantiles : sequence of float
        Quantile levels in (0, 1).
    prefix : str, default="subsidence"
        Output column prefix.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping like:
        {
            "subsidence_q10": ...,
            "subsidence_q50": ...,
            ...
        }
    """
    mu = np.asarray(mean, dtype=float)
    sig = np.asarray(scale, dtype=float)

    if mu.shape != sig.shape:
        raise ValueError(
            "`mean` and `scale` must share the same shape."
        )

    nd = NormalDist()
    out: dict[str, np.ndarray] = {}

    for q in quantiles:
        qq = float(q)
        if not (0.0 < qq < 1.0):
            raise ValueError(
                "Quantiles must lie strictly in (0, 1)."
            )
        z = float(nd.inv_cdf(qq))
        name = f"{prefix}_q{int(round(100 * qq))}"
        out[name] = mu + z * sig

    return out


def make_spatial_quantile_frame(
    support: SpatialSupport,
    *,
    year: int,
    forecast_step: int,
    mean: np.ndarray,
    scale: np.ndarray,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    actual: np.ndarray | None = None,
    prefix: str = "subsidence",
    unit: str = "mm",
) -> pd.DataFrame:
    """
    Pack one spatial mean/scale state into a forecast table.

    Parameters
    ----------
    support : SpatialSupport
        Support container.
    year : int
        Calendar year for `coord_t`.
    forecast_step : int
        Forecast step or horizon index.
    mean : np.ndarray
        Central field.
    scale : np.ndarray
        Positive spread field.
    quantiles : sequence of float
        Quantile levels to export.
    actual : np.ndarray or None, default=None
        Optional observed target values.
    prefix : str, default="subsidence"
        Column prefix.
    unit : str, default="mm"
        Unit label.

    Returns
    -------
    pd.DataFrame
        Forecast-ready long table for one year.
    """
    qcols = quantile_columns_from_mean_scale(
        mean,
        scale,
        quantiles=quantiles,
        prefix=prefix,
    )

    data: dict[str, Any] = {
        "sample_idx": support.sample_idx.astype(int),
        "forecast_step": np.full(
            support.sample_idx.size,
            int(forecast_step),
            dtype=int,
        ),
        "coord_t": np.full(
            support.sample_idx.size,
            int(year),
            dtype=int,
        ),
        "coord_x": support.coord_x.astype(float),
        "coord_y": support.coord_y.astype(float),
        f"{prefix}_unit": [str(unit)]
        * support.sample_idx.size,
    }

    for name, vals in qcols.items():
        data[name] = np.asarray(vals, dtype=float)

    if actual is not None:
        data[f"{prefix}_actual"] = np.asarray(
            actual,
            dtype=float,
        )

    return pd.DataFrame(data)


def make_noisy_actual(
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    rng: np.random.Generator,
    noise_scale: float = 1.0,
) -> np.ndarray:
    """
    Sample an observed field from mean + spread.

    Parameters
    ----------
    mean : np.ndarray
        Mean field.
    scale : np.ndarray
        Positive spread field.
    rng : np.random.Generator
        Random generator.
    noise_scale : float, default=1.0
        Global multiplier applied to the spread.

    Returns
    -------
    np.ndarray
        Noisy observed field.
    """
    mu = np.asarray(mean, dtype=float)
    sig = np.asarray(scale, dtype=float)

    if mu.shape != sig.shape:
        raise ValueError(
            "`mean` and `scale` must share the same shape."
        )

    return mu + rng.normal(
        0.0,
        float(noise_scale) * sig,
        size=mu.shape,
    )


# -------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_script_dirs() -> None:
    ensure_dir(cfg.OUT_DIR)
    ensure_dir(cfg.FIG_DIR)


def as_path(src: Any) -> Path:
    if isinstance(src, Path):
        return src
    return Path(str(src)).expanduser()


def canonical_city(name: str) -> str:
    if not name:
        return name
    k = str(name).strip().lower()
    return cfg.CITY_CANON.get(k, name)


def label(key: str, *, with_unit: bool = True) -> str:
    base = cfg.LABELS.get(key, key.replace("_", " ").title())
    if not with_unit:
        return base
    u = cfg.UNITS.get(key)
    return f"{base} ({u})" if u else base


def phys_label(key: str, *, with_unit: bool = True) -> str:
    base = cfg.PHYS_LABELS.get(key, key)
    if not with_unit:
        return base
    u = cfg.PHYS_UNITS.get(key)
    return f"{base} ({u})" if u else base


# -------------------------------------------------------------------
# Matplotlib style (centralized)
# Replaces repeated set_style() across figure scripts.
# -------------------------------------------------------------------
def set_paper_style(
    *,
    fontsize: int = cfg.PAPER_FONT,
    dpi: int = cfg.PAPER_DPI,
) -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": fontsize,
            "axes.labelsize": fontsize,
            "axes.titlesize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# -------------------------------------------------------------------
# Robust file discovery from a "src"
# -------------------------------------------------------------------
def _iter_candidates(
    root: Path,
    patterns: Sequence[str],
) -> Iterable[Path]:
    for pat in patterns:
        yield from root.rglob(pat)


def find_preferred(
    src: Any,
    patterns: Sequence[str],
    *,
    must_exist: bool = False,
) -> Path | None:
    """
    Try patterns in order, returning the first match.

    Unlike find_latest() with multiple patterns,
    this respects priority order.
    """
    root = as_path(src)
    if root.is_file():
        root = root.parent

    for pat in patterns:
        p = find_latest(root, [pat], must_exist=False)
        if p is not None:
            return p

    if must_exist:
        raise FileNotFoundError(
            f"No match under {root} for {patterns}"
        )

    return None


def find_eval_diag_json(src: Any) -> Path | None:
    pats = cfg.PATTERNS.get("eval_diag_json", ())
    if not pats:
        return None
    return find_preferred(src, pats)


def find_latest(
    src: Any,
    patterns: Sequence[str],
    *,
    must_exist: bool = False,
) -> Path | None:
    """
    Find newest file matching any of patterns under src.

    - If src is a file: returns it if it matches any pattern.
    - If src is a directory: searches recursively and returns
      the most recently modified candidate.
    """
    p = as_path(src)

    if p.is_file():
        name = p.name
        ok = any(_glob_match(name, pat) for pat in patterns)
        if ok:
            return p
        if must_exist:
            raise FileNotFoundError(str(p))
        return None

    if not p.exists():
        if must_exist:
            raise FileNotFoundError(str(p))
        return None

    cands = list(_iter_candidates(p, patterns))
    if not cands:
        if must_exist:
            raise FileNotFoundError(
                f"No match under {p} for {patterns}"
            )
        return None

    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0]


def _glob_match(name: str, pattern: str) -> bool:
    import fnmatch

    return fnmatch.fnmatch(name, pattern)


@dataclass
class Artifacts:
    src: Path
    phys_json: Path | None = None
    eval_diag_json: Path | None = None
    forecast_test_csv: Path | None = None
    forecast_test_future_csv: Path | None = None
    forecast_val_csv: Path | None = None
    forecast_future_csv: Path | None = None
    physics_payload: Path | None = None
    coords_npz: Path | None = None


def detect_artifacts(src: Any) -> Artifacts:
    """
    Detect common v3.2 artifacts starting from a src path.

    Example:
    - src = ".../results/nansha_..._stage1/train_YYYYmmdd-HHMMSS"
    - We auto-locate:
      geoprior_eval_phys_*.json, eval_diagnostics.json,
      *_calibrated.csv, *_future.csv, physics payload, etc.
    """
    root = as_path(src)
    if root.is_file():
        root = root.parent

    out = Artifacts(src=root)

    out.phys_json = find_latest(
        root, cfg.PATTERNS["phys_json"]
    )
    out.eval_diag_json = find_eval_diag_json(root)

    out.forecast_val_csv = find_latest(
        root,
        cfg.PATTERNS["forecast_val_csv"],
    )
    out.forecast_future_csv = find_latest(
        root,
        cfg.PATTERNS["forecast_future_csv"],
    )
    # out.forecast_val_csv = find_preferred(
    #     root,
    #     cfg.PATTERNS["forecast_val_csv"],
    # )

    # out.forecast_test_csv = find_preferred(
    #     root,
    #     cfg.PATTERNS["forecast_test_csv"],
    # )

    out.physics_payload = find_latest(
        root,
        cfg.PATTERNS["physics_payload"],
    )
    out.forecast_test_csv = find_latest(
        root,
        cfg.PATTERNS["forecast_test_csv"],
    )
    out.forecast_test_future_csv = find_latest(
        root,
        cfg.PATTERNS["forecast_test_future_csv"],
    )
    out.coords_npz = find_latest(
        root, cfg.PATTERNS["coords_npz"]
    )
    return out


# -------------------------------------------------------------------
# Loading helpers (JSON / CSV)
# -------------------------------------------------------------------
def safe_load_json(path: Any | None) -> dict[str, Any]:
    if not path:
        return {}
    p = as_path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def load_forecast_csv(path: Any) -> pd.DataFrame:
    """
    Load calibrated forecast CSV and enforce basic schema.

    Expected columns are consistent with the uncertainty scripts:
      sample_idx, forecast_step,
      subsidence_q10, subsidence_q50, subsidence_q90,
      subsidence_actual
    """
    p = as_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    df = pd.read_csv(p)

    needed = {
        "sample_idx",
        "forecast_step",
        "subsidence_q10",
        "subsidence_q50",
        "subsidence_q90",
        "subsidence_actual",
    }
    missing = [c for c in needed if c not in df.columns]
    # if missing:
    #     raise KeyError(f"Missing columns in {p}: {missing}")
    if missing:
        if (
            "subsidence_actual" in missing
            and "future" in p.name.lower()
        ):
            raise KeyError(
                f"{p} looks like a future "
                "forecast (no actual). "
                "Use *_eval*_calibrated.csv."
            )
        raise KeyError(f"Missing columns in {p}: {missing}")

    for c in [
        "subsidence_q10",
        "subsidence_q50",
        "subsidence_q90",
        "subsidence_actual",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["forecast_step"] = pd.to_numeric(
        df["forecast_step"],
        errors="coerce",
    )
    df = df.dropna(subset=["forecast_step"]).copy()
    return df


# -------------------------------------------------------------------
# GeoPrior JSON harmonization
# - v3.2 may produce both:
#   * raw JSON (subs_metrics_unit="m")
#   * interpretable JSON (subs_metrics_unit="mm")
# We standardize to "mm" for plotting/tables.
# -------------------------------------------------------------------


def phys_json_to_mm(meta: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize GeoPrior phys JSON to "mm" subs metrics.

    v3.2 may produce:
    - interpretable JSON: already in mm (no-op)
    - raw JSON: often distance-like metrics in meters

    We convert:
    - distance-like: MAE / RMSE / sharpness  -> × scale
    - squared:       MSE                     -> × scale²

    This conversion touches (if present):
    - metrics_evaluate: subs_pred_* keys
    - point_metrics
    - per_horizon: mae/mse/rmse/sharpness
    - interval_metrics: sharpness*
    - interval_calibration: sharpness* keys
    """
    if not meta:
        return {}

    out = dict(meta)
    units = dict(out.get("units") or {})

    u = str(units.get("subs_metrics_unit", "")).lower()
    if u == "mm":
        return out

    # Prefer an explicit factor if provided.
    # Common: 1000 for m -> mm.
    scale = units.get("subs_factor_si_to_real", 1000.0)
    try:
        scale = float(scale)
    except Exception:
        scale = 1000.0

    if not np.isfinite(scale) or scale <= 0:
        scale = 1000.0

    scale2 = float(scale) ** 2

    def _scale_dist(d: dict[str, Any], key: str) -> None:
        v = d.get(key, None)
        if isinstance(v, int | float) and np.isfinite(v):
            d[key] = float(v) * float(scale)

    def _scale_mse(d: dict[str, Any], key: str) -> None:
        v = d.get(key, None)
        if isinstance(v, int | float) and np.isfinite(v):
            d[key] = float(v) * float(scale2)

    def _scale_map(
        d: dict[str, Any],
        *,
        dist_keys: tuple[str, ...],
        mse_keys: tuple[str, ...],
    ) -> dict[str, Any]:
        if not isinstance(d, dict):
            return {}
        dd = dict(d)
        for k in dist_keys:
            if k in dd:
                _scale_dist(dd, k)
        for k in mse_keys:
            if k in dd:
                _scale_mse(dd, k)
        return dd

    # ---------------------------------------------------------
    # metrics_evaluate (subs_pred_*)
    # ---------------------------------------------------------
    me = dict(out.get("metrics_evaluate") or {})
    if isinstance(me, dict) and me:
        for k in list(me.keys()):
            if not str(k).startswith("subs_pred_"):
                continue

            kk = str(k).lower()

            # mse family: mse, mse_q50, etc.
            if "mse" in kk:
                _scale_mse(me, k)
                continue

            # distance-like: mae/rmse/sharpness, incl *_q50.
            if "mae" in kk or "rmse" in kk or "sharp" in kk:
                _scale_dist(me, k)

        out["metrics_evaluate"] = me

    # ---------------------------------------------------------
    # point_metrics
    # ---------------------------------------------------------
    pm = dict(out.get("point_metrics") or {})
    if isinstance(pm, dict) and pm:
        pm = _scale_map(
            pm,
            dist_keys=("mae", "rmse"),
            mse_keys=("mse",),
        )
        out["point_metrics"] = pm

    # ---------------------------------------------------------
    # per_horizon metrics
    # (only scale numeric dict-of-dict blocks)
    # ---------------------------------------------------------
    ph = dict(out.get("per_horizon") or {})
    if isinstance(ph, dict) and ph:
        ph2 = dict(ph)

        def _scale_hdict(
            hd: dict[str, Any], mul: float
        ) -> dict[str, Any]:
            outd = dict(hd)
            for hk, hv in outd.items():
                if isinstance(
                    hv, int | float
                ) and np.isfinite(hv):
                    outd[hk] = float(hv) * float(mul)
            return outd

        for key, mul in [
            ("mae", scale),
            ("rmse", scale),
            ("sharpness80", scale),
            ("sharpness", scale),
            ("mse", scale2),
        ]:
            blk = ph2.get(key, None)
            if isinstance(blk, dict):
                ph2[key] = _scale_hdict(blk, float(mul))

        out["per_horizon"] = ph2

    # ---------------------------------------------------------
    # interval_metrics (if present)
    # ---------------------------------------------------------
    im = dict(out.get("interval_metrics") or {})
    if isinstance(im, dict) and im:
        for k in list(im.keys()):
            if "sharp" in str(k).lower():
                _scale_dist(im, k)
        out["interval_metrics"] = im

    # ---------------------------------------------------------
    # interval_calibration
    # NOTE: these often contain both SI and phys fields.
    # Raw JSON can store these in meters; scale all sharpness*
    # keys to mm to make plotting consistent.
    # ---------------------------------------------------------
    ic = dict(out.get("interval_calibration") or {})
    if isinstance(ic, dict) and ic:
        for k in list(ic.keys()):
            if "sharp" in str(k).lower():
                _scale_dist(ic, k)
        out["interval_calibration"] = ic

    units["subs_metrics_unit"] = "mm"
    out["units"] = units

    return out


# -------------------------------------------------------------------
# Metric picking (GeoPrior JSON = primary; eval_diag = fallback)
# Mirrors your Figure-2 collection logic.
# -------------------------------------------------------------------
def pick_point_metrics(
    phys_json: dict[str, Any],
    fallback: dict[str, Any],
) -> tuple[float, float, float]:
    r2 = mae = mse = np.nan

    if phys_json:
        pm = phys_json.get("point_metrics", {}) or {}
        if pm:
            r2 = pm.get("r2", r2)
            mae = pm.get("mae", mae)
            mse = pm.get("mse", mse)

        me = phys_json.get("metrics_evaluate", {}) or {}
        if np.isnan(mae):
            mae = me.get("subs_pred_mae", mae)
        if np.isnan(mse):
            mse = me.get("subs_pred_mse", mse)

    flat = flatten_eval_diag(fallback) if fallback else {}
    if np.isnan(r2):
        r2 = flat.get("r2", r2)
    if np.isnan(mae):
        mae = flat.get("mae", mae)
    if np.isnan(mse):
        mse = flat.get("mse", mse)

    return (to_float(r2), to_float(mae), to_float(mse))


def pick_interval_metrics(
    phys_json: dict[str, Any],
    fallback: dict[str, Any],
) -> tuple[float, float]:
    """
    Pick interval metrics with calibrated preference.

    Preference order:
      1) phys_json.interval_calibration:
         - calibrated_phys (paper / physical)
         - calibrated      (often SI)
      2) phys_json.interval_metrics
      3) phys_json.metrics_evaluate (may be uncalibrated)
      4) eval diagnostics JSON (flattened)

    Returns
    -------
    coverage80 : float
        Unitless.
    sharpness80 : float
        Returned in mm when we can infer conversion.
    """
    cov = float("nan")
    shp = float("nan")
    shp_src = ""

    def _take(cur: float, v: Any) -> float:
        if np.isfinite(cur):
            return cur
        fv = to_float(v)
        return fv if np.isfinite(fv) else cur

    def _take_tag(
        cur: float, v: Any, tag: str
    ) -> tuple[float, str]:
        if np.isfinite(cur):
            return cur, tag
        fv = to_float(v)
        if np.isfinite(fv):
            return fv, tag
        return cur, ""

    # Conversion factor (SI -> "real") when available.
    # Common: 1000 for m -> mm.
    def _si_to_real_factor() -> float:
        if not phys_json:
            return 1000.0

        units = phys_json.get("units") or {}
        fac = units.get("subs_factor_si_to_real", None)
        try:
            fac = float(fac)
        except Exception:
            fac = 1000.0

        if not np.isfinite(fac) or fac <= 0:
            fac = 1000.0
        return float(fac)

    fac = _si_to_real_factor()

    if phys_json:
        ic = phys_json.get("interval_calibration") or {}
        if isinstance(ic, dict):
            cov = _take(
                cov,
                ic.get("coverage80_calibrated_phys"),
            )
            cov = _take(
                cov,
                ic.get("coverage80_calibrated"),
            )

            shp, tag = _take_tag(
                shp,
                ic.get("sharpness80_calibrated_phys"),
                "ic_phys",
            )
            if not tag:
                shp, tag = _take_tag(
                    shp,
                    ic.get("sharpness80_calibrated"),
                    "ic_si",
                )
            if tag:
                shp_src = tag

        im = phys_json.get("interval_metrics") or {}
        if isinstance(im, dict):
            cov = _take(cov, im.get("coverage80"))
            shp, tag = _take_tag(
                shp,
                im.get("sharpness80"),
                "im",
            )
            if tag:
                shp_src = tag

        me = phys_json.get("metrics_evaluate") or {}
        if isinstance(me, dict):
            cov = _take(
                cov,
                me.get("subs_pred_coverage80"),
            )
            shp, tag = _take_tag(
                shp,
                me.get("subs_pred_sharpness80"),
                "me",
            )
            if tag:
                shp_src = tag

    flat = flatten_eval_diag(fallback) if fallback else {}
    cov = _take(cov, flat.get("coverage80"))

    if not np.isfinite(shp):
        shp, tag = _take_tag(
            shp,
            flat.get("sharpness80"),
            "diag",
        )
        if tag:
            shp_src = tag

    # ---------------------------------------------------------
    # Unit harmonization for sharpness
    # - If we used SI-calibrated value from interval_calibration,
    #   convert to mm using fac.
    # - If we fell back to diagnostics and it looks SI (< 1),
    #   convert as well.
    # ---------------------------------------------------------
    if np.isfinite(shp):
        if shp_src == "ic_si":
            shp = float(shp) * float(fac)

        elif shp_src == "diag" and float(shp) < 1.0:
            shp = float(shp) * float(fac)

    return (float(cov), float(shp))


def flatten_eval_diag(
    diag: dict[str, Any],
) -> dict[str, float]:
    """
    Flatten eval diagnostics into keys used by plots.

    Supports:
      1) Flat: {"mae":..., "coverage80":...}
      2) "__overall__" block:
         {"2020.0": {...}, "__overall__": {...}}
      3) eval_after:
         {"overall_key": "__overall__",
          "eval_after": {"__overall__": {...}}}
    """
    if not diag:
        return {}

    out: dict[str, float] = {}

    def _set_if(dst: str, v: Any) -> None:
        if dst in out:
            return
        fv = to_float(v)
        if np.isfinite(fv):
            out[dst] = fv

    blk = diag.get("__overall__")
    if isinstance(blk, dict):
        _set_if("mae", blk.get("overall_mae"))
        _set_if("mse", blk.get("overall_mse"))
        _set_if("rmse", blk.get("overall_rmse"))
        _set_if("r2", blk.get("overall_r2"))
        _set_if("coverage80", blk.get("coverage80"))
        _set_if("sharpness80", blk.get("sharpness80"))

    for k in ["r2", "mae", "mse", "rmse"]:
        if k in diag:
            _set_if(k, diag.get(k))

    for k in ["coverage80", "sharpness80"]:
        if k in diag:
            _set_if(k, diag.get(k))

    overall_key = str(diag.get("overall_key") or "")
    eval_after = diag.get("eval_after") or {}
    if isinstance(eval_after, dict):
        if overall_key and overall_key in eval_after:
            maybe = eval_after.get(overall_key) or {}
            if isinstance(maybe, dict):
                eval_after = maybe

        cov = eval_after.get("coverage", None)
        shp = eval_after.get("sharpness", None)

        if "coverage80" not in out and cov is not None:
            _set_if("coverage80", cov)

        if "sharpness80" not in out and shp is not None:
            _set_if("sharpness80", shp)

    return out


def to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def resolve_out_out(out: str) -> Path:
    """
    Resolve tabular output with user-path semantics.

    Bare relative names go to scripts/out/.
    Relative paths with an explicit parent are kept.
    """
    return cfg.resolve_user_artifact_path(
        out,
        kind="out",
    )


def find_phys_json(src: Any) -> Path | None:
    """
    Prefer interpretable GeoPrior JSON when available.
    """
    pats = cfg.PATTERNS.get("phys_json", ())
    if not pats:
        return None

    root = as_path(src)
    if root.is_file():
        root = root.parent

    p0 = find_latest(root, [pats[0]])
    if p0 is not None:
        return p0

    if len(pats) > 1:
        return find_latest(root, [pats[1]])

    return None
