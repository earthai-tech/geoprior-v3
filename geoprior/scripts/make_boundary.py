r"""Script helpers for building study-area boundary artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import config as cfg
from . import utils

if TYPE_CHECKING:
    import geopandas as gpd  # noqa


_CITY_A = cfg.CITY_CANON.get("ns", "Nansha")
_CITY_B = cfg.CITY_CANON.get("zh", "Zhongshan")


def _require_geopandas():
    try:
        import geopandas as gpd  # noqa
    except Exception as e:
        raise SystemExit(
            "Need geopandas installed for boundary export. "
            f"Error: {e}"
        ) from e
    return gpd


def _require_shapely_boundary_tools():
    try:
        from shapely import concave_hull
        from shapely.geometry import MultiPoint
    except Exception as e:
        raise SystemExit(
            "Need shapely installed for boundary export. "
            f"Error: {e}"
        ) from e
    return MultiPoint, concave_hull


def _pick_paths(
    art: utils.Artifacts,
    split: str,
) -> tuple[Path | None, Path | None]:
    if split == "val":
        return art.forecast_val_csv, art.forecast_future_csv
    if split == "test":
        return (
            art.forecast_test_csv,
            art.forecast_test_future_csv,
        )
    if (
        art.forecast_test_csv is not None
        and art.forecast_test_future_csv is not None
    ):
        return (
            art.forecast_test_csv,
            art.forecast_test_future_csv,
        )
    return art.forecast_val_csv, art.forecast_future_csv


def _load_xy(path: str) -> np.ndarray:
    df = pd.read_csv(utils.as_path(path))
    utils.ensure_columns(df, aliases=cfg._BASE_ALIASES)

    for c in ("coord_x", "coord_y"):
        if c not in df.columns:
            raise KeyError(f"{path}: missing {c}")

    x = pd.to_numeric(
        df["coord_x"], errors="coerce"
    ).to_numpy(float)
    y = pd.to_numeric(
        df["coord_y"], errors="coerce"
    ).to_numpy(float)

    m = np.isfinite(x) & np.isfinite(y)
    return np.column_stack([x[m], y[m]])


def _resolve_city(
    *,
    city: str,
    src: str | None,
    eval_csv: str | None,
    future_csv: str | None,
    split: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {"name": city}

    if eval_csv and future_csv:
        out["eval_csv"] = str(utils.as_path(eval_csv))
        out["future_csv"] = str(utils.as_path(future_csv))
        return out

    if not src:
        raise ValueError(
            f"{city}: provide --*-src or --*-eval/--*-future"
        )

    art = utils.detect_artifacts(src)
    ev, fu = _pick_paths(art, split)

    if ev is None or fu is None:
        raise FileNotFoundError(
            f"{city}: missing eval/future under {src}"
        )

    out["eval_csv"] = str(ev)
    out["future_csv"] = str(fu)
    return out


def _poly_from_points(
    xy: np.ndarray,
    *,
    method: str,
    alpha: float,
):
    MultiPoint, concave_hull = (
        _require_shapely_boundary_tools()
    )

    pts = MultiPoint([tuple(p) for p in xy])

    if method == "convex":
        return pts.convex_hull

    try:
        return concave_hull(pts, ratio=float(alpha))
    except Exception:
        return pts.convex_hull


def _make_boundary_gdf(
    *,
    city: str,
    poly: Any,
):
    gpd = _require_geopandas()  # noqa
    return gpd.GeoDataFrame(
        [{"city": city, "geometry": poly}],
        geometry="geometry",
        crs=None,
    )


def _slug_city(city: str) -> str:
    return str(city).strip().lower().replace(" ", "_")


def _out_stem(
    out_arg: str,
    *,
    city: str,
) -> Path:
    base = utils.resolve_out_out(out_arg)

    if base.suffix:
        base = base.with_suffix("")

    stem = base.parent / (f"{base.name}_{_slug_city(city)}")
    stem.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    return stem


def _write_checked(
    gdf: Any,
    path: Path,
    *,
    driver: str | None = None,
) -> Path:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if driver is None:
        gdf.to_file(path)
    else:
        gdf.to_file(path, driver=driver)

    if not path.exists():
        raise FileNotFoundError(
            "Boundary export reported success but "
            f"file is missing: {path}"
        )

    return path.resolve()


def make_boundary_main(
    argv: list[str] | None = None,
    *,
    prog: str | None = None,
) -> list[Path]:
    ap = argparse.ArgumentParser(
        prog=prog or "make-boundary",
        description=(
            "Create a boundary polygon from forecast points."
        ),
    )
    utils.add_city_flags(ap, default_both=True)

    ap.add_argument("--ns-src", type=str, default=None)
    ap.add_argument("--zh-src", type=str, default=None)
    ap.add_argument("--ns-eval", type=str, default=None)
    ap.add_argument("--zh-eval", type=str, default=None)
    ap.add_argument("--ns-future", type=str, default=None)
    ap.add_argument("--zh-future", type=str, default=None)

    ap.add_argument(
        "--split",
        choices=["auto", "val", "test"],
        default="auto",
    )

    ap.add_argument(
        "--method",
        choices=["convex", "concave"],
        default="convex",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Concave hull ratio (0..1).",
    )

    ap.add_argument(
        "--format",
        choices=["geojson", "shp", "both"],
        default="geojson",
    )

    ap.add_argument(
        "--out",
        type=str,
        default="boundary",
        help=(
            "Output stem/path. Bare names go to "
            "scripts/out; explicit folders are kept."
        ),
    )

    args = ap.parse_args(argv)
    utils.ensure_script_dirs()

    cities0 = utils.resolve_cities(args) or [
        _CITY_A,
        _CITY_B,
    ]

    jobs: list[dict[str, Any]] = []

    if _CITY_A in cities0:
        jobs.append(
            _resolve_city(
                city=_CITY_A,
                src=args.ns_src,
                eval_csv=args.ns_eval,
                future_csv=args.ns_future,
                split=args.split,
            )
        )

    if _CITY_B in cities0:
        jobs.append(
            _resolve_city(
                city=_CITY_B,
                src=args.zh_src,
                eval_csv=args.zh_eval,
                future_csv=args.zh_future,
                split=args.split,
            )
        )

    written: list[Path] = []

    for j in jobs:
        city = str(j["name"])
        xy1 = _load_xy(j["eval_csv"])
        xy2 = _load_xy(j["future_csv"])
        xy = np.vstack([xy1, xy2])

        poly = _poly_from_points(
            xy,
            method=str(args.method),
            alpha=float(args.alpha),
        )

        gdf = _make_boundary_gdf(
            city=city,
            poly=poly,
        )

        stem = _out_stem(args.out, city=city)

        if args.format in ("geojson", "both"):
            p = _write_checked(
                gdf,
                stem.with_suffix(".geojson"),
                driver="GeoJSON",
            )
            written.append(p)
            print(f"[OK] wrote {p}")

        if args.format in ("shp", "both"):
            p = _write_checked(
                gdf,
                stem.with_suffix(".shp"),
            )
            written.append(p)
            print(f"[OK] wrote {p}")

    return written


def main(
    argv: list[str] | None = None,
) -> None:
    make_boundary_main(argv)


if __name__ == "__main__":
    main()
