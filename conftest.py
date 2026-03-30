from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

TIME_STEPS = 5
FORECAST_HORIZON = 3
TRAIN_END_YEAR = 2022
FORECAST_START_YEAR = 2023
DEFAULT_CITY = "nansha"
DEFAULT_MODEL = "GeoPriorSubsNet"
RNG_SEED = 123

LITHOLOGIES = (
    "Residual (Saprolitic) Soil",
    "Sandstone",
    "Siltstone",
    "Mudstone",
)

LITHO_CLASSES = (
    "Mixed Clastics",
    "Fine-Grained Soil",
    "Coarse-Grained Soil",
)

RAW_PANEL_COLUMNS = (
    "longitude",
    "latitude",
    "year",
    "lithology",
    "GWL",
    "rainfall_mm",
    "soil_thickness",
    "normalized_urban_load_proxy",
    "subsidence",
    "subsidence_cum",
    "city",
    "lithology_class",
    "GWL_depth_bgs",
    "GWL_depth_bgs_m",
    "GWL_depth_bgs_z",
    "soil_thickness_censored",
    "soil_thickness_imputed",
    "soil_thickness_eff",
    "urban_load_global",
    "z_surf",
    "z_surf_m",
    "head_m",
)

DYNAMIC_FEATURES = (
    "GWL_depth_bgs_m__si",
    "subsidence_cum__si",
    "rainfall_mm",
    "urban_load_global",
    "soil_thickness_censored",
)

FUTURE_FEATURES = ("rainfall_mm",)

TARGET_KEYS = ("subs_pred", "gwl_pred")
INPUT_KEYS = (
    "static_features",
    "dynamic_features",
    "future_features",
    "coords",
    "H_field",
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "heavy_cli: mark tests that exercise"
        " the real stage workflow.",
    )
    config.addinivalue_line(
        "markers",
        "stage_artifacts: uses synthetic stage"
        " artifact trees.",
    )
    config.addinivalue_line(
        "markers",
        "script_artifacts: uses isolated geoprior._scripts"
        " output roots.",
    )
    config.addinivalue_line(
        "markers",
        "fast_plots: uses stubbed figure writing for speed.",
    )


@pytest.fixture(autouse=True)
def _test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", str(RNG_SEED))
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "3")
    monkeypatch.setenv("MPLBACKEND", "Agg")

    import matplotlib
    matplotlib.use("Agg", force=True)

    try:
        import matplotlib.pyplot as plt
        plt.switch_backend("Agg")
    except Exception:
        pass


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


@pytest.fixture
def city_panel_df(rng: np.random.Generator) -> pd.DataFrame:
    return make_city_panel(rng=rng)


@pytest.fixture
def write_city_csv(
    tmp_path: Path,
    city_panel_df: pd.DataFrame,
) -> Callable[..., Path]:
    def _write(
        *,
        city: str = DEFAULT_CITY,
        filename: str | None = None,
        df: pd.DataFrame | None = None,
    ) -> Path:
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        name = filename
        if name is None:
            name = (
                f"{city}_final_main_std.harmonized."
                "cleaned.with_zsurf.csv"
            )

        out = data_dir / name
        frame = city_panel_df.copy() if df is None else df.copy()
        frame["city"] = city
        frame.to_csv(out, index=False)
        return out

    return _write


@pytest.fixture
def natcom_config_dict(
    tmp_path: Path,
    write_city_csv: Callable[..., Path],
) -> dict[str, Any]:
    data_csv = write_city_csv()
    return build_natcom_config(
        data_dir=data_csv.parent,
        base_output_dir=tmp_path / "results",
        city=DEFAULT_CITY,
        model=DEFAULT_MODEL,
    )


@pytest.fixture
def write_natcom_config(
    tmp_path: Path,
) -> Callable[..., dict[str, Any]]:
    def _write(
        config: dict[str, Any],
        *,
        root_name: str = "nat.com",
    ) -> dict[str, Any]:
        root = tmp_path / root_name
        root.mkdir(parents=True, exist_ok=True)

        config_py = root / "config.py"
        config_json = root / "config.json"

        write_python_assignments(config_py, config)
        config_json.write_text(
            json.dumps(
                {
                    "city": config["CITY_NAME"],
                    "model": config["MODEL_NAME"],
                    "config": config,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "root": root,
            "config_py": config_py,
            "config_json": config_json,
        }

    return _write


@pytest.fixture
def make_stage1_bundle(
    tmp_path: Path,
) -> Callable[..., dict[str, Any]]:
    def _make(
        *,
        city: str = DEFAULT_CITY,
        model: str = DEFAULT_MODEL,
        n_groups: int = 10,
        start_year: int = 2015,
        end_year: int = 2025,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(RNG_SEED)
        panel = make_city_panel(
            city=city,
            n_groups=n_groups,
            start_year=start_year,
            end_year=end_year,
            rng=rng,
        )

        config = build_natcom_config(
            data_dir=tmp_path / "data",
            base_output_dir=tmp_path / "results",
            city=city,
            model=model,
        )

        data_dir = Path(config["DATA_DIR"])
        data_dir.mkdir(parents=True, exist_ok=True)
        input_csv = data_dir / config["BIG_FN"]
        panel.to_csv(input_csv, index=False)

        out = build_stage1_artifact_tree(
            tmp_path=tmp_path,
            panel=panel,
            city=city,
            model=model,
            config=config,
        )
        out["input_csv"] = input_csv
        return out

    return _make


@pytest.fixture
def mini_stage1_bundle(
    make_stage1_bundle: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    return make_stage1_bundle()


@pytest.fixture
def patch_cli_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., dict[str, Any]]:
    def _patch(
        module: Any,
        name: str,
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "calls": [],
        }

        def _fake(
            argv: list[str] | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            state["calls"].append(
                {
                    "argv": [] if argv is None else list(argv),
                    "args": args,
                    "kwargs": kwargs,
                }
            )

        monkeypatch.setattr(module, name, _fake)
        state["callable"] = _fake
        return state

    return _patch


@pytest.fixture
def assert_stage1_layout() -> Callable[[Path], None]:
    def _assert(run_dir: Path) -> None:
        artifacts_dir = run_dir / "artifacts"
        holdout_dir = artifacts_dir / "holdout"
        manifest_path = run_dir / "manifest.json"
        assert manifest_path.exists()

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        city = manifest["city"]

        required = [
            manifest_path,
            run_dir / f"{city}_01_raw.csv",
            run_dir / f"{city}_02_clean.csv",
            run_dir / f"{city}_03_scaled.csv",
            artifacts_dir / "train_inputs.npz",
            artifacts_dir / "train_targets.npz",
            artifacts_dir / "val_inputs.npz",
            artifacts_dir / "val_targets.npz",
            artifacts_dir / "test_inputs.npz",
            artifacts_dir / "test_targets.npz",
            artifacts_dir / f"{city}_main_scaler.joblib",
            artifacts_dir / f"{city}_coord_scaler.joblib",
            artifacts_dir / f"{city}_ohe_lithology.joblib",
            artifacts_dir
            / f"{city}_ohe_lithology_class.joblib",
            holdout_dir / "train_groups.csv",
            holdout_dir / "val_groups.csv",
            holdout_dir / "test_groups.csv",
        ]

        missing = [str(p) for p in required if not p.exists()]
        assert not missing, missing

    return _assert


def make_city_panel(
    *,
    city: str = DEFAULT_CITY,
    n_groups: int = 10,
    start_year: int = 2015,
    end_year: int = 2025,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    rng = rng or np.random.default_rng(RNG_SEED)
    years = np.arange(start_year, end_year + 1, dtype=int)
    base_lon = 113.40
    base_lat = 22.15
    rows: list[dict[str, Any]] = []

    for gid in range(n_groups):
        lon = base_lon + 0.01 * (gid % 5)
        lat = base_lat + 0.01 * (gid // 5)
        lith = LITHOLOGIES[gid % len(LITHOLOGIES)]
        lith_cls = LITHO_CLASSES[gid % len(LITHO_CLASSES)]
        z_surf_m = 5.0 + 0.25 * gid
        thick = 8.0 + 0.8 * gid
        urban = min(0.12 + 0.06 * gid, 0.98)
        cum = 0.0

        for idx, year in enumerate(years):
            rain = 1450.0 + 35.0 * idx + 7.5 * gid
            gwl = 1.2 + 0.08 * idx + 0.03 * gid
            step = 1.0 + 0.12 * idx + 0.04 * gid
            step += float(rng.normal(0.0, 0.02))
            step = max(step, 0.05)
            cum += step

            cens = bool(thick >= 30.0)
            eff = min(thick, 30.0)
            head = z_surf_m - gwl

            rows.append(
                {
                    "longitude": lon,
                    "latitude": lat,
                    "year": int(year),
                    "lithology": lith,
                    "GWL": gwl,
                    "rainfall_mm": rain,
                    "soil_thickness": thick,
                    "normalized_urban_load_proxy": urban,
                    "subsidence": step,
                    "subsidence_cum": cum,
                    "city": city,
                    "lithology_class": lith_cls,
                    "GWL_depth_bgs": gwl,
                    "GWL_depth_bgs_m": gwl,
                    "GWL_depth_bgs_z": (gwl - 2.0) / 0.8,
                    "soil_thickness_censored": cens,
                    "soil_thickness_imputed": thick,
                    "soil_thickness_eff": eff,
                    "urban_load_global": urban,
                    "z_surf": z_surf_m,
                    "z_surf_m": z_surf_m,
                    "head_m": head,
                }
            )

    df = pd.DataFrame.from_records(rows)
    return df.loc[:, list(RAW_PANEL_COLUMNS)].copy()


def build_natcom_config(
    *,
    data_dir: str | Path,
    base_output_dir: str | Path,
    city: str,
    model: str,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    out_dir = Path(base_output_dir)
    big_fn = (
        f"{city}_final_main_std.harmonized."
        "cleaned.with_zsurf.csv"
    )
    small_fn = f"{city}_2000.cleaned.with_zsurf.csv"

    return {
        "CITY_NAME": city,
        "MODEL_NAME": model,
        "DATA_DIR": str(data_dir),
        "DATASET_VARIANT": "with_zsurf",
        "BIG_FN_TEMPLATE": (
            "{city}_final_main_std.harmonized."
            "cleaned.{variant}.csv"
        ),
        "SMALL_FN_TEMPLATE": (
            "{city}_2000.cleaned.{variant}.csv"
        ),
        "BIG_FN": big_fn,
        "SMALL_FN": small_fn,
        "ALL_CITIES_PARQUET": "natcom_all_cities.parquet",
        "TRAIN_END_YEAR": TRAIN_END_YEAR,
        "FORECAST_START_YEAR": FORECAST_START_YEAR,
        "FORECAST_HORIZON_YEARS": FORECAST_HORIZON,
        "TIME_STEPS": TIME_STEPS,
        "MODE": "tft_like",
        "TIME_COL": "year",
        "LON_COL": "longitude",
        "LAT_COL": "latitude",
        "SUBSIDENCE_COL": "subsidence_cum",
        "H_FIELD_COL_NAME": "soil_thickness",
        "GWL_COL": "GWL_depth_bgs_m",
        "GWL_KIND": "depth_bgs",
        "GWL_SIGN": "down_positive",
        "USE_HEAD_PROXY": False,
        "Z_SURF_COL": "z_surf_m",
        "INCLUDE_Z_SURF_AS_STATIC": True,
        "HEAD_COL": "head_m",
        "GWL_DYN_INDEX": 0,
        "NORMALIZE_COORDS": True,
        "KEEP_COORDS_RAW": False,
        "SHIFT_RAW_COORDS": True,
        "SCALE_H_FIELD": False,
        "SCALE_GWL": False,
        "SCALE_Z_SURF": False,
        "SUBSIDENCE_KIND": "cumulative",
        "OPTIONAL_NUMERIC_FEATURES": [
            (
                "rainfall_mm",
                "rainfall",
                "rain_mm",
                "precip_mm",
            ),
            (
                "urban_load_global",
                "normalized_density",
                "urban_load",
            ),
        ],
        "OPTIONAL_CATEGORICAL_FEATURES": [
            ("lithology", "geology"),
            "lithology_class",
        ],
        "ALREADY_NORMALIZED_FEATURES": [
            "urban_load_global",
        ],
        "FUTURE_DRIVER_FEATURES": [
            (
                "rainfall_mm",
                "rainfall",
                "rain_mm",
                "precip_mm",
            )
        ],
        "DYNAMIC_FEATURE_NAMES": None,
        "FUTURE_FEATURE_NAMES": None,
        "CENSORING_SPECS": [
            {
                "col": "soil_thickness",
                "direction": "right",
                "cap": 30.0,
                "tol": 1e-6,
                "flag_suffix": "_censored",
                "eff_suffix": "_eff",
                "eff_mode": "clip",
                "eps": 0.02,
                "impute": {
                    "by": ["year"],
                    "func": "median",
                },
                "flag_threshold": 0.5,
            }
        ],
        "INCLUDE_CENSOR_FLAGS_AS_DYNAMIC": True,
        "INCLUDE_CENSOR_FLAGS_AS_FUTURE": False,
        "USE_EFFECTIVE_H_FIELD": True,
        "BUILD_FUTURE_NPZ": False,
        "SPLIT_SEED": 42,
        "VAL_FRAC": 0.2,
        "TEST_FRAC": 0.2,
        "TRACK_AUX_METRICS": False,
        "COORD_MODE": "degrees",
        "COORD_SRC_EPSG": 4326,
        "COORD_TARGET_EPSG": 32649,
        "BASE_OUTPUT_DIR": str(out_dir),
    }


def build_stage1_artifact_tree(
    *,
    tmp_path: Path,
    panel: pd.DataFrame,
    city: str,
    model: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    run_dir = (
        Path(config["BASE_OUTPUT_DIR"])
        / f"{city}_{model}_stage1"
    )
    artifacts_dir = run_dir / "artifacts"
    holdout_dir = artifacts_dir / "holdout"

    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)

    raw_df = panel.copy()
    clean_df = panel.sort_values(
        ["longitude", "latitude", "year"]
    ).reset_index(drop=True)
    proc = add_processed_columns(clean_df)
    proc, static_features = add_static_ohe(proc)

    groups = build_group_splits(proc)
    arrays = build_sequence_arrays(
        df=proc,
        static_features=static_features,
        train_groups=groups["train"],
        val_groups=groups["val"],
        test_groups=groups["test"],
    )

    raw_csv = run_dir / f"{city}_01_raw.csv"
    clean_csv = run_dir / f"{city}_02_clean.csv"
    scaled_csv = run_dir / f"{city}_03_scaled.csv"
    raw_df.to_csv(raw_csv, index=False)
    clean_df.to_csv(clean_csv, index=False)
    proc.to_csv(scaled_csv, index=False)

    main_scaler_path = artifacts_dir / f"{city}_main_scaler.joblib"
    coord_scaler_path = (
        artifacts_dir / f"{city}_coord_scaler.joblib"
    )
    ohe_lith_path = (
        artifacts_dir / f"{city}_ohe_lithology.joblib"
    )
    ohe_class_path = (
        artifacts_dir
        / f"{city}_ohe_lithology_class.joblib"
    )
    seq_path = (
        artifacts_dir
        / f"{city}_train_sequences_T{TIME_STEPS}"
        f"_H{FORECAST_HORIZON}.joblib"
    )

    save_stage1_helpers(
        df=proc,
        main_scaler_path=main_scaler_path,
        coord_scaler_path=coord_scaler_path,
        ohe_lith_path=ohe_lith_path,
        ohe_class_path=ohe_class_path,
        seq_path=seq_path,
        arrays=arrays,
        static_features=static_features,
    )

    npz_paths = save_npz_artifacts(
        artifacts_dir=artifacts_dir,
        arrays=arrays,
    )
    holdout_paths = save_holdout_groups(
        holdout_dir=holdout_dir,
        train_groups=groups["train"],
        val_groups=groups["val"],
        test_groups=groups["test"],
    )

    manifest = build_manifest(
        city=city,
        model=model,
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        static_features=static_features,
        arrays=arrays,
        npz_paths=npz_paths,
        holdout_paths=holdout_paths,
        main_scaler_path=main_scaler_path,
        coord_scaler_path=coord_scaler_path,
        ohe_lith_path=ohe_lith_path,
        ohe_class_path=ohe_class_path,
        seq_path=seq_path,
        proc=proc,
        groups=groups,
    )
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return {
        "run_dir": run_dir,
        "artifacts_dir": artifacts_dir,
        "holdout_dir": holdout_dir,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "raw_df": raw_df,
        "clean_df": clean_df,
        "scaled_df": proc,
        "config": config,
        "arrays": arrays,
        "group_splits": groups,
    }


def add_processed_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mean_lat = float(out["latitude"].mean())
    x_scale = 111_320.0 * math.cos(math.radians(mean_lat))
    y_scale = 110_540.0

    out["year_numeric_coord"] = out["year"].astype(float)
    out["x_m"] = (
        out["longitude"] - out["longitude"].min()
    ) * x_scale
    out["y_m"] = (
        out["latitude"] - out["latitude"].min()
    ) * y_scale

    out["subsidence_cum__si"] = out["subsidence_cum"]
    out["GWL_depth_bgs_m__si"] = out["GWL_depth_bgs_m"]
    out["head_m__si"] = out["head_m"]
    out["soil_thickness_eff__si"] = out[
        "soil_thickness_eff"
    ]
    out["z_surf_m__si"] = out["z_surf_m"]
    return out


def add_static_ohe(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    ohe_lith = make_ohe()
    lith = ohe_lith.fit_transform(out[["lithology"]])
    lith_cols = list(
        ohe_lith.get_feature_names_out(["lithology"])
    )

    ohe_cls = make_ohe()
    cls = ohe_cls.fit_transform(out[["lithology_class"]])
    cls_cols = list(
        ohe_cls.get_feature_names_out([
            "lithology_class"
        ])
    )

    for idx, col in enumerate(lith_cols):
        out[col] = lith[:, idx]
    for idx, col in enumerate(cls_cols):
        out[col] = cls[:, idx]

    static_features = lith_cols + cls_cols + ["z_surf_m__si"]
    return out, static_features


def build_group_splits(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    groups = (
        df[["longitude", "latitude"]]
        .drop_duplicates()
        .sort_values(["longitude", "latitude"])
        .reset_index(drop=True)
    )
    n = len(groups)
    n_train = max(1, int(round(n * 0.6)))
    n_val = max(1, int(round(n * 0.2)))
    n_val = min(n_val, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_train = max(1, n_train - 1)

    train = groups.iloc[:n_train].copy()
    val = groups.iloc[n_train : n_train + n_val].copy()
    test = groups.iloc[n_train + n_val :].copy()
    return {"train": train, "val": val, "test": test}


def build_sequence_arrays(
    *,
    df: pd.DataFrame,
    static_features: list[str],
    train_groups: pd.DataFrame,
    val_groups: pd.DataFrame,
    test_groups: pd.DataFrame,
) -> dict[str, Any]:
    frames = {
        "train": train_groups,
        "val": val_groups,
        "test": test_groups,
    }
    arrays: dict[str, Any] = {}
    for split, groups in frames.items():
        inputs, targets = make_split_arrays(
            df=df,
            group_df=groups,
            static_features=static_features,
        )
        arrays[f"{split}_inputs"] = inputs
        arrays[f"{split}_targets"] = targets
    return arrays


def make_split_arrays(
    *,
    df: pd.DataFrame,
    group_df: pd.DataFrame,
    static_features: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    static_rows: list[np.ndarray] = []
    dynamic_rows: list[np.ndarray] = []
    future_rows: list[np.ndarray] = []
    coords_rows: list[np.ndarray] = []
    h_rows: list[np.ndarray] = []
    subs_rows: list[np.ndarray] = []
    gwl_rows: list[np.ndarray] = []

    for row in group_df.itertuples(index=False):
        mask = (
            (df["longitude"] == row.longitude)
            & (df["latitude"] == row.latitude)
        )
        g = df.loc[mask].sort_values("year")
        hist = g[
            g["year"].between(
                TRAIN_END_YEAR - TIME_STEPS + 1,
                TRAIN_END_YEAR,
            )
        ]
        fut = g[
            g["year"].between(
                FORECAST_START_YEAR,
                FORECAST_START_YEAR + FORECAST_HORIZON - 1,
            )
        ]
        full_fut = g[
            g["year"].between(
                TRAIN_END_YEAR - TIME_STEPS + 1,
                FORECAST_START_YEAR
                + FORECAST_HORIZON
                - 1,
            )
        ]

        if len(hist) != TIME_STEPS:
            continue
        if len(fut) != FORECAST_HORIZON:
            continue
        if len(full_fut) != TIME_STEPS + FORECAST_HORIZON:
            continue

        static_rows.append(
            hist.iloc[0][static_features].to_numpy(
                dtype=np.float32,
            )
        )
        dynamic_rows.append(
            hist.loc[:, DYNAMIC_FEATURES].to_numpy(
                dtype=np.float32,
            )
        )
        future_rows.append(
            full_fut.loc[:, FUTURE_FEATURES].to_numpy(
                dtype=np.float32,
            )
        )
        coords_rows.append(
            fut.loc[
                :, ["year_numeric_coord", "x_m", "y_m"]
            ].to_numpy(dtype=np.float32)
        )
        h_rows.append(
            fut.loc[:, ["soil_thickness_eff__si"]].to_numpy(
                dtype=np.float32,
            )
        )
        subs_rows.append(
            fut.loc[:, ["subsidence_cum__si"]].to_numpy(
                dtype=np.float32,
            )
        )
        gwl_rows.append(
            fut.loc[:, ["head_m__si"]].to_numpy(
                dtype=np.float32,
            )
        )

    inputs = {
        "static_features": np.stack(static_rows),
        "dynamic_features": np.stack(dynamic_rows),
        "future_features": np.stack(future_rows),
        "coords": np.stack(coords_rows),
        "H_field": np.stack(h_rows),
    }
    targets = {
        "subs_pred": np.stack(subs_rows),
        "gwl_pred": np.stack(gwl_rows),
    }
    return inputs, targets


def save_stage1_helpers(
    *,
    df: pd.DataFrame,
    main_scaler_path: Path,
    coord_scaler_path: Path,
    ohe_lith_path: Path,
    ohe_class_path: Path,
    seq_path: Path,
    arrays: dict[str, Any],
    static_features: list[str],
) -> None:
    main_scaler = MinMaxScaler()
    main_scaler.fit(
        df[[
            "rainfall_mm",
            "soil_thickness_censored",
        ]].astype(float)
    )
    joblib.dump(main_scaler, main_scaler_path)

    coord_scaler = MinMaxScaler()
    coord_scaler.fit(
        df[["year_numeric_coord", "x_m", "y_m"]]
        .astype(float)
    )
    joblib.dump(coord_scaler, coord_scaler_path)

    lith_ohe = make_ohe()
    lith_ohe.fit(df[["lithology"]])
    joblib.dump(lith_ohe, ohe_lith_path)

    class_ohe = make_ohe()
    class_ohe.fit(df[["lithology_class"]])
    joblib.dump(class_ohe, ohe_class_path)

    sequence_payload = {
        "train_inputs": arrays["train_inputs"],
        "train_targets": arrays["train_targets"],
        "static_features": static_features,
    }
    joblib.dump(sequence_payload, seq_path)


def save_npz_artifacts(
    *,
    artifacts_dir: Path,
    arrays: dict[str, Any],
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        inp = artifacts_dir / f"{split}_inputs.npz"
        tgt = artifacts_dir / f"{split}_targets.npz"
        np.savez(inp, **arrays[f"{split}_inputs"])
        np.savez(tgt, **arrays[f"{split}_targets"])
        out[f"{split}_inputs_npz"] = inp
        out[f"{split}_targets_npz"] = tgt
    return out


def save_holdout_groups(
    *,
    holdout_dir: Path,
    train_groups: pd.DataFrame,
    val_groups: pd.DataFrame,
    test_groups: pd.DataFrame,
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for split, frame in {
        "train": train_groups,
        "val": val_groups,
        "test": test_groups,
    }.items():
        path = holdout_dir / f"{split}_groups.csv"
        frame.to_csv(path, index=False)
        out[f"{split}_groups_csv"] = path
    return out


def build_manifest(
    *,
    city: str,
    model: str,
    run_dir: Path,
    artifacts_dir: Path,
    static_features: list[str],
    arrays: dict[str, Any],
    npz_paths: dict[str, Path],
    holdout_paths: dict[str, Path],
    main_scaler_path: Path,
    coord_scaler_path: Path,
    ohe_lith_path: Path,
    ohe_class_path: Path,
    seq_path: Path,
    proc: pd.DataFrame,
    groups: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    scaler_cols = ["rainfall_mm", "soil_thickness_censored"]
    coord_ranges = {
        "t": float(
            proc["year_numeric_coord"].max()
            - proc["year_numeric_coord"].min()
        ),
        "x": float(proc["x_m"].max() - proc["x_m"].min()),
        "y": float(proc["y_m"].max() - proc["y_m"].min()),
    }

    return {
        "schema_version": "3.2",
        "timestamp": "2026-03-30 00:00:00",
        "city": city,
        "model": model,
        "stage": "stage1",
        "config": {
            "TIME_STEPS": TIME_STEPS,
            "FORECAST_HORIZON_YEARS": FORECAST_HORIZON,
            "MODE": "tft_like",
            "TRAIN_END_YEAR": TRAIN_END_YEAR,
            "FORECAST_START_YEAR": FORECAST_START_YEAR,
            "cols": {
                "time": "year",
                "lon": "longitude",
                "lat": "latitude",
                "time_numeric": "year_numeric_coord",
                "time_used": "year_numeric_coord",
                "x_base": "x_m",
                "y_base": "y_m",
                "x_used": "x_m",
                "y_used": "y_m",
                "subs_raw": "subsidence_cum",
                "subs_model": "subsidence_cum__si",
                "depth_raw": "GWL_depth_bgs_m",
                "head_raw": "head_m",
                "depth_model": "GWL_depth_bgs_m__si",
                "head_model": "head_m__si",
                "h_field_raw": "soil_thickness_eff",
                "h_field_model": "soil_thickness_eff__si",
                "z_surf_raw": "z_surf_m",
                "z_surf_static": "z_surf_m__si",
            },
            "features": {
                "static": static_features,
                "dynamic": list(DYNAMIC_FEATURES),
                "future": list(FUTURE_FEATURES),
                "group_id_cols": [
                    "longitude",
                    "latitude",
                ],
            },
            "indices": {
                "gwl_dyn_index": 0,
                "subs_dyn_index": 1,
                "gwl_dyn_name": DYNAMIC_FEATURES[0],
                "z_surf_static_index": (
                    len(static_features) - 1
                ),
            },
            "conventions": {
                "gwl_kind": "depth_bgs",
                "gwl_sign": "down_positive",
                "use_head_proxy": False,
                "time_units": "year",
                "gwl_driver_kind": "depth",
                "gwl_target_kind": "head",
                "gwl_driver_sign": "down_positive",
                "gwl_target_sign": "up_positive",
            },
            "scaler_info": {
                name: {
                    "scaler_path": str(main_scaler_path),
                    "all_features": scaler_cols,
                    "idx": idx,
                }
                for idx, name in enumerate(scaler_cols)
            },
            "scaling_kwargs": {
                "subs_scale_si": 1.0,
                "subs_bias_si": 0.0,
                "head_scale_si": 1.0,
                "head_bias_si": 0.0,
                "H_scale_si": 1.0,
                "H_bias_si": 0.0,
                "subsidence_kind": "cumulative",
                "allow_subs_residual": True,
                "coords_normalized": True,
                "coord_order": ["t", "x", "y"],
                "coord_ranges": coord_ranges,
                "coord_mode": "degrees",
                "coord_src_epsg": 4326,
                "coord_target_epsg": 32649,
                "coord_epsg_used": 32649,
                "coords_in_degrees": False,
                "cons_residual_units": "second",
                "gw_residual_units": "second",
                "dt_min_units": 1e-6,
                "Q_wrt_normalized_time": False,
                "Q_in_si": False,
                "Q_in_per_second": False,
                "Q_kind": "per_volume",
                "Q_length_in_si": False,
                "drainage_mode": "double",
                "clip_global_norm": 5.0,
                "mv_prior_mode": "calibrate",
                "mv_weight": 0.001,
                "mv_schedule_unit": "epoch",
                "mv_delay_epochs": 1,
                "mv_warmup_epochs": 2,
                "track_aux_metrics": False,
                "gwl_dyn_index": 0,
                "subs_dyn_index": 1,
                "gwl_dyn_name": DYNAMIC_FEATURES[0],
                "z_surf_static_index": (
                    len(static_features) - 1
                ),
                "gwl_col": DYNAMIC_FEATURES[0],
                "gwl_dyn_col": DYNAMIC_FEATURES[0],
                "gwl_target_col": "head_m__si",
                "subs_model_col": "subsidence_cum__si",
                "z_surf_col": "z_surf_m__si",
            },
            "feature_registry": {
                "optional_numeric_declared": [
                    [
                        "rainfall_mm",
                        "rainfall",
                        "rain_mm",
                        "precip_mm",
                    ],
                    [
                        "urban_load_global",
                        "normalized_density",
                        "urban_load",
                    ],
                ],
                "optional_categorical_declared": [
                    ["lithology", "geology"],
                    "lithology_class",
                ],
                "already_normalized": [
                    "urban_load_global",
                ],
                "future_drivers_declared": [
                    "rainfall_mm",
                ],
                "resolved_optional_numeric": [
                    "rainfall_mm",
                    "urban_load_global",
                ],
                "resolved_optional_categorical": [
                    "lithology",
                    "lithology_class",
                ],
            },
            "censoring": {
                "specs": [
                    {
                        "col": "soil_thickness",
                        "direction": "right",
                        "cap": 30.0,
                        "tol": 1e-6,
                        "flag_suffix": "_censored",
                        "eff_suffix": "_eff",
                        "eff_mode": "clip",
                        "eps": 0.02,
                        "impute": {
                            "by": ["year"],
                            "func": "median",
                        },
                        "flag_threshold": 0.5,
                    }
                ],
                "report": {
                    "soil_thickness": {
                        "direction": "right",
                        "cap": 30.0,
                        "flag_col": (
                            "soil_thickness_censored"
                        ),
                        "eff_col": "soil_thickness_eff",
                        "eff_mode": "clip",
                        "censored_rate": float(
                            proc[
                                "soil_thickness_censored"
                            ].mean()
                        ),
                    }
                },
                "use_effective_h_field": True,
                "flags_as_dynamic": True,
                "flags_as_future": False,
            },
            "units_provenance": {
                "subs_unit_to_si_applied_stage1": 1.0,
                "thickness_unit_to_si_applied_stage1": 1.0,
                "head_unit_to_si_assumed_stage1": 1.0,
                "z_surf_unit_to_si_assumed_stage1": 1.0,
            },
            "holdout": {
                "strategy": "random",
                "seed": 42,
                "val_frac": 0.2,
                "test_frac": 0.2,
                "group_cols": [
                    "longitude",
                    "latitude",
                ],
                "coord_cols_for_blocking": {
                    "x_col": "x_m",
                    "y_col": "y_m",
                    "epsg_used": 32649,
                },
                "block_size_m": None,
                "requirements": {
                    "time_col": "year",
                    "train_end_year": TRAIN_END_YEAR,
                    "forecast_start_year": (
                        FORECAST_START_YEAR
                    ),
                    "time_steps": TIME_STEPS,
                    "horizon": FORECAST_HORIZON,
                    "years_required_for_train_windows": (
                        TIME_STEPS + FORECAST_HORIZON
                    ),
                    "note": (
                        "Synthetic fixture for CLI smoke"
                        " tests."
                    ),
                },
                "group_counts": {
                    "valid_for_train": int(len(proc)),
                    "valid_for_forecast": int(len(proc)),
                    "kept_for_processing": int(len(proc)),
                    "train_groups": int(
                        len(groups["train"])
                    ),
                    "val_groups": int(len(groups["val"])),
                    "test_groups": int(
                        len(groups["test"])
                    ),
                },
                "row_counts_hist": {
                    "train_rows": int(
                        len(groups["train"]) * TIME_STEPS
                    ),
                    "val_rows": int(
                        len(groups["val"]) * TIME_STEPS
                    ),
                    "test_rows": int(
                        len(groups["test"]) * TIME_STEPS
                    ),
                },
                "sequence_counts": {
                    "train_seq": int(
                        arrays["train_inputs"][
                            "coords"
                        ].shape[0]
                    ),
                    "val_seq": int(
                        arrays["val_inputs"][
                            "coords"
                        ].shape[0]
                    ),
                    "test_seq": int(
                        arrays["test_inputs"][
                            "coords"
                        ].shape[0]
                    ),
                },
                "artifacts": {
                    key: str(path)
                    for key, path in holdout_paths.items()
                },
            },
        },
        "artifacts": {
            "csv": {
                "raw": str(run_dir / f"{city}_01_raw.csv"),
                "clean": str(run_dir / f"{city}_02_clean.csv"),
                "scaled": str(
                    run_dir / f"{city}_03_scaled.csv"
                ),
            },
            "encoders": {
                "ohe": {
                    "lithology": str(ohe_lith_path),
                    "lithology_class": str(
                        ohe_class_path
                    ),
                },
                "coord_scaler": str(coord_scaler_path),
                "main_scaler": str(main_scaler_path),
                "scaled_ml_numeric_cols": [
                    "rainfall_mm",
                    "soil_thickness_censored",
                ],
            },
            "sequences": {
                "joblib_train_sequences": str(seq_path),
                "dims": {
                    "output_subsidence_dim": 1,
                    "output_gwl_dim": 1,
                },
            },
            "numpy": {
                key: str(path)
                for key, path in npz_paths.items()
            },
            "shapes": build_shapes_block(arrays),
        },
        "paths": {
            "run_dir": str(run_dir),
            "artifacts_dir": str(artifacts_dir),
        },
        "versions": {
            "python": (
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "tensorflow": "stub",
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": "fixture",
        },
    }


def build_shapes_block(
    arrays: dict[str, Any],
) -> dict[str, dict[str, list[int]]]:
    out: dict[str, dict[str, list[int]]] = {}
    for split in ("train", "val", "test"):
        out[f"{split}_inputs"] = {
            key: list(value.shape)
            for key, value in arrays[
                f"{split}_inputs"
            ].items()
        }
        out[f"{split}_targets"] = {
            key: list(value.shape)
            for key, value in arrays[
                f"{split}_targets"
            ].items()
        }
    return out


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
        )
    except TypeError:
        return OneHotEncoder(
            sparse=False,
            handle_unknown="ignore",
        )


def write_python_assignments(
    path: Path,
    config: dict[str, Any],
) -> None:
    lines = [
        "# Auto-generated by tests/conftest.py",
        "",
    ]
    for key, value in config.items():
        lines.append(f"{key} = {repr(value)}")
    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ------------------------------------------------------------------
# Optional helpers for geoprior._scripts tests.
# These are opt-in so the existing stage tests keep the exact same
# behavior.
# ------------------------------------------------------------------

def _write_placeholder_artifact(path: Path) -> Path:
    """Create a tiny placeholder file for fast plot tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext == ".png":
        import base64

        png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
            "AAAAC0lEQVR42mP8/x8AAwMCAO+a4WQAAAAASUVORK5CYII="
        )
        path.write_bytes(base64.b64decode(png_b64))
        return path

    if ext == ".svg":
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>\n',
            encoding="utf-8",
        )
        return path

    if ext == ".pdf":
        path.write_bytes(
            b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
        )
        return path

    if ext == ".eps":
        path.write_text(
            "%!PS-Adobe-3.0 EPSF-3.0\n"
            "%%BoundingBox: 0 0 1 1\n"
            "showpage\n",
            encoding="utf-8",
        )
        return path

    path.touch()
    return path


def _default_script_output_dirs(root: Path) -> dict[str, Path]:
    """Return a standard isolated output tree for script tests."""
    scripts_dir = root / "scripts"
    out = {
        "root": root,
        "scripts_dir": scripts_dir,
        "figs_dir": scripts_dir / "figs",
        "tables_dir": scripts_dir / "tables",
        "exports_dir": scripts_dir / "exports",
        "results_dir": scripts_dir / "results",
        "data_dir": scripts_dir / "data",
    }
    for path in out.values():
        path.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def scripts_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    """
    Isolated workspace for ``geoprior._scripts`` tests.

    The fixture changes the working directory to a temporary root and
    pre-creates the common ``scripts/`` output folders. Existing tests
    are unaffected unless they opt into this fixture.
    """
    root = tmp_path / "scripts-workspace"
    paths = _default_script_output_dirs(root)
    monkeypatch.chdir(root)
    return paths


@pytest.fixture
def patch_script_output_roots(
    monkeypatch: pytest.MonkeyPatch,
    scripts_workspace: dict[str, Path],
) -> dict[str, Path]:
    """
    Route relative script outputs into the temporary workspace.

    This patches ``geoprior._scripts.config.resolve_user_artifact_path``
    when available. Absolute output paths are preserved.
    """
    try:
        import importlib

        cfg = importlib.import_module("geoprior._scripts.config")
    except Exception:
        return scripts_workspace

    def _resolve(
        out: str | os.PathLike[str],
        *,
        kind: str | None = None,
    ) -> Path:
        path = Path(os.fspath(out)).expanduser()
        if path.is_absolute():
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        kind_key = str(kind or "result").strip().lower()
        if kind_key in {"fig", "figure", "plot"}:
            base = scripts_workspace["figs_dir"]
        elif kind_key in {"table", "tables", "summary"}:
            base = scripts_workspace["tables_dir"]
        elif kind_key in {"export", "exports", "geojson"}:
            base = scripts_workspace["exports_dir"]
        elif kind_key in {"data", "dataset", "input"}:
            base = scripts_workspace["data_dir"]
        else:
            base = scripts_workspace["results_dir"]

        out_path = base / path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

    monkeypatch.setattr(
        cfg,
        "resolve_user_artifact_path",
        _resolve,
        raising=False,
    )
    return scripts_workspace


@pytest.fixture
def fast_script_figures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Replace expensive Matplotlib file rendering with placeholders.

    Use this only in tests that assert file creation or dispatch,
    not in tests that inspect real pixel content.
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    def _fake_savefig(
        self,
        fname,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        del self, args, kwargs
        _write_placeholder_artifact(Path(os.fspath(fname)))

    monkeypatch.setattr(Figure, "savefig", _fake_savefig)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)


@pytest.fixture
def script_test_env(
    patch_script_output_roots: dict[str, Path],
) -> dict[str, Path]:
    """
    Convenience fixture for most ``geoprior._scripts`` tests.

    It provides an isolated workspace and patched relative output
    resolution, while leaving figure rendering unchanged.
    """
    return patch_script_output_roots


@pytest.fixture
def collect_script_outputs(
    scripts_workspace: dict[str, Path],
) -> Callable[..., list[Path]]:
    """Collect files written inside the isolated script workspace."""

    def _collect(*patterns: str) -> list[Path]:
        pats = patterns or ("*",)
        seen: dict[str, Path] = {}
        for key in (
            "figs_dir",
            "tables_dir",
            "exports_dir",
            "results_dir",
            "data_dir",
        ):
            base = scripts_workspace[key]
            for pat in pats:
                for path in base.rglob(pat):
                    if path.is_file():
                        seen[str(path.resolve())] = path.resolve()
        return sorted(seen.values())

    return _collect


@pytest.fixture
def assert_script_outputs(
    collect_script_outputs: Callable[..., list[Path]],
) -> Callable[..., list[Path]]:
    """Assert that one or more script outputs were created."""

    def _assert(*patterns: str) -> list[Path]:
        found = collect_script_outputs(*patterns)
        assert found, (
            "No script outputs were created"
            f" for patterns={patterns or ('*',)}."
        )
        return found

    return _assert
