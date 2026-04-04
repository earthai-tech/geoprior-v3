# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — NATCOM config template
#
# This file is rendered by:
#     python -m scripts init-config
#
# Place it at:
#     geoprior/resources/natcom_config_template.py
#
# Notes
# -----
# - Keep placeholders only for values the bootstrap script supplies.
# - Keep the file limited to assignments and comments.
# - Avoid imports or helper code.
r"""Template configuration for NATCOM-based workflows."""

# ================================================================
# 1) CORE EXPERIMENT SETUP
# ================================================================

# ---------------------------------------------------------------
# 1.1 City / model identifiers
# ---------------------------------------------------------------
CITY_NAME = "$CITY_NAME"
MODEL_NAME = "$MODEL_NAME"


# ---------------------------------------------------------------
# 1.2 Data root and file patterns
# ---------------------------------------------------------------
DATA_DIR = "$DATA_DIR"
DATASET_VARIANT = "$DATASET_VARIANT"

BIG_FN_TEMPLATE = (
    "{city}_final_main_std.harmonized.cleaned.{variant}.csv"
)
SMALL_FN_TEMPLATE = "{city}_2000.cleaned.{variant}.csv"

BIG_FN = BIG_FN_TEMPLATE.format(
    city=CITY_NAME,
    variant=DATASET_VARIANT,
)
SMALL_FN = SMALL_FN_TEMPLATE.format(
    city=CITY_NAME,
    variant=DATASET_VARIANT,
)

ALL_CITIES_PARQUET = "$ALL_CITIES_PARQUET"


# ---------------------------------------------------------------
# 1.3 Temporal windows
# ---------------------------------------------------------------
TRAIN_END_YEAR = "$TRAIN_END_YEAR"
FORECAST_START_YEAR = "$FORECAST_START_YEAR"
FORECAST_HORIZON_YEARS = "$FORECAST_HORIZON_YEARS"
TIME_STEPS = "$TIME_STEPS"
MODE = "$MODE"


# ---------------------------------------------------------------
# 1.4 Column names and groundwater conventions
# ---------------------------------------------------------------
TIME_COL = "$TIME_COL"
LON_COL = "$LON_COL"
LAT_COL = "$LAT_COL"
SUBSIDENCE_COL = "$SUBSIDENCE_COL"
H_FIELD_COL_NAME = "soil_thickness"
GWL_COL = "$GWL_COL"

GWL_KIND = "$GWL_KIND"
GWL_SIGN = "$GWL_SIGN"
USE_HEAD_PROXY = "$USE_HEAD_PROXY"

Z_SURF_COL = "$Z_SURF_COL"
INCLUDE_Z_SURF_AS_STATIC = True
HEAD_COL = "$HEAD_COL"
GWL_DYN_INDEX = None

NORMALIZE_COORDS = True
KEEP_COORDS_RAW = "$KEEP_COORDS_RAW"
SHIFT_RAW_COORDS = True

SCALE_H_FIELD = False
SCALE_GWL = False
SCALE_Z_SURF = False

SUBSIDENCE_KIND = "cumulative"


# ================================================================
# 2) FEATURE REGISTRY (Stage-1 -> Stage-2 handshake)
# ================================================================
OPTIONAL_NUMERIC_FEATURES = ["$OPTIONAL_NUMERIC_FEATURES"]

OPTIONAL_CATEGORICAL_FEATURES = [
    ("lithology", "geology"),
    "lithology_class",
]

ALREADY_NORMALIZED_FEATURES = [
    "urban_load_global",
]

FUTURE_DRIVER_FEATURES = ["$FUTURE_DRIVER_FEATURES"]

DYNAMIC_FEATURE_NAMES = None
FUTURE_FEATURE_NAMES = None


# ================================================================
# 3) CENSORING / EFFECTIVE-THICKNESS CONFIGURATION
# ================================================================
CENSORING_SPECS = [
    {
        "col": H_FIELD_COL_NAME,
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
    },
]

INCLUDE_CENSOR_FLAGS_AS_DYNAMIC = True
INCLUDE_CENSOR_FLAGS_AS_FUTURE = False
USE_EFFECTIVE_H_FIELD = True
BUILD_FUTURE_NPZ = False


# ================================================================
# 4) HOLDOUT STRATEGY
# ================================================================
SPLIT_SEED = 42
VAL_FRAC = 0.2
TEST_FRAC = 0.1
HOLDOUT_STRATEGY = "random"
HOLDOUT_BLOCK_M = 2000.0


# ================================================================
# 5) MODEL ARCHITECTURE DEFAULTS (Stage-2)
# ================================================================
ATTENTION_LEVELS = ["cross", "hierarchical", "memory"]

EMBED_DIM = 32
HIDDEN_UNITS = "$HIDDEN_UNITS"
LSTM_UNITS = ["$LSTM_UNITS"]
ATTENTION_UNITS = "$ATTENTION_UNITS"
NUMBER_HEADS = "$NUMBER_HEADS"
DROPOUT_RATE = "$DROPOUT_RATE"
USE_BATCH_NORM = "$USE_BATCH_NORM"
USE_VSN = "$USE_VSN"
VSN_UNITS = 32

QUANTILES = [0.1, 0.5, 0.9]
SUBS_WEIGHTS = {0.1: 3.0, 0.5: 1.0, 0.9: 3.0}
GWL_WEIGHTS = {0.1: 1.5, 0.5: 1.0, 0.9: 1.5}


# ================================================================
# 6) PHYSICS + TRAINING SCHEDULE
# ================================================================
PDE_MODE_CONFIG = "$PDE_MODE_CONFIG"
PHYSICS_BASELINE_MODE = "none"
SCALE_PDE_RESIDUALS = True

LAMBDA_CONS = 1.0
LAMBDA_GW = 0.10
LAMBDA_PRIOR = 0.20
LAMBDA_SMOOTH = 0.01
LAMBDA_BOUNDS = 0.05
LAMBDA_MV = 0.01
LAMBDA_Q = 0.0

PHYSICS_WARMUP_STEPS = 1800
PHYSICS_RAMP_STEPS = 4800

OFFSET_MODE = "mul"
LAMBDA_OFFSET = 1.0
USE_LAMBDA_OFFSET_SCHEDULER = False
LAMBDA_OFFSET_UNIT = "epoch"
LAMBDA_OFFSET_WHEN = "begin"
LAMBDA_OFFSET_WARMUP = 5
LAMBDA_OFFSET_START = 0.1
LAMBDA_OFFSET_END = 1.0
LAMBDA_OFFSET_SCHEDULE = None

TRAINING_STRATEGY = "data_first"

Q_POLICY_PHYSICS_FIRST = "warmup_off"
Q_WARMUP_EPOCHS_PHYSICS_FIRST = 5
Q_RAMP_EPOCHS_PHYSICS_FIRST = 5
SUBS_RESID_POLICY_PHYSICS_FIRST = "warmup_off"
SUBS_RESID_WARMUP_EPOCHS_PHYSICS_FIRST = 5
SUBS_RESID_RAMP_EPOCHS_PHYSICS_FIRST = 5
LOSS_WEIGHT_GWL_PHYSICS_FIRST = 0.5
LAMBDA_Q_PHYSICS_FIRST = 1e-5

LOSS_WEIGHT_GWL_DATA_FIRST = 0.8
LAMBDA_Q_DATA_FIRST = 5e-4
Q_POLICY_DATA_FIRST = "warmup_off"
Q_WARMUP_EPOCHS_DATA_FIRST = 2
Q_RAMP_EPOCHS_DATA_FIRST = 6
SUBS_RESID_POLICY_DATA_FIRST = "warmup_off"
SUBS_RESID_WARMUP_EPOCHS_DATA_FIRST = 1
SUBS_RESID_RAMP_EPOCHS_DATA_FIRST = 4
LOG_Q_DIAGNOSTICS = True

PHYSICS_BOUNDS = {
    "H_min": 0.1,
    "H_max": 30.0,
    "K_min": 1e-12,
    "K_max": 1e-7,
    "Ss_min": 1e-6,
    "Ss_max": 1e-3,
    "tau_min": 7.0 * 86400.0,
    "tau_max": 300.0 * 31556952.0,
}

PHYSICS_BOUNDS_MODE = "soft"
BOUNDS_LOSS_KIND = "both"
BOUNDS_BETA = 20.0
BOUNDS_GUARD = 5.0
BOUNDS_W = 1.0
BOUNDS_INCLUDE_TAU = True
BOUNDS_TAU_W = 1.0
TIME_UNITS = "year"
DEBUG_PHYSICS_GRADS = False

CONS_DRAWDOWN_MODE = "smooth_relu"
CONS_DRAWDOWN_RULE = "ref_minus_mean"
CONS_STOP_GRAD_REF = True
CONS_DRAWDOWN_ZERO_AT_ORIGIN = False
CONS_DRAWDOWN_CLIP_MAX = None
CONS_RELU_BETA = 20.0

HEAD_UNIT_TO_SI = 1.0
SUBS_SCALE_SI = 1.0
SUBS_BIAS_SI = 0.0
HEAD_SCALE_SI = None
HEAD_BIAS_SI = None
Z_SURF_UNIT_TO_SI = 1.0
SUBS_UNIT_TO_SI = 1e-3
THICKNESS_UNIT_TO_SI = 1.0
H_FIELD_MIN_SI = 0.1
AUTO_SI_AFFINE_FROM_STAGE1 = True

COORD_MODE = "degrees"
UTM_EPSG = 32649
COORD_SRC_EPSG = 4326


# ================================================================
# 7) GEOPRIOR SCALAR PARAMETERS
# ================================================================
GEOPRIOR_INIT_MV = 1e-7
GEOPRIOR_INIT_KAPPA = 1.0
GEOPRIOR_GAMMA_W = 9810.0
GEOPRIOR_KAPPA_MODE = "kb"
GEOPRIOR_USE_EFFECTIVE_H = True
GEOPRIOR_HD_FACTOR = 0.6
GEOPRIOR_H_REF = "auto"

CONSOLIDATION_STEP_RESIDUAL_METHOD = "exact"
CONSOLIDATION_RESIDUAL_UNITS = "second"
CONS_SCALE_FLOOR = 3e-11
GW_RESIDUAL_UNITS = "second"
GW_SCALE_FLOOR = 1e-12
ALLOW_SUBS_RESIDUAL = True
DT_MIN_UNITS = 1e-6
Q_WRT_NORMALIZED_TIME = False
Q_IN_SI = False
Q_IN_PER_SECOND = False
Q_KIND = "per_volume"
Q_LENGTH_IN_SI = False
DRAINAGE_MODE = "double"
SCALING_ERROR_POLICY = "raise"
CLIP_GLOBAL_NORM = 5.0


# ================================================================
# 8) TRAINING LOOP DEFAULTS
# ================================================================
EPOCHS = "$EPOCHS"
BATCH_SIZE = "$BATCH_SIZE"
LEARNING_RATE = "$LEARNING_RATE"
VERBOSE = 1


# ================================================================
# 9) HARDWARE / RUNTIME
# ================================================================
TF_DEVICE_MODE = "auto"


# ================================================================
# 10) SCALING / BOOKKEEPING
# ================================================================
SCALES = {
    "subsidence": "$SUBS_SCALE",
    "gwl": "$GWL_SCALE",
}
AUDIT_STAGES = ["stage1", "stage2", "stage3"]

H_FIELD_COL_NAME = "$H_FIELD_COL_NAME"
INCLUDE_Z_SURF_AS_STATIC = "$INCLUDE_Z_SURF_AS_STATIC"
NORMALIZE_COORDS = "$NORMALIZE_COORDS"
SCALE_GWL = "$SCALE_GWL"
SCALE_H_FIELD = "$SCALE_H_FIELD"
SCALE_Z_SURF = "$SCALE_Z_SURF"
