# SPDX-License-Identifier: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>
# Adapted from: earthai-tech/fusionlab-learn — https://github.com/earthai-tech/fusionlab-learn
# Modified for GeoPrior-v3 API.
r"""Metric registry helpers for evaluation workflows."""

import importlib
from collections.abc import Callable

# ---------------------------------------------------------------------
# Canonical metric registry: public name -> fully-qualified path
# ---------------------------------------------------------------------
_METRIC_PATHS: dict[str, str] = {
    # Coverage / sharpness / intervals
    "coverage_score": "geoprior.metrics.coverage_score",
    "weighted_interval_score": "geoprior.metrics.weighted_interval_score",
    "mean_interval_width_score": "geoprior.metrics.mean_interval_width_score",
    # Calibration / distribution
    "quantile_calibration_error": "geoprior.metrics.quantile_calibration_error",
    "continuous_ranked_probability_score": "geoprior.metrics.continuous_ranked_probability_score",
    # Dynamics / temporal
    "prediction_stability_score": "geoprior.metrics.prediction_stability_score",
    "time_weighted_mean_absolute_error": "geoprior.metrics.time_weighted_mean_absolute_error",
    "time_weighted_accuracy_score": "geoprior.metrics.time_weighted_accuracy_score",
    "time_weighted_interval_score": "geoprior.metrics.time_weighted_interval_score",
    # Other
    "theils_u_score": "geoprior.metrics.theils_u_score",
    # Optional: PIT (user is expected to implement this)
    # e.g. geoprior.metrics.probability_inverse_transform
    "probability_inverse_transform": "geoprior.metrics.probability_inverse_transform",
}

# ---------------------------------------------------------------------
# Alias map: common abbreviations / short names -> canonical key
# (after normalization)
# ---------------------------------------------------------------------
_ALIAS_MAP: dict[str, str] = {
    # Prediction stability
    "pss": "prediction_stability_score",
    "predictionstability": "prediction_stability_score",
    "prediction_stability": "prediction_stability_score",
    # Coverage / sharpness
    "coverage": "coverage_score",
    "cov": "coverage_score",
    "sharpness": "mean_interval_width_score",
    "miw": "mean_interval_width_score",
    # Interval scores
    "wis": "weighted_interval_score",
    "weightedinterval": "weighted_interval_score",
    # Time-weighted errors / scores
    "twmae": "time_weighted_mean_absolute_error",
    "twae": "time_weighted_mean_absolute_error",
    "twa": "time_weighted_accuracy_score",
    "twacc": "time_weighted_accuracy_score",
    "twis": "time_weighted_interval_score",
    "twinterval": "time_weighted_interval_score",
    # CRPS and PIT
    "crps": "continuous_ranked_probability_score",
    "pit": "probability_inverse_transform",
}


def _normalize_name(name: str) -> str:
    """
    Normalize a metric identifier to a compact, case-insensitive key.

    Example
    -------
    'PSS', 'prediction_stability', 'Prediction-Stability' → 'predictionstability'
    """
    if not isinstance(name, str):
        raise ValueError(
            f"Metric name must be a string, got {type(name)!r}"
        )
    # lower + strip whitespace + drop separators
    key = name.strip().lower()
    for ch in (" ", "_", "-"):
        key = key.replace(ch, "")
    return key


def _canonical_name(name: str) -> str:
    """
    Resolve a user-facing metric name or alias to a canonical registry key.

    Resolution order:
    1. Try direct match in _METRIC_PATHS (exact key).
    2. Normalize + look up in _ALIAS_MAP.
    3. Fall back to the original name (so you can still use the
       full canonical key like 'prediction_stability_score').
    """
    if name in _METRIC_PATHS:
        return name

    norm = _normalize_name(name)
    if norm in _ALIAS_MAP:
        return _ALIAS_MAP[norm]

    # Fall back: maybe user already passed canonical key
    return name


def get_metric(name: str) -> Callable:
    """
    Return the metric function for `name`, lazily importing its module.

    Parameters
    ----------
    name : str
        Metric identifier. This can be:
        - a canonical name (e.g. 'prediction_stability_score'),
        - or a short alias (e.g. 'pss', 'crps', 'sharpness', 'coverage').

    Aliases
    -------
    Some common aliases (case-insensitive, '_'/'-' ignored):

    - 'pss'              -> 'prediction_stability_score'
    - 'crps'             -> 'continuous_ranked_probability_score'
    - 'sharpness', 'miw' -> 'mean_interval_width_score'
    - 'coverage', 'cov'  -> 'coverage_score'
    - 'twmae', 'twae'    -> 'time_weighted_mean_absolute_error'
    - 'twa', 'twacc'     -> 'time_weighted_accuracy_score'
    - 'twis', 'twinterval'
                         -> 'time_weighted_interval_score'
    - 'pit'              -> 'probability_inverse_transform'
                           (expects a metric function implemented at
                           ``geoprior.metrics.probability_inverse_transform``)

    Returns
    -------
    Callable
        The metric function corresponding to `name`.

    Raises
    ------
    ValueError
        If `name` is unknown (not in the registry or aliases), or if the
        resolved metric function cannot be imported. In that case, the
        caller is expected to provide the metric function explicitly.
    """
    original = name
    canonical = _canonical_name(name)

    if canonical not in _METRIC_PATHS:
        valid = ", ".join(sorted(_METRIC_PATHS.keys()))
        aliases = ", ".join(sorted(set(_ALIAS_MAP.keys())))
        raise ValueError(
            f"Unknown metric '{original}'.\n"
            f"  Known canonical names: {valid}\n"
            f"  Known aliases: {aliases}\n"
            "If you are using a custom metric, please pass the function "
            "object directly instead of a string."
        )

    path = _METRIC_PATHS[canonical]
    module_name, func_name = path.rsplit(".", 1)

    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        raise ValueError(
            f"Failed to import module '{module_name}' for metric '{original}' "
            f"(resolved as '{canonical}'). Underlying error: {exc}\n"
            "Either implement this metric in geoprior.metrics, or pass a "
            "custom metric function directly."
        ) from exc

    try:
        func = getattr(mod, func_name)
    except AttributeError as exc:
        raise ValueError(
            f"Metric '{original}' resolved to '{path}', but '{func_name}' "
            f"was not found in module '{module_name}'.\n"
            "Make sure the function is implemented, or pass a custom "
            "metric function instead."
        ) from exc

    return func
