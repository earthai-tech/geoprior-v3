# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# https://lkouadio.com
# Copyright (c) 2026-present
# Author: LKouadio <etanoyau@gmail.com>
r"""Geospatial utility helpers for GeoPrior workflows."""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import (
    Any,
    Literal,
)

import numpy as np
import pandas as pd

from ..core.checks import (
    check_spatial_columns,
    exist_features,
)
from ..core.handlers import columns_manager
from ..core.io import SaveFile
from ..logging import get_logger
from .generic_utils import vlog

logger = get_logger(__name__)

PathLike = str | os.PathLike[str]
DataSource = PathLike | pd.DataFrame


__all__ = [
    "augment_city_spatiotemporal_data",
    "augment_series_features",
    "augment_spatiotemporal_data",
    "generate_dummy_pinn_data",
    "interpolate_temporal_gaps",
    "resolve_spatial_columns",
    "merge_frames_to_file",
    "unpack_frames_from_file",
]


def resolve_spatial_columns(
    df, spatial_cols=None, lon_col=None, lat_col=None
):
    """
    Helper to validate and resolve spatial columns.

    Accepts either explicit lon/lat columns or a
    list of spatial_cols. Returns (lon_col, lat_col).

    - If lon_col and lat_col are both provided, they
      take precedence (warn if spatial_cols also set).
    - Else if spatial_cols is provided, it must yield
      exactly two column names.
    - Otherwise, error is raised.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame for feature checks.
    spatial_cols : list[str] or None
        Two-element list of [lon_col, lat_col].
    lon_col : str or None
        Name of longitude column.
    lat_col : str or None
        Name of latitude column.

    Returns
    -------
    (lon_col, lat_col) : tuple of str
        Validated column names for longitude and
        latitude.

    Raises
    ------
    ValueError
        If neither lon/lat nor valid spatial_cols is
        provided, or if spatial_cols len != 2.
    """
    # Case 1: explicit lon/lat
    if lon_col is not None and lat_col is not None:
        if spatial_cols:
            warnings.warn(
                "Both lon_col/lat_col and spatial_cols set;"
                " spatial_cols will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        exist_features(
            df,
            features=[lon_col, lat_col],
            name="Longitude/Latitude",
        )
        return lon_col, lat_col

    # Case 2: spatial_cols provided
    if spatial_cols:
        spatial_cols = columns_manager(
            spatial_cols, empty_as_none=False
        )
        check_spatial_columns(df, spatial_cols=spatial_cols)
        exist_features(
            df, features=spatial_cols, name="Spatial columns"
        )
        if len(spatial_cols) != 2:
            raise ValueError(
                "spatial_cols must contain exactly two"
                " column names"
            )
        lon, lat = spatial_cols
        return lon, lat

    # Neither provided
    raise ValueError(
        "Either lon_col & lat_col, or spatial_cols"
        " must be provided."
    )


@SaveFile
def interpolate_temporal_gaps(
    series_df: pd.DataFrame,
    time_col: str,
    value_cols: list[str],
    freq: str | None = None,
    method: str = "linear",
    order: int | None = None,
    fill_limit: int | None = None,
    fill_limit_direction: str = "forward",
    savefile: str | None = None,
) -> pd.DataFrame:
    r"""
    Interpolates missing values in specified columns of a time series
    DataFrame.

    This function is designed to work on a DataFrame representing a time
    series for a single spatial group (e.g., one monitoring location),
    sorted by time. If :code:`freq` is provided, the DataFrame’s index is
    first reindexed to that frequency, which can create NaN values for
    missing time steps. These NaNs, along with any pre-existing NaNs in
    :code:`value_cols`, are then interpolated.

    Let :math:`t_1 < t_2 < \dots < t_n` be the original timestamps. If
    :code:`freq` yields a new index :math:`\{t_i'\}` that includes times
    not in the original, NaNs appear at those :math:`t_i'`. Then for each
    column :math:`v` in :math:`\{\text{value\_cols}\}`, we perform:

    .. math::
       v(t) \;=\; 
       \begin{cases}
         \text{interpolate}(v,\;t;\;\text{method},\;\dots)
         & \text{for } t \in \{t_i'\}\,,\\
         v(t) & \text{if } t \in \{t_1,\dots,t_n\}\text{ and not NaN.}
       \end{cases}

    Parameters
    ----------
    series_df : pd.DataFrame
        Input DataFrame for a single time series, ideally sorted by
        :code:`time_col`. The :code:`time_col` should be convertible to
        datetime.
    time_col : str
        Name of the column containing datetime information.
    value_cols : List[str]
        List of column names whose missing values (NaNs) should be
        interpolated.
    freq : str or None, default None
        The desired frequency for the time series (e.g., 'D' for daily,
        'MS' for month start, 'AS' for year start). If provided, the
        DataFrame is reindexed to this frequency before interpolation.
        This helps identify and fill gaps where entire time steps are
        missing.
    method : str, default 'linear'
        Interpolation method to use. Passed to
        `pandas.DataFrame.interpolate()`. Common methods: 'linear',
        'time', 'polynomial', 'spline'. If 'polynomial' or 'spline',
        :code:`order` must be specified.
    order : int or None, default None
        Order for polynomial or spline interpolation. Required if
        :code:`method` is 'polynomial' or 'spline'.
    fill_limit : int or None, default None
        Maximum number of consecutive NaNs to fill. Passed to
        `pandas.DataFrame.interpolate()`.
    fill_limit_direction : str, default 'forward'
        Direction for :code:`fill_limit` ('forward', 'backward',
        'both'). Passed to `pandas.DataFrame.interpolate()`.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns interpolated. If :code:`freq`
        was used, the DataFrame will have a DatetimeIndex. Other columns
        not in :code:`value_cols` will be forward-filled after reindexing
        if :code:`freq` is set, to propagate their last known values into
        new empty rows.

    Raises
    ------
    TypeError
        If :code:`series_df` is not a DataFrame or if
        :code:`value_cols` is not a list of strings.
        Also if :code:`time_col` is missing from the DataFrame.
    ValueError
        If :code:`order` is required but not provided for 'polynomial'
        or 'spline'.

    Examples
    --------
    >>> import pandas as pd
    >>> from geoprior.utils.geo_utils import interpolate_temporal_gaps
    >>> # Sample time series with missing dates
    >>> df = pd.DataFrame({
    ...     'date': ['2020-01-01', '2020-01-03', '2020-01-06'],
    ...     'value': [1.0, None, 4.0]
    ... })
    >>> df
             date  value
    0 2020-01-01    1.0
    1 2020-01-03    NaN
    2 2020-01-06    4.0
    >>> result = interpolate_temporal_gaps(
    ...     df, time_col='date', value_cols=['value'], freq='D'
    ... )
    >>> result.head()
             date  value
    0 2020-01-01    1.0
    1 2020-01-02    2.0
    2 2020-01-03    3.0
    3 2020-01-04    3.0
    4 2020-01-05    3.5

    Notes
    -----
    - Ensure :code:`series_df` pertains to a single spatial group and is
      sorted by time for meaningful interpolation.
    - The 'time' method for interpolation requires the index to be a
      DatetimeIndex.
    - Polynomial or spline methods require :code:`order` to be specified.

    See Also
    --------
    pandas.DataFrame.interpolate : Core interpolation method.
    pandas.DataFrame.asfreq : Reindex DataFrame to fixed frequency.
    """
    if not isinstance(series_df, pd.DataFrame):
        raise TypeError(
            "Input 'series_df' must be a pandas DataFrame."
        )
    if not isinstance(value_cols, list) or not all(
        isinstance(col, str) for col in value_cols
    ):
        raise TypeError(
            "'value_cols' must be a list of strings."
        )
    if time_col not in series_df.columns:
        raise ValueError(
            f"Time column '{time_col}' not found in DataFrame."
        )

    df_interpolated = series_df.copy()

    # Convert time column to datetime and set as index for interpolation
    try:
        df_interpolated[time_col] = pd.to_datetime(
            df_interpolated[time_col]
        )
    except Exception as e:
        logger.error(
            f"Could not convert time column '{time_col}' to datetime: {e}"
        )
        raise

    df_interpolated = df_interpolated.sort_values(by=time_col)
    # Store original index name if it exists, to restore it later if needed.
    original_index_name = df_interpolated.index.name
    df_interpolated = df_interpolated.set_index(
        time_col, drop=False
    )

    original_columns = (
        series_df.columns.tolist()
    )  # Keep track of original columns

    if freq:
        try:
            # Reindex to the specified frequency. This may create new rows
            # with NaNs.
            df_interpolated = df_interpolated.asfreq(freq)

            # Forward fill other columns that are not being interpolated
            # to propagate their values into the new empty rows created by
            # asfreq.
            other_cols = [
                col
                for col in original_columns
                if col not in value_cols and col != time_col
            ]
            if other_cols:
                df_interpolated[other_cols] = df_interpolated[
                    other_cols
                ].ffill()

            # The time_col itself might become NaN in new rows if it was
            # dropped when setting index and then re-added. Ensure it's
            # filled from the new index.
            if time_col not in df_interpolated.columns:
                # if it was dropped
                df_interpolated[time_col] = (
                    df_interpolated.index
                )
            else:
                # if it was kept, ensure it is also ffilled from the new
                # index
                df_interpolated[time_col] = (
                    df_interpolated.index
                )

        except ValueError as e:
            logger.error(
                f"Invalid frequency string: '{freq}'. Error: {e}. "
                "Skipping reindexing."
            )
            # If reindexing fails, return a copy of the original to avoid
            # partial modification
            return series_df.copy()
        except Exception as e:
            logger.error(
                f"Error during reindexing with frequency '{freq}': {e}"
            )
            return series_df.copy()

    # Perform interpolation
    # Check if value_cols exist after potential reindexing
    missing_value_cols = [
        vc
        for vc in value_cols
        if vc not in df_interpolated.columns
    ]
    if missing_value_cols:
        logger.warning(
            f"Value columns not found in DataFrame after potential "
            f"reindexing: {missing_value_cols}. Skipping interpolation for "
            f"these."
        )
        value_cols = [
            vc
            for vc in value_cols
            if vc in df_interpolated.columns
        ]

    if not value_cols:
        logger.warning(
            "No valid value_cols left to interpolate. Returning processed "
            "DataFrame."
        )
        # If freq was applied, df_interpolated is reindexed and possibly
        # ffilled
        # If not, it's a copy of series_df with sorted datetime index
        df_to_return = df_interpolated.reset_index(drop=True)
        if (
            original_index_name
        ):  # Restore original index name if it existed
            df_to_return.index.name = original_index_name
        return df_to_return

    interpolate_kwargs = {
        "method": method,
        "limit": fill_limit,
        "limit_direction": fill_limit_direction,
    }
    if method in ["polynomial", "spline"]:
        if order is None:
            raise ValueError(
                f"Order must be specified for '{method}' interpolation."
            )
        interpolate_kwargs["order"] = order
    if method == "time":
        if not isinstance(
            df_interpolated.index, pd.DatetimeIndex
        ):
            logger.warning(
                "Interpolation method 'time' requires a DatetimeIndex. "
                "Consider setting `freq` or ensuring `time_col` is a "
                "DatetimeIndex."
            )
            # Fallback or let pandas handle it/error out

    try:
        df_interpolated[value_cols] = df_interpolated[
            value_cols
        ].interpolate(**interpolate_kwargs)
    except Exception as e:
        logger.error(
            f"Error during interpolation of columns {value_cols}: {e}"
        )
        # Return the DataFrame as it is before the erroring interpolation
        # attempt
        df_to_return = df_interpolated.reset_index(drop=True)
        if original_index_name:
            df_to_return.index.name = original_index_name
        return df_to_return

    # Reset index to restore time_col as a column and get default integer index
    df_interpolated = df_interpolated.reset_index(drop=True)
    if (
        original_index_name
    ):  # Restore original index name if it existed
        df_interpolated.index.name = original_index_name

    # Ensure original column order if possible, and all original columns
    # are present
    # This is important if asfreq added/removed rows and columns were
    # just ffilled
    final_cols_ordered = []
    for col in original_columns:
        if col in df_interpolated.columns:
            final_cols_ordered.append(col)
    # Add any new columns that might have been created (e.g. if time_col
    # was index and then reset)
    # though current logic tries to preserve original_columns.
    for col in df_interpolated.columns:
        if col not in final_cols_ordered:
            final_cols_ordered.append(col)

    return df_interpolated[final_cols_ordered]


@SaveFile
def augment_series_features(
    series_df: pd.DataFrame,
    feature_cols: list[str],
    noise_level: float = 0.01,
    noise_type: str = "gaussian",
    random_seed: int | None = None,
    savefile: str | None = None,
) -> pd.DataFrame:
    """
    Add random noise to selected numeric feature columns.
    
    Parameters
    ----------
    series_df : pandas.DataFrame
        Input DataFrame representing one or more time series.
    feature_cols : list of str
        Feature columns to augment.
    noise_level : float, optional
        Magnitude of the added noise. For Gaussian noise it scales the
        feature standard deviation, and for uniform noise it scales the
        feature range.
    noise_type : {'gaussian', 'uniform'}, optional
        Type of noise distribution to use.
    random_seed : int or None, optional
        Seed for reproducible noise generation.
    savefile : str or None, optional
        Optional output path handled by the decorator.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with noise added to the selected feature columns.
    
    Raises
    ------
    ValueError
        If requested feature columns are missing or ``noise_type`` is
        invalid.
    TypeError
        If the main inputs are of the wrong type.
    
    Notes
    -----
    Non-numeric columns are skipped, and constant or invalid numeric ranges
    are left unchanged.
    """
    if not isinstance(series_df, pd.DataFrame):
        raise TypeError(
            "Input 'series_df' must be a pandas DataFrame."
        )
    if not isinstance(feature_cols, list) or not all(
        isinstance(col, str) for col in feature_cols
    ):
        raise TypeError(
            "'feature_cols' must be a list of strings."
        )

    missing_cols = [
        col
        for col in feature_cols
        if col not in series_df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Feature columns not found in DataFrame: {missing_cols}"
        )

    if noise_type not in ["gaussian", "uniform"]:
        raise ValueError(
            f"Invalid noise_type: '{noise_type}'. "
            "Choose 'gaussian' or 'uniform'."
        )

    if random_seed is not None:
        np.random.seed(random_seed)

    df_augmented = series_df.copy()

    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(
            df_augmented[col]
        ):
            logger.warning(
                f"Column '{col}' is not numeric. "
                "Skipping noise augmentation for this column."
            )
            continue

        feature_values = df_augmented[col].values
        if noise_type == "gaussian":
            # Scale noise by the standard deviation of the feature
            col_std = np.std(feature_values)
            if col_std == 0 or np.isnan(
                col_std
            ):  # Avoid division by zero
                noise = 0
                if np.isnan(col_std):
                    logger.debug(
                        f"Std of column '{col}' is NaN. Adding zero noise."
                    )
            else:
                noise = np.random.normal(
                    0,
                    col_std * noise_level,
                    size=feature_values.shape,
                )
        elif noise_type == "uniform":
            # Scale noise by the range of the feature
            col_min = np.min(feature_values)
            col_max = np.max(feature_values)
            col_range = col_max - col_min
            if col_range == 0 or np.isnan(
                col_range
            ):  # Avoid zero range
                noise = 0
                if np.isnan(col_range):
                    logger.debug(
                        f"Range of column '{col}' is NaN. Adding zero noise."
                    )
            else:
                noise = np.random.uniform(
                    -col_range * noise_level / 2.0,
                    col_range * noise_level / 2.0,
                    size=feature_values.shape,
                )

        df_augmented[col] = feature_values + noise.astype(
            feature_values.dtype
        )  # Ensure dtype consistency

    return df_augmented


@SaveFile
def augment_spatiotemporal_data(
    df: pd.DataFrame,
    mode: str,
    group_by_cols: list[str] | None = None,
    time_col: str | None = None,
    value_cols_interpolate: list[str] | None = None,
    feature_cols_augment: list[str] | None = None,
    interpolation_kwargs: dict[str, Any] | None = None,
    augmentation_kwargs: dict[str, Any] | None = None,
    savefile: str | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Apply interpolation, feature augmentation, or both to grouped data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input spatiotemporal DataFrame.
    mode : {'interpolate', 'augment_features', 'both'}
        Processing mode. Use interpolation only, feature augmentation only,
        or interpolation followed by augmentation.
    group_by_cols : list of str or None, optional
        Grouping columns used for per-location processing.
    time_col : str or None, optional
        Time column required when interpolation is requested.
    value_cols_interpolate : list of str or None, optional
        Value columns to interpolate when interpolation is enabled.
    feature_cols_augment : list of str or None, optional
        Feature columns to perturb when augmentation is enabled.
    interpolation_kwargs : dict or None, optional
        Keyword arguments forwarded to ``interpolate_temporal_gaps``.
    augmentation_kwargs : dict or None, optional
        Keyword arguments forwarded to ``augment_series_features``.
    savefile : str or None, optional
        Optional output path handled by the decorator.
    verbose : bool, optional
        Whether to emit progress information.
    
    Returns
    -------
    pandas.DataFrame
        Processed DataFrame assembled from all groups.
    
    Raises
    ------
    ValueError
        If ``mode`` is invalid or required arguments for the selected mode
        are missing.
    
    Notes
    -----
    Groups are processed independently and concatenated afterward.
    """
    if mode not in [
        "interpolate",
        "augment_features",
        "both",
    ]:
        raise ValueError(
            "Invalid mode. Choose from 'interpolate', "
            "'augment_features', or 'both'."
        )

    processed_df = df.copy()
    interpolation_kwargs = interpolation_kwargs or {}
    augmentation_kwargs = augmentation_kwargs or {}

    if mode in ["interpolate", "both"]:
        if not all(
            [group_by_cols, time_col, value_cols_interpolate]
        ):
            raise ValueError(
                "For 'interpolate' or 'both' mode, 'group_by_cols', "
                "'time_col', and 'value_cols_interpolate' must be provided."
            )

        logger.info(
            f"Starting temporal interpolation for groups: "
            f"{group_by_cols}..."
        )
        interpolated_groups = []
        grouped = processed_df.groupby(
            group_by_cols, group_keys=True
        )  # group_keys=True for safety

        num_groups = len(grouped)
        for i, (name, group_df) in enumerate(grouped):
            if verbose:  # Simple print for progress, logger can be used too
                print(
                    f"  Interpolating group {i + 1}/{num_groups}: {name}",
                    end="\r",
                )

            interpolated_group = interpolate_temporal_gaps(
                series_df=group_df,
                time_col=time_col,
                value_cols=value_cols_interpolate,
                **interpolation_kwargs,
            )
            interpolated_groups.append(interpolated_group)

        if verbose:
            print("\nInterpolation of all groups complete.")

        if interpolated_groups:
            processed_df = pd.concat(
                interpolated_groups, ignore_index=True
            )
            logger.info(
                "Temporal interpolation applied to all groups."
            )
        else:
            logger.warning(
                "No groups found for interpolation or all groups were empty."
            )

    if mode in ["augment_features", "both"]:
        if not feature_cols_augment:
            raise ValueError(
                "For 'augment_features' or 'both' mode, "
                "'feature_cols_augment' must be provided."
            )
        logger.info(
            f"Starting feature augmentation for columns: "
            f"{feature_cols_augment}..."
        )
        processed_df = augment_series_features(
            series_df=processed_df,
            feature_cols=feature_cols_augment,
            **augmentation_kwargs,
        )
        logger.info("Feature augmentation applied.")

    return processed_df


def augment_city_spatiotemporal_data(
    df: pd.DataFrame,
    city: str,
    mode: str = "interpolate",
    group_by_cols: list[str] | None = None,
    time_col: str | None = None,
    value_cols_interpolate: list[str] | None = None,
    feature_cols_augment: list[str] | None = None,
    interpolation_config: dict[str, Any] | None = None,
    augmentation_config: dict[str, Any] | None = None,
    target_name: str | None = None,
    interpolate_target: bool = False,
    verbose: bool = True,
    coordinate_precision: int | None = None,
    savefile: str | None = None,
) -> pd.DataFrame:
    """
    Apply grouped spatiotemporal augmentation with city-aware defaults.
    
    This is a convenience wrapper around ``augment_spatiotemporal_data``. It
    validates the requested city, optionally rounds coordinates before
    grouping, and forwards interpolation and augmentation configuration
    dictionaries.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing spatial, temporal, and feature columns.
    city : {'nansha', 'zhongshan'}
        City identifier used for validation and defaults.
    mode : {'interpolate', 'augment_features', 'both'}, optional
        Processing mode forwarded to ``augment_spatiotemporal_data``.
    group_by_cols : list of str or None, optional
        Grouping columns for interpolation.
    time_col : str or None, optional
        Time column used for interpolation.
    value_cols_interpolate : list of str or None, optional
        Columns to interpolate.
    feature_cols_augment : list of str or None, optional
        Columns to augment with noise.
    interpolation_config : dict or None, optional
        Keyword arguments for ``interpolate_temporal_gaps``. Typical values
        include ``{'freq': 'AS', 'method': 'linear'}``.
    augmentation_config : dict or None, optional
        Keyword arguments for ``augment_series_features``. Typical values
        include ``{'noise_level': 0.01, 'noise_type': 'gaussian'}``.
    target_name : str or None, optional
        Optional target column name used when inferring default feature sets.
    interpolate_target : bool, optional
        Whether the target should be included in default interpolation
        columns.
    verbose : bool, optional
        Whether to emit progress information.
    coordinate_precision : int or None, optional
        Decimal precision applied to coordinates before grouping.
    savefile : str or None, optional
        Optional output CSV path handled by the decorator.
    
    Returns
    -------
    pandas.DataFrame
        Augmented DataFrame.
    
    Raises
    ------
    ValueError
        If ``city`` or ``mode`` is invalid, or if required arguments are
        missing for the selected mode.
    TypeError
        If the main inputs are of the wrong type.
    """
    # Validate DataFrame type
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input 'df' must be a pandas DataFrame."
        )

    # Validate city parameter
    city_lower = city.strip().lower()
    if city_lower not in ["nansha", "zhongshan"]:
        raise ValueError(
            "`city` must be 'nansha' or 'zhongshan'."
        )
    logger.info(
        f"Processing data for city: {city_lower.capitalize()}."
    )

    # Copy input to avoid modifying original
    df_city = df.copy()

    # Ensure 'year' (or specified time_col) is datetime
    # Default time_col for processing, can be overridden by user
    _time_col = time_col or "year"
    if _time_col not in df_city.columns:
        raise ValueError(
            f"Time column '{_time_col}' not found in DataFrame."
        )
    try:
        # Attempt to convert to datetime if not already.
        # If 'year' is int like 2015, format='%Y' makes it YYYY-01-01
        first_val = df_city[_time_col].iloc[0]
        if (
            pd.api.types.is_integer(first_val)
            and 1000 < first_val < 3000
        ):
            df_city[_time_col] = pd.to_datetime(
                df_city[_time_col], format="%Y"
            )
        else:
            df_city[_time_col] = pd.to_datetime(
                df_city[_time_col]
            )
        logger.debug(f"Ensured '{_time_col}' is datetime.")
    except Exception as e:
        logger.warning(
            f"Could not convert time column '{_time_col}' to datetime: {e}. "
            "Proceeding, but interpolation might behave unexpectedly."
        )

    # --- Define default parameters ---
    _group_by_cols = group_by_cols or [
        "longitude",
        "latitude",
    ]

    _group_by_cols = columns_manager(
        _group_by_cols, empty_as_none=False
    )
    # Default columns to exclude from auto-selection
    default_exclude_cols = set(_group_by_cols + [_time_col])
    # Add known categorical or ID-like columns (expand as needed)
    known_non_numeric_or_id_cols = {
        "lithology",
        "city",
        "lithology_class",
        # 'density_concentration', 'rainfall_category',
        # 'building_concentration', 'soil_thickness'
    }
    default_exclude_cols.update(known_non_numeric_or_id_cols)

    if target_name and target_name in df_city.columns:
        if not interpolate_target:
            # Exclude target from interpolation by default
            default_exclude_cols.add(target_name)

    numeric_cols = df_city.select_dtypes(
        include=np.number
    ).columns.tolist()

    _value_cols_interpolate = value_cols_interpolate
    if _value_cols_interpolate is None:
        _value_cols_interpolate = [
            col
            for col in numeric_cols
            if col not in default_exclude_cols
        ]
        if (
            interpolate_target
            and target_name
            and target_name in numeric_cols
        ):
            _value_cols_interpolate.append(target_name)
        logger.info(
            f"Defaulting 'value_cols_interpolate' to: "
            f"{_value_cols_interpolate}"
        )

    _feature_cols_augment = feature_cols_augment
    if _feature_cols_augment is None:
        augment_exclude_cols = default_exclude_cols.copy()
        if target_name:
            augment_exclude_cols.add(target_name)
        _feature_cols_augment = [
            col
            for col in numeric_cols
            if col not in augment_exclude_cols
        ]
        logger.info(
            f"Defaulting 'feature_cols_augment' to: "
            f"{_feature_cols_augment}"
        )

    _interpolation_config = interpolation_config or {
        "freq": "AS",
        "method": "linear",
    }
    _augmentation_config = augmentation_config or {
        "noise_level": 0.01,
        "noise_type": "gaussian",
        "random_seed": None,
    }

    # --- Coordinate Precision ---
    if coordinate_precision is not None:
        if not (
            isinstance(coordinate_precision, int)
            and coordinate_precision >= 0
        ):
            raise ValueError(
                "'coordinate_precision' must be a "
                "non-negative integer."
            )
        if (
            "longitude" in df_city.columns
            and "latitude" in df_city.columns
        ):
            df_city["longitude"] = df_city["longitude"].round(
                coordinate_precision
            )
            df_city["latitude"] = df_city["latitude"].round(
                coordinate_precision
            )
            logger.info(
                f"Coordinates rounded to {coordinate_precision} "
                "decimal places."
            )
        else:
            logger.warning(
                "Longitude/Latitude columns not found for rounding, "
                "but coordinate_precision was set."
            )

    logger.info(
        f"Original DataFrame shape for augmentation: {df_city.shape}"
    )

    # --- Call the core augmentation function ---
    try:
        df_augmented = augment_spatiotemporal_data(
            df=df_city,
            mode=mode,
            group_by_cols=_group_by_cols,
            time_col=_time_col,
            value_cols_interpolate=_value_cols_interpolate,
            feature_cols_augment=_feature_cols_augment,
            interpolation_kwargs=_interpolation_config,
            augmentation_kwargs=_augmentation_config,
            verbose=verbose,
        )
        logger.info(
            f"Augmented DataFrame shape: {df_augmented.shape}"
        )

        if savefile:
            try:
                # Ensure directory exists for savefile
                save_dir = os.path.dirname(savefile)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                df_augmented.to_csv(savefile, index=False)
                logger.info(
                    f"Augmented DataFrame saved to: {savefile}"
                )
            except Exception as e_save:
                logger.error(
                    f"Failed to save augmented DataFrame to '{savefile}': "
                    f"{e_save}"
                )

        return df_augmented

    except ValueError as ve:
        logger.error(
            f"ValueError during core augmentation process: {ve}"
        )
        raise
    except TypeError as te:
        logger.error(
            f"TypeError during core augmentation process: {te}"
        )
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during core augmentation: {e}"
        )
        raise


def generate_dummy_pinn_data(
    n_samples: int,
    *,
    year_range: tuple[float, float] | None = None,
    coords_range: tuple[
        tuple[float, float], tuple[float, float]
    ]
    | None = None,
    subs_range: tuple[float, float] | None = None,
    gwl_range: tuple[float, float] | None = None,
    rainfall_range: tuple[float, float] | None = None,
    vars_range: dict | None = None,
) -> dict[str, np.ndarray]:
    """
    Generate dummy PINN data dictionary with specified or default ranges.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    year_range : tuple[float, float], optional
        (min_year, max_year) for integer years. Default (2000, 2025).
    coords_range : tuple[tuple[float, float], tuple[float, float]], optional
        ((lon_min, lon_max), (lat_min, lat_max)). Default ((113.0, 113.8),
        (22.3, 22.8)).
    subs_range : tuple[float, float], optional
        (mean_subsidence, std_subsidence) for normal distribution. Default
        (-20, 15).
    gwl_range : tuple[float, float], optional
        (mean_gwl, std_gwl) for normal distribution. Default (2.5, 1.0).
    rainfall_range : tuple[float, float], optional
        (min_rain, max_rain) for uniform distribution. Default (500, 2500).
    vars_range : dict, optional
        Dictionary that may contain any of the keys:
        'year_range', 'coords_range', 'subs_range', 'gwl_range',
        'rainfall_range'. Missing keys will fall back to defaults or to
        explicitly passed arguments.

    Returns
    -------
    dummy_data_dict : dict[str, np.ndarray]
        Dictionary with keys:
            - "year" : integer years array
            - "longitude" : float longitudes array
            - "latitude" : float latitudes array
            - "subsidence" : float subsidence values array
            - "GWL" : float groundwater level values array
            - "rainfall_mm" : float rainfall values array
    """
    # Default ranges
    _def_year = (2000, 2025)
    _def_coords = ((113.0, 113.8), (22.3, 22.8))
    _def_subs = (-20.0, 15.0)
    _def_gwl = (2.5, 1.0)
    _def_rain = (500.0, 2500.0)

    # Merge vars_range if provided
    vr = vars_range or {}
    yr_rng = (
        year_range
        if year_range is not None
        else vr.get("year_range", _def_year)
    )
    crd_rng = (
        coords_range
        if coords_range is not None
        else vr.get("coords_range", _def_coords)
    )
    sbs_rng = (
        subs_range
        if subs_range is not None
        else vr.get("subs_range", _def_subs)
    )
    gwl_rng = (
        gwl_range
        if gwl_range is not None
        else vr.get("gwl_range", _def_gwl)
    )
    rnf_rng = (
        rainfall_range
        if rainfall_range is not None
        else vr.get("rainfall_range", _def_rain)
    )

    # Unpack ranges
    yr_min, yr_max = float(yr_rng[0]), float(yr_rng[1])
    (lon_min, lon_max), (lat_min, lat_max) = crd_rng
    subs_mean, subs_std = sbs_rng
    gwl_mean, gwl_std = gwl_rng
    rain_min, rain_max = rnf_rng

    # Generate year integers
    years = np.random.randint(
        int(yr_min), int(yr_max), size=n_samples
    )

    # Generate coordinates uniformly
    longitudes = np.random.uniform(
        float(lon_min), float(lon_max), size=n_samples
    )
    latitudes = np.random.uniform(
        float(lat_min), float(lat_max), size=n_samples
    )

    # Generate subsidence via normal distribution
    subsidence = np.random.normal(
        float(subs_mean), float(subs_std), size=n_samples
    )

    # Generate GWL via normal distribution
    gwl = np.random.normal(
        float(gwl_mean), float(gwl_std), size=n_samples
    )

    # Generate rainfall uniformly
    rainfall_mm = np.random.uniform(
        float(rain_min), float(rain_max), size=n_samples
    )

    return {
        "year": years,
        "longitude": longitudes.astype(np.float32),
        "latitude": latitudes.astype(np.float32),
        "subsidence": subsidence.astype(np.float32),
        "GWL": gwl.astype(np.float32),
        "rainfall_mm": rainfall_mm.astype(np.float32),
    }


def merge_frames_to_file(
    sources: Iterable[DataSource],
    output_path: PathLike,
    *,
    output_format: Literal[
        "parquet", "csv", "feather", "pickle"
    ] = "parquet",
    compression: str | None = "snappy",
    check_columns: Literal[
        "strict", "subset", "union"
    ] = "strict",
    excel_mode: Literal[
        "all_sheets", "first_sheet"
    ] = "all_sheets",
    sheet_names: Iterable[str] | None = None,
    add_source_label: bool = True,
    source_col: str = "source",
    sort_by: Iterable[str] | None = None,
    drop_duplicates: bool = False,
    reset_index: bool = True,
    save_kwargs: dict[str, Any] | None = None,
    verbose: int = 1,
) -> pd.DataFrame:
    """
    Merge multiple NATCOM city datasets into a single compressed file.

    Parameters
    ----------
    sources : iterable of {path-like, DataFrame}
        Input sources. Each element can be:

        * A path to a CSV file (e.g. ``"nansha_final...csv"``),
        * A path to an Excel workbook (one or many sheets per city),
        * A pre-loaded :class:`~pandas.DataFrame`.

    output_path : path-like
        Destination file path. If ``output_format='parquet'`` and
        the suffix is missing, ``'.parquet'`` is appended.

    output_format : {'parquet', 'csv', 'feather', 'pickle'}, optional
        Output format. Default is ``'parquet'`` for compact,
        columnar storage (recommended for Code Ocean).

    compression : str or None, optional
        Compression to use for the chosen format.

        * For ``'parquet'`` this is passed to
          :meth:`pandas.DataFrame.to_parquet` (e.g. ``'snappy'``,
          ``'gzip'``, ``'brotli'``).
        * For ``'csv'`` it is passed to
          :meth:`pandas.DataFrame.to_csv` via the ``compression``
          keyword if non-``None``.
        * Ignored for ``'feather'`` and ``'pickle'`` (Feather uses
          its own defaults; pickle rarely benefits from extra
          compression at this layer).

    check_columns : {'strict', 'subset', 'union'}, optional
        How to handle column consistency across sources:

        * ``'strict'`` (default): all sources must have exactly the
          same set of columns (order may differ). Columns are then
          aligned to the order of the first DataFrame. A mismatch
          raises :class:`ValueError`.

        * ``'subset'``: all columns in the *first* DataFrame must
          exist in each subsequent DataFrame. Extra columns in later
          sources are dropped. Missing required columns raise
          :class:`ValueError`.

        * ``'union'``: columns are unioned across all sources. Any
          missing column in a particular source is added and filled
          with ``NaN`` before concatenation.

    excel_mode : {'all_sheets', 'first_sheet'}, optional
        Behaviour when a source is an Excel workbook:

        * ``'all_sheets'`` (default): read *all* sheets and treat
          each sheet as a separate DataFrame to merge.
        * ``'first_sheet'``: read only the first sheet.

        If ``sheet_names`` is provided, it takes precedence.

    sheet_names : iterable of str, optional
        Explicit sheet names to read from Excel workbooks. If
        provided, only these sheets are read.

    add_source_label : bool, optional
        If ``True`` (default), add a column named ``source_col`` to
        each chunk before concatenation. For path-like inputs, the
        label is derived from the file name and, when applicable,
        sheet name (e.g. ``"nansha_final_main_std.harmonized.csv"`` or
        ``"zhongshan.xlsx:Sheet1"``). For pre-loaded DataFrames, the
        label is ``'<in_memory>'``.

    source_col : str, optional
        Name of the column storing the source label when
        ``add_source_label=True``.

    sort_by : iterable of str, optional
        Optional column(s) to sort the merged DataFrame by at the
        end (e.g. ``['city', 'year', 'longitude', 'latitude']``).

    drop_duplicates : bool, optional
        If ``True``, drop duplicate rows at the end (after sorting).

    reset_index : bool, optional
        If ``True`` (default), reset index after concatenation.

    save_kwargs : dict, optional
        Extra keyword arguments forwarded to the corresponding
        ``to_*`` writer (e.g. ``to_parquet``, ``to_csv``,
        ``to_feather``, ``to_pickle``).

    verbose : int, optional
        Verbosity level. ``0`` = silent, ``>=1`` prints basic
        progress information.

    Returns
    -------
    merged : pandas.DataFrame
        The merged DataFrame (also written to disk).

    Raises
    ------
    ValueError
        If ``check_columns='strict'`` or ``'subset'`` and a column
        mismatch is detected.
    Examples
    ----------
    >>> from geoprior.utils.geo_utils import merge_frames_to_file
    >>> merge_frames_to_file(
    ...    sources=[
    ...        "nansha_final_main_std.harmonized.csv",
    ...        "zhongshan_final_main_std.harmonized.csv",
    ...    ],
    ...    output_path="natcom_all_cities",
    ...    output_format="parquet",
    ...    compression="snappy",
    ...    sort_by=["city", "year", "longitude", "latitude"],
    ... )

    Notes
    -----
    * All inputs are read fully into memory before concatenation.
      This is acceptable for the NATCOM subsidence datasets
      (``O(10^6 - 10^7)`` rows) but can be refactored to a
      streaming/row-group approach if needed later.
    * Using ``output_format='parquet'`` with compression (e.g.
      ``'snappy'``) is recommended for Code Ocean to minimise disk
      usage while keeping I/O efficient.
    """
    save_kwargs = dict(save_kwargs or {})
    sources = list(sources)
    if not sources:
        raise ValueError(
            "No sources provided to merge_city_frames_to_file()."
        )

    out_path = Path(output_path)
    fmt = output_format.lower()

    if fmt not in {"parquet", "csv", "feather", "pickle"}:
        raise ValueError(
            f"Unsupported output_format={output_format!r}. "
            "Expected one of {'parquet','csv','feather','pickle'}."
        )

    def _iter_frames_from_source(
        src: DataSource,
    ) -> list[pd.DataFrame]:
        """Internal: expand a single source into one or more frames."""
        if isinstance(src, pd.DataFrame):
            label = "<in_memory>"
            return [(src.copy(), label)]

        p = Path(src)
        if not p.exists():
            raise FileNotFoundError(f"Source not found: {p}")

        suffix = p.suffix.lower()
        frames_with_labels: list[
            tuple[pd.DataFrame, str]
        ] = []

        if suffix in {".csv", ".txt"}:
            df = pd.read_csv(p)
            frames_with_labels.append((df, p.name))
        elif suffix in {".xls", ".xlsx"}:
            # Decide which sheets to read.
            if sheet_names is not None:
                sheets_to_read = list(sheet_names)
            elif excel_mode == "first_sheet":
                sheets_to_read = [
                    0
                ]  # first sheet by position
            else:  # 'all_sheets'
                # Read all sheets into a dict
                all_sheets = pd.read_excel(p, sheet_name=None)
                for sh_name, sh_df in all_sheets.items():
                    frames_with_labels.append(
                        (sh_df, f"{p.name}:{sh_name}")
                    )
                return [*frames_with_labels]

            for sh in sheets_to_read:
                sh_df = pd.read_excel(p, sheet_name=sh)
                label = (
                    f"{p.name}:{sh}"
                    if not isinstance(sh, int)
                    else f"{p.name}:sheet{sh}"
                )
                frames_with_labels.append((sh_df, label))
        else:
            raise ValueError(
                f"Unsupported file extension for source {p!s}. "
                "Only CSV (.csv) and Excel (.xls, .xlsx) are allowed."
            )

        return frames_with_labels

    # 1) Collect and validate/align columns
    frames: list[pd.DataFrame] = []
    base_cols = None

    for src in sources:
        for df_chunk, label in _iter_frames_from_source(src):
            if base_cols is None:
                base_cols = list(df_chunk.columns)
                if verbose:
                    print(
                        f"[merge_city_frames] Base columns "
                        f"({len(base_cols)}): {base_cols[:6]}..."
                    )
            else:
                cols_now = list(df_chunk.columns)
                set_base = set(base_cols)
                set_now = set(cols_now)

                if check_columns == "strict":
                    if set_now != set_base:
                        missing = sorted(set_base - set_now)
                        extra = sorted(set_now - set_base)
                        raise ValueError(
                            "Column mismatch under 'strict' mode.\n"
                            f"  Missing in {label}: {missing}\n"
                            f"  Extra in {label}: {extra}"
                        )
                    # Align column order
                    df_chunk = df_chunk.reindex(
                        columns=base_cols
                    )
                elif check_columns == "subset":
                    # ensure *required* columns (base) exist
                    missing = sorted(set_base - set_now)
                    if missing:
                        raise ValueError(
                            "Column mismatch under 'subset' mode.\n"
                            f"  Missing required in {label}: {missing}"
                        )
                    # drop any extra columns and align order
                    df_chunk = df_chunk.reindex(
                        columns=base_cols
                    )
                elif check_columns == "union":
                    # union columns so far; fill missing with NaN
                    union_cols = list(set_base | set_now)
                    for c in union_cols:
                        if c not in df_chunk.columns:
                            df_chunk[c] = pd.NA
                    if set_base != set(union_cols):
                        # update base definition & expand existing frames
                        for i, f in enumerate(frames):
                            for c in union_cols:
                                if c not in f.columns:
                                    frames[i][c] = pd.NA
                        base_cols = union_cols
                    df_chunk = df_chunk.reindex(
                        columns=base_cols
                    )
                else:
                    raise ValueError(
                        f"Unknown check_columns={check_columns!r}"
                    )

            if add_source_label:
                if source_col in df_chunk.columns:
                    # avoid clobbering an existing column silently
                    df_chunk[f"{source_col}_orig"] = df_chunk[
                        source_col
                    ]
                df_chunk[source_col] = label

            frames.append(df_chunk)

    if not frames:
        raise RuntimeError(
            "No frames were collected from the given sources."
        )

    # 2) Concatenate and post-process
    merged = pd.concat(
        frames, axis=0, ignore_index=reset_index
    )

    if sort_by:
        merged = merged.sort_values(
            list(sort_by), ignore_index=reset_index
        )

    if drop_duplicates:
        merged = merged.drop_duplicates(
            ignore_index=reset_index
        )

    if not reset_index:
        merged = merged  # keep concatenated index as-is
    else:
        merged = merged.reset_index(drop=True)

    # 3) Save to disk (compressed, if requested)
    if fmt == "parquet":
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".parquet")
        if verbose:
            print(
                f"[merge_frames] Writing Parquet -> {out_path} "
                f"(compression={compression!r})"
            )
        merged.to_parquet(
            out_path,
            compression=compression,
            index=False,
            **save_kwargs,
        )
    elif fmt == "csv":
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".csv")
        if verbose:
            print(
                f"[merge_frames] Writing CSV -> {out_path} "
                f"(compression={compression!r})"
            )
        if compression is not None:
            save_kwargs.setdefault("compression", compression)
        merged.to_csv(out_path, index=False, **save_kwargs)
    elif fmt == "feather":
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".feather")
        if verbose:
            print(
                f"[merge_frames] Writing Feather -> {out_path}"
            )
        # pandas uses pyarrow under the hood
        merged.to_feather(out_path, **save_kwargs)
    elif fmt == "pickle":
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".pkl")
        if verbose:
            print(
                f"[merge_frames] Writing Pickle -> {out_path}"
            )
        merged.to_pickle(out_path, **save_kwargs)

    if verbose:
        print(
            f"[merge_frames] Done. Merged shape="
            f"{merged.shape}, saved as '{fmt}' at: {out_path}"
        )

    return merged


def unpack_frames_from_file(
    merged: PathLike | pd.DataFrame,
    *,
    group_col: str = "city",
    output_dir: PathLike | None = None,
    output_format: Literal[
        "csv", "parquet", "feather", "pickle"
    ] = "csv",
    compression: str | None = None,
    use_source_col: bool = True,
    source_col: str = "source",
    filename_pattern: str = "{group_value}_split",
    drop_columns: Iterable[str] | None = None,
    keep_columns: Iterable[str] | None = None,
    save: bool = True,
    return_dict: bool = True,
    save_kwargs: dict[str, Any] | None = None,
    verbose: int = 1,
    logger: None,
) -> dict[Any, pd.DataFrame]:
    """
    Reverse of `merge_city_frames_to_file`: split an aggregated NATCOM
    dataset into per-city frames (and optionally write them to disk).

    Parameters
    ----------
    merged : path-like or DataFrame
        Aggregated dataset. If path-like, the format is inferred from
        the file suffix:

        - ``.parquet`` → :func:`pandas.read_parquet`
        - ``.csv``     → :func:`pandas.read_csv`
        - ``.feather`` → :func:`pandas.read_feather`
        - ``.pkl``/``.pickle`` → :func:`pandas.read_pickle`

        If a :class:`~pandas.DataFrame` is passed, it is used directly.

    group_col : str, optional
        Column used to split the dataset (default: ``'city'``). Each
        unique value defines one output chunk.

    output_dir : path-like, optional
        Directory where per-group files are written. If ``None`` and
        ``merged`` is a path, the directory of ``merged`` is used.
        If ``merged`` is a DataFrame and ``output_dir`` is ``None``,
        the current working directory is used.

    output_format : {'csv', 'parquet', 'feather', 'pickle'}, optional
        Output format for per-group files. Default is ``'csv'``.

    compression : str or None, optional
        Compression to use when writing:

        - For ``'csv'``, forwarded to :meth:`DataFrame.to_csv` as the
          ``compression`` argument (e.g. ``'gzip'``).
        - For ``'parquet'``, forwarded to :meth:`DataFrame.to_parquet`
          (e.g. ``'snappy'``, ``'gzip'``).
        - Ignored for ``'feather'`` and ``'pickle'`` (these use their
          own defaults).

    use_source_col : bool, optional
        If ``True`` (default) and a column named ``source_col`` exists,
        the helper tries to reconstruct the *original* file name for
        each group:

        - If a group has a single unique, non-null `source` value that
          looks like a filename (e.g. ``'nansha_final_main_std.harmonized.csv'``),
          that base name is used for the output (with its suffix
          adjusted to match ``output_format`` if needed).

        - If there are multiple unique `source` labels within a group,
          it falls back to ``filename_pattern``.

    source_col : str, optional
        Name of the column containing the source label (default:
        ``'source'``). This should match the column created in
        :func:`merge_frames_to_file` when ``add_source_label=True``.

    filename_pattern : str, optional
        Pattern used when no suitable `source` label is available.
        The following placeholders are supported:

        - ``{group_value}`` : the group value as a string
        - ``{group_col}``   : the name of the grouping column

        Example:
        ``filename_pattern="{group_col}_{group_value}_data"`` →
        ``"city_Nansha_data.csv"``.

    drop_columns : iterable of str, optional
        Columns to drop from each group before saving/returning
        (e.g. ``['source']`` if you don't want the bookkeeping column).

    keep_columns : iterable of str, optional
        If provided, only these columns are kept (all others are
        dropped *after* any ``drop_columns`` processing is applied).

    save : bool, optional
        If ``True`` (default), write each group to disk as a separate
        file. If ``False``, no files are written; only the dict of
        DataFrames is returned (if ``return_dict=True``).

    return_dict : bool, optional
        If ``True`` (default), return a mapping
        ``{group_value: group_df}``. If ``False``, an empty dict is
        returned (useful when you only care about side-effect files).

    save_kwargs : dict, optional
        Extra keyword arguments forwarded to the respective writer:
        :meth:`DataFrame.to_csv`, :meth:`DataFrame.to_parquet`,
        :meth:`DataFrame.to_feather`, or :meth:`DataFrame.to_pickle`.

    verbose : int, optional
        Verbosity level. ``0`` = silent, ``>=1`` prints progress
        information.

    Returns
    -------
    out : dict
        Dictionary mapping each group value to the corresponding
        :class:`~pandas.DataFrame`. Empty if ``return_dict=False``.

    Raises
    ------
    ValueError
        If ``group_col`` is not present in the merged dataset.

    Examples
    ---------
    >>> from geoprior.utils.geo_utils import unpack_frames_from_file
    >>> unpack_frames_from_file(
    ...     "natcom_all_cities.parquet",
    ...     group_col="city",
    ...     output_format="csv",
    ... )
    # -> writes e.g. 'nansha_final_main_std.harmonized.csv',
    #    'zhongshan_final_main_std.harmonized.csv' (if `source` labels exist),
    #    and returns a dict: {'Nansha': df_nansha, 'Zhongshan': df_zhongshan}
    """

    def _log(mess, verbose=verbose, logger=logger):
        vlog(mess, verbose=verbose, logger=logger)

    save_kwargs = dict(save_kwargs or {})

    # 1) Load merged DataFrame
    df: pd.DataFrame
    merged_path: Path | None = None

    if isinstance(merged, pd.DataFrame):
        df = merged.copy()
    else:
        merged_path = Path(merged)
        if not merged_path.exists():
            raise FileNotFoundError(
                f"Merged file not found: {merged_path}"
            )

        suffix = merged_path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(merged_path)
        elif suffix == ".csv":
            df = pd.read_csv(merged_path)
        elif suffix in {".feather", ".ft"}:
            df = pd.read_feather(merged_path)
        elif suffix in {".pkl", ".pickle"}:
            df = pd.read_pickle(merged_path)
        else:
            raise ValueError(
                f"Unsupported merged file extension {suffix!r}. "
                "Expected one of: '.parquet', '.csv', '.feather', '.pkl', '.pickle'."
            )

    if group_col not in df.columns:
        raise ValueError(
            f"group_col={group_col!r} not found in merged DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # 2) Resolve output directory
    if output_dir is not None:
        out_dir = Path(output_dir)
    elif merged_path is not None:
        out_dir = merged_path.parent
    else:
        out_dir = Path.cwd()

    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = output_format.lower()
    if fmt not in {"csv", "parquet", "feather", "pickle"}:
        raise ValueError(
            f"Unsupported output_format={output_format!r}. "
            "Expected one of {'csv','parquet','feather','pickle'}."
        )

    def _ext_for_format() -> str:
        if fmt == "csv":
            return ".csv"
        elif fmt == "parquet":
            return ".parquet"
        elif fmt == "feather":
            return ".feather"
        else:  # 'pickle'
            return ".pkl"

    def _filename_from_source_label(
        label: str,
        group_value: Any,
    ) -> str:
        """
        Try to infer a good filename from the `source` label. Adjust
        the suffix to match `output_format`, but keep the stem.
        """
        p = Path(str(label))
        stem = p.stem  # 'nansha_final_main_std.harmonized'
        ext = _ext_for_format()
        # Special case: if output_format is csv and original already ends
        # with .csv, keep exactly that file name.
        if fmt == "csv" and p.suffix.lower() == ".csv":
            return p.name
        return stem + ext

    # 3) Split, post-process, and save
    out_frames: dict[Any, pd.DataFrame] = {}

    groups = df.groupby(group_col, dropna=False)
    if verbose:
        _log(
            f"[unpack_{group_col}_frames] Splitting merged DataFrame of shape "
            f"{df.shape} into {len(groups)} group(s) by {group_col!r}..."
        )

    for gval, gdf in groups:
        chunk = gdf.copy()

        # Optional column dropping/keeping
        if drop_columns:
            cols_to_drop = [
                c for c in drop_columns if c in chunk.columns
            ]
            if cols_to_drop:
                chunk = chunk.drop(columns=cols_to_drop)

        if keep_columns is not None:
            keep_set = [
                c for c in keep_columns if c in chunk.columns
            ]
            chunk = chunk[keep_set]

        # Decide file name
        filename: str | None = None
        if save:
            if use_source_col and source_col in gdf.columns:
                labels = (
                    gdf[source_col]
                    .dropna()
                    .astype(str)
                    .unique()
                )
                if labels.size == 1:
                    # Nice case: one clear source label for this group
                    filename = _filename_from_source_label(
                        labels[0], gval
                    )
                elif labels.size > 1 and verbose:
                    _log(
                        f"[unpack_{group_col}_frames] Group {group_col}={gval!r} "
                        f"has multiple source labels {labels.tolist()}; "
                        "falling back to filename_pattern."
                    )

            if filename is None:
                # Fallback to pattern
                gv_str = str(gval).strip().replace(" ", "_")
                base = filename_pattern.format(
                    group_value=gv_str,
                    group_col=group_col,
                )
                if not any(
                    base.lower().endswith(ext)
                    for ext in [
                        ".csv",
                        ".parquet",
                        ".feather",
                        ".pkl",
                        ".pickle",
                    ]
                ):
                    base += _ext_for_format()
                filename = base

            out_path = out_dir / filename

            if verbose:
                _log(
                    f"[unpack_{group_col}_frames] Writing group {group_col}={gval!r} "
                    f"to {out_path} (format={fmt}, n={len(chunk)})"
                )

            # Save according to format
            if fmt == "csv":
                if compression is not None:
                    save_kwargs.setdefault(
                        "compression", compression
                    )
                chunk.to_csv(
                    out_path, index=False, **save_kwargs
                )
            elif fmt == "parquet":
                chunk.to_parquet(
                    out_path,
                    compression=compression,
                    index=False,
                    **save_kwargs,
                )
            elif fmt == "feather":
                chunk.to_feather(out_path, **save_kwargs)
            else:  # 'pickle'
                chunk.to_pickle(out_path, **save_kwargs)

        if return_dict:
            out_frames[gval] = chunk

    if verbose:
        _log(
            f"[unpack_{group_col}_frames] Done. Produced {len(out_frames)} "
            f"group DataFrame(s). Files saved in: {out_dir}"
        )

    return out_frames
