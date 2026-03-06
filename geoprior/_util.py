# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026-present Laurent Kouadio

from __future__ import annotations

import logging
import os
import warnings
from typing import Optional

from ._geopriorlog import geopriorlog


def _default_config_path() -> Optional[str]:
    """
    Return the packaged YAML logging config path if available.

    Requires geoprior/_flog.yml to be shipped as package data.
    """
    try:
        from importlib import resources

        p = resources.files("geoprior").joinpath("_glog.yml")
        with resources.as_file(p) as path:
            return str(path)
    except Exception:
        return None


def initialize_logging(
    config_path: Optional[str] = None,
    use_default_logger: bool = True,
    verbose: bool = False,
) -> None:
    """
    Initialize GeoPrior-v3 structured logging.

    If `config_path` is None, this tries (in order):
    1) env var GEOPRIOR_LOG_CONFIG
    2) packaged geoprior/_flog.yml (if present)
    3) fallback to a basic console logger

    Parameters
    ----------
    config_path : Optional[str]
        Path to a YAML/INI logging config.
    use_default_logger : bool
        If True and no config is found, install a basic logger.
    verbose : bool
        If True, prints which config is being used.
    """
    if config_path is None:
        config_path = os.getenv("GEOPRIOR_LOG_CONFIG")

    if config_path is None:
        config_path = _default_config_path()

    try:
        geopriorlog.load_configuration(
            config_path=config_path,
            use_default_logger=use_default_logger,
            verbose=verbose,
        )
    except Exception as e:
        warnings.warn(
            "GeoPrior-v3 logging initialization failed: "
            f"{e}. Falling back to basic console logging.",
            RuntimeWarning,
            stacklevel=2,
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )