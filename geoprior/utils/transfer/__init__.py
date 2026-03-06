# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""geoprior.utils.transfer

Reusable helpers for cross-city transferability analyses.

Design goals
------------
- Pure, dependency-light helpers (numpy/pandas).
- Robust I/O for your xfer folder layout.
- Plot-prep utilities (Pareto front, retention, risk).

Typical usage
-------------
>>> from geoprior.utils.transfer import xfer_io
>>> from geoprior.utils.transfer import xfer_metrics
>>> from geoprior.utils.transfer import xfer_risk
"""

from . import xfer_io
from . import xfer_metrics
from . import xfer_risk
from . import xfer_utils
from . import xfer_units

__all__ = [
    "xfer_io",
    "xfer_metrics",
    "xfer_risk",
    "xfer_utils", 
    "xfer_units"
]
