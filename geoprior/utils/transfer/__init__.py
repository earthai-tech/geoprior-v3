"""

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

from . import (
    xfer_io,
    xfer_metrics,
    xfer_risk,
    xfer_units,
    xfer_utils,
)

__all__ = [
    "xfer_io",
    "xfer_metrics",
    "xfer_risk",
    "xfer_utils",
    "xfer_units",
]
