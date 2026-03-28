# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Initializes the `nn` subpackage, dynamically selecting the backend.

This module checks for the presence of TensorFlow/Keras and configures
a central `KERAS_DEPS` object. If the backend is available, `KERAS_DEPS`
becomes a lazy loader for real Keras/TensorFlow components. If not, it
becomes a dummy object generator that raises helpful `ImportError`
messages at runtime.

This allows other modules in the `nn` subpackage to be imported without
crashing, even if heavy dependencies are not installed.
"""

import os

# filter out TF INFO and WARNING messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or "3"
# Disable oneDNN custom operations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from .._deps import check_backends
from ..compat._config import Config as config
from ..compat._config import (
    check_keras_backend,
    configure_dependencies,
    import_keras_dependencies,
)

_HAS_KT = check_backends("keras_tuner")["keras_tuner"]

# Set default configuration
config.INSTALL_DEPS = False
config.WARN_STATUS = "warn"

# Custom message for missing dependencies
EXTRA_MSG = (
    "`nn` sub-package expects the `tensorflow` or"
    " `keras` library to be installed."
)
# Configure and install dependencies if needed
configure_dependencies(
    install_dependencies=config.INSTALL_DEPS
)

# Lazy-load Keras dependencies
KERAS_DEPS = import_keras_dependencies(
    extra_msg=EXTRA_MSG, error="ignore"
)

# Check if TensorFlow or Keras is installed
KERAS_BACKEND = check_keras_backend(error="ignore")


def dependency_message(module_name):
    """
    Generate a custom message for missing dependencies.

    Parameters
    ----------
    module_name : str
        The name of the module that requires the dependencies.

    Returns
    -------
    str
        A message indicating the required dependencies.
    """
    return (
        f"`{module_name}` needs either the `tensorflow`"
        " or `keras` package to be installed. Please install"
        " one of these packages to use this function."
    )


__all__ = []


# if KERAS_BACKEND:
from ._shapes import (
    _logs_to_py,
    debug_quantile_crossing_np,
    debug_tensor_interval,
    debug_val_interval,
)
from .calibration import (
    apply_calibrator_to_subs,
    fit_interval_calibrator_on_val,
)
from .callbacks import (
    FrozenValQuantileLogger,
    FrozenValQuantileMonitor,
    LambdaOffsetScheduler,
)
from .keras_metrics import (
    MAEQ50,
    MSEQ50,
    Coverage80,
    Sharpness80,
    _to_py,
    coverage80_fn,
    sharpness80_fn,
)
from .losses import make_weighted_pinball
from .op import extract_physical_parameters
from .plotting import plot_history_in
from .subsidence import (
    GeoPriorSubsNet,
    PoroElasticSubsNet,
    autoplot_geoprior_history,
    debug_model_reload,
    finalize_scaling_kwargs,
    load_physics_payload,
    override_scaling_kwargs,
    plot_physics_values_in,
)
from .subsidence.identifiability import (
    derive_K_from_tau_np,
    ident_audit_dict,
)
from .subsidence.payloads import (
    identifiability_diagnostics_from_payload,
    summarise_effective_params,
)
from .tuners import SubsNetTuner

__all__ = [
    "KERAS_DEPS",
    "KERAS_BACKEND",
    "GeoPriorSubsNet",
    "PoroElasticSubsNet",
    "_logs_to_py",
    "debug_quantile_crossing_np",
    "debug_tensor_interval",
    "debug_val_interval",
    "make_weighted_pinball",
    "Coverage80",
    "MAEQ50",
    "MSEQ50",
    "Sharpness80",
    "_to_py",
    "coverage80_fn",
    "sharpness80_fn",
    "apply_calibrator_to_subs",
    "fit_interval_calibrator_on_val",
    "LambdaOffsetScheduler",
    "plot_history_in",
    "extract_physical_parameters",
    "finalize_scaling_kwargs",
    "debug_model_reload",
    "autoplot_geoprior_history",
    "plot_physics_values_in",
    "load_physics_payload",
    "override_scaling_kwargs",
    "identifiability_diagnostics_from_payload",
    "load_physics_payload",
    "summarise_effective_params",
    "FrozenValQuantileMonitor",
    "FrozenValQuantileLogger",
    "make_weighted_pinball",
    "FrozenValQuantileMonitor",
    "FrozenValQuantileLogger",
    "derive_K_from_tau_np",
    "ident_audit_dict",
    "SubsNetTuner",
]
