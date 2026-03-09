# License: Apache-2.0
# Copyright (c) 2026-present
# Author: LKouadio <etanoyau@gmail.com>

"""
A subpackage for hyperparameter tuning of geoprior models using Keras Tuner.
"""

from ._config import check_keras_tuner_is_available

HAS_KT = check_keras_tuner_is_available(error="ignore")

if HAS_KT:
    from ...compat.kt import KerasTunerDependencies

    KT_DEPS = KerasTunerDependencies()
else:
    from ..._dummies import DummyKT

    KT_DEPS = DummyKT()

if HAS_KT:
    from ._geoprior_tuner import SubsNetTuner

    __all__ = [
        "HAS_KT",
        "KT_DEPS",
        "SubsNetTuner",
    ]
