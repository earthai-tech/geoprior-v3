from __future__ import annotations

import numpy as np
import pytest

from tests._helpers import DummyModel, import_module_group

pytest.importorskip("tensorflow")

mod = import_module_group("maths")


def test_log_clip_constraint_replaces_non_finite_then_clips():
    c = mod.LogClipConstraint(-2.0, 1.0)
    x = np.array([np.nan, -10.0, 0.5, 5.0], dtype=np.float32)
    y = c(x)
    if hasattr(y, "numpy"):
        y = y.numpy()
    np.testing.assert_allclose(
        y, np.array([-2.0, -2.0, 0.5, 1.0], dtype=np.float32)
    )


def test_q_to_gw_source_term_si_per_volume_passthrough_when_already_si():
    out = mod.q_to_gw_source_term_si(
        DummyModel(),
        np.array([[2.0]], dtype=np.float32),
        Ss_field=None,
        H_field=None,
        coords_normalized=False,
        t_range_units=None,
        time_units="year",
        scaling_kwargs={
            "Q_kind": "per_volume",
            "Q_in_per_second": True,
        },
    )
    if hasattr(out, "numpy"):
        out = out.numpy()
    np.testing.assert_allclose(
        out, np.array([[2.0]], dtype=np.float32)
    )


def test_q_to_gw_source_term_si_recharge_rate_divides_by_effective_thickness():
    out = mod.q_to_gw_source_term_si(
        DummyModel(),
        np.array([[4.0]], dtype=np.float32),
        Ss_field=None,
        H_field=np.array([[2.0]], dtype=np.float32),
        coords_normalized=False,
        t_range_units=None,
        time_units="year",
        scaling_kwargs={
            "Q_kind": "recharge_rate",
            "Q_length_in_si": True,
        },
    )
    if hasattr(out, "numpy"):
        out = out.numpy()
    np.testing.assert_allclose(
        out, np.array([[2.0]], dtype=np.float32)
    )


def test_q_to_gw_source_term_si_head_rate_uses_mv_gammaw_fallback():
    model = DummyModel(_mv_value=lambda: 2.0, gamma_w=3.0)
    out = mod.q_to_gw_source_term_si(
        model,
        np.array([[5.0]], dtype=np.float32),
        Ss_field=None,
        H_field=None,
        coords_normalized=False,
        t_range_units=None,
        time_units="year",
        scaling_kwargs={
            "Q_kind": "head_rate",
            "Q_length_in_si": True,
        },
    )
    if hasattr(out, "numpy"):
        out = out.numpy()
    np.testing.assert_allclose(
        out, np.array([[30.0]], dtype=np.float32)
    )


def test_q_to_gw_source_term_si_recharge_requires_H_field():
    with pytest.raises(ValueError, match="requires H_field"):
        mod.q_to_gw_source_term_si(
            DummyModel(),
            np.array([[1.0]], dtype=np.float32),
            Ss_field=None,
            H_field=None,
            coords_normalized=False,
            t_range_units=None,
            time_units="year",
            scaling_kwargs={
                "Q_kind": "recharge_rate",
                "Q_length_in_si": True,
            },
        )
