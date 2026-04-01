from __future__ import annotations

import importlib

import numpy as np
import pytest

MODULE_CANDIDATES = [
    "geoprior.models.subsidence.stability",
]


def _import_target():
    last = None
    for name in MODULE_CANDIDATES:
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover
            last = exc
    raise last  # type: ignore[misc]


@pytest.fixture(scope="module")
def mod():
    return _import_target()


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def test_clamp_physics_logits_clips_all_supported_channels(
    mod,
):
    K = mod.tf_constant(
        [-20.0, 0.0, 20.0], dtype=mod.tf_float32
    )
    Ss = mod.tf_constant(
        [-99.0, 1.0, 99.0], dtype=mod.tf_float32
    )
    tau = mod.tf_constant(
        [-50.0, 0.5, 50.0], dtype=mod.tf_float32
    )
    Q = mod.tf_constant(
        [-1e9, 0.0, 1e9], dtype=mod.tf_float32
    )

    Kc, Ssc, tauc, Qc = mod.clamp_physics_logits(
        K, Ss, tau, Q
    )

    np.testing.assert_allclose(
        _to_numpy(Kc), [-15.0, 0.0, 15.0]
    )
    np.testing.assert_allclose(
        _to_numpy(Ssc), [-15.0, 1.0, 15.0]
    )
    np.testing.assert_allclose(
        _to_numpy(tauc), [-15.0, 0.5, 15.0]
    )
    np.testing.assert_allclose(
        _to_numpy(Qc), [-1e5, 0.0, 1e5]
    )


def test_sanitize_scales_replaces_bad_values_and_clamps(mod):
    scales = {
        "good": mod.tf_constant(
            [1.0, 2.0], dtype=mod.tf_float32
        ),
        "bad": mod.tf_constant(
            [np.nan, np.inf, -np.inf], dtype=mod.tf_float32
        ),
        "tiny": mod.tf_constant(
            [1e-12], dtype=mod.tf_float32
        ),
        "huge": mod.tf_constant([1e12], dtype=mod.tf_float32),
    }

    out = mod.sanitize_scales(
        scales, min_scale=1e-6, max_scale=1e6
    )

    np.testing.assert_allclose(
        _to_numpy(out["good"]), [1.0, 2.0]
    )
    np.testing.assert_allclose(
        _to_numpy(out["bad"]), [1.0, 1.0, 1.0]
    )
    np.testing.assert_allclose(_to_numpy(out["tiny"]), [1e-6])
    np.testing.assert_allclose(_to_numpy(out["huge"]), [1e6])


def test_compute_physics_warmup_gate_covers_warmup_ramp_and_saturation(
    mod,
):
    assert float(
        _to_numpy(mod.compute_physics_warmup_gate(0, 10, 5))
    ) == pytest.approx(0.0)
    assert float(
        _to_numpy(mod.compute_physics_warmup_gate(10, 10, 5))
    ) == pytest.approx(0.0)
    assert float(
        _to_numpy(mod.compute_physics_warmup_gate(12, 10, 5))
    ) == pytest.approx(0.4)
    assert float(
        _to_numpy(mod.compute_physics_warmup_gate(15, 10, 5))
    ) == pytest.approx(1.0)
    assert float(
        _to_numpy(mod.compute_physics_warmup_gate(999, 10, 5))
    ) == pytest.approx(1.0)


def test_filter_nan_gradients_handles_dense_sparse_and_none(
    mod,
):
    dense = mod.tf_constant(
        [1.0, np.nan, np.inf, -2.0], dtype=mod.tf_float32
    )
    sparse = mod.tf_IndexedSlices(
        values=mod.tf_constant(
            [[np.nan, 1.0], [np.inf, -3.0]],
            dtype=mod.tf_float32,
        ),
        indices=mod.tf_constant([0, 3]),
        dense_shape=mod.tf_constant([5, 2]),
    )

    out = mod.filter_nan_gradients([dense, sparse, None])

    np.testing.assert_allclose(
        _to_numpy(out[0]), [1.0, 0.0, 0.0, -2.0]
    )
    np.testing.assert_allclose(
        _to_numpy(out[1].values), [[0.0, 1.0], [0.0, -3.0]]
    )
    assert out[1].indices.shape[0] == 2
    assert out[2] is None
