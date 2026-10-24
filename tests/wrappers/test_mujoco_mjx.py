"""Wrapper tests: MuJoCo / MJX adapters with duck-typed doubles.

Real ``mujoco`` is an optional dependency; extraction logic is validated
against fakes, and JIT-compilation of the MJX path is validated in pure JAX.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bioafferent.core.exceptions import ConfigurationError, UnsupportedBackendError
from bioafferent.wrappers.base import MechanicalState
from bioafferent.wrappers.mjx import (
    MJXAfferents,
    default_mjx_models,
    extract_mjx_mechanics,
    mjx_afferent_step,
)


class _FakeData:
    def __init__(self, n=6):
        self.actuator_length = np.linspace(0.09, 0.11, n)
        self.actuator_velocity = np.full(n, 0.01)
        self.actuator_force = np.full(n, 25.0)


class _BareData:
    pass


@pytest.fixture()
def _mujoco_present(monkeypatch):
    """Pretend mujoco is importable so pure extraction logic is testable."""
    import bioafferent.wrappers.mujoco as mj

    monkeypatch.setattr(mj, "mujoco", object())
    return mj


def test_mujoco_defaults_model_everything(_mujoco_present):
    from bioafferent.wrappers.mujoco import extract_mujoco_mechanics

    mech = extract_mujoco_mechanics(_FakeData())
    assert mech.length.shape == (6,)
    assert bool(jnp.all(mech.length > 0.0))
    assert bool(jnp.all(mech.force >= 0.0))


def test_mujoco_subset_and_rest_lengths(_mujoco_present):
    from bioafferent.wrappers.mujoco import extract_mujoco_mechanics

    mech = extract_mujoco_mechanics(_FakeData(), actuator_ids=(0, 2, 4), rest_lengths=(0.1, 0.1, 0.1))
    assert mech.length.shape == (3,)


def test_mujoco_rejects_bad_ids(_mujoco_present):
    from bioafferent.wrappers.mujoco import extract_mujoco_mechanics

    with pytest.raises(ConfigurationError):
        extract_mujoco_mechanics(_FakeData(), actuator_ids=(0, 99))


def test_mujoco_unsupported_structure(_mujoco_present):
    from bioafferent.wrappers.mujoco import extract_mujoco_mechanics

    with pytest.raises(UnsupportedBackendError):
        extract_mujoco_mechanics(_BareData())


def test_mujoco_missing_backend_raises():
    import bioafferent.wrappers.mujoco as mj

    if mj.mujoco is not None:
        pytest.skip("real mujoco installed; missing-backend path N/A")
    from bioafferent.wrappers.mujoco import extract_mujoco_mechanics

    with pytest.raises(UnsupportedBackendError, match="not installed"):
        extract_mujoco_mechanics(_FakeData())


def test_mjx_extract_defaults():
    data = _FakeData()
    data.actuator_length = jnp.asarray(data.actuator_length)
    data.actuator_velocity = jnp.asarray(data.actuator_velocity)
    data.actuator_force = jnp.asarray(data.actuator_force)
    mech = extract_mjx_mechanics(data)
    assert isinstance(mech, MechanicalState)
    assert mech.length.shape == (6,)
    assert bool(jnp.all(mech.force >= 0.0))


def test_mjx_extract_subset():
    data = _FakeData()
    mech = extract_mjx_mechanics(
        data,
        actuator_ids=jnp.array([0, 2, 4]),
        rest_lengths=jnp.array([0.10, 0.10, 0.10]),
    )
    assert mech.length.shape == (3,)


def test_mjx_extract_unsupported():
    with pytest.raises(UnsupportedBackendError):
        extract_mjx_mechanics(_BareData())


def test_mjx_afferent_step_jit():
    spindle, gto = default_mjx_models()
    s_state = spindle.init_state((3,))
    g_state = gto.init_state((3,))
    mech = MechanicalState(
        length=jnp.full((3,), 1.02),
        velocity=jnp.full((3,), 0.04),
        force=jnp.full((3,), 20.0),
    )
    ref = mjx_afferent_step(s_state, g_state, mech, spindle, gto, 40.0, 30.0)
    jit_step = jax.jit(lambda ss, gs, m: mjx_afferent_step(ss, gs, m, spindle, gto, 40.0, 30.0))
    _, _, aff = jit_step(s_state, g_state, mech)
    assert isinstance(aff, MJXAfferents)
    assert bool(jnp.allclose(aff.ia_rate, ref[2].ia_rate, atol=1e-6))
    assert bool(jnp.allclose(aff.ib_rate, ref[2].ib_rate, atol=1e-6))


def test_mjx_full_path_scan():
    spindle, gto = default_mjx_models()

    def body(carry, k):
        ss, gs = carry
        mech = MechanicalState(
            length=jnp.full((2,), 1.0) + 0.0001 * k,
            velocity=jnp.full((2,), 0.05),
            force=jnp.full((2,), 5.0),
        )
        nss, ngs, aff = mjx_afferent_step(ss, gs, mech, spindle, gto)
        return (nss, ngs), aff.ia_rate

    (_, _), ia = jax.lax.scan(body, (spindle.init_state((2,)), gto.init_state((2,))), jnp.arange(50))
    assert ia.shape == (50, 2)
    assert bool(jnp.isfinite(ia).all())
