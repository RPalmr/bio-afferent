"""Unit tests: invalid inputs produce informative errors."""

import jax.numpy as jnp
import pytest

from bioafferent import MuscleSpindle
from bioafferent.core.exceptions import InvalidStateError, ShapeError


def test_spindle_rejects_nan_length(spindle):
    with pytest.raises(ValueError, match="length"):
        spindle.step(spindle.init_state(()), jnp.nan, 0.0)


def test_spindle_rejects_inf_velocity(spindle):
    with pytest.raises(ValueError, match="velocity"):
        spindle.step(spindle.init_state(()), 1.0, jnp.inf)


def test_spindle_rejects_negative_gamma(spindle):
    with pytest.raises(InvalidStateError, match="gamma"):
        spindle.step(spindle.init_state(()), 1.0, 0.0, gamma_dynamic=-3.0)


def test_spindle_rejects_nonpositive_length(spindle):
    with pytest.raises(InvalidStateError, match="length"):
        spindle.step(spindle.init_state(()), 0.0, 0.0)


def test_spindle_rejects_bad_dt(spindle):
    with pytest.raises(InvalidStateError, match="dt"):
        spindle.step(spindle.init_state(()), 1.0, 0.0, dt=0.0)
    with pytest.raises(InvalidStateError, match="dt"):
        spindle.step(spindle.init_state(()), 1.0, 0.0, dt=-0.001)


def test_spindle_rejects_shape_mismatch(spindle):
    with pytest.raises(ShapeError):
        spindle.step(spindle.init_state((3,)), jnp.ones((4,)), jnp.zeros((4,)))


def test_spindle_rejects_wrong_state_type(spindle, gto):
    with pytest.raises(InvalidStateError):
        spindle.step(gto.init_state(()), 1.0, 0.0)  # type: ignore[arg-type]


def test_gto_rejects_negative_force(gto):
    with pytest.raises(InvalidStateError, match="force"):
        gto.step(gto.init_state(()), -5.0)


def test_gto_rejects_nan(gto):
    with pytest.raises(ValueError, match="force"):
        gto.step(gto.init_state(()), jnp.nan)


def test_spindle_wrong_state_object():
    with pytest.raises(InvalidStateError):
        MuscleSpindle().step(object(), 1.0, 0.0)  # type: ignore[arg-type]
