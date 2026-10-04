"""Unit tests: integrators converge to analytic solutions."""

import jax.numpy as jnp
import pytest

from bioafferent.core import integration as integ
from bioafferent.core.exceptions import ConfigurationError


def _relax_rhs(target=2.0, tau=0.05):
    def rhs(s):
        return jnp.full_like(s, target), jnp.full_like(s, tau)

    return rhs


def test_exponential_exact_for_constant_target():
    x0 = jnp.array([0.0, 1.0])
    out = integ.exponential_step(_relax_rhs(), x0, 0.01)
    expected = 2.0 + (x0 - 2.0) * float(jnp.exp(-0.01 / 0.05))
    assert bool(jnp.allclose(out, expected, atol=1e-7))


def test_euler_first_order_accuracy():
    rhs = _relax_rhs()
    x0 = jnp.array([0.0])
    out = integ.euler_step(lambda s: (rhs(s)[0] - s) / rhs(s)[1], x0, 1e-4)
    expected = 2.0 + (x0 - 2.0) * float(jnp.exp(-1e-4 / 0.05))
    assert bool(jnp.allclose(out, expected, atol=1e-5))


def test_rk2_second_order_beats_euler():
    def f(s):
        return -3.0 * s  # dx/dt = -3x, exact x0*exp(-3t)

    x0, dt = jnp.array([1.0]), 0.01
    exact = float(jnp.exp(-3.0 * dt))
    e_euler = abs(float(integ.euler_step(f, x0, dt)[0]) - exact)
    e_rk2 = abs(float(integ.rk2_step(f, x0, dt)[0]) - exact)
    assert e_rk2 < e_euler
    assert e_rk2 < 1e-5  # midpoint RK2 global error is O(dt^2); dt=0.01


def test_integrate_dispatch():
    x0 = jnp.array([0.5])
    rhs = _relax_rhs()
    assert integ.integrate(rhs, x0, 0.01, "exponential").shape == (1,)
    with pytest.raises(ConfigurationError):
        integ.integrate(rhs, x0, 0.01, "bogus")


def test_exponential_stable_for_huge_dt():
    out = integ.exponential_step(_relax_rhs(tau=0.05), jnp.array([0.0]), 10.0)
    assert bool(jnp.isfinite(out).all())
    assert abs(float(out[0]) - 2.0) < 1e-6
