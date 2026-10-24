"""Numerical integrators decoupled from physiological equations.

Architecture::

    state_next = integrator.step(rhs_fn, state, dt)

where ``rhs_fn(state) -> time_derivative`` is a pure function supplied by the
model in ``dynamics.py``. Models never implement their own solver loop; all
time stepping goes through :class:`Integrator`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from bioafferent.core.exceptions import ConfigurationError, NumericalStabilityError


@dataclass(frozen=True)
class IntegrationConfig:
    """Numerical integration settings.

    Attributes:
        dt: default timestep in seconds (must be > 0).
        method: one of ``{"euler", "rk2", "exponential"}``.
        check_finite: raise :class:`NumericalStabilityError` when a step
            produces non-finite values.
    """

    dt: float = 0.001
    method: str = "exponential"
    check_finite: bool = True

    def __post_init__(self) -> None:
        if not self.dt > 0.0:
            raise ConfigurationError(f"dt must be strictly positive, got {self.dt}. Use e.g. dt=0.001 (1 ms).")
        if self.method not in ("euler", "rk2", "exponential"):
            raise ConfigurationError(
                f"Unknown integration method {self.method!r}. Choose from 'euler', 'rk2', 'exponential'."
            )


RhsFn = Callable[[Array], Array]


def euler_step(rhs_fn: RhsFn, state: Array, dt: float) -> Array:
    """Explicit Euler: ``x + dt * f(x)``."""
    return state + dt * rhs_fn(state)


def rk2_step(rhs_fn: RhsFn, state: Array, dt: float) -> Array:
    """Midpoint RK2: second-order, still fully explicit and JIT friendly."""
    k1 = rhs_fn(state)
    k2 = rhs_fn(state + 0.5 * dt * k1)
    return state + dt * k2


def exponential_step(rhs_fn: Callable[[Array], tuple[Array, Array]], state: Array, dt: float) -> Array:
    """Exponential-Euler for relaxational dynamics ``dx/dt = (target - x)/tau``.

    The ``rhs_fn`` must return the tuple ``(target, tau)`` with shapes matching
    ``state``: ``target`` is the quasi-static target and ``tau`` the (positive)
    time constant in seconds. The update
    ``x + (target - x) * (1 - exp(-dt / tau))`` is unconditionally stable for
    any ``dt`` and exact for constant targets, which is why it is the default
    for the spindle/GTO first-order relaxational dynamics.
    """
    target, tau = rhs_fn(state)
    tau_safe = jnp.maximum(tau, 1e-9)
    alpha = 1.0 - jnp.exp(-dt / tau_safe)
    return state + (target - state) * alpha


def integrate(rhs_fn: Callable[[Array], Any], state: Array, dt: float, method: str) -> Array:
    """Dispatch to the selected explicit integrator."""
    if method == "euler":
        return euler_step(rhs_fn, state, dt)
    if method == "rk2":
        return rk2_step(rhs_fn, state, dt)
    if method == "exponential":
        return exponential_step(rhs_fn, state, dt)
    raise ConfigurationError(f"Unknown integration method {method!r}. Choose from 'euler', 'rk2', 'exponential'.")


def check_stability(value: Array, context: str) -> Array:
    """Raise :class:`NumericalStabilityError` on non-finite values (host check).

    Must be called *outside* JIT-compiled paths (it synchronizes). Hot paths
    instead clamp/avoid NaN construction; this is used by tests and by the
    non-compiled validation helpers.
    """
    bad = jnp.logical_not(jnp.all(jnp.isfinite(value)))
    if bool(jax.device_get(bad)):
        raise NumericalStabilityError(
            f"Non-finite values detected {context}. Reduce dt, check input ranges, or file a regression test."
        )
    return value
