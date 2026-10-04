"""Shared fixtures and helpers for the Bioafferent test suite."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from bioafferent import (
    AfferentEncoder,
    EncoderConfig,
    GolgiTendonOrgan,
    GTOConfig,
    MuscleSpindle,
    SpindleConfig,
)

DT = 0.001


@pytest.fixture()
def spindle() -> MuscleSpindle:
    return MuscleSpindle(SpindleConfig())


@pytest.fixture()
def gto() -> GolgiTendonOrgan:
    return GolgiTendonOrgan(GTOConfig())


@pytest.fixture()
def encoder_poisson() -> AfferentEncoder:
    return AfferentEncoder(EncoderConfig(mode="poisson"))


def ramp_and_hold(
    n_base: int = 500,
    n_ramp: int = 1100,
    n_hold: int = 1100,
    l0: float = 0.95,
    rate: float = 0.11,
    dt: float = DT,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vannucci-2017-style stretch: hold at 0.95 L0, ramp at 0.11 L0/s, hold.

    Returns ``(length, velocity)`` time series, velocity via finite difference.
    """
    base = jnp.full((n_base,), l0)
    ramp = l0 + rate * dt * jnp.arange(1, n_ramp + 1)
    hold = jnp.full((n_hold,), ramp[-1])
    length = jnp.concatenate([base, ramp, hold])
    velocity = jnp.concatenate([jnp.zeros((1,)), jnp.diff(length) / dt])
    return length, velocity
