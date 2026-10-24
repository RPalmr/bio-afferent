"""Unit tests: spike-train decoding."""

import jax
import jax.numpy as jnp
import pytest

from bioafferent import decode_rates
from bioafferent.core.exceptions import ConfigurationError, ShapeError


def _regular_spikes(n_steps=1000, period=10):
    spikes = jnp.zeros((n_steps,), dtype=bool)
    return spikes.at[::period].set(True)


def test_decode_regular_train():
    dt = 0.001
    spikes = _regular_spikes()
    rates = decode_rates(spikes, dt=dt, window_s=0.05)
    assert rates.shape == spikes.shape
    assert float(rates[0]) == pytest.approx(1000.0)
    assert float(rates[9]) == pytest.approx(100.0)
    assert float(rates[500]) == pytest.approx(100.0)
    assert bool(jnp.all(rates >= 0.0))


def test_decode_multichannel_and_silent_units():
    spikes = jnp.stack([_regular_spikes(), jnp.zeros((1000,), dtype=bool)], axis=1)
    rates = decode_rates(spikes, dt=0.001, window_s=0.05)
    assert rates.shape == (1000, 2)
    assert float(rates[500, 0]) == pytest.approx(100.0)
    assert bool(jnp.all(rates[:, 1] == 0.0))


def test_decode_treats_nonzero_as_spike():
    numeric = _regular_spikes().astype(jnp.float32) * 2.0
    expected = decode_rates(_regular_spikes(), dt=0.001, window_s=0.01)
    actual = decode_rates(numeric, dt=0.001, window_s=0.01)
    assert bool(jnp.array_equal(actual, expected))


def test_decode_jittable():
    spikes = _regular_spikes()
    f = jax.jit(lambda s: decode_rates(s, dt=0.001, window_s=0.01))
    rates = f(spikes)
    assert rates.shape == spikes.shape


def test_decode_validates_inputs():
    with pytest.raises(ShapeError, match="leading time axis"):
        decode_rates(jnp.array(True))
    with pytest.raises(ConfigurationError, match="dt"):
        decode_rates(jnp.zeros((4,), dtype=bool), dt=0.0)
    with pytest.raises(ConfigurationError, match="window_s"):
        decode_rates(jnp.zeros((4,), dtype=bool), window_s=0.0)
