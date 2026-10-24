"""Spike-train decoding: causal windowed firing-rate estimates."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from bioafferent.core.exceptions import ConfigurationError, ShapeError


def decode_rates(
    spikes: object,
    dt: float = 0.001,
    window_s: float = 0.05,
) -> Array:
    """Decode firing rates from binary spikes with a causal boxcar window.

    Args:
        spikes: boolean (or numeric, non-zero treated as a spike) array with a
            leading time axis, shape ``(time, ...)``.
        dt: timestep in seconds.
        window_s: decoding window in seconds.

    Returns:
        Rates in Hz with the same shape as ``spikes``. Each sample uses the
        spikes in the trailing window, dividing by the number of samples
        actually available so start-up estimates are unbiased.

    The operation is a vectorised cumulative sum, so it is ``jit``/``vmap``
    compatible and has no Python loop over time.
    """
    if not dt > 0.0:
        raise ConfigurationError(f"dt must be strictly positive, got {dt}.")
    if not window_s > 0.0:
        raise ConfigurationError(f"window_s must be strictly positive, got {window_s}.")
    arr = jnp.asarray(spikes)
    if arr.ndim == 0:
        raise ShapeError(
            "spikes must have a leading time axis, e.g. shape (time,) or "
            "(time, n_units)."
        )
    window = max(1, int(round(window_s / dt)))
    dtype = jnp.result_type(arr, 1.0)
    counts_in = (arr != 0).astype(dtype)
    cumulative = jnp.cumsum(counts_in, axis=0)
    padded = jnp.concatenate([jnp.zeros_like(cumulative[:1]), cumulative], axis=0)
    t = jnp.arange(arr.shape[0])
    start = jnp.maximum(t - window + 1, 0)
    counts = padded[t + 1] - padded[start]
    samples = ((t - start + 1) * dt).reshape((-1,) + (1,) * (arr.ndim - 1))
    return counts / samples
