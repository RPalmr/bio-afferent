"""Explicit JAX PRNG utilities. No global RNG state anywhere in the library."""

from __future__ import annotations

import jax
from jax import Array


def split_key(key: Array, n: int = 2) -> tuple[Array, ...]:
    """Split a PRNG key into ``n`` independent keys. Returns a tuple."""
    return tuple(jax.random.split(key, n))


def next_key(key: Array) -> tuple[Array, Array]:
    """Return ``(new_key, subkey)``: advance the stream and hand out a subkey."""
    new_key, subkey = jax.random.split(key)
    return new_key, subkey
