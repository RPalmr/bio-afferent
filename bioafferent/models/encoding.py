"""Spike encoding: continuous firing rates to discrete spike trains.

Three encoders, all with explicit JAX PRNG keys (never global RNG state):

- ``poisson``: homogeneous Poisson; per-step spike probability ``r * dt``.
  Exact in the ``r * dt << 1`` regime; inputs violating ``r * dt <= 1`` are
  rejected so probabilities stay valid.
- ``inhomogeneous``: time-varying-rate Poisson via per-step Bernoulli
  thinning with the instantaneous rate. Exact for piecewise-constant rates
  at the simulation timestep; same ``r * dt <= 1`` validity requirement,
  documented in ``docs/physiology.md``.
- ``gamma``: gamma-renewal (Erlang) process of integer ``order``: ISIs drawn
  from ``Gamma(shape=order, mean=1/r)``. Implemented exactly via a
  time-to-next-spike residual carried in :class:`EncoderState`.

All steps are pure, elementwise, and ``jit``/``vmap`` compatible. ``EncoderState``
is only advanced by the ``gamma`` mode; Poisson modes return it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from bioafferent.core.exceptions import ConfigurationError, InvalidStateError
from bioafferent.core.state import EncoderState
from bioafferent.core.validation import (
    as_array,
    check_broadcastable,
    check_dtype_support,
    check_finite,
    is_tracer,
)


@dataclass(frozen=True)
class EncoderConfig:
    """Immutable encoder configuration."""

    mode: str = "poisson"  # one of {"poisson", "inhomogeneous", "gamma"}
    gamma_order: int = 4  # renewal order (shape); >= 1, integer
    max_rate_hz: float = 1000.0  # validity cap for r * dt <= 1 checks
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.mode not in ("poisson", "inhomogeneous", "gamma"):
            raise ConfigurationError(
                f"Unknown encoder mode {self.mode!r}. "
                "Choose 'poisson', 'inhomogeneous', or 'gamma'."
            )
        if not isinstance(self.gamma_order, int) or self.gamma_order < 1:
            raise ConfigurationError(
                f"gamma_order must be a positive integer, got {self.gamma_order}."
            )
        if not self.max_rate_hz > 0.0:
            raise ConfigurationError("max_rate_hz must be strictly positive.")
        if self.dtype not in ("float32", "float64"):
            raise ConfigurationError(
                f"dtype must be 'float32' or 'float64', got {self.dtype!r}."
            )


class EncoderOutput(NamedTuple):
    """Per-step encoder readout; arrays share batch shape ``(...)``."""

    spikes: jax.Array  # bool array: True where a spike occurred this step
    rate: jax.Array  # Hz input rate echoed for convenience


class AfferentEncoder:
    """Vectorized spike encoder: ``step(key, state, rate, dt)``."""

    def __init__(self, config: EncoderConfig | None = None) -> None:
        self.config = config or EncoderConfig()
        check_dtype_support(self.config.dtype, "AfferentEncoder")

    def init_state(self, shape: tuple[int, ...] | int = ()) -> EncoderState:
        """Zero residual state with batch shape ``shape``."""
        if isinstance(shape, int):
            shape = (shape,)
        dtype = jnp.float32 if self.config.dtype == "float32" else jnp.float64
        return EncoderState(residual=jnp.zeros(tuple(shape), dtype=dtype))

    def step(
        self,
        key: jax.Array,
        state: EncoderState,
        rate: object,
        dt: float | None = None,
    ) -> tuple[jax.Array, EncoderState, EncoderOutput]:
        """Encode one timestep.

        Args:
            key: JAX PRNG key; consumed functionally, advanced key returned.
            state: current :class:`EncoderState` with batch shape ``(...)``.
            rate: firing rate in Hz (scalar / broadcastable to state shape).
            dt: timestep in seconds (default 1 ms).

        Returns:
            ``(new_key, new_state, output)``.
        """
        c = self.config
        dt_val = 0.001 if dt is None else float(dt)
        if not dt_val > 0.0:
            raise InvalidStateError(f"dt must be strictly positive, got {dt_val}.")
        if not isinstance(state, EncoderState):
            raise InvalidStateError(
                f"Expected EncoderState, got {type(state).__name__}."
            )
        batch = tuple(state.residual.shape)
        dtype = state.residual.dtype
        r = as_array(rate, "rate", dtype)
        check_broadcastable("rate", r, batch)
        check_finite("rate", r)
        r = jnp.broadcast_to(r, batch)
        if not is_tracer(r) and bool(jnp.any(r < 0.0)):
            raise InvalidStateError("firing rate must be non-negative (Hz).")
        lam = r * dt_val
        if not is_tracer(lam) and bool(jnp.any(lam > 1.0)):
            raise InvalidStateError(
                f"rate*dt must be <= 1 for Bernoulli-step encoding, got max "
                f"{float(jnp.max(lam)):.3f}. Reduce dt or rate "
                f"(dt={dt_val}, max rate={float(jnp.max(r)):.1f} Hz)."
            )

        new_key, subkey = jax.random.split(key)
        if c.mode in ("poisson", "inhomogeneous"):
            spikes = jax.random.bernoulli(subkey, lam).astype(bool)
            return new_key, state, EncoderOutput(spikes=spikes, rate=r)
        # Gamma-renewal: countdown time-to-next-spike; on crossing, emit a
        # spike and draw Gamma(order, scale=1/(r*order)) ISI (mean 1/r).
        # Exact-zero residuals mark fresh/silent units: they draw an ISI
        # without spiking so simulations never emit a spurious t=0 spike and
        # zero-rate units stay silent with residual pinned at zero.
        fresh = state.residual <= 0.0
        waiting = state.residual - dt_val
        crossed = waiting <= 0.0
        active = r > 0.0
        fired = crossed & active & (~fresh)
        safe_rate = jnp.maximum(r, 1e-12)
        isi = jax.random.gamma(subkey, float(c.gamma_order), shape=batch) / (
            safe_rate * c.gamma_order
        )
        isi = isi.astype(dtype)
        draw = jnp.where(active, isi, jnp.zeros(batch, dtype=dtype))
        new_residual = jnp.where(crossed | fresh, draw, jnp.maximum(waiting, 0.0))
        return (
            new_key,
            EncoderState(residual=new_residual),
            EncoderOutput(spikes=fired, rate=r),
        )
