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

All modes support an optional absolute refractory period
(``EncoderConfig.refractory_s``), enforced through a countdown carried in
:class:`EncoderState`.

All steps are pure, elementwise, and ``jit``/``vmap`` compatible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from bioafferent.core.exceptions import (
    ConfigurationError,
    InvalidStateError,
    ShapeError,
)
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
    refractory_s: float = 0.0  # absolute refractory period in seconds
    relative_refractory_s: float = 0.0  # relative refractory duration after absolute
    relative_refractory_factor: float = 1.0  # probability multiplier
    # during relative refractory (0=full suppression, 1=no effect)
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
        if not self.refractory_s >= 0.0 or not math.isfinite(self.refractory_s):
            raise ConfigurationError(
                "refractory_s must be finite and non-negative, got "
                f"{self.refractory_s}."
            )
        if (
            not self.relative_refractory_s >= 0.0
            or not math.isfinite(self.relative_refractory_s)
        ):
            raise ConfigurationError(
                "relative_refractory_s must be finite and non-negative, got "
                f"{self.relative_refractory_s}."
            )
        if (
            not (0.0 <= self.relative_refractory_factor <= 1.0)
            or not math.isfinite(self.relative_refractory_factor)
        ):
            raise ConfigurationError(
                "relative_refractory_factor must be in [0, 1], got "
                f"{self.relative_refractory_factor}."
            )
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
        zeros = jnp.zeros(tuple(shape), dtype=dtype)
        return EncoderState(residual=zeros, relative_residual=zeros)

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
        # Relative refractory tracking: separate countdown from absolute
        # When absolute > 0: full suppression. When absolute == 0 but relative > 0:
        # reduced probability by factor. Both count down independently.
        abs_rem = jnp.maximum(state.residual - dt_val, 0.0)
        rel_rem = jnp.where(
            state.residual > 0.0,
            jnp.zeros_like(state.relative_residual),  # relative only after absolute
            jnp.maximum(state.relative_residual - dt_val, 0.0),
        )
        if c.mode in ("poisson", "inhomogeneous"):
            if c.refractory_s <= 0.0 and c.relative_refractory_s <= 0.0:
                spikes = jax.random.bernoulli(subkey, lam).astype(bool)
                return new_key, state, EncoderOutput(spikes=spikes, rate=r)
            available = state.residual <= 0.0
            in_relative = (state.residual <= 0.0) & (state.relative_residual > 0.0)
            prob = jnp.where(
                in_relative,
                jnp.clip(lam * c.relative_refractory_factor, 0.0, 1.0),
                jnp.where(available, lam, 0.0),
            )
            spikes = jax.random.bernoulli(subkey, prob).astype(bool) & available
            # If spike fired: reset both countdowns. Otherwise: count down.
            new_abs_rem = jnp.where(
                spikes,
                jnp.where(
                    c.refractory_s > 0.0,
                    jnp.asarray(c.refractory_s, dtype=dtype),
                    jnp.zeros_like(abs_rem),
                ),
                abs_rem,
            )
            new_rel_rem = jnp.where(
                spikes,
                jnp.where(
                    c.relative_refractory_s > 0.0,
                    jnp.asarray(c.relative_refractory_s, dtype=dtype),
                    jnp.zeros_like(rel_rem),
                ),
                rel_rem,
            )
            return (
                new_key,
                EncoderState(residual=new_abs_rem, relative_residual=new_rel_rem),
                EncoderOutput(spikes=spikes, rate=r),
            )
        # Gamma-renewal with relative refractory: suppress emission probability
        # by factor when in relative refractory period.
        fresh_gamma = state.residual <= 0.0
        waiting_gamma = state.residual - dt_val
        crossed_gamma = waiting_gamma <= 0.0
        active_gamma = r > 0.0
        in_rel_gamma = (state.residual <= 0.0) & (state.relative_residual > 0.0)
        emission_factor = jnp.where(in_rel_gamma, c.relative_refractory_factor, 1.0)
        allowed_gamma = crossed_gamma & active_gamma & (~fresh_gamma)
        suppress_key, subkey_gamma = jax.random.split(subkey)
        suppress_gamma = (
            jax.random.bernoulli(suppress_key, emission_factor)
            .astype(bool)
        )
        fired_gamma = allowed_gamma & suppress_gamma
        safe_rate = jnp.maximum(r, 1e-12)
        isi_gamma = (
            jax.random.gamma(subkey_gamma, float(c.gamma_order), shape=batch)
            / (safe_rate * c.gamma_order)
        )
        isi_gamma = isi_gamma.astype(dtype)
        draw_gamma = jnp.where(
            active_gamma,
            jnp.maximum(isi_gamma, jnp.asarray(c.refractory_s, dtype=dtype)),
            jnp.zeros(batch, dtype=dtype),
        )
        # Suppressed by relative refractory: continue counting down
        new_residual_gamma_abs = jnp.where(
            fired_gamma,
            jnp.where(
                c.refractory_s > 0.0,
                jnp.asarray(c.refractory_s, dtype=dtype),
                jnp.zeros(batch, dtype=dtype),
            ),
            jnp.where(
                allowed_gamma & (~fired_gamma) & in_rel_gamma,
                jnp.maximum(waiting_gamma, 0.0),
                jnp.where(
                    crossed_gamma | fresh_gamma,
                    draw_gamma,
                    jnp.maximum(waiting_gamma, 0.0),
                ),
            ),
        )
        new_residual_gamma_rel = jnp.where(
            fired_gamma,
            jnp.where(
                c.relative_refractory_s > 0.0,
                jnp.asarray(c.relative_refractory_s, dtype=dtype),
                jnp.zeros(batch, dtype=dtype),
            ),
            jnp.where(
                allowed_gamma & (~fired_gamma) & in_rel_gamma,
                jnp.maximum(state.relative_residual - dt_val, 0.0),
                jnp.where(
                    fresh_gamma,
                    jnp.zeros(batch, dtype=dtype),
                    jnp.maximum(state.relative_residual - dt_val, 0.0),
                ),
            ),
        )
        return (
            new_key,
            EncoderState(
                residual=new_residual_gamma_abs,
                relative_residual=new_residual_gamma_rel,
            ),
            EncoderOutput(spikes=fired_gamma, rate=r),
        )


def population_encode(
    values: object,
    centers: object,
    widths: object,
    max_rate_hz: float = 100.0,
) -> jax.Array:
    """Gaussian population code: ``values`` to per-unit rates in Hz.

    Args:
        values: scalar or array with shape ``(...)``.
        centers: preferred values, shape ``(n_units,)``.
        widths: tuning widths (same units as ``values``), shape ``(n_units,)``.
        max_rate_hz: peak firing rate at the preferred value.

    Returns:
        Rates with shape ``(..., n_units)``, each in ``[0, max_rate_hz]``.
        Pass them to :class:`AfferentEncoder` to generate spike trains.
    """
    if not max_rate_hz > 0.0:
        raise ConfigurationError("max_rate_hz must be strictly positive.")
    x = jnp.asarray(values)
    c = jnp.asarray(centers)
    w = jnp.asarray(widths)
    if c.ndim != 1:
        raise ShapeError(f"centers must be 1-D, got shape {tuple(c.shape)}.")
    if w.shape != c.shape:
        raise ShapeError(
            f"widths must match centers shape {tuple(c.shape)}, got {tuple(w.shape)}."
        )
    dtype = jnp.result_type(x, c, w, 1.0)
    x, c, w = x.astype(dtype), c.astype(dtype), w.astype(dtype)
    if not is_tracer(w) and bool(jnp.any(w <= 0.0)):
        raise ConfigurationError("widths must be strictly positive.")
    z = (x[..., None] - c) / w
    return max_rate_hz * jnp.exp(-0.5 * z * z)
