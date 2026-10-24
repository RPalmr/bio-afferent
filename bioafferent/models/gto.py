"""Golgi tendon organ model: muscle-tendon force to Ib firing rate."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bioafferent.core import dynamics as dyn
from bioafferent.core import integration as integ
from bioafferent.core.exceptions import InvalidStateError
from bioafferent.core.state import GTOState
from bioafferent.core.validation import (
    as_array,
    check_broadcastable,
    check_dtype_support,
    check_finite,
    is_tracer,
)
from bioafferent.models.gto_parameters import GTOConfig


class GTOOutput(NamedTuple):
    """Per-step GTO readout; arrays share batch shape ``(...)``."""

    ib_rate: jax.Array  # Hz
    force_filtered: jax.Array  # N


class GolgiTendonOrgan:
    """Vectorized GTO: ``step(state, force, dt)``."""

    def __init__(self, config: GTOConfig | None = None) -> None:
        self.config = config or GTOConfig()
        check_dtype_support(self.config.dtype, "GolgiTendonOrgan")

    @property
    def dtype(self):  # type: ignore[no-untyped-def]
        return self.config.numpy_dtype

    def init_state(self, shape: tuple[int, ...] | int = ()) -> GTOState:
        """Create a zero (resting) state with batch shape ``shape``."""
        if isinstance(shape, int):
            shape = (shape,)
        shape = tuple(shape)
        return GTOState(force_filtered=jnp.zeros(shape, dtype=self.dtype))

    def step(
        self,
        state: GTOState,
        force: object,
        dt: float | None = None,
    ) -> tuple[GTOState, GTOOutput]:
        """Advance the GTO by one timestep.

        Args:
            state: current :class:`GTOState` with batch shape ``(...)``.
            force: muscle-tendon force in newton (scalar / broadcastable).
            dt: timestep in seconds (default 1 ms).

        Returns:
            ``(new_state, output)`` with ``ib_rate`` in Hz.
        """
        c = self.config
        dt_val = 0.001 if dt is None else float(dt)
        if not dt_val > 0.0:
            raise InvalidStateError(f"dt must be strictly positive, got {dt_val}.")
        if not isinstance(state, GTOState):
            raise InvalidStateError(f"Expected GTOState, got {type(state).__name__}.")
        batch = tuple(state.force_filtered.shape)
        F = as_array(force, "force", self.dtype)
        check_broadcastable("force", F, batch)
        check_finite("force", F)
        F = jnp.broadcast_to(F, batch)
        if not is_tracer(F) and bool(jnp.any(F < 0.0)):
            raise InvalidStateError("force must be non-negative (tendon compression is unmodelled).")

        def _target(s: jax.Array) -> tuple[jax.Array, jax.Array]:
            tau = jnp.full(batch, c.tau_force, dtype=s.dtype)
            return jnp.broadcast_to(F, batch).astype(s.dtype), tau

        if c.integrator == "exponential":
            f_new = integ.exponential_step(_target, state.force_filtered, dt_val)
        elif c.integrator == "euler":
            f_new = integ.euler_step(lambda s: (F - s) / c.tau_force, state.force_filtered, dt_val)
        else:
            f_new = integ.rk2_step(lambda s: (F - s) / c.tau_force, state.force_filtered, dt_val)
        ib = dyn.gto_transduction(f_new, c.threshold, c.gain, c.saturation_force, c.baseline, c.max_rate)
        return GTOState(force_filtered=f_new), GTOOutput(ib_rate=ib, force_filtered=f_new)

    def rollout(self, state: GTOState, force: jax.Array, dt: float = 0.001) -> tuple[GTOState, GTOOutput]:
        """Run a ``(time, ...)`` force sequence with ``jax.lax.scan``."""

        def _body(s: GTOState, f: jax.Array) -> tuple[GTOState, GTOOutput]:
            return self.step(s, f, dt)

        return jax.lax.scan(_body, state, force)
