"""Muscle spindle model: Ia / II afferents with gamma fusimotor drives.

Formulation record (full details in ``docs/physiology.md``):

- Structure follows Mileusnic et al. (2006): three intrafusal fiber models
  (bag1, bag2, chain) driven by fascicle length and fusimotor activation;
  dynamic gamma drive acts on bag1, static gamma drive on bag2/chain; the
  secondary (II) ending sums bag2 + chain; the primary (Ia) ending combines
  all three with partial occlusion.
- Tension dynamics use the first-order reduction of Vannucci et al. (2017):
  quasi-static tension targets with velocity-dependent (damping) terms and
  activation-dependent stiffness/damping/lag, integrated with the
  ``exponential`` (unconditionally stable) integrator by default.
- Fusimotor frequency-to-activation transduction uses a Hill equation plus
  first-order low-pass filtering (Mileusnic et al. 2006 structure; rise times
  after Vannucci et al. 2017: 90% in 343 ms dynamic / 471 ms static).

Shape convention: every input is broadcast against the batch shape ``(...)``
of the state, e.g. scalar (single muscle), ``(muscles,)`` (multi-muscle), or
``(batch, muscles)`` (vectorized environments x muscles).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bioafferent.core import dynamics as dyn
from bioafferent.core import integration as integ
from bioafferent.core.exceptions import InvalidStateError
from bioafferent.core.state import SpindleState
from bioafferent.core.validation import (
    as_array,
    check_broadcastable,
    check_dtype_support,
    check_finite,
    is_tracer,
)
from bioafferent.models.spindle_parameters import SpindleConfig, SpindleOutput


class MuscleSpindle:
    """Vectorized muscle spindle: ``step(state, length, velocity, ...)``."""

    def __init__(self, config: SpindleConfig | None = None) -> None:
        self.config = config or SpindleConfig()
        check_dtype_support(self.config.dtype, "MuscleSpindle")
        c = self.config
        self._k_rest = c.fiber_array("stiffness_rest")
        self._k_gain = c.fiber_array("stiffness_gain")
        self._b_rest = c.fiber_array("damping_rest")
        self._b_gain = c.fiber_array("damping_gain")
        self._tau_rest = c.fiber_array("tau_fiber_rest")
        self._tau_gain = c.fiber_array("tau_fiber_gain")
        self._l_thr = c.fiber_array("length_threshold")
        self._k_pre = c.fiber_array("prestretch_gain")
        self._g = c.fiber_array("receptor_gain")
        self._p = c.fiber_array("receptor_exponent")
        self._t0 = c.fiber_array("receptor_threshold")
        self._w = c.fiber_array("ia_weights")

    @property
    def dtype(self):  # type: ignore[no-untyped-def]
        return self.config.numpy_dtype

    def init_state(self, shape: tuple[int, ...] | int = ()) -> SpindleState:
        """Create a zero state with batch shape ``shape``."""
        if isinstance(shape, int):
            shape = (shape,)
        shape = tuple(shape)
        tension = jnp.zeros(shape + (3,), dtype=self.dtype)
        act = jnp.zeros(shape, dtype=self.dtype)
        return SpindleState(
            tension=tension, activation_dynamic=act, activation_static=act
        )

    def _check_state(self, state: SpindleState) -> tuple[int, ...]:
        if not isinstance(state, SpindleState):
            raise InvalidStateError(
                f"Expected SpindleState, got {type(state).__name__}. "
                "Build it with spindle.init_state(shape)."
            )
        if state.tension.shape[-1:] != (3,):
            raise InvalidStateError(
                f"SpindleState.tension trailing dim must be 3 (bag1, bag2, chain), "
                f"got shape {tuple(state.tension.shape)}."
            )
        batch = tuple(state.tension.shape[:-1])
        if (
            tuple(state.activation_dynamic.shape) != batch
            or tuple(state.activation_static.shape) != batch
        ):
            raise InvalidStateError(
                "SpindleState activation shapes "
                f"{tuple(state.activation_dynamic.shape)}/"
                f"{tuple(state.activation_static.shape)} do not match tension "
                f"batch shape {batch}."
            )
        return batch

    def step(
        self,
        state: SpindleState,
        length: object,
        velocity: object,
        gamma_dynamic: object = 0.0,
        gamma_static: object = 0.0,
        dt: float | None = None,
    ) -> tuple[SpindleState, SpindleOutput]:
        """Advance the spindle by one timestep.

        Args:
            state: current :class:`SpindleState` with batch shape ``(...)``.
            length: fascicle length in units of rest length (L/L0).
            velocity: stretch velocity in L0/s (positive = lengthening).
            gamma_dynamic: dynamic fusimotor rate in Hz (acts on bag1).
            gamma_static: static fusimotor rate in Hz (acts on bag2/chain).
            dt: timestep in seconds (default 1 ms).

        Returns:
            ``(new_state, output)`` with ``SpindleOutput`` rates in Hz.

        Raises:
            InvalidStateError / ShapeError / ValueError on bad state/shapes.
        """
        c = self.config
        dt_val = 0.001 if dt is None else float(dt)
        if not dt_val > 0.0:
            raise InvalidStateError(
                f"dt must be strictly positive, got {dt_val}. Use e.g. dt=0.001."
            )
        batch = self._check_state(state)
        L = as_array(length, "length", self.dtype)
        V = as_array(velocity, "velocity", self.dtype)
        gd = as_array(gamma_dynamic, "gamma_dynamic", self.dtype)
        gs = as_array(gamma_static, "gamma_static", self.dtype)
        for name, arr in (
            ("length", L),
            ("velocity", V),
            ("gamma_dynamic", gd),
            ("gamma_static", gs),
        ):
            check_broadcastable(name, arr, batch)
            check_finite(name, arr)
        L = jnp.broadcast_to(L, batch)
        V = jnp.broadcast_to(V, batch)
        gd = jnp.broadcast_to(gd, batch)
        gs = jnp.broadcast_to(gs, batch)
        if not is_tracer(gd) and bool(jnp.any(gd < 0.0)):
            raise InvalidStateError("gamma rates must be non-negative (Hz).")
        if not is_tracer(gs) and bool(jnp.any(gs < 0.0)):
            raise InvalidStateError("gamma rates must be non-negative (Hz).")
        if not is_tracer(L) and bool(jnp.any(L <= 0.0)):
            raise InvalidStateError("length must be strictly positive (L/L0).")

        # Fusimotor transduction: Hill steady state + exponential low-pass.
        a_d_inf = dyn.hill_activation(gd, c.hill_half_rate_dynamic, c.hill_exponent)
        a_s_inf = dyn.hill_activation(gs, c.hill_half_rate_static, c.hill_exponent)
        alpha_d = 1.0 - jnp.exp(-dt_val / c.tau_activation_dynamic)
        alpha_s = 1.0 - jnp.exp(-dt_val / c.tau_activation_static)
        a_d = state.activation_dynamic + (a_d_inf - state.activation_dynamic) * alpha_d
        a_s = state.activation_static + (a_s_inf - state.activation_static) * alpha_s
        a_d = jnp.clip(a_d, 0.0, 1.0)
        a_s = jnp.clip(a_s, 0.0, 1.0)
        activation = jnp.stack([a_d, a_s, a_s * c.chain_static_scale], axis=-1)

        # Tension dynamics via the configured integrator (default exponential).
        def _targets(t: jax.Array) -> tuple[jax.Array, jax.Array]:
            return dyn.first_order_targets(
                t,
                L,
                V,
                activation,
                self._k_rest,
                self._k_gain,
                self._b_rest,
                self._b_gain,
                self._tau_rest,
                self._tau_gain,
                self._l_thr,
                self._k_pre,
            )

        if c.integrator == "exponential":
            tension = integ.exponential_step(_targets, state.tension, dt_val)
        elif c.integrator == "euler":

            def _euler_rhs(t: jax.Array) -> jax.Array:
                tgt, tau = _targets(t)
                return (tgt - t) / jnp.maximum(tau, 1e-9)

            tension = integ.euler_step(_euler_rhs, state.tension, dt_val)
        else:  # rk2

            def _rk_rhs(t: jax.Array) -> jax.Array:
                tgt, tau = _targets(t)
                return (tgt - t) / jnp.maximum(tau, 1e-9)

            tension = integ.rk2_step(_rk_rhs, state.tension, dt_val)
        tension = jnp.maximum(tension, 0.0)

        potentials = dyn.receptor_potentials(tension, self._g, self._p, self._t0)
        ia_rate, ii_rate = dyn.combine_afferents(
            potentials,
            self._w,
            c.occlusion,
            c.baseline_ia,
            c.baseline_ii,
            c.max_rate,
        )
        new_state = SpindleState(
            tension=tension, activation_dynamic=a_d, activation_static=a_s
        )
        return new_state, SpindleOutput(
            ia_rate=ia_rate,
            ii_rate=ii_rate,
            tension=tension,
            activation_dynamic=a_d,
            activation_static=a_s,
        )

    def rollout(
        self,
        state: SpindleState,
        length: jax.Array,
        velocity: jax.Array,
        gamma_dynamic: jax.Array | float = 0.0,
        gamma_static: jax.Array | float = 0.0,
        dt: float = 0.001,
    ) -> tuple[SpindleState, SpindleOutput]:
        """Run a full ``(time, ...)`` input sequence with ``jax.lax.scan`.

        Leading axis of ``length``/``velocity`` is time; remaining axes must
        broadcast to the state batch shape. Returns final state and stacked
        outputs (each leaf gains a leading time axis).
        """

        def _body(
            s: SpindleState, inputs: tuple[jax.Array, ...]
        ) -> tuple[SpindleState, SpindleOutput]:
            L, V, gd, gs = inputs
            return self.step(s, L, V, gd, gs, dt)

        gd_arr = jnp.broadcast_to(jnp.asarray(gamma_dynamic), length.shape)
        gs_arr = jnp.broadcast_to(jnp.asarray(gamma_static), length.shape)
        return jax.lax.scan(_body, state, (length, velocity, gd_arr, gs_arr))
