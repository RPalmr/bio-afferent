"""MJX-path example using duck-typed MJX-style data (no mujoco needed).

With real mujoco installed, replace ``FakeMJXData`` by ``mjx_data`` from
``mujoco.mjx``. Pattern: extract mechanics to plain JAX arrays, then run the
JIT-compiled afferent advance (no host round-trips, no device transfers).
"""

import jax
import jax.numpy as jnp

from bioafferent.wrappers.mjx import (
    default_mjx_models,
    extract_mjx_mechanics,
    mjx_afferent_step,
)


class FakeMJXData:
    actuator_length = jnp.array([0.098, 0.101, 0.103])
    actuator_velocity = jnp.array([0.005, 0.010, 0.002])
    actuator_force = jnp.array([12.0, 20.0, 8.0])


spindle, gto = default_mjx_models()
s_state, g_state = spindle.init_state((3,)), gto.init_state((3,))
step = jax.jit(lambda ss, gs, m: mjx_afferent_step(ss, gs, m, spindle, gto, 40.0, 30.0))
for _ in range(500):
    s_state, g_state, aff = step(s_state, g_state, extract_mjx_mechanics(FakeMJXData()))
print("Ia:", aff.ia_rate, "\nII:", aff.ii_rate, "\nIb:", aff.ib_rate)
