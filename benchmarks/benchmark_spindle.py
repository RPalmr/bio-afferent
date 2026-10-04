"""Benchmark: spindle step across batch sizes (single muscle per unit)."""

import jax
import jax.numpy as jnp
from _common import measure, report

from bioafferent import MuscleSpindle


def run_batch(spindle: MuscleSpindle, batch: int) -> None:
    f = jax.jit(lambda s, L, V: spindle.step(s, L, V, 40.0, 30.0, 0.001))
    st = spindle.init_state((batch,))
    L = jnp.full((batch,), 1.02)
    V = jnp.full((batch,), 0.03)
    jax.block_until_ready(f(st, L, V))  # compile
    report("spindle", measure(lambda: f(st, L, V)), f"batch={batch}")


spindle = MuscleSpindle()
for batch in (1, 10, 100, 1000):
    run_batch(spindle, batch)
