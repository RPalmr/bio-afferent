"""Benchmark: GTO step across batch sizes."""

import jax
import jax.numpy as jnp
from _common import measure, report

from bioafferent import GolgiTendonOrgan


def run_batch(gto: GolgiTendonOrgan, batch: int) -> None:
    f = jax.jit(lambda s, F: gto.step(s, F, 0.001))
    st = gto.init_state((batch,))
    F = jnp.full((batch,), 20.0)
    jax.block_until_ready(f(st, F))
    report("gto", measure(lambda: f(st, F)), f"batch={batch}")


gto = GolgiTendonOrgan()
for batch in (1, 10, 100, 1000):
    run_batch(gto, batch)
