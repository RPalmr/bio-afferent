"""Performance tests: vectorization scaling and latency smoke checks.

Full benchmark methodology (warmup, compile/steady-state split, backend
report) lives in ``benchmarks/``; these tests guard against gross
regressions in scaling behavior on CPU.
"""

import time

import jax
import jax.numpy as jnp
import pytest


def _steady(fn, warmup=3, reps=10):
    for _ in range(warmup):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


@pytest.mark.parametrize("batch", [1, 32])
def test_spindle_step_latency_scales_sublinearly(spindle, batch):
    f = jax.jit(
        lambda s, L, V: spindle.step(s, L, V, 40.0, 30.0, 0.001),
        static_argnums=(),
    )
    st = spindle.init_state((batch,))
    L, V = jnp.full((batch,), 1.02), jnp.full((batch,), 0.03)
    med = _steady(lambda: f(st, L, V))
    assert med < 0.05  # 50 ms budget per step on CPU for small batches


def test_vectorized_1000_envs_runs(spindle):
    batch, muscles = 1000, 4
    f = jax.jit(lambda s, L, V: spindle.step(s, L, V, 40.0, 30.0, 0.001))
    st = spindle.init_state((batch, muscles))
    L = jnp.full((batch, muscles), 1.02)
    V = jnp.full((batch, muscles), 0.03)
    jax.block_until_ready(f(st, L, V))
    med = _steady(lambda: f(st, L, V), warmup=2, reps=5)
    print(f"\n1000x4 spindle step median: {med * 1e3:.3f} ms")
    assert med < 0.5
