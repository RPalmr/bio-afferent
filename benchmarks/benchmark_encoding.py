"""Benchmark: spike encoding modes."""

import jax
import jax.numpy as jnp
from _common import measure, report

from bioafferent import AfferentEncoder, EncoderConfig


def run_mode(mode: str) -> None:
    enc = AfferentEncoder(EncoderConfig(mode=mode))  # type: ignore[arg-type]
    f = jax.jit(lambda k, s, r: enc.step(k, s, r, 0.001))
    key, st = jax.random.PRNGKey(0), enc.init_state((1000,))
    r = jnp.full((1000,), 50.0)
    jax.block_until_ready(f(key, st, r))
    report(mode, measure(lambda: f(key, st, r)), "batch=1000")


for mode in ("poisson", "inhomogeneous", "gamma"):
    run_mode(mode)
