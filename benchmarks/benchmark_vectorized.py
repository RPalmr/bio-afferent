"""Benchmark: vectorized envs x muscles (spindle + GTO + MJX path)."""

import jax
import jax.numpy as jnp
from _common import measure, report

from bioafferent.wrappers.base import MechanicalState
from bioafferent.wrappers.mjx import default_mjx_models, mjx_afferent_step


def run_case(batch: int, muscles: int) -> None:
    spindle, gto = default_mjx_models()
    f = jax.jit(
        lambda ss, gs, m: mjx_afferent_step(ss, gs, m, spindle, gto, 40.0, 30.0)
    )
    mech = MechanicalState(
        length=jnp.full((batch, muscles), 1.02),
        velocity=jnp.full((batch, muscles), 0.03),
        force=jnp.full((batch, muscles), 20.0),
    )
    ss = spindle.init_state((batch, muscles))
    gs = gto.init_state((batch, muscles))
    jax.block_until_ready(f(ss, gs, mech))
    report(
        "mjx-path",
        measure(lambda: f(ss, gs, mech)),
        f"envs={batch} muscles={muscles}",
    )


for batch, muscles in ((1, 1), (32, 6), (1000, 4), (1000, 20)):
    run_case(batch, muscles)
