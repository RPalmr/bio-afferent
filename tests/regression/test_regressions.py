"""Regression tests: bugs found during development stay fixed.

R1: ``first_order_targets`` broadcast ``(...,3)`` activation against
    ``(...,3)`` fiber params (was ``activation[..., None]`` -> ``(...,3,3)``).
R2: host-side ``bool(...)`` validation raised ``TracerBoolConversionError``
    under ``jax.jit``; validation is now tracer-aware.
R3: gamma-renewal encoder spiked silent (0 Hz) units and emitted a spurious
    t=0 spike on fresh states.
"""

import jax
import jax.numpy as jnp

from bioafferent import AfferentEncoder, EncoderConfig


def test_r1_batched_fiber_broadcast(spindle):
    st, out = spindle.step(
        spindle.init_state((4,)),
        jnp.ones((4,)) * 1.02,
        jnp.ones((4,)) * 0.05,
        50.0,
        30.0,
        0.001,
    )
    assert out.tension.shape == (4, 3)
    st0, out0 = spindle.step(spindle.init_state(()), 1.02, 0.05, 50.0, 30.0)
    assert bool(jnp.allclose(out.ia_rate, out0.ia_rate))


def test_r2_step_jittable_with_python_floats(spindle, gto):
    f = jax.jit(lambda s: spindle.step(s, 1.02, 0.05, 50.0, 30.0, 0.001))
    _, out = f(spindle.init_state(()))
    assert bool(jnp.isfinite(out.ia_rate))
    g = jax.jit(lambda s: gto.step(s, 20.0, 0.001))
    _, gout = g(gto.init_state(()))
    assert bool(jnp.isfinite(gout.ib_rate))


def test_r3_gamma_silent_and_fresh():
    enc = AfferentEncoder(EncoderConfig(mode="gamma"))
    key, st = jax.random.PRNGKey(0), enc.init_state((8,))
    for _ in range(10):
        key, st, out = enc.step(key, st, jnp.zeros((8,)), 0.001)
        assert not bool(out.spikes.any())
        assert bool(jnp.all(st.residual == 0.0))
    _, _, out = enc.step(
        jax.random.PRNGKey(1), enc.init_state((8,)), jnp.full((8,), 80.0), 0.001
    )
    assert not bool(out.spikes.any())
