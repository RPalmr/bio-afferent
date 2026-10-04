"""Integration tests: JAX transforms, batch equivalence, scan rollouts."""

import jax
import jax.numpy as jnp

from bioafferent import AfferentEncoder, EncoderConfig
from tests.conftest import ramp_and_hold


def test_spindle_jit_matches_eager(spindle):
    st = spindle.init_state((3,))
    args = (jnp.full((3,), 1.03), jnp.full((3,), 0.02), 40.0, 30.0, 0.001)
    ref_st, ref_out = spindle.step(st, *args)
    jit_step = jax.jit(lambda s, L, V: spindle.step(s, L, V, 40.0, 30.0, 0.001))
    jst, jout = jit_step(st, args[0], args[1])
    assert bool(jnp.allclose(ref_out.ia_rate, jout.ia_rate, atol=1e-6))
    assert bool(jnp.allclose(ref_st.tension, jst.tension, atol=1e-6))


def test_spindle_vmap_matches_scalar(spindle):
    L = jnp.array([0.98, 1.0, 1.03, 1.06])
    V = jnp.array([0.0, 0.01, 0.05, -0.02])
    batched_st, batched_out = spindle.step(spindle.init_state((4,)), L, V, 40.0, 30.0)
    singles = [
        spindle.step(spindle.init_state(()), float(li), float(vi), 40.0, 30.0)
        for li, vi in zip(L, V, strict=True)
    ]
    assert bool(
        jnp.allclose(
            batched_out.ia_rate, jnp.array([o.ia_rate for _, o in singles]), atol=1e-6
        )
    )
    assert bool(
        jnp.allclose(
            batched_out.ii_rate, jnp.array([o.ii_rate for _, o in singles]), atol=1e-6
        )
    )


def test_spindle_scan_rollout_matches_loop(spindle):
    length, velocity = ramp_and_hold(n_base=50, n_ramp=100, n_hold=50)
    final, stacked = spindle.rollout(
        spindle.init_state(()), length, velocity, 40.0, 30.0, 0.001
    )
    assert stacked.ia_rate.shape == length.shape
    st = spindle.init_state(())
    for L, V in zip(length, velocity, strict=True):
        st, out = spindle.step(st, float(L), float(V), 40.0, 30.0, 0.001)
    assert bool(jnp.allclose(stacked.ia_rate[-1], out.ia_rate, atol=1e-5))
    assert bool(jnp.allclose(final.tension, st.tension, atol=1e-5))


def test_scan_is_jittable(spindle):
    length, velocity = ramp_and_hold(n_base=20, n_ramp=40, n_hold=20)
    f = jax.jit(lambda s: spindle.rollout(s, length, velocity, 40.0, 30.0, 0.001))
    final, stacked = f(spindle.init_state(()))
    assert stacked.ii_rate.shape == length.shape
    assert bool(jnp.isfinite(stacked.ia_rate).all())


def test_batch_env_x_muscle(spindle):
    st = spindle.init_state((8, 6))  # 8 envs x 6 muscles
    st2, out = spindle.step(
        st, jnp.full((8, 6), 1.02), jnp.full((8, 6), 0.03), 50.0, 25.0
    )
    assert out.ia_rate.shape == (8, 6)
    assert bool(jnp.isfinite(out.ia_rate).all())


def test_gto_jit_vmap_scan(gto):
    F = jnp.array([0.0, 5.0, 50.0, 500.0])
    _, ref = gto.step(gto.init_state((4,)), F)
    f = jax.jit(jax.vmap(lambda s, x: gto.step(s, x), in_axes=(0, 0)))
    _, out = f(gto.init_state((4,)), F)
    assert bool(jnp.allclose(ref.ib_rate, out.ib_rate, atol=1e-6))
    forces = jnp.linspace(0.0, 100.0, 300)
    _, stacked = gto.rollout(gto.init_state(()), forces)
    assert stacked.ib_rate.shape == (300,)


def test_encoder_scan_over_rates():
    enc = AfferentEncoder(EncoderConfig(mode="inhomogeneous"))
    rates = jnp.concatenate([jnp.full((100,), 10.0), jnp.full((100,), 80.0)])

    def body(carry, r):
        key, st = carry
        key, st, out = enc.step(key, st, r, 0.001)
        return (key, st), out.spikes

    (_, _), spikes = jax.lax.scan(
        body, (jax.random.PRNGKey(0), enc.init_state(())), rates
    )
    assert spikes.shape == (200,)
    assert int(spikes[100:].sum()) > int(spikes[:100].sum())
