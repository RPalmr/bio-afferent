"""Unit tests: Afferents facade — shapes, parity, reset, seeding."""

import jax.numpy as jnp
import pytest

from bioafferent import AfferentRates, Afferents, EncoderConfig, ShapeError


def test_facade_shapes():
    out = Afferents(3).step(1.02, 0.03, 20.0)
    assert isinstance(out, AfferentRates)
    assert out.ia_rate.shape == (3,)
    assert out.spikes is None


def test_facade_matches_functional_api():
    from bioafferent import GolgiTendonOrgan, MuscleSpindle

    aff = Afferents(2)
    out = None
    for _ in range(100):
        out = aff.step(1.03, 0.02, 25.0, gamma_dynamic=40.0, gamma_static=30.0)
    sp, gto = MuscleSpindle(), GolgiTendonOrgan()
    ss, gs = sp.init_state((2,)), gto.init_state((2,))
    for _ in range(100):
        ss, so = sp.step(ss, 1.03, 0.02, 40.0, 30.0)
        gs, go = gto.step(gs, 25.0)
    assert bool(jnp.allclose(out.ia_rate, so.ia_rate, atol=1e-6))
    assert bool(jnp.allclose(out.ib_rate, go.ib_rate, atol=1e-6))


def test_facade_reset_and_reseed():
    aff = Afferents(2, encoder_config=EncoderConfig(mode="poisson"), seed=1)
    for _ in range(50):
        out1 = aff.step(1.05, 0.05, 30.0)
    aff.reset()
    aff.reseed(1)
    for _ in range(50):
        out2 = aff.step(1.05, 0.05, 30.0)
    assert out1.spikes is not None and out2.spikes is not None
    assert bool(jnp.array_equal(out1.spikes, out2.spikes))


def test_facade_rejects_bad_muscle_count():
    with pytest.raises(ValueError):
        Afferents(0)


def test_facade_rollout_matches_step():
    n_steps = 20
    length = jnp.linspace(1.0, 1.05, n_steps)
    velocity = jnp.full((n_steps,), 0.01)
    force = jnp.full((n_steps,), 20.0)

    rolled = Afferents(2)
    out = rolled.rollout(length, velocity, force, gamma_dynamic=40.0, gamma_static=30.0)

    stepped = Afferents(2)
    manual = [
        stepped.step(length[t], velocity[t], force[t], 40.0, 30.0)
        for t in range(n_steps)
    ]
    for field in ("ia_rate", "ii_rate", "ib_rate"):
        expected = jnp.stack([getattr(o, field) for o in manual])
        assert bool(jnp.allclose(getattr(out, field), expected, atol=1e-6))
    assert out.ia_rate.shape == (n_steps, 2)
    assert out.spikes is None

    next_rolled = rolled.step(1.02, 0.0, 10.0)
    next_stepped = stepped.step(1.02, 0.0, 10.0)
    assert bool(jnp.allclose(next_rolled.ia_rate, next_stepped.ia_rate, atol=1e-6))


def test_facade_rollout_encoder_matches_step():
    n_steps = 30
    length = jnp.linspace(1.0, 1.1, n_steps)[:, None]
    cfg = EncoderConfig(mode="poisson")

    rolled = Afferents(2, encoder_config=cfg, seed=5)
    out = rolled.rollout(length, 0.02, 25.0)

    stepped = Afferents(2, encoder_config=cfg, seed=5)
    manual = [stepped.step(length[t], 0.02, 25.0) for t in range(n_steps)]
    assert out.spikes is not None
    assert out.spikes.shape == (n_steps, 6)
    expected = jnp.stack([o.spikes for o in manual])
    assert bool(jnp.array_equal(out.spikes, expected))


def test_facade_rollout_validates_sequence_shapes():
    aff = Afferents(2)
    with pytest.raises(ShapeError, match="leading time axis"):
        aff.rollout(1.0)
    with pytest.raises(ShapeError, match="leading axis"):
        aff.rollout(jnp.ones((5, 2)), velocity=jnp.ones((4, 2)))
    with pytest.raises(ShapeError, match="broadcast-compatible"):
        aff.rollout(jnp.ones((5, 2)), force=jnp.ones((5, 3)))
