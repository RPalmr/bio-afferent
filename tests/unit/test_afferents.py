"""Unit tests: Afferents facade — shapes, parity, reset, seeding."""

import jax.numpy as jnp

from bioafferent import AfferentRates, Afferents, EncoderConfig


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
    import pytest

    with pytest.raises(ValueError):
        Afferents(0)
