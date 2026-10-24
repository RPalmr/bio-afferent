"""Unit tests: spike encoding — shapes, reproducibility, statistics."""

import jax
import jax.numpy as jnp
import pytest

from bioafferent import AfferentEncoder, EncoderConfig, ShapeError, population_encode
from bioafferent.core.exceptions import ConfigurationError, InvalidStateError


@pytest.mark.parametrize("mode", ["poisson", "inhomogeneous", "gamma"])
def test_encoder_shapes(mode):
    enc = AfferentEncoder(EncoderConfig(mode=mode))
    st = enc.init_state((2, 3))
    key = jax.random.PRNGKey(0)
    key2, st2, out = enc.step(key, st, jnp.full((2, 3), 50.0), 0.001)
    assert out.spikes.shape == (2, 3)
    assert out.spikes.dtype == bool
    assert st2.residual.shape == (2, 3)
    assert key2.shape == key.shape


@pytest.mark.parametrize("mode", ["poisson", "inhomogeneous", "gamma"])
def test_encoder_reproducible(mode):
    enc = AfferentEncoder(EncoderConfig(mode=mode))
    key = jax.random.PRNGKey(42)

    def run():
        return enc.step(key, enc.init_state((8,)), jnp.full((8,), 60.0), 0.001)

    _, _, o1 = run()
    _, _, o2 = run()
    assert bool(jnp.array_equal(o1.spikes, o2.spikes))


def test_different_keys_differ_statistically():
    enc = AfferentEncoder(EncoderConfig(mode="poisson"))
    outs = []
    for seed in range(6):
        key = jax.random.PRNGKey(seed)
        st = enc.init_state((200,))
        spikes = []
        for _ in range(50):
            key, st, out = enc.step(key, st, jnp.full((200,), 40.0), 0.001)
            spikes.append(out.spikes)
        outs.append(jnp.stack(spikes))
    counts = jnp.array([o.sum() for o in outs])
    assert len(set(int(c) for c in counts)) > 1  # not all identical


def test_poisson_mean_rate():
    enc = AfferentEncoder(EncoderConfig(mode="poisson"))
    key, st = jax.random.PRNGKey(0), enc.init_state((400,))
    total, steps = 0, 400
    for _ in range(steps):
        key, st, out = enc.step(key, st, jnp.full((400,), 50.0), 0.001)
        total += int(out.spikes.sum())
    measured = total / (400 * steps * 0.001)
    assert measured == pytest.approx(50.0, rel=0.15)


def test_gamma_mean_rate_and_regularity():
    enc = AfferentEncoder(EncoderConfig(mode="gamma", gamma_order=8))
    key, st = jax.random.PRNGKey(1), enc.init_state((60,))
    counts = []
    for _ in range(2000):
        key, st, out = enc.step(key, st, jnp.full((60,), 50.0), 0.001)
        counts.append(out.spikes)
    counts = jnp.stack(counts)
    measured = float(counts.sum()) / (60 * 2000 * 0.001)
    assert measured == pytest.approx(50.0, rel=0.15)
    # gamma-renewal (order 8) is more regular than Poisson: Fano factor < 1
    # measured over 100 ms windows
    windows = counts.reshape(20, 100, 60).sum(axis=1)
    fano = float(windows.var() / jnp.maximum(windows.mean(), 1e-9))
    assert fano < 1.0


def test_zero_rate_never_spikes():
    for mode in ("poisson", "inhomogeneous", "gamma"):
        enc = AfferentEncoder(EncoderConfig(mode=mode))
        key, st = jax.random.PRNGKey(0), enc.init_state((4,))
        for _ in range(20):
            key, st, out = enc.step(key, st, jnp.zeros((4,)), 0.001)
            assert not bool(out.spikes.any())


def test_gamma_no_spurious_first_spike():
    enc = AfferentEncoder(EncoderConfig(mode="gamma"))
    _, _, out = enc.step(
        jax.random.PRNGKey(0), enc.init_state((16,)), jnp.full((16,), 80.0), 0.001
    )
    assert not bool(out.spikes.any())


def test_rejects_negative_rate():
    enc = AfferentEncoder()
    with pytest.raises(InvalidStateError):
        enc.step(jax.random.PRNGKey(0), enc.init_state((2,)), jnp.full((2,), -5.0))


def test_rejects_rate_dt_above_one():
    enc = AfferentEncoder()
    with pytest.raises(InvalidStateError):
        enc.step(jax.random.PRNGKey(0), enc.init_state(()), 2000.0, 0.001)


def test_encoder_jit_vmap():
    enc = AfferentEncoder(EncoderConfig(mode="poisson"))
    f = jax.jit(
        jax.vmap(
            lambda k, r: enc.step(k, enc.init_state(()), r, 0.001),
            in_axes=(0, 0),
        )
    )
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    _, _, out = f(keys, jnp.full((8,), 30.0))
    assert out.spikes.shape == (8,)


@pytest.mark.parametrize("mode", ["poisson", "gamma"])
def test_refractory_prevents_close_spikes(mode):
    refractory = 0.005
    dt = 0.001
    enc = AfferentEncoder(EncoderConfig(mode=mode, refractory_s=refractory))
    key, st = jax.random.PRNGKey(0), enc.init_state(())
    spike_times = []
    for i in range(1000):
        key, st, out = enc.step(key, st, 300.0, dt)
        if bool(out.spikes):
            spike_times.append(i * dt)
    assert len(spike_times) > 5
    gaps = jnp.diff(jnp.asarray(spike_times))
    assert bool(jnp.all(gaps >= refractory - 1e-6))


@pytest.mark.parametrize("mode", ["poisson", "gamma"])
def test_zero_rate_with_refractory_stays_silent(mode):
    enc = AfferentEncoder(EncoderConfig(mode=mode, refractory_s=0.005))
    key, st = jax.random.PRNGKey(0), enc.init_state((4,))
    for _ in range(20):
        key, st, out = enc.step(key, st, jnp.zeros((4,)), 0.001)
        assert not bool(out.spikes.any())


@pytest.mark.parametrize("value", [-1e-3, float("nan"), float("inf")])
def test_refractory_config_rejects_invalid(value):
    with pytest.raises(ConfigurationError):
        EncoderConfig(refractory_s=value)


def test_refractory_encoder_jittable():
    enc = AfferentEncoder(EncoderConfig(mode="poisson", refractory_s=0.002))
    f = jax.jit(lambda k, s: enc.step(k, s, 80.0, 0.001))
    key, st = jax.random.PRNGKey(0), enc.init_state((4,))
    key, st, out = f(key, st)
    assert out.spikes.shape == (4,)
    assert st.residual.shape == (4,)


def test_population_encode_tuning_curves():
    centers = jnp.array([-1.0, 0.0, 1.0])
    widths = jnp.array([0.5, 0.5, 0.5])
    rates = population_encode(0.0, centers, widths, max_rate_hz=80.0)
    assert rates.shape == (3,)
    assert int(jnp.argmax(rates)) == 1
    assert float(rates[1]) == pytest.approx(80.0)
    assert bool(jnp.all(rates <= 80.0))
    batch = population_encode(jnp.array([-1.0, 0.0, 1.0]), centers, widths)
    assert batch.shape == (3, 3)
    assert bool(jnp.allclose(jnp.diag(batch), 100.0))


def test_population_encode_validates_inputs():
    centers = jnp.array([-1.0, 0.0, 1.0])
    with pytest.raises(ShapeError):
        population_encode(0.0, centers, jnp.ones((2,)))
    with pytest.raises(ConfigurationError):
        population_encode(0.0, centers, jnp.array([0.5, -0.5, 0.5]))
    with pytest.raises(ConfigurationError):
        population_encode(0.0, centers, jnp.ones((3,)), max_rate_hz=0.0)


def test_population_encode_jittable():
    centers = jnp.linspace(-1.0, 1.0, 5)
    f = jax.jit(lambda x: population_encode(x, centers, jnp.full((5,), 0.4)))
    assert f(0.2).shape == (5,)
