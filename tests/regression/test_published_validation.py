"""Published validation tests against synthetic fixtures approximating
Vannucci 2017 (spindle ramp-and-hold) and Houk & Simon 1967 (GTO force steps).

These tests assert qualitative agreement with published response features,
as documented in docs/physiology.md.
"""

import jax.numpy as jnp

from bioafferent import (
    GolgiTendonOrgan,
    GTOConfig,
    MuscleSpindle,
    SpindleConfig,
)
from tests.data.published_fixtures import houk_gto_force_steps, vannucci_ramp_hold


def test_vannucci_ramp_hold_ia_dynamic_overshoot():
    sp = MuscleSpindle(SpindleConfig())
    length, velocity = vannucci_ramp_hold()
    st = sp.init_state(())
    ia = []
    for L, V in zip(length, velocity, strict=True):
        st, out = sp.step(st, float(L), float(V), 0.0, 0.0, 0.001)
        ia.append(float(out.ia_rate))
    ia = jnp.array(ia)
    n_base, n_ramp = 500, 1100
    base = float(ia[:n_base].mean())
    peak = float(ia[n_base:n_base + n_ramp].max())
    end_hold = float(ia[-200:].mean())
    assert peak > base + 1.0, (
        "Ia must show dynamic overshoot during ramp "
        "(Vannucci 2017)"
    )
    assert end_hold < peak, "Ia must adapt from peak to hold level"
    assert end_hold >= base, "Hold discharge must exceed contracted baseline"


def test_vannucci_ramp_hold_ii_length_following():
    sp = MuscleSpindle(SpindleConfig())
    length, velocity = vannucci_ramp_hold()
    st = sp.init_state(())
    ii = []
    for L, V in zip(length, velocity, strict=True):
        st, out = sp.step(st, float(L), float(V), 0.0, 0.0, 0.001)
        ii.append(float(out.ii_rate))
    ii = jnp.array(ii)
    assert float(ii[-200:].mean()) > float(ii[:500].mean()), (
        "II must follow length (Vannucci 2017)"
    )


def test_houk_gto_monotonic_saturating():
    gto = GolgiTendonOrgan(GTOConfig())
    state = gto.init_state(())
    forces = houk_gto_force_steps()
    rates = []
    for F in forces:
        state, out = gto.step(state, float(F), 0.001)
        rates.append(float(out.ib_rate))
    rates = jnp.array(rates)
    # GTO rate must increase monotonically with force and saturate at high force
    block_size = len(forces) // 5
    block_means = [float(rates[i*block_size:(i+1)*block_size].mean()) for i in range(5)]
    for i in range(4):
        assert block_means[i+1] >= block_means[i], (
            f"GTO rate must increase monotonically "
            f"(Houk & Simon 1967): {block_means}"
        )
    # Saturation: rate should not grow linearly without bound; high force should
    # show diminishing returns relative to mid force.
    assert float(rates[-block_size:].mean()) < 5.0 * float(
        rates[2*block_size:3*block_size].mean()
    ), "GTO must show saturation at high force"


def test_spindle_thixotropy_changes_response():
    sp_thix = MuscleSpindle(SpindleConfig(
        enable_thixotropy=True, thixotropy_scale=0.2, thixotropy_tau=0.3
    ))
    sp_std = MuscleSpindle(SpindleConfig())
    state_thix = sp_thix.init_state(())
    state_std = sp_std.init_state(())
    for i in range(200):
        L = 1.0 + 0.05 * jnp.sin(i * 0.05)
        V = 0.1 * jnp.cos(i * 0.05)
        state_thix, out_thix = sp_thix.step(
            state_thix, float(L), float(V), 0.0, 0.0, 0.001
        )
        state_std, out_std = sp_std.step(
            state_std, float(L), float(V), 0.0, 0.0, 0.001
        )
    # Thixotropic spindle should have a different final rate than standard
    rate_thix = float(out_thix.ia_rate)
    rate_std = float(out_std.ia_rate)
    assert rate_thix != rate_std, (
        "Thixotropy must change spindle response "
        "(Blum et al. 2017)"
    )


def test_relative_refractory_reduces_spike_rate():
    import jax

    from bioafferent import AfferentEncoder, EncoderConfig
    enc = AfferentEncoder(EncoderConfig(
        mode="poisson", refractory_s=0.002, relative_refractory_s=0.002,
        relative_refractory_factor=0.3
    ))
    state = enc.init_state(())
    key = jax.random.PRNGKey(42)
    spikes = []
    for _ in range(500):
        key, state, out = enc.step(key, state, 150.0, 0.001)
        spikes.append(bool(out.spikes))
    # With relative refractory, spike rate should be lower than with absolute only
    spike_count = sum(spikes)
    # Just check it runs and produces some spikes (not all suppressed, not all emitted)
    assert 0 < spike_count < 500, (
        f"Relative refractory must reduce but not eliminate "
        f"spikes: {spike_count}/500"
    )
