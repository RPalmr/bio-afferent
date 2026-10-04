"""Physiological tests: spindle ramp-and-hold, gamma modulation, adaptation.

Anchors (qualitative, sourced in docs/physiology.md):

- Vannucci et al. 2017 validation stretch: baseline 0.95 L0, ramp at
  0.11 L0/s, hold; dynamic/static gamma at 70 spk/s conditions.
- Ia is velocity-sensitive (dynamic overshoot on ramp, bag1/dynamic-gamma
  dependent); II is length-following (bag2+chain, static-gamma dependent).
- Fusimotor drive increases sensitivity; static gamma provides discharge at
  short (contracted) lengths.
"""

import jax.numpy as jnp

from bioafferent import MuscleSpindle, SpindleConfig
from tests.conftest import ramp_and_hold


def _run(sp, gd=0.0, gs=0.0, dt=0.001):
    length, velocity = ramp_and_hold()
    st = sp.init_state(())
    ia, ii = [], []
    for L, V in zip(length, velocity, strict=True):
        st, out = sp.step(st, float(L), float(V), gd, gs, dt)
        ia.append(float(out.ia_rate))
        ii.append(float(out.ii_rate))
    return length, jnp.array(ia), jnp.array(ii)


def test_ramp_hold_dynamic_overshoot_and_adaptation(spindle):
    _, ia, _ = _run(spindle)
    n_base, n_ramp = 500, 1100
    base = float(ia[:n_base].mean())
    peak = float(ia[n_base : n_base + n_ramp].max())
    end_hold = float(ia[-200:].mean())
    assert peak > base + 1.0, "Ia must show a dynamic overshoot during the ramp"
    assert end_hold < peak, "Ia must adapt from ramp peak to hold level"
    assert end_hold >= base, "hold discharge must exceed contracted baseline"


def test_hold_response_reflects_length(spindle):
    _, ia, ii = _run(spindle)
    assert float(ii[-200:].mean()) > float(ii[:500].mean())
    assert float(ia[-200:].mean()) > float(ia[:500].mean())


def test_dynamic_gamma_boosts_velocity_sensitivity(spindle):
    _, ia_lo, _ = _run(spindle, gd=0.0, gs=0.0)
    _, ia_hi, _ = _run(spindle, gd=70.0, gs=0.0)
    n_base, n_ramp = 500, 1100

    def overshoot(t):
        return float(t[n_base : n_base + n_ramp].max() - t[:n_base].mean())

    assert overshoot(ia_hi) > 1.5 * overshoot(ia_lo)


def test_static_gamma_raises_resting_and_hold_discharge(spindle):
    # Settled resting discharge at short length (L=0.95), then hold level.
    def settle(gd, gs, L=0.95, steps=3000):
        st = spindle.init_state(())
        for _ in range(steps):
            st, out = spindle.step(st, L, 0.0, gd, gs, 0.001)
        return float(out.ia_rate), float(out.ii_rate)

    ia_lo, ii_lo = settle(0.0, 0.0)
    ia_hi, ii_hi = settle(0.0, 70.0)
    assert ii_hi > ii_lo + 2.0, "static gamma must drive resting II discharge"
    assert ia_hi > ia_lo + 1.0, "static gamma must raise resting Ia discharge"
    _, _, ii_hold_lo = None, None, _run(spindle, 0.0, 0.0)[2][-200:].mean()
    ii_hold_hi = _run(spindle, 0.0, 70.0)[2][-200:].mean()
    assert float(ii_hold_hi) > float(ii_hold_lo) + 5.0


def test_ii_mostly_length_not_velocity(spindle):
    _, ia, ii = _run(spindle, gd=70.0, gs=0.0)
    n_base, n_ramp = 500, 1100
    ia_gain = float(ia[n_base : n_base + n_ramp].max()) / max(
        float(ia[-200:].mean()), 1e-6
    )
    ii_gain = float(ii[n_base : n_base + n_ramp].max()) / max(
        float(ii[-200:].mean()), 1e-6
    )
    assert ia_gain > ii_gain, "Ia must be relatively more velocity-driven than II"


def test_shortening_silences_afferent(spindle):
    st = spindle.init_state(())
    for _ in range(1500):
        st, out = spindle.step(st, 0.97, -0.1, 0.0, 0.0, 0.001)
    assert float(out.ia_rate) <= spindle.config.baseline_ia + 1.0
    assert float(out.ii_rate) <= spindle.config.baseline_ii + 1.0


def test_float64_runs():
    import jax

    was_enabled = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    try:
        sp64 = MuscleSpindle(SpindleConfig(dtype="float64"))
        st = sp64.init_state(())
        st, out = sp64.step(st, 1.02, 0.05, 40.0, 30.0, 0.001)
        assert out.ia_rate.dtype == jnp.float64
        assert bool(jnp.isfinite(out.ia_rate))
    finally:
        jax.config.update("jax_enable_x64", was_enabled)
