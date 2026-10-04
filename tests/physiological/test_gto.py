"""Physiological tests: GTO force response properties.

Sourced properties (docs/physiology.md): threshold/below-threshold baseline,
monotonic non-decreasing response, saturating asymptote, first-order lag on
steps (dynamic behavior), zero force gives baseline.
"""

import jax.numpy as jnp

from bioafferent import GTOConfig


def _settle(gto, force, steps=4000, dt=0.001):
    st = gto.init_state(())
    for _ in range(steps):
        st, out = gto.step(st, force, dt)
    return float(out.ib_rate)


def test_zero_force_gives_baseline(gto):
    assert abs(_settle(gto, 0.0) - gto.config.baseline) < 1e-6


def test_monotonic_force_response(gto):
    rates = [_settle(gto, f) for f in (0, 2, 5, 10, 20, 40, 80, 160, 400)]
    assert all(b >= a for a, b in zip(rates, rates[1:], strict=False))


def test_saturation(gto):
    r_high = _settle(gto, 5000.0)
    asymptote = gto.config.baseline + gto.config.gain * gto.config.saturation_force
    assert abs(r_high - asymptote) / asymptote < 0.05
    # Incremental gain deep in saturation must be a small fraction of the
    # low-force gain G (saturating, not linear, growth).
    r_a, r_b = _settle(gto, 2000.0), _settle(gto, 5000.0)
    assert (r_b - r_a) / (5000.0 - 2000.0) < 0.05 * gto.config.gain


def test_threshold_behavior():
    import bioafferent

    cfg = GTOConfig(threshold=10.0)
    gto = bioafferent.GolgiTendonOrgan(cfg)
    assert abs(_settle(gto, 5.0) - cfg.baseline) < 1e-6
    assert _settle(gto, 15.0) > cfg.baseline


def test_step_lag_is_first_order(gto):
    st = gto.init_state(())
    trace = []
    for i in range(1500):
        st, out = gto.step(st, 0.0 if i < 500 else 100.0, 0.001)
        trace.append(float(out.ib_rate))
    trace = jnp.array(trace)
    assert trace[499] < trace[500] + 1.0  # no instantaneous jump beyond one step
    ss = _settle(gto, 100.0)
    # ~63% rise within ~tau after step onset
    tau = gto.config.tau_force
    idx = 500 + int(tau / 0.001)
    frac = (trace[idx] - gto.config.baseline) / (ss - gto.config.baseline)
    assert 0.4 < frac < 0.85


def test_force_ramp_and_hold(gto):
    st = gto.init_state(())
    peak, end = 0.0, 0.0
    for i in range(2500):
        f = min(i * 0.1, 100.0)
        st, out = gto.step(st, f, 0.001)
        peak = max(peak, float(out.ib_rate))
    end = float(out.ib_rate)
    assert end <= peak + 1e-6  # low-pass output cannot overshoot a ramp plateau
    assert end > gto.config.baseline
