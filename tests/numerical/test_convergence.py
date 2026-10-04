"""Numerical tests: timestep convergence (dt, dt/2, dt/4)."""

import jax.numpy as jnp

from bioafferent import GolgiTendonOrgan, MuscleSpindle, SpindleConfig


def _stretch_protocol(t):
    """Analytic ramp-and-hold: 0.5 s base @0.95, 1.1 s ramp @0.11 L0/s, hold."""
    ramp = 0.95 + 0.11 * jnp.clip(t - 0.5, 0.0, 1.1)
    length = jnp.where(t < 0.5, 0.95, ramp)
    velocity = jnp.where((t >= 0.5) & (t <= 1.6), 0.11, 0.0)
    return length, velocity


def _run_spindle(sp, dt):
    n = int(round(2.7 / dt))
    st = sp.init_state(())
    ia, ii = [], []
    for i in range(n):
        L, V = _stretch_protocol(i * dt)
        st, out = sp.step(st, float(L), float(V), 40.0, 30.0, dt)
        ia.append(float(out.ia_rate))
        ii.append(float(out.ii_rate))
    return jnp.array(ia), jnp.array(ii)


def test_spindle_converges_with_dt():
    sp = MuscleSpindle(SpindleConfig())
    ia1, ii1 = _run_spindle(sp, 0.001)
    ia2, ii2 = _run_spindle(sp, 0.0005)
    ia4, ii4 = _run_spindle(sp, 0.00025)
    peak = float(jnp.max(ia1))
    d12 = float(jnp.max(jnp.abs(ia1 - ia2[::2])))
    d24 = float(jnp.max(jnp.abs(ia2 - ia4[::2])))
    assert d24 < d12, "error must shrink when dt is halved (1st-order dynamics)"
    assert d24 < 0.02 * peak
    assert float(jnp.max(jnp.abs(ii2 - ii4[::2]))) < 0.02 * float(jnp.max(ii1))


def _run_gto(gto, dt):
    n = int(round(1.5 / dt))
    st = gto.init_state(())
    trace = []
    for i in range(n):
        f = 50.0 if i * dt < 0.75 else 120.0
        st, out = gto.step(st, f, dt)
        trace.append(float(out.ib_rate))
    return jnp.array(trace)


def test_gto_converges_with_dt():
    gto = GolgiTendonOrgan()
    o1 = _run_gto(gto, 0.001)
    o2 = _run_gto(gto, 0.0005)
    o4 = _run_gto(gto, 0.00025)
    peak = float(jnp.max(o1))
    d12 = float(jnp.max(jnp.abs(o1 - o2[::2])))
    d24 = float(jnp.max(jnp.abs(o2 - o4[::2])))
    assert d24 < d12
    assert d24 < 0.02 * peak
