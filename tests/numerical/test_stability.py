"""Numerical tests: stability across input regimes and timesteps."""

import jax.numpy as jnp
import pytest

from bioafferent import GolgiTendonOrgan, MuscleSpindle


def _assert_finite_state(state, out, prefix=""):
    for arr in (state.tension, out.ia_rate, out.ii_rate):
        assert bool(jnp.isfinite(arr).all()), f"non-finite {prefix}"
    assert bool(jnp.all(out.ia_rate >= 0.0))
    assert bool(jnp.all(out.ii_rate >= 0.0))


@pytest.mark.parametrize("integrator", ["euler", "rk2", "exponential"])
@pytest.mark.parametrize("dt", [0.00025, 0.001, 0.004])
def test_spindle_regimes_stable(integrator, dt):
    from bioafferent import SpindleConfig

    sp = MuscleSpindle(SpindleConfig(integrator=integrator))
    cases = [
        (1.0, 0.0, 0.0, 0.0),  # rest
        (1.0, 0.0, 120.0, 100.0),  # rest + strong fusimotor
        (1.05, 0.0, 0.0, 0.0),  # constant stretch
        (1.05, 0.2, 60.0, 40.0),  # fast ramp + gamma
        (0.96, -0.2, 60.0, 40.0),  # fast shortening
        (1.15, 0.5, 150.0, 150.0),  # extreme stretch
    ]
    for L, V, gd, gs in cases:
        st = sp.init_state(())
        for _ in range(2000):
            st, out = sp.step(st, L, V, gd, gs, dt)
        _assert_finite_state(st, out, f"L={L} V={V} dt={dt} {integrator}")


def test_spindle_long_run_no_drift():
    import numpy as np

    from bioafferent import SpindleConfig

    sp = MuscleSpindle(SpindleConfig())
    st = sp.init_state(())
    for _ in range(20000):  # 20 s simulated
        st, out = sp.step(st, 1.02, 0.0, 30.0, 20.0, 0.001)
    assert bool(jnp.isfinite(out.ia_rate))
    assert 0.0 <= float(out.ia_rate) <= 300.0
    assert 0.0 <= float(out.ii_rate) <= 300.0
    assert bool(np.all(np.asarray(st.tension) >= 0.0))


@pytest.mark.parametrize("dt", [0.00025, 0.001, 0.004])
def test_gto_regimes_stable(dt):
    gto = GolgiTendonOrgan()
    for F in (0.0, 1.0, 20.0, 200.0, 2000.0):
        st = gto.init_state(())
        for _ in range(3000):
            st, out = gto.step(st, F, dt)
        assert bool(jnp.isfinite(out.ib_rate))
        assert 0.0 <= float(out.ib_rate) <= 250.0
