"""User-facing verification script for Bioafferent. Run:  python3 user_check.py.

Exercises the public API end to end (spindle, GTO, spike encoding, JAX
transforms, and the Gymnasium wrapper when installed) with plain asserts.
"""

import sys

import jax
import jax.numpy as jnp

from bioafferent import (
    AfferentEncoder,
    EncoderConfig,
    GolgiTendonOrgan,
    MuscleSpindle,
)


def check(name, cond):
    assert cond, f"FAILED: {name}"
    print(f"  ok: {name}")


print("== 1. spindle ramp-and-hold ==")
spindle = MuscleSpindle()
state = spindle.init_state(())
ia_trace = []
for i in range(2700):
    t = i * 0.001
    if t < 0.5:
        L, V = 0.95, 0.0
    elif t < 1.6:
        L, V = 0.95 + 0.11 * (t - 0.5), 0.11
    else:
        L, V = 1.071, 0.0
    state, out = spindle.step(
        state, L, V, gamma_dynamic=40.0, gamma_static=30.0, dt=0.001
    )
    ia_trace.append(float(out.ia_rate))
ia_trace = jnp.array(ia_trace)
print(f"  baseline={float(ia_trace[:500].mean()):.1f} Hz  "
      f"peak={float(ia_trace[500:1600].max()):.1f} Hz  "
      f"hold={float(ia_trace[-200:].mean()):.1f} Hz  II={float(out.ii_rate):.1f} Hz")
check("dynamic overshoot then adaptation",
      ia_trace[500:1600].max() > ia_trace[-200:].mean() > ia_trace[:500].mean())

print("== 2. GTO force response ==")
gto = GolgiTendonOrgan()
gstate = gto.init_state(())
rates = {}
for force in (0.0, 20.0, 200.0, 5000.0):
    for _ in range(3000):
        gstate, gout = gto.step(gstate, force, dt=0.001)
    rates[force] = float(gout.ib_rate)
print("  Ib rates:", {f: round(r, 1) for f, r in rates.items()})
check("monotonic + saturating",
      rates[0.0] <= rates[20.0] <= rates[200.0] <= rates[5000.0] <= 250.0)

print("== 3. spike encoding ==")
key = jax.random.PRNGKey(0)
for mode in ("poisson", "inhomogeneous", "gamma"):
    enc = AfferentEncoder(EncoderConfig(mode=mode))
    estate = enc.init_state((50,))
    total = 0
    for _ in range(1000):  # 1 s at 50 Hz -> expect ~50 spikes/unit
        key, estate, eout = enc.step(key, estate, jnp.full((50,), 50.0), 0.001)
        total += int(eout.spikes.sum())
    mean_isi_rate = total / 50.0
    print(f"  {mode:14s} ~{mean_isi_rate:.1f} spikes/unit/s (expect ~50)")
    check(f"{mode} mean rate within 20%", abs(mean_isi_rate - 50.0) < 10.0)

print("== 4. jax.jit / jax.vmap ==")
jitted = jax.jit(lambda s: spindle.step(s, 1.02, 0.05, 40.0, 30.0, 0.001))
_, jout = jitted(spindle.init_state(()))
check("jit runs and is finite", bool(jnp.isfinite(jout.ia_rate)))
batched = jax.vmap(lambda s, x: spindle.step(s, x, 0.05, 40.0, 30.0, 0.001))
_, vout = batched(spindle.init_state((8,)), jnp.full((8,), 1.02))
check("vmap over 8 muscles", vout.ia_rate.shape == (8,))

print("== 5. gymnasium wrapper ==")
try:
    import numpy as np

    from bioafferent.wrappers.gymnasium import AfferentWrapper, DictInfoAdapter
    import gymnasium as gym

    class ToyEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(-1, 1, (2,), np.float32)
            self.action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(2, np.float32), self._info()

        def step(self, a):
            return np.zeros(2, np.float32), 0.0, False, False, self._info()

        def _info(self):
            return {
                "muscle_length": np.full(2, 1.02),
                "muscle_velocity": np.full(2, 0.02),
                "muscle_force": np.full(2, 15.0),
            }

    env = AfferentWrapper(ToyEnv(), DictInfoAdapter(2), 2,
                          observation_mode="dict", seed=0)
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(np.zeros(1, np.float32))
    check("dict obs has afferents",
          set(obs) >= {"obs", "ia_rate", "ii_rate", "ib_rate"})
    print(f"  Ia={obs['ia_rate']} II={obs['ii_rate']} Ib={obs['ib_rate']}")
except ImportError:
    print("  skipped: gymnasium not installed")

print("\nALL USER CHECKS PASSED")
