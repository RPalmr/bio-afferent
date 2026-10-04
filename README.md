# Bioafferent

JAX-based proprioceptive sensory feedback for physics simulators and
reinforcement-learning environments.

Converts mechanical muscle/joint state (length, velocity, force) into
biologically motivated afferent signals:

- muscle spindle primary afferents (**Ia**) and secondary afferents (**II**),
  following the structural formulation of Mileusnic et al. (2006) with the
  first-order reduction of Vannucci et al. (2017);
- Golgi tendon organ afferents (**Ib**) from muscle-tendon force, combining the
  static properties reported by Houk & Simon with first-order dynamics in the
  style of Lin & Crago (2002);
- continuous firing rates plus discrete spike trains (Poisson, inhomogeneous
  Poisson, gamma-renewal encoding) with explicit JAX PRNG handling.

Core models are pure JAX (jit/vmap/scan compatible) and independent of any
simulator. Optional adapters cover Gymnasium, MuJoCo, and MuJoCo MJX.

> Scientific status: the spindle implements the *structure* of the published
> models (three intrafusal fiber types, polar/sensory tension dynamics,
> fusimotor drives, partial occlusion of Ia). Exact ODE coefficients are
> documented engineering parameters, **not** the copyrighted Table 1 values of
> Mileusnic et al. See `docs/physiology.md` for the full formulation record.

## Installation

```bash
pip install bioafferent
```

With simulator integrations:

```bash
pip install "bioafferent[gymnasium]"   # Gymnasium wrapper
pip install "bioafferent[mujoco]"      # MuJoCo / MJX adapters
```

From source:

```bash
git clone https://github.com/appliedCL/bioafferent.git
cd bioafferent
pip install -e .
```

Requires Python ≥3.10; JAX and NumPy install automatically.

## Quick start

```python
import jax.numpy as jnp
from bioafferent import MuscleSpindle, SpindleConfig

spindle = MuscleSpindle(SpindleConfig())
state = spindle.init_state(shape=())
for _ in range(1000):
    state, out = spindle.step(
        state, length=1.02, velocity=0.0, gamma_dynamic=0.3, gamma_static=0.2, dt=0.001
    )
print(float(out.ia_rate), float(out.ii_rate))
```

Prefer fewer moving parts? `Afferents(n_muscles=2).step(length, velocity,
force)` hides states and defaults the rest; `wrap(gym.make("CartPole-v1"))`
does the same for Gymnasium envs. See `examples/`, `docs/usage.md`, and
`docs/` for details.
