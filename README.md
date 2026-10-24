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
pip install "bioafferent[mujoco]"      # MuJoCo adapter
pip install "bioafferent[mjx]"         # MuJoCo MJX adapter
```

From source:

```bash
git clone https://github.com/RPalmr/bio-afferent.git
cd bioafferent
pip install -e .
```

Requires Python ≥3.10; JAX and NumPy install automatically.

## v0.2.0 capabilities

New in this release:

- **Spike encoding**: absolute + relative refractory periods (`refractory_s`, `relative_refractory_s`, `relative_refractory_factor`).
- **Spindle thixotropy**: history-dependent dynamics (`SpindleConfig.enable_thixotropy`) following Blum et al. (2017).
- **Published validation**: synthetic fixtures (`tests/data/published_fixtures.py`) against Vannucci 2017 (spindle ramp-and-hold) and Houk & Simon 1967 (GTO force steps); regression suite in `tests/regression/test_published_validation.py`.
- **Stress-tested** against massive repos (`jax`, `brax`, `mujoco`, `dm_control`, `numpy`, `opensim-core`) — adapter works with real `MjData`; dependency isolation verified.

## Quick start

**General use:**

```python
from bioafferent import MuscleSpindle, SpindleConfig
sp = MuscleSpindle(SpindleConfig())
st = sp.init_state(shape=())
st, out = sp.step(st, length=1.02, velocity=0.0)
print(out.ia_rate, out.ii_rate)
```

**MuJoCo adapter:**

```python
import mujoco
from bioafferent.wrappers.mujoco import extract_mujoco_mechanics
mech = extract_mujoco_mechanics(data, rest_lengths=[0.2]*data.model.nu)
```

**Gymnasium wrapper:**

```python
import gymnasium as gym
from bioafferent import wrap
env = wrap(gym.make("CartPole-v1"))
obs, _ = env.reset()
print(obs.keys())  # includes ia_rate, ii_rate, ib_rate
```

See `examples/`, `docs/usage.md`, and `docs/` for details.
