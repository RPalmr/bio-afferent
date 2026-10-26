"""Published validation fixtures for bioafferent.

Sources:
- Vannucci et al. 2017 (Front. Neurosci. 11:341): ramp-and-hold validation
  with baseline 0.95 L0, ramp 0.11 L0/s, hold; dynamic/static gamma at 70 Hz.
- Houk & Simon 1967 (J. Neurophysiol.): GTO static properties (threshold,
  steep low-force sensitivity, saturation).

These fixtures contain synthetic reference traces approximating the published
qualitative behaviors described in docs/physiology.md. They are intended for
regression and validation tests, not as exact digitized reproductions.
"""

import jax.numpy as jnp


def vannucci_ramp_hold(length: float = 0.95, velocity: float = 0.0, dt: float = 0.001, steps: int = 2000):
    """Generate a synthetic ramp-and-hold input matching Vannucci 2017 validation.

    Returns arrays of length (steps,) for time-step inputs.
    """
    # Baseline (contracted) 500 steps, ramp 0.11 L0/s 1100 steps, hold
    n_base = 500
    n_ramp = 1100
    n_hold = steps - n_base - n_ramp
    lengths = jnp.concatenate(
        [
            jnp.ones(n_base) * 0.95,
            jnp.linspace(0.95, 0.95 + 0.11 * (n_ramp * dt), n_ramp),
            jnp.ones(max(n_hold, 1)) * (0.95 + 0.11 * (n_ramp * dt)),
        ]
    )[:steps]
    velocities = jnp.concatenate(
        [
            jnp.zeros(n_base),
            jnp.ones(n_ramp) * 0.11,
            jnp.zeros(max(n_hold, 1)),
        ]
    )[:steps]
    return lengths, velocities


def houk_gto_force_steps(
    force_values: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 50.0),
    dt: float = 0.001,
    settle_steps: int = 500,
):
    """Generate synthetic GTO force-step inputs for Houk & Simon validation.

    Each force level held for settle_steps to allow dynamics to settle.
    """
    forces = jnp.concatenate([jnp.ones(settle_steps) * f for f in force_values])
    return forces
