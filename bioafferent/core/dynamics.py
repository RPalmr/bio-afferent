"""Pure physiological right-hand-side functions (no solver logic here).

This module contains only elementwise JAX operations so every function is
``jit``/``vmap``/``scan`` compatible. Time stepping lives in
`bioafferent.core.integration`; state containers live in `bioafferent.core.state`.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def hill_activation(rate_hz: Array, half_rate: float, exponent: float) -> Array:
    """Steady-state fusimotor activation from motoneuron rate (Hill equation).

    ``a_inf = r^n / (r^n + r_half^n)`` in ``[0, 1]``. This is the static part
    of the Mileusnic et al. (2006) fusimotor drive transduction (biochemical
    Hill-type equation); low-pass filtering to full dynamics is applied by the
    spindle model so the frequency-to-activation path has finite rise time.
    """
    r = jnp.maximum(rate_hz, 0.0)
    num = jnp.power(r, exponent)
    den = num + jnp.power(jnp.asarray(half_rate), exponent)
    return num / jnp.maximum(den, 1e-12)


def first_order_targets(
    tensions: Array,
    length: Array,
    velocity: Array,
    activation: Array,
    stiffness_rest: Array,
    stiffness_gain: Array,
    damping_rest: Array,
    damping_gain: Array,
    tau_rest: Array,
    tau_gain: Array,
    length_threshold: Array,
    prestretch_gain: Array,
) -> tuple[Array, Array]:
    """Quasi-static tension targets and time constants for the fiber triplet.

    All fiber parameters have shape ``(3,)`` (bag1, bag2, chain) and broadcast
    against batch dimensions ``(...)`` of the inputs.

    - Effective stretch ``s = max(L - L_thr + k_pre * a, 0)``: fusimotor
      activation contracts the polar region, preloading the sensory region
      even at constant muscle length (the mechanism behind gamma-driven
      resting discharge).
    - ``target = K(a) * s + B(a) * V`` floored at zero, where
      ``K(a) = K0 * (1 + cK * a)`` and ``B(a) = B0 * (1 + cB * a)``.
    - ``tau = tau0 / (1 + c_tau * a)``: fusimotor drive stiffens the fiber and
      therefore shortens its mechanical lag.
    """
    stretch = jnp.maximum(
        length[..., None] - length_threshold + prestretch_gain * activation, 0.0
    )
    stiff = stiffness_rest * (1.0 + stiffness_gain * activation)
    damp = damping_rest * (1.0 + damping_gain * activation)
    tau = tau_rest / (1.0 + tau_gain * activation)
    target = stiff * stretch + damp * velocity[..., None]
    return jnp.maximum(target, 0.0), tau


def receptor_potentials(
    tensions: Array,
    gain: Array,
    exponent: Array,
    threshold: Array,
) -> Array:
    """Nonlinear sensory transduction per fiber: ``G * max(T - T0, 0)^p``."""
    drive = jnp.maximum(tensions - threshold, 0.0)
    return gain * jnp.power(drive + 1e-12, exponent)


def combine_afferents(
    potentials: Array,
    ia_weights: Array,
    occlusion: float,
    baseline_ia: float,
    baseline_ii: float,
    max_rate: float,
) -> tuple[Array, Array]:
    """Combine fiber receptor potentials into Ia / II firing rates (Hz).

    - ``II = P_bag2 + P_chain`` (direct sum; Mileusnic et al. 2006 structure).
    - ``Ia = S / (1 + k_occ * S)`` with ``S = w . P`` over all three fibers:
      a saturating combination that models the experimentally reported partial
      occlusion of convergent bag1/bag2/chain inputs onto the primary ending.
    - Baselines added, output clipped to ``[0, max_rate]``.
    """
    p_bag1, p_bag2, p_chain = potentials[..., 0], potentials[..., 1], potentials[..., 2]
    weighted = ia_weights[0] * p_bag1 + ia_weights[1] * p_bag2 + ia_weights[2] * p_chain
    ia = weighted / (1.0 + occlusion * weighted)
    ii = p_bag2 + p_chain
    ia_rate = jnp.clip(ia + baseline_ia, 0.0, max_rate)
    ii_rate = jnp.clip(ii + baseline_ii, 0.0, max_rate)
    return ia_rate, ii_rate


def gto_transduction(
    force_filtered: Array,
    threshold: float,
    gain: float,
    saturation_force: float,
    baseline: float,
    max_rate: float,
) -> Array:
    """Force-to-Ib transduction: threshold + gain with saturating asymptote.

    ``Ib = b + G * x / (1 + x / F_sat)`` with ``x = max(F - F_thr, 0)``.
    Monotonic non-decreasing, linear with slope ``G`` (Hz/N) at low force,
    asymptote ``b + G * F_sat``. Encodes the Houk & Simon static properties
    (threshold, high low-force sensitivity, saturation) applied to the
    low-pass filtered force (Lin & Crago style dynamics handled by the caller).
    """
    x = jnp.maximum(force_filtered - threshold, 0.0)
    rate = baseline + gain * x / (1.0 + x / jnp.maximum(saturation_force, 1e-12))
    return jnp.clip(rate, 0.0, max_rate)
