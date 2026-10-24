# Parameter provenance

Status vocabulary:

- **Published**: the numeric value is taken from, or derived directly from, the
  cited source.
- **Engineering calibration**: the equation structure follows the cited source,
  but the coefficient is this library's documented calibration choice.
- **Numerical**: implementation setting with no physiological claim.

No spindle coefficient is copied from Table 1 of Mileusnic et al. (2006); those
values are copyrighted and paywalled. Every field below is overridable through
`SpindleConfig` / `GTOConfig`, and the defaults are the values used for the
figures in `docs/_static/`.

## SpindleConfig

Per-fiber values are ordered `(bag1, bag2, chain)`. Length is `L/L0`,
velocity is `L0/s`, rates are Hz, time constants are seconds.

| Field | Default | Role | Provenance |
|---|---|---|---|
| `hill_half_rate_dynamic` | `60.0` | Dynamic gamma rate at half activation | Engineering calibration (Hill structure after Mileusnic 2006) |
| `hill_half_rate_static` | `60.0` | Static gamma rate at half activation | Engineering calibration (Hill structure after Mileusnic 2006) |
| `hill_exponent` | `2.0` | Fusimotor Hill cooperativity | Engineering calibration |
| `tau_activation_dynamic` | `343 ms / ln(10)` | Dynamic fusimotor low-pass | **Published**: 90% rise in 343 ms, Vannucci et al. 2017 |
| `tau_activation_static` | `471 ms / ln(10)` | Static fusimotor low-pass | **Published**: 90% rise in 471 ms, Vannucci et al. 2017 |
| `chain_static_scale` | `0.829` | Static drive scaling on chain fibers | Engineering calibration |
| `stiffness_rest` | `(8.0, 15.0, 10.0)` | Resting fiber stiffness | Engineering calibration |
| `stiffness_gain` | `(2.5, 2.0, 1.5)` | Activation-dependent stiffness | Engineering calibration |
| `damping_rest` | `(40.0, 4.0, 3.0)` | Resting fiber damping | Engineering calibration |
| `damping_gain` | `(6.0, 2.0, 1.5)` | Activation-dependent damping | Engineering calibration |
| `tau_fiber_rest` | `(0.060, 0.120, 0.150)` | Resting fiber relaxation time | Engineering calibration |
| `tau_fiber_gain` | `(2.0, 1.5, 1.0)` | Activation-dependent relaxation | Engineering calibration |
| `length_threshold` | `(0.98, 0.97, 0.97)` | Fiber recruitment length | Engineering calibration |
| `prestretch_gain` | `(0.06, 0.08, 0.06)` | Polar preload per unit activation | Engineering adaptation |
| `receptor_gain` | `(13.0, 14.0, 10.0)` | Power-law transducer gain, Hz/unit | Engineering adaptation |
| `receptor_exponent` | `(0.7, 0.6, 0.6)` | Power-law transducer exponent | Engineering adaptation |
| `receptor_threshold` | `(0.5, 0.4, 0.4)` | Tension threshold for discharge | Engineering adaptation |
| `ia_weights` | `(1.0, 0.6, 0.4)` | Ia convergence weights | Engineering calibration |
| `occlusion` | `0.005` | Ia partial-occlusion saturation | Engineering adaptation |
| `baseline_ia` | `5.0` | Ia resting discharge, Hz | Engineering calibration |
| `baseline_ii` | `5.0` | II resting discharge, Hz | Engineering calibration |
| `max_rate` | `300.0` | Hard firing-rate cap, Hz | Numerical |
| `integrator` | `"exponential"` | Tension ODE solver | Numerical |
| `dtype` | `"float32"` | Compute precision | Numerical |

Structural sources: Mileusnic et al. (2006) for the three-fiber spindle
architecture, gamma drives, and Ia/II convergence; Vannucci et al. (2017) for
the first-order tension reduction and fusimotor rise times.

## GTOConfig

Force is in newton, rates are Hz, time constants are seconds.

| Field | Default | Role | Provenance |
|---|---|---|---|
| `tau_force` | `0.050` | First-order force filter | Engineering calibration (first-order style of Lin & Crago 2002) |
| `threshold` | `1.0` | Ib force threshold | Engineering calibration (threshold property after Houk & Simon 1967) |
| `gain` | `4.0` | Low-force slope, Hz/N | Engineering calibration |
| `saturation_force` | `60.0` | Saturating asymptote scale, N | Engineering calibration |
| `baseline` | `2.0` | Resting Ib discharge, Hz | Engineering calibration |
| `max_rate` | `250.0` | Hard firing-rate cap, Hz | Numerical |
| `integrator` | `"exponential"` | Force filter solver | Numerical |
| `dtype` | `"float32"` | Compute precision | Numerical |

Houk & Simon (1967) report the static response properties (threshold, steep
low-force sensitivity, saturation); they do not supply these curve constants.
Lin & Crago (2002) motivate first-order force dynamics; the time constant here
is an engineering choice.

## Calibration targets

The defaults are calibrated to the qualitative features of the Vannucci 2017
validation stretch (baseline `0.95 L0`, ramp `0.11 L0/s`, gamma drive
`70 spk/s`):

- Ia produces a dynamic overshoot with adaptation to a tonic hold rate;
- II follows length with much weaker velocity sensitivity;
- dynamic gamma boosts Ia velocity sensitivity by roughly `1.9x`;
- static gamma raises resting and hold discharge on both channels;
- GTO responses are monotonic and saturating in force.

These are qualitative targets, not a fit to digitised published traces. A
numeric validation suite is tracked in `todo.md`.
