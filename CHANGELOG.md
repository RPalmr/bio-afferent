# Changelog

All notable changes to this project are documented here.

## [0.1.1] - 2026-10-24

- Add `Afferents.rollout` using `jax.lax.scan`.
- Add an absolute refractory period to `EncoderConfig`.
- Add `decode_rates` and `population_encode` helpers.
- Add per-field parameter provenance documentation.
- Add `py.typed`, a Python 3.10-3.13 CI matrix, MuJoCo CI coverage, a docs
  build job, and a trusted-publishing workflow.
- Add a separate `mjx` extra that installs `mujoco-mjx`.
- Fix repository URLs and remove scratch scripts.

## [0.1.0] - 2026-10-03

- Initial release.
- Mileusnic-structure muscle spindle (Ia/II) with dynamic/static gamma drives.
- Golgi tendon organ (Ib) force transducer with saturation dynamics.
- Poisson / inhomogeneous-Poisson / gamma-renewal spike encoding (JAX PRNG).
- JAX `jit` / `vmap` / `lax.scan` compatible core; float32 default.
- Gymnasium `AfferentWrapper`, MuJoCo and MJX adapters (optional dependencies).
- Unit, numerical, physiological, integration, wrapper, and regression tests.
- Benchmarks, examples, and scientific documentation.
