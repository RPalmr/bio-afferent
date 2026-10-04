# Changelog

All notable changes to this project are documented here.

## [0.1.0] - 2026-10-03

- Initial release.
- Mileusnic-structure muscle spindle (Ia/II) with dynamic/static gamma drives.
- Golgi tendon organ (Ib) force transducer with saturation dynamics.
- Poisson / inhomogeneous-Poisson / gamma-renewal spike encoding (JAX PRNG).
- JAX `jit` / `vmap` / `lax.scan` compatible core; float32 default.
- Gymnasium `AfferentWrapper`, MuJoCo and MJX adapters (optional dependencies).
- Unit, numerical, physiological, integration, wrapper, and regression tests.
- Benchmarks, examples, and scientific documentation.
