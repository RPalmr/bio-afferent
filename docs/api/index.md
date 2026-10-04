# API

Public surface (`bioafferent/__init__.py`):

- Models: `MuscleSpindle`, `GolgiTendonOrgan`, `AfferentEncoder`.
- Configs: `SpindleConfig`, `GTOConfig`, `EncoderConfig`, `IntegrationConfig`.
- States: `SpindleState`, `GTOState`, `EncoderState`.
- Errors: `BioafferentError` (+ `PySpindleError` alias), `ConfigurationError`,
  `ParameterError`, `NumericalStabilityError`, `InvalidStateError`,
  `ShapeError`, `UnsupportedBackendError`.

Step signatures:

- `MuscleSpindle.step(state, length, velocity, gamma_dynamic=0.0,
  gamma_static=0.0, dt=0.001)` — length in L/L0, velocity in L0/s, gamma in Hz.
- `GolgiTendonOrgan.step(state, force, dt=0.001)` — force in newton.
- `AfferentEncoder.step(key, state, rate, dt=0.001)` — rate in Hz; returns
  `(new_key, new_state, output)`.

All steps broadcast scalar / `(muscles,)` / `(batch, muscles)` inputs against
the state batch shape. `rollout(state, ...)` runs `(time, ...)` sequences via
`jax.lax.scan`. Wrappers: `AfferentWrapper`, `DictInfoAdapter`,
`MuJoCoMuscleSet`/`extract_mujoco_mechanics`, `MJXMuscleConfig`/
`extract_mjx_mechanics`/`mjx_afferent_step`.
