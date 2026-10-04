# Architecture

Dependency direction: `core <- models <- wrappers`.

- **core**: `exceptions` (domain errors), `validation` (shape/finite checks;
  data checks skip gracefully under JIT tracing), `state` (immutable JAX
  pytree states: `SpindleState`, `GTOState`, `EncoderState`), `dynamics`
  (pure elementwise equations), `integration` (Euler / RK2 / exponential-Euler
  steppers; models supply RHS, never solver loops).
- **models**: `spindle` (`MuscleSpindle`), `gto` (`GolgiTendonOrgan`),
  `encoding` (`AfferentEncoder`); immutable validated configs
  (`SpindleConfig`, `GTOConfig`, `EncoderConfig`, `IntegrationConfig`).
- **wrappers**: `base` (`MechanicalState`, `MechanicalAdapter` protocol),
  `gymnasium` (`wrap` zero-config entry point, `AfferentWrapper`,
  `DictInfoAdapter`), `mujoco` (CPU adapter, optional dep), `mjx`
  (JAX-native path, optional dep).
- **configs**: default factories. **utils**: typing, PRNG helpers, SI units,
  logging (quiet by default).

Shape convention: batch shape `(...)` — scalar (single muscle), `(muscles,)`,
or `(batch, muscles)` (envs x muscles). Time series add a leading time axis.
States carry the batch shape; inputs broadcast against it.

Randomness: explicit JAX PRNG keys threaded through `AfferentEncoder.step`
(`new_key, new_state, output`). No global RNG, no `numpy.random` in JAX paths.

dtype: float32 default; float64 requires JAX x64 mode (loud error otherwise).
