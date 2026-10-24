# TODO

Post-0.1.0 implementation work, ordered by impact. `main` is ahead of the
`v0.1.0` tag; the next release is 0.1.1.

## Scientific validation

- [ ] Add a reproducible validation suite against digitised published traces
      (Vannucci 2017 ramp-and-hold for Ia/II, Houk & Simon 1967 force steps for
      Ib). Assert rate trajectories numerically; keep fixtures under
      `tests/data/`.
- [x] Add a parameter provenance table mapping every `SpindleConfig` and
      `GTOConfig` field to a published source or an explicit engineering
      calibration choice. Link it from `docs/physiology.md`.
- [ ] Add a calibration script that regenerates the figures in `docs/_static/`
      and the numbers in `docs/performance.md` from the current defaults.
- [ ] Implement muscle history / thixotropy effects (Blum et al. 2017) behind a
      config flag, with tests.

## Simulator and vectorised RL integration

- [ ] Add an end-to-end MyoSuite or MuJoCo Playground/MJX example that trains a
      policy on Ia/II/Ib observations.
- [x] Add a `lax.scan` rollout API for spindle + GTO + encoder.
- [ ] Support `gym.vector.VectorEnv` (or document the MJX path as the supported
      vectorised route) and remove per-step `jax.device_get` host syncs where
      possible.
- [ ] Replace `_AutoJoints` heuristics with an explicit observation-to-muscle
      mapping. The wrapper currently assumes paired `(position, velocity)`
      entries, one agonist/antagonist pair per joint, and constant tendon force.
- [ ] Make MJX extraction use model rest lengths and force calibration. With
      `rest_lengths=None`, `extract_mjx_mechanics` normalises by the current
      snapshot, so length depends on the pose at the first call.

## Encoding and decoding

- [x] Add an absolute refractory period to `EncoderConfig`.
- [ ] Add a relative refractory option (reduced spike probability after the
      absolute dead time).
- [x] Add spike-to-rate decoding and population/ensemble encoding helpers, with
      tests for ISI distributions and Fano factor.
- [ ] Reduce gamma-renewal cost (0.63 ms per 1000 units in
      `docs/performance.md`) or make Poisson the documented fast path more
      prominently.

## CI, packaging, and docs

- [x] Install `mujoco` in CI so the real adapter tests run.
- [x] Add a Python 3.10-3.13 CI matrix.
- [ ] Add a JAX-version CI matrix.
- [x] Build the Sphinx docs in CI with warnings as errors.
- [ ] Verify and publish the ReadTheDocs build.
- [x] Add `bioafferent/py.typed`.
- [ ] Single-source the version: `pyproject.toml` duplicates
      `bioafferent/_version.py`. Add a trusted-publishing release workflow and
      bump to 0.1.1.
- [x] Document `MechanicalState`, `MechanicalAdapter`, and `MJXAfferents` in
      `docs/api.rst`.
