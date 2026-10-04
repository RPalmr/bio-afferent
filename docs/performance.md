# Performance

Methodology (`benchmarks/`): JIT warmup separated from steady-state timing,
`jax.block_until_ready` synchronization, median/mean/stdev over 25 reps,
backend/hardware/dtype/batch reported with every claim.

Design notes:

- All hot paths are elementwise; no Python loops inside compiled regions;
  rollouts use `jax.lax.scan`.
- States are flat pytrees; no per-step allocation beyond JAX buffers.
- Boundary crossings (NumPy in wrappers) are documented per call site; the
  MJX path is transfer-free.
- float32 default; float64 roughly 2x memory traffic (CPU) — enable only with
  justification.

Target: <1 ms additional steady-state sensory-model overhead for ~1000
vectorized environments. Measured results are recorded here after execution
(see verification report); the target is a design goal, not an assumed result.

## Measured results (2026-10-03)

Hardware: Apple Silicon (arm64), macOS; backend: JAX CPU (`cpu` platform);
dtype float32; 25 reps after 5-rep warmup, `block_until_ready` synced.

| workload | batch | median/step |
|---|---|---|
| spindle step | 1 | 0.036 ms |
| spindle step | 100 | 0.039 ms |
| spindle step | 1000 | 0.076 ms |
| GTO step | 1000 | 0.013 ms |
| Poisson encoding | 1000 | 0.051 ms |
| inhomog. Poisson encoding | 1000 | 0.050 ms |
| gamma-renewal encoding | 1000 | 0.634 ms |
| MJX path (spindle+GTO) 1000 envs x 4 muscles | 4000 units | 0.336 ms |
| MJX path (spindle+GTO) 1000 envs x 20 muscles | 20000 units | 0.334 ms |

Target verdict: **met** for the continuous-rate path (0.34 ms at
1000x20, well under 1 ms). Poisson-encoded full path also under 1 ms.
Known cost: gamma-renewal sampling (`jax.random.gamma`) dominates at large
batches (0.63 ms alone at 1000 units); prefer Poisson modes when
microsecond-scale overhead matters at >=1000 units.

Single-env Gymnasium wrapper: ~1.8 ms/step (2 muscles, Poisson encoding),
dominated by eager per-call dispatch and host syncs — expected for
Python-side single environments; it is not the vectorized path (use MJX +
`vmap`/`scan` for throughput).
