# Validation harness

This directory is the research-grade verification and validation (V&V) harness
for bioafferent. It is **not** a claim that the model is validated. It is the
machinery required to make such a claim falsifiable.

## Status

| Component | Status |
|---|---|
| Independent NumPy reference implementation | implemented |
| JAX vs reference verification | implemented |
| dt / precision / integrator checks | implemented |
| Dataset schema + provenance hashing | implemented |
| Condition-level split + leakage checks | implemented |
| Rate / spike / decoding metrics | implemented |
| Calibration + held-out evaluation | implemented |
| Blind freeze/predict/score workflow | implemented |
| MuJoCo/MJX semantics + parity checks | implemented |
| Digitised published traces | **not included** |
| Executed calibration/validation study | **not done** |
| Independent lab blind validation | **not done** |

Do not use the word "validated" in the README, paper, or pitch until the
pre-registered study in [`protocol.md`](protocol.md) has been executed and
reported, including failures.

## Quick start

```bash
# 1. Code verification (no data required)
python -m validation.verification

# 2. MuJoCo/MJX semantics (requires mujoco/mjx)
python -m validation.simulator

# 3. Create a dataset template and fill in provenance
python -m validation.io validation/data/my_dataset

# 4. Validate a dataset directory
python -m validation.io check validation/data/my_dataset

# 5. Fit on the calibration split, evaluate on validation
python -m validation.calibrate \
    --data-root validation/data \
    --model spindle \
    --output validation/results/fit_spindle.json

# 6. Blind workflow
python -m validation.blind freeze --model spindle --output validation/results/frozen.json
python -m validation.blind predict \
    --model-json validation/results/frozen.json \
    --data validation/data/blind_experiment \
    --output validation/results/predictions
python -m validation.blind score \
    --predictions validation/results/predictions/predictions.csv \
    --truth validation/data/blind_experiment \
    --criteria validation/criteria.example.json \
    --output validation/results/score.json
```

## Layout

```
validation/
    protocol.md          pre-registration template (fill before fitting)
    io.py                dataset schema, provenance, hashing, CLI
    splits.py            condition-level splits and leakage checks
    metrics.py           rate, spike-train, decoding, acceptance metrics
    reference.py         independent NumPy implementation (verification)
    verification.py      JAX-vs-reference and numerical checks
    calibrate.py         bounded parameter fitting + held-out evaluation
    blind.py             freeze/predict/score prospective validation
    simulator.py         MuJoCo/MJX semantics and parity checks
    data/                digitised datasets (not bundled)
    results/             generated reports (git-ignored)
```

## Dataset format

A dataset is a directory with:

- `metadata.json` — citation, DOI, licence, species, muscle, preparation,
  normalisation, digitisation method/date/error, split, conditions;
- `traces.csv` — `trial_id, condition, time_s, length_norm,
  velocity_norm_s, force_n, gamma_dynamic_hz, gamma_static_hz, ia_rate_hz,
  ii_rate_hz, ib_rate_hz`;
- `spikes.csv` (optional) — `trial_id, afferent, time_s`;
- `MANIFEST.json` (optional) — SHA-256 hashes.

A dataset missing provenance or digitisation error is rejected by
`validation.io`. Blind datasets must be inputs-only until predictions are
frozen; `load_dataset` refuses to expose their target columns.

## What counts as validation

A study is validation, not calibration, only if:

1. The protocol was frozen before fitting.
2. Calibration and validation conditions are disjoint.
3. Primary metrics and acceptance thresholds were pre-registered.
4. The model was evaluated unchanged on held-out conditions.
5. Model error is compared with empirical trial-to-trial variability.
6. Negative results and deviations are reported.
7. The blind split, if used, was predicted before unsealing.

## Dependencies

The harness uses NumPy for metrics, SciPy for calibration, and MuJoCo/MJX for
simulator checks:

```bash
pip install "bioafferent[validation,mujoco,mjx]"
```

## Scientific corrections to record

- Numerical parameter values are facts and are not copyrightable; reuse
  published constants with citation, and do not reproduce a table's layout
  verbatim.
- The current defaults are engineering calibration choices. They must be
  replaced or explicitly labelled before any claim of reproducing Mileusnic
  et al. (2006) or Vannucci et al. (2017).
- Blum et al. (2017) is *PLoS Comput. Biol.* 13(9):e1005767, not *eLife*
  6:e23240. Lin & Crago (2002) is *Ann. Biomed. Eng.* 30:68–83, not
  *J. Biomech.*
