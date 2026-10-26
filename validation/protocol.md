# Bioafferent validation protocol (pre-registration template)

**Status:** template — not yet executed. Filling this document in and freezing
it before calibration is the point. Do not describe bioafferent as
"validated" until every pre-registered criterion below has been met and the
results are reported, including failures.

| Field | Value |
|---|---|
| Protocol version | `0.1-template` |
| Date frozen | `[YYYY-MM-DD]` |
| Model commit | `[git rev-parse HEAD]` |
| Package version | `[bioafferent.__version__]` |
| Responsible researcher | `[name, affiliation, contact]` |
| Independent reviewer | `[name, affiliation]` |
| Data custodian / lab partner | `[name, affiliation]` |
| OSF / registration DOI | `[DOI or "not registered"]` |

## 1. Scope and claims

This protocol tests whether the bioafferent spindle (Ia/II) and Golgi tendon
organ (Ib) models reproduce held-out experimental afferent recordings within
pre-specified tolerances, and whether the implementation is numerically
correct.

**In scope**

- Species: `[e.g. cat]`
- Muscles: `[e.g. soleus, medial gastrocnemius]`
- Preparations: `[e.g. decerebrate in vivo, in situ, human microneurography]`
- Conditions: `[length ramps, force steps, fusimotor stimulation, ...]`
- Signals: `[Ia rate, II rate, Ib rate, spike times]`

**Out of scope**

- Claims about unmeasured species/muscles.
- Closed-loop motor control or behavioural predictions.
- Any condition not represented in the pre-registered validation split.

## 2. Hypotheses

State directional, falsifiable hypotheses before seeing validation outcomes.

- **H1 (spindle rate):** Ia and II rate trajectories on held-out ramp-and-hold
  conditions agree with experiment better than the pre-specified null model
  (constant mean rate) by the primary metric.
- **H2 (spindle dynamics):** Ia peak timing and adaptation index fall within
  the pre-specified tolerance on held-out velocities.
- **H3 (GTO):** Ib force-rate curves on held-out force steps are monotonic and
  within tolerance of the measured curve.
- **H4 (information):** A linear decoder trained on model afferent rates
  predicts held-out length/velocity/force at least as well as a decoder trained
  on the empirical rates (non-inferiority margin `[FILL]`).
- **H5 (simulator path):** MJX and CPU mechanics agree, and the MuJoCo adapter
  maps signed muscle force to non-negative tendon force correctly.

## 3. Data sources and inclusion criteria

For every dataset, record citation, DOI/URL, figure/panel, licence, species,
muscle, preparation, normalisation (`L0`, force units), digitisation method,
digitizer, date, number of independent digitisation repeats, and per-channel
digitisation error. The schema is enforced by `validation/io.py`.

Inclusion criteria:

- Raw or digitised afferent rate/spike data with a documented mechanical input
  (length, velocity, or force).
- Units and normalisation are recoverable.
- Licence permits redistribution or the data can be held by the lab partner
  with a blind-scoring agreement.
- The condition can be assigned to exactly one split.

Exclusion criteria:

- Figures with unresolved axis calibration or ambiguous units.
- Conditions that overlap a calibration condition.
- Data used to tune the model by any prior process (must be declared).

### 3.1 Digitisation

- Use at least two independent digitisation passes per trace.
- Report per-channel digitisation error (rate, time, force).
- Store the digitised CSV, not only the figure.
- Never use the validation/blind traces to choose filter cutoffs, parameter
  bounds, or model structure.

## 4. Splits

Splits are assigned by **condition**, never by random time samples.

| Split | Purpose | Conditions |
|---|---|---|
| Calibration | Fit free parameters | `[list]` |
| Validation | Held-out model comparison | `[list]` |
| Blind | Prospective prediction, outcomes sealed | `[list]` |

Rules:

- A condition may appear in only one split (`validation/splits.py` enforces
  this).
- The validation split is untouched until the calibration configuration and
  code commit are frozen.
- The blind split outcomes are not visible to the modeller until predictions
  are hashed and delivered (`validation/blind.py`).

## 5. Model configuration

### 5.1 Published constants (frozen)

Numerical parameter values from published sources are facts and may be reused
with citation; do not reproduce a table's layout verbatim. Record every
published value and its source in `docs/parameters.md`.

| Parameter | Value | Source | Frozen? |
|---|---|---|---|
| `[name]` | `[value]` | `[citation, table/equation]` | yes |

### 5.2 Free parameters (calibrated)

Only parameters without a defensible published value may be fitted. List them
and justify each one.

| Parameter | Bounds | Rationale |
|---|---|---|
| `[name]` | `[lower, upper]` | `[why free]` |

### 5.3 Fixed numerical settings

- Integrator: `[exponential]`
- Timestep: `[0.001 s]`
- Precision: `[float64 for fitting, float32 for deployment]`
- Parameter provenance: `[link to generated table]`

## 6. Code verification (before calibration)

Run `python -m validation.verification --json` and archive the report.

- JAX vs independent NumPy reference: max absolute error `[threshold]`.
- Timestep convergence: halving `dt` reduces error `[threshold]`.
- float32 vs float64: rate difference `[threshold]`.
- Integrator comparison at small `dt`: `[threshold]`.
- MuJoCo/MJX parity: `python -m validation.simulator --json`.

If verification fails, stop: do not calibrate or validate.

## 7. Calibration

- Fit only the parameters in §5.2, using only the calibration split.
- Optimiser: `[bounded least squares / differential evolution + local]`.
- Report the objective, number of function evaluations, convergence status, and
  parameter posterior or profile-likelihood interval.
- Check structural identifiability (profile likelihood or Fisher information).
  Parameters that are not identifiable must be fixed and reported as such.
- Save the fitted configuration and the calibration data hash
  (`validation/calibrate.py` writes both).

## 8. Validation

Evaluate the frozen configuration on the validation split, then on the blind
split after unsealing. Report all trials and all channels, including failures.

### 8.1 Primary metrics

| Metric | Channel | Criterion | Rationale |
|---|---|---|---|
| NRMSE (range-normalised) | Ia | `<= [0.15]` | rate trajectory accuracy |
| NRMSE (range-normalised) | II | `<= [0.20]` | rate trajectory accuracy |
| NRMSE (range-normalised) | Ib | `<= [0.15]` | force-rate accuracy |
| Peak timing error | Ia | `<= [0.02 s]` | dynamic fidelity |
| Adaptation index error | Ia | `<= [0.15]` | adaptation fidelity |
| Cross-correlation lag | all | `|lag| <= [0.02 s]` | no systematic delay |
| Decoding R² non-inferiority | all | `>= empirical - [0.05]` | information preservation |

Secondary metrics: MAE, bias, Pearson r, R², ISI KS statistic, Victor–Purpura
distance, van Rossum distance, CV of ISI, Fano factor, spike-rate error.

### 8.2 Biological variability reference

Compare model error to the empirical trial-to-trial variability of the same
condition. A model whose error is within the animal's own variability is
preferable to one that merely beats a constant baseline. Report both.

### 8.3 Baselines and ablations

- Constant-mean-rate null model.
- Linear rate model (no spindle structure).
- No-thixotropy and no-occlusion ablations.
- First-order vs full published model (if implemented).
- Shuffled-input negative control.

Report AIC/BIC or WAIC where likelihoods are comparable. Do not select the
model on the validation split.

### 8.4 Sensitivity and uncertainty

- Sobol or Morris sensitivity for free parameters.
- Bootstrap confidence intervals over trials (report `n`, seeds, and CIs).
- If Bayesian calibration is used, posterior predictive checks and 95%
  credible intervals.
- Report parameter uncertainty, not only point estimates.

## 9. Blind prediction procedure

1. Freeze the model: `python -m validation.blind freeze --model spindle --output frozen.json`.
2. Send the lab the frozen-model hash and request inputs-only traces.
3. Run `predict` and deliver the hashed predictions.
4. The lab unseals the measured outputs.
5. Run `score` with the pre-registered criteria; publish the score file and
   hash.
6. An independent reviewer repeats `score` from the archived artifacts.

Any change to the model, parameters, or code after step 1 invalidates the
blind test; start a new freeze and disclose the previous attempt.

## 10. Statistical analysis

- Primary analysis: pre-registered metrics and criteria, no optional stopping.
- Multiplicity: Holm–Bonferroni across primary channels/conditions.
- Effect sizes with confidence intervals, not only p-values.
- Equivalence/non-inferiority tests where the claim is "as good as".
- Report all exclusions and missing data.

## 11. Deviations log

Record every deviation from this protocol, with date, reason, and whether it
was before or after unblinding.

| Date | Deviation | Reason | Before unblinding? |
|---|---|---|---|
| | | | |

## 12. Reporting and reproducibility

Archive:

- this protocol and its registration DOI;
- the exact code commit and package version;
- frozen configurations and hashes;
- digitised data and provenance (or a blind-scoring agreement);
- verification, calibration, validation, and blind-score reports;
- environment lockfile and one-command reproduction instructions.

Report negative and null results. A failed pre-registered criterion is a
result, not a reason to re-tune on the validation split.

## 13. Stop rules

Stop and report failure if:

- code verification fails;
- calibration does not converge or parameters are unidentifiable;
- any primary criterion fails on the blind split;
- the data provenance or licence cannot be established.

## 14. Ethics and licensing

- Respect data licences and any lab data-sharing agreement.
- Do not redistribute restricted raw traces.
- Cite the original experimental papers and the model papers.
- Declare conflicts of interest and funding.

## 15. References

- Mileusnic, M.P. et al. (2006). Mathematical models of proprioceptors. II.
  *J. Neurophysiol.* 96:1696–1708.
- Vannucci, L. et al. (2017). Proprioceptive feedback through a neuromorphic
  muscle spindle model. *Front. Neurosci.* 11:341.
- Houk, J. & Simon, W. (1967). Responses of Golgi tendon organs to forces
  applied to the tendon of the soleus muscle. *J. Neurophysiol.*
- Lin, C.C.K. & Crago, P.E. (2002). Structural model of the muscle spindle.
  *Ann. Biomed. Eng.* 30:68–83. doi:10.1114/1.1433488.
- Lin, C.C.K. & Crago, P.E. (2002). Neural and mechanical contributions to the
  stretch reflex: a model synthesis. *Ann. Biomed. Eng.* 30:54–67.
  doi:10.1114/1.1432692.
- Blum, K.P., Lamotte D'Incamps, B., Zytnicki, D., & Ting, L.H. (2017). Force
  encoding in muscle spindles during stretch of passive muscle.
  *PLoS Comput. Biol.* 13(9):e1005767. doi:10.1371/journal.pcbi.1005767.
- Blum, K.P. et al. (2020). Diverse and complex muscle spindle afferent firing
  properties emerge from multiscale muscle mechanics. *eLife* 9:e55177.
- ASME V&V 20-2009, *Standard for Verification and Validation in Computational
  Fluid Dynamics and Heat Transfer* (V&V vocabulary, adapted).
