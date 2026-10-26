# Validation data

No digitised published traces are bundled with this repository. That is
deliberate: fabricated or unattributed fixtures are not admissible evidence.

To add a dataset:

1. Create a directory with `python -m validation.io validation/data/<dataset_id>`.
2. Fill in `metadata.template.json` and copy it to `metadata.json`.
3. Copy `traces.template.csv` to `traces.csv` and add the digitised samples.
4. Add `spikes.csv` only if the source reports spike times.
5. Validate with `python -m validation.io check validation/data/<dataset_id>`.

Minimum provenance requirements:

- citation and DOI/URL/figure/panel;
- licence;
- species, muscle, preparation;
- normalisation (`L/L0`, `L0/s`, force units);
- digitisation method, digitizer, date, independent repeats, per-channel error;
- split (`calibration`, `validation`, or `blind`) and condition names.

Condition names must be unique across splits. A condition used for calibration
cannot reappear in validation or blind data.

Do not commit restricted raw traces. Keep them outside the repository and
record the data-sharing agreement in `metadata.json`; the blind workflow only
needs inputs until predictions are frozen.
