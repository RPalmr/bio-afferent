"""Digitised-trace schema, provenance validation, and content hashing.

A validation dataset is a directory::

    dataset_id/
        metadata.json   # provenance, units, split, digitisation error
        traces.csv      # one row per time sample, long format
        spikes.csv      # optional: trial_id, afferent, time_s
        MANIFEST.json   # optional content hashes

The schema is deliberately strict: a trace without a citation, licence,
normalisation record, and digitisation error is not admissible evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0"
SPLITS = ("calibration", "validation", "blind")

INPUT_COLUMNS = (
    "length_norm",
    "velocity_norm_s",
    "force_n",
    "gamma_dynamic_hz",
    "gamma_static_hz",
)
TARGET_COLUMNS = ("ia_rate_hz", "ii_rate_hz", "ib_rate_hz")
TRACE_COLUMNS = ("trial_id", "condition", "time_s", *INPUT_COLUMNS, *TARGET_COLUMNS)
SPIKE_COLUMNS = ("trial_id", "afferent", "time_s")
AFFERENTS = ("ia", "ii", "ib")

REQUIRED_METADATA = (
    "schema_version",
    "dataset_id",
    "split",
    "source",
    "license",
    "species",
    "muscle",
    "preparation",
    "normalization",
    "digitization",
    "conditions",
)
REQUIRED_SOURCE = ("citation",)
REQUIRED_NORMALIZATION = ("length", "velocity", "force")
REQUIRED_DIGITIZATION = ("method", "digitizer", "date", "independent_repeats", "error")


class DatasetError(ValueError):
    """Raised when a validation dataset violates the schema."""


@dataclass(frozen=True)
class Trial:
    """One experimental condition/trial with inputs, targets, and spikes."""

    trial_id: str
    condition: str
    time_s: np.ndarray
    inputs: dict[str, np.ndarray] = field(default_factory=dict)
    targets: dict[str, np.ndarray] = field(default_factory=dict)
    spikes: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def dt(self) -> float:
        if self.time_s.size < 2:
            raise DatasetError(f"Trial {self.trial_id!r} needs at least two samples to infer dt.")
        diffs = np.diff(self.time_s)
        if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-12):
            raise DatasetError(f"Trial {self.trial_id!r} is not uniformly sampled.")
        return float(diffs[0])


@dataclass(frozen=True)
class Dataset:
    """A validated dataset directory."""

    metadata: dict[str, Any]
    trials: tuple[Trial, ...]
    source_path: Path | None = None

    @property
    def split(self) -> str:
        return str(self.metadata["split"])

    @property
    def dataset_id(self) -> str:
        return str(self.metadata["dataset_id"])

    @property
    def conditions(self) -> tuple[str, ...]:
        return tuple(sorted({trial.condition for trial in self.trials}))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialise {type(value).__name__}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetError(f"{path} must contain a JSON object.")
    return payload


def _parse_float(value: str | None, *, field_name: str, trial_id: str) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise DatasetError(f"Non-numeric {field_name!r} in trial {trial_id!r}: {value!r}") from exc
    if not np.isfinite(parsed):
        raise DatasetError(f"Non-finite {field_name!r} in trial {trial_id!r}.")
    return parsed


def _read_traces(path: Path, *, allow_targets: bool) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise DatasetError(f"Missing traces file: {path}")
    grouped: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DatasetError(f"{path} has no header.")
        unknown = set(reader.fieldnames) - set(TRACE_COLUMNS)
        if unknown:
            raise DatasetError(f"Unknown trace columns: {sorted(unknown)}")
        if "trial_id" not in reader.fieldnames or "time_s" not in reader.fieldnames:
            raise DatasetError("traces.csv must contain 'trial_id' and 'time_s'.")
        for row_number, row in enumerate(reader, start=2):
            trial_id = (row.get("trial_id") or "").strip()
            if not trial_id:
                raise DatasetError(f"Blank trial_id at {path}:{row_number}.")
            condition = (row.get("condition") or "").strip()
            if not condition:
                raise DatasetError(f"Blank condition at {path}:{row_number}.")
            time = _parse_float(row.get("time_s"), field_name="time_s", trial_id=trial_id)
            if time is None:
                raise DatasetError(f"Blank time_s at {path}:{row_number}.")
            if not allow_targets:
                exposed = [name for name in TARGET_COLUMNS if (row.get(name) or "").strip()]
                if exposed:
                    raise DatasetError(
                        f"Blind/input-only data contains target values {exposed} at {path}:{row_number}; "
                        "seal the outcomes until predictions are frozen."
                    )
            record = grouped.setdefault(trial_id, {"condition": condition, "rows": []})
            if record["condition"] != condition:
                raise DatasetError(f"Trial {trial_id!r} mixes conditions.")
            record["rows"].append(row)
    return grouped


def _read_spikes(path: Path) -> dict[str, dict[str, list[float]]]:
    if not path.exists():
        return {}
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DatasetError(f"{path} has no header.")
        unknown = set(reader.fieldnames) - set(SPIKE_COLUMNS)
        if unknown:
            raise DatasetError(f"Unknown spike columns: {sorted(unknown)}")
        for row_number, row in enumerate(reader, start=2):
            trial_id = (row.get("trial_id") or "").strip()
            afferent = (row.get("afferent") or "").strip().lower()
            if not trial_id:
                raise DatasetError(f"Blank trial_id at {path}:{row_number}.")
            if afferent not in AFFERENTS:
                raise DatasetError(f"Unknown afferent {afferent!r} at {path}:{row_number}; expected {AFFERENTS}.")
            time = _parse_float(row.get("time_s"), field_name="time_s", trial_id=trial_id)
            if time is None:
                raise DatasetError(f"Blank spike time at {path}:{row_number}.")
            grouped[trial_id][afferent].append(float(time))
    return {trial: dict(channels) for trial, channels in grouped.items()}


def load_dataset(path: str | Path, *, allow_blind_targets: bool = False) -> Dataset:
    """Load and validate a dataset directory.

    ``allow_blind_targets`` must remain false until predictions are frozen and
    the blind outcomes are unsealed for scoring.
    """
    root = Path(path)
    if root.is_file():
        root = root.parent
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise DatasetError(f"Missing metadata.json in {root}.")
    metadata = _read_json(metadata_path)
    validate_metadata(metadata)
    grouped = _read_traces(root / "traces.csv", allow_targets=allow_blind_targets or metadata["split"] != "blind")
    spikes = _read_spikes(root / "spikes.csv")
    trials: list[Trial] = []
    for trial_id, record in grouped.items():
        rows = record["rows"]
        rows.sort(key=lambda row: float(row["time_s"]))
        time = np.array([float(row["time_s"]) for row in rows], dtype=float)
        if time.size < 2:
            raise DatasetError(f"Trial {trial_id!r} needs at least two samples.")
        if np.any(np.diff(time) <= 0.0):
            raise DatasetError(f"Trial {trial_id!r} time must be strictly increasing.")
        inputs: dict[str, np.ndarray] = {}
        targets: dict[str, np.ndarray] = {}
        for name in INPUT_COLUMNS + TARGET_COLUMNS:
            if name not in rows[0]:
                continue
            values = [_parse_float(row.get(name), field_name=name, trial_id=trial_id) for row in rows]
            if any(value is None for value in values):
                continue
            array = np.array(values, dtype=float)
            if name in INPUT_COLUMNS:
                inputs[name] = array
            else:
                if not allow_blind_targets and metadata["split"] == "blind":
                    raise DatasetError(f"Blind trial {trial_id!r} contains target {name!r}.")
                targets[name] = array
        if not inputs:
            raise DatasetError(f"Trial {trial_id!r} has no input channels.")
        if not targets and not allow_blind_targets and metadata["split"] != "blind":
            raise DatasetError(f"Trial {trial_id!r} has no target channels.")
        trial_spikes = {
            name: np.sort(np.asarray(values, dtype=float)) for name, values in spikes.get(trial_id, {}).items()
        }
        trials.append(
            Trial(
                trial_id=trial_id,
                condition=str(record["condition"]),
                time_s=time,
                inputs=inputs,
                targets=targets,
                spikes=trial_spikes,
            )
        )
    for trial_id in spikes:
        if trial_id not in grouped:
            raise DatasetError(f"spikes.csv references unknown trial {trial_id!r}.")
    dataset = Dataset(metadata=metadata, trials=tuple(sorted(trials, key=lambda t: t.trial_id)), source_path=root)
    validate_dataset(dataset)
    return dataset


def validate_metadata(metadata: dict[str, Any]) -> None:
    """Raise :class:`DatasetError` listing every missing/malformed field."""
    errors: list[str] = []
    for key in REQUIRED_METADATA:
        if key not in metadata:
            errors.append(f"missing metadata field {key!r}")
    if errors:
        raise DatasetError("; ".join(errors))
    if metadata["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}, got {metadata['schema_version']!r}")
    if metadata["split"] not in SPLITS:
        errors.append(f"split must be one of {SPLITS}, got {metadata['split']!r}")
    if not isinstance(metadata["source"], dict):
        errors.append("source must be an object")
    else:
        for key in REQUIRED_SOURCE:
            if not metadata["source"].get(key):
                errors.append(f"source.{key} is required")
        if not any(metadata["source"].get(key) for key in ("doi", "url", "figure")):
            errors.append("source needs at least one of doi/url/figure")
    if not metadata.get("license"):
        errors.append("license is required")
    if not isinstance(metadata["normalization"], dict):
        errors.append("normalization must be an object")
    else:
        for key in REQUIRED_NORMALIZATION:
            if not metadata["normalization"].get(key):
                errors.append(f"normalization.{key} is required")
    if not isinstance(metadata["digitization"], dict):
        errors.append("digitization must be an object")
    else:
        for key in REQUIRED_DIGITIZATION:
            if key not in metadata["digitization"]:
                errors.append(f"digitization.{key} is required")
        repeats = metadata["digitization"].get("independent_repeats")
        if not isinstance(repeats, int) or repeats < 1:
            errors.append("digitization.independent_repeats must be a positive integer")
        errors_in = metadata["digitization"].get("error")
        if not isinstance(errors_in, dict) or not errors_in:
            errors.append("digitization.error must be a non-empty object")
    conditions = metadata.get("conditions")
    if not isinstance(conditions, list) or not conditions or not all(isinstance(c, str) and c for c in conditions):
        errors.append("conditions must be a non-empty list of strings")
    if errors:
        raise DatasetError("; ".join(errors))


def validate_dataset(dataset: Dataset) -> None:
    """Validate cross-field invariants and condition provenance."""
    validate_metadata(dataset.metadata)
    declared = set(dataset.metadata["conditions"])
    seen_ids: set[str] = set()
    for trial in dataset.trials:
        if trial.trial_id in seen_ids:
            raise DatasetError(f"Duplicate trial_id {trial.trial_id!r}.")
        seen_ids.add(trial.trial_id)
        if trial.condition not in declared:
            raise DatasetError(f"Trial {trial.trial_id!r} condition {trial.condition!r} is not in metadata.conditions.")
        n = trial.time_s.size
        for name, array in {**trial.inputs, **trial.targets}.items():
            if array.shape != (n,):
                raise DatasetError(
                    f"Trial {trial.trial_id!r} channel {name!r} has shape {array.shape}, expected {(n,)}."
                )
            if not np.all(np.isfinite(array)):
                raise DatasetError(f"Trial {trial.trial_id!r} channel {name!r} contains non-finite values.")
        for name, times in trial.spikes.items():
            if times.ndim != 1 or not np.all(np.isfinite(times)):
                raise DatasetError(f"Trial {trial.trial_id!r} spikes {name!r} must be a finite 1-D array.")
            if np.any(np.diff(times) < 0.0):
                raise DatasetError(f"Trial {trial.trial_id!r} spikes {name!r} must be sorted.")
    if not dataset.trials:
        raise DatasetError("Dataset has no trials.")


def save_dataset(dataset: Dataset, path: str | Path, *, overwrite: bool = False) -> Path:
    """Write a dataset directory in the canonical schema."""
    root = Path(path)
    if root.exists():
        if not overwrite:
            raise DatasetError(f"{root} already exists; pass overwrite=True to replace it.")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    validate_dataset(dataset)
    (root / "metadata.json").write_text(
        json.dumps(dataset.metadata, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    with (root / "traces.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_COLUMNS)
        writer.writeheader()
        for trial in dataset.trials:
            for index in range(trial.time_s.size):
                row: dict[str, Any] = {
                    "trial_id": trial.trial_id,
                    "condition": trial.condition,
                    "time_s": f"{trial.time_s[index]:.17g}",
                }
                for name in INPUT_COLUMNS + TARGET_COLUMNS:
                    array = trial.inputs.get(name, trial.targets.get(name))
                    row[name] = "" if array is None else f"{array[index]:.17g}"
                writer.writerow(row)
    spike_trials = [trial for trial in dataset.trials if trial.spikes]
    if spike_trials:
        with (root / "spikes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SPIKE_COLUMNS)
            writer.writeheader()
            for trial in spike_trials:
                for afferent, times in trial.spikes.items():
                    for time in times:
                        writer.writerow({"trial_id": trial.trial_id, "afferent": afferent, "time_s": f"{time:.17g}"})
    write_manifest(root)
    return root


def dataset_hash(dataset: Dataset) -> str:
    """Stable SHA-256 over metadata and every array in the dataset."""
    digest = hashlib.sha256()
    digest.update(json.dumps(dataset.metadata, sort_keys=True, default=_json_default).encode("utf-8"))
    for trial in dataset.trials:
        digest.update(trial.trial_id.encode("utf-8"))
        digest.update(trial.condition.encode("utf-8"))
        digest.update(trial.time_s.astype(np.float64).tobytes())
        for name in sorted(trial.inputs):
            digest.update(name.encode("utf-8"))
            digest.update(trial.inputs[name].astype(np.float64).tobytes())
        for name in sorted(trial.targets):
            digest.update(name.encode("utf-8"))
            digest.update(trial.targets[name].astype(np.float64).tobytes())
        for name in sorted(trial.spikes):
            digest.update(name.encode("utf-8"))
            digest.update(trial.spikes[name].astype(np.float64).tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(root: str | Path) -> dict[str, str]:
    """Write SHA-256 hashes for the canonical dataset files."""
    root = Path(root)
    files = {}
    for name in ("metadata.json", "traces.csv", "spikes.csv"):
        path = root / name
        if path.exists():
            files[name] = _sha256_file(path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "files": files,
        "combined_sha256": hashlib.sha256(
            "".join(f"{name}:{digest}" for name, digest in sorted(files.items())).encode("utf-8")
        ).hexdigest(),
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify_manifest(root: str | Path) -> bool:
    """True when every file hash in ``MANIFEST.json`` still matches."""
    root = Path(root)
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.exists():
        raise DatasetError(f"Missing MANIFEST.json in {root}.")
    manifest = _read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise DatasetError("MANIFEST.json has no files.")
    for name, expected in files.items():
        path = root / name
        if not path.exists() or _sha256_file(path) != expected:
            return False
    return True


def write_template(path: str | Path) -> Path:
    """Create an empty, clearly-marked template dataset directory."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "replace-me",
        "split": "calibration",
        "source": {
            "citation": "Author et al. (Year), Journal, volume:pages",
            "doi": "10.xxxx/xxxxx",
            "url": "",
            "figure": "Figure 1",
            "panel": "A",
        },
        "license": "replace-me (e.g. CC BY 4.0)",
        "species": "replace-me",
        "muscle": "replace-me",
        "preparation": "in_vivo | in_situ | simulation",
        "normalization": {
            "length": "L/L0",
            "velocity": "L0/s",
            "force": "N",
            "l0_m": None,
            "notes": "How L0 and force were normalised.",
        },
        "digitization": {
            "method": "WebPlotDigitizer 5.x",
            "digitizer": "replace-me",
            "date": str(date.today()),
            "independent_repeats": 2,
            "error": {"ia_rate_hz": 1.0, "ii_rate_hz": 1.0, "time_s": 0.005},
        },
        "conditions": ["replace-me"],
        "notes": "Template only. Not admissible until provenance is complete.",
    }
    (root / "metadata.template.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "traces.template.csv").write_text(",".join(TRACE_COLUMNS) + "\n", encoding="utf-8")
    (root / "spikes.template.csv").write_text(",".join(SPIKE_COLUMNS) + "\n", encoding="utf-8")
    (root / "README.md").write_text(
        "# Dataset template\n\n"
        "Copy the templates to `metadata.json`, `traces.csv`, and (optionally)\n"
        "`spikes.csv`, then fill in every provenance field. Run\n"
        "`python -m validation.io check <this-directory>` to validate.\n\n"
        "Do not delete `digitization.error`: it is used to weight residuals and\n"
        "to decide whether model error is within measurement error.\n",
        encoding="utf-8",
    )
    return root


def discover_datasets(root: str | Path) -> tuple[Dataset, ...]:
    """Load every dataset directory immediately below ``root``."""
    base = Path(root)
    if not base.is_dir():
        raise DatasetError(f"{base} is not a directory.")
    datasets = []
    for candidate in sorted(base.iterdir()):
        if candidate.is_dir() and (candidate / "metadata.json").exists():
            datasets.append(load_dataset(candidate))
    if not datasets:
        raise DatasetError(f"No datasets with metadata.json found below {base}.")
    return tuple(datasets)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate a bioafferent dataset directory.")
    parser.add_argument("path", help="Dataset directory")
    parser.add_argument("--allow-blind-targets", action="store_true")
    args = parser.parse_args()
    dataset = load_dataset(args.path, allow_blind_targets=args.allow_blind_targets)
    print(f"OK {dataset.dataset_id} split={dataset.split} trials={len(dataset.trials)}")
    print(f"conditions: {', '.join(dataset.conditions)}")
    print(f"sha256: {dataset_hash(dataset)}")


if __name__ == "__main__":
    _main()
