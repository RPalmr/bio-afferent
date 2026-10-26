"""Blind-prediction workflow: freeze, predict, then unseal and score.

The strongest validation design is prospective: freeze the model and its
parameters, obtain held-out inputs from a laboratory without seeing the
measured afferent outputs, emit predictions, and only then unseal the outcomes.
This module makes that workflow auditable with content hashes and a manifest.

Typical use::

    python -m validation.blind freeze --model spindle --output frozen.json
    python -m validation.blind predict --model-json frozen.json \
        --data validation/data/blind_experiment --output validation/results/predictions
    # ... laboratory unseals the outcomes ...
    python -m validation.blind score --predictions validation/results/predictions/predictions.csv \
        --truth validation/data/blind_experiment --output validation/results/score.json
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from bioafferent._version import __version__
from bioafferent.models.gto import GolgiTendonOrgan
from bioafferent.models.gto_parameters import GTOConfig
from bioafferent.models.provenance import parameter_table
from bioafferent.models.spindle import MuscleSpindle
from bioafferent.models.spindle_parameters import SpindleConfig
from validation.io import Dataset, Trial, dataset_hash, load_dataset
from validation.metrics import aggregate_metrics, check_acceptance, rate_metrics

PREDICTION_COLUMNS = ("trial_id", "time_s", "ia_rate_hz", "ii_rate_hz", "ib_rate_hz")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def freeze_model(
    config: SpindleConfig | GTOConfig,
    path: str | Path,
    *,
    model: str = "spindle",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an immutable, hashed model snapshot for blind prediction."""
    if model not in ("spindle", "gto"):
        raise ValueError("model must be 'spindle' or 'gto'.")
    config_dict = dataclasses.asdict(config)
    payload = {
        "schema_version": "1.0",
        "model": model,
        "package_version": __version__,
        "git_commit": _git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_dict,
        "config_sha256": sha256_bytes(_canonical_json(config_dict).encode("utf-8")),
        "provenance": parameter_table(config),
        "metadata": metadata or {},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_frozen(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("Unsupported frozen-model schema.")
    expected = sha256_bytes(_canonical_json(payload["config"]).encode("utf-8"))
    if payload.get("config_sha256") != expected:
        raise ValueError("Frozen-model config hash mismatch; the snapshot was edited.")
    return payload


def _config_from_payload(payload: dict[str, Any]) -> SpindleConfig | GTOConfig:
    if payload["model"] == "spindle":
        return SpindleConfig(**payload["config"])
    if payload["model"] == "gto":
        return GTOConfig(**payload["config"])
    raise ValueError(f"Unknown frozen model {payload['model']!r}.")


def predict_trial(model: str, config: SpindleConfig | GTOConfig, trial: Trial) -> dict[str, np.ndarray]:
    """Run a frozen model on one inputs-only trial."""
    predictions: dict[str, np.ndarray] = {}
    if model == "spindle":
        length = trial.inputs.get("length_norm")
        velocity = trial.inputs.get("velocity_norm_s")
        if length is None or velocity is None:
            raise ValueError(f"Trial {trial.trial_id!r} lacks length/velocity inputs.")
        spindle = MuscleSpindle(config)  # type: ignore[arg-type]
        state = spindle.init_state(())
        _, out = spindle.rollout(
            state,
            jnp.asarray(length),
            jnp.asarray(velocity),
            jnp.asarray(trial.inputs.get("gamma_dynamic_hz", 0.0)),
            jnp.asarray(trial.inputs.get("gamma_static_hz", 0.0)),
            trial.dt,
        )
        predictions["ia_rate_hz"] = np.asarray(out.ia_rate)
        predictions["ii_rate_hz"] = np.asarray(out.ii_rate)
    elif model == "gto":
        force = trial.inputs.get("force_n")
        if force is None:
            raise ValueError(f"Trial {trial.trial_id!r} lacks force input.")
        gto = GolgiTendonOrgan(config)  # type: ignore[arg-type]
        state = gto.init_state(())
        _, out = gto.rollout(state, jnp.asarray(force), trial.dt)
        predictions["ib_rate_hz"] = np.asarray(out.ib_rate)
    else:
        raise ValueError("model must be 'spindle' or 'gto'.")
    return predictions


def predict_dataset(
    dataset: Dataset,
    frozen: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Write predictions CSV and a hash manifest for an inputs-only dataset."""
    config = _config_from_payload(frozen)
    model = str(frozen["model"])
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for trial in dataset.trials:
        if trial.targets or trial.spikes:
            raise ValueError(f"Blind trial {trial.trial_id!r} contains outcomes; seal them before predicting.")
        predictions = predict_trial(model, config, trial)
        for index, time in enumerate(trial.time_s):
            row: dict[str, Any] = {"trial_id": trial.trial_id, "time_s": f"{time:.10g}"}
            for column in PREDICTION_COLUMNS[2:]:
                values = predictions.get(column)
                row[column] = "" if values is None else f"{values[index]:.10g}"
            rows.append(row)
    prediction_path = root / "predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": "1.0",
        "frozen_model_sha256": sha256_bytes(_canonical_json(frozen).encode("utf-8")),
        "input_dataset_sha256": dataset_hash(dataset),
        "predictions_sha256": sha256_file(prediction_path),
        "n_rows": len(rows),
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return prediction_path


def load_predictions(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("predictions.csv has no header.")
        unknown = set(reader.fieldnames) - set(PREDICTION_COLUMNS)
        if unknown:
            raise ValueError(f"Unknown prediction columns: {sorted(unknown)}")
        for row in reader:
            trial_id = (row.get("trial_id") or "").strip()
            if not trial_id:
                raise ValueError("Blank trial_id in predictions.csv.")
            record = grouped.setdefault(trial_id, {"time_s": []})
            record["time_s"].append(float(row["time_s"]))
            for column in PREDICTION_COLUMNS[2:]:
                value = (row.get(column) or "").strip()
                if value:
                    record.setdefault(column, []).append(float(value))
    out: dict[str, dict[str, np.ndarray]] = {}
    for trial_id, record in grouped.items():
        n_time = len(record["time_s"])
        for name, values in record.items():
            if len(values) != n_time:
                raise ValueError(f"Trial {trial_id!r} column {name!r} has {len(values)} values for {n_time} times.")
        out[trial_id] = {name: np.asarray(values, dtype=float) for name, values in record.items()}
    return out


def score_predictions(
    predictions_path: str | Path,
    truth: Dataset,
    *,
    criteria: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score predictions against unsealed outcomes, trial by trial."""
    predictions = load_predictions(predictions_path)
    records: list[dict[str, float]] = []
    details: list[dict[str, Any]] = []
    for trial in truth.trials:
        if trial.trial_id not in predictions:
            raise ValueError(f"Missing predictions for trial {trial.trial_id!r}.")
        pred = predictions[trial.trial_id]
        if pred["time_s"].shape != trial.time_s.shape or not np.allclose(pred["time_s"], trial.time_s):
            raise ValueError(f"Prediction times for trial {trial.trial_id!r} do not match the truth.")
        for target in ("ia_rate_hz", "ii_rate_hz", "ib_rate_hz"):
            if target not in trial.targets or target not in pred:
                continue
            record = rate_metrics(trial.targets[target], pred[target], trial.dt)
            records.append(record)
            details.append({"trial_id": trial.trial_id, "channel": target, **record})
    if not records:
        raise ValueError("No overlapping target channels between predictions and truth.")
    result: dict[str, Any] = {
        "predictions_sha256": sha256_file(predictions_path),
        "truth_dataset_sha256": dataset_hash(truth),
        "aggregate": aggregate_metrics(records),
        "per_trial": details,
    }
    if criteria is not None:
        means = {name: values["mean"] for name, values in result["aggregate"].items()}
        result["acceptance"] = check_acceptance(means, criteria)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description="Blind-validation workflow for bioafferent.")
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze", help="Write a frozen model snapshot.")
    freeze.add_argument("--model", choices=("spindle", "gto"), default="spindle")
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--metadata", default=None, help="Optional JSON metadata file.")

    predict = sub.add_parser("predict", help="Emit predictions for an inputs-only dataset.")
    predict.add_argument("--model-json", required=True)
    predict.add_argument("--data", required=True)
    predict.add_argument("--output", required=True)

    score = sub.add_parser("score", help="Score predictions after outcomes are unsealed.")
    score.add_argument("--predictions", required=True)
    score.add_argument("--truth", required=True)
    score.add_argument("--criteria", default=None, help="Optional JSON acceptance criteria.")
    score.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "freeze":
        config = SpindleConfig() if args.model == "spindle" else GTOConfig()
        metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8")) if args.metadata else {}
        payload = freeze_model(config, args.output, model=args.model, metadata=metadata)
        print(json.dumps({"output": args.output, "config_sha256": payload["config_sha256"]}, indent=2))
    elif args.command == "predict":
        frozen = load_frozen(args.model_json)
        dataset = load_dataset(args.data, allow_blind_targets=False)
        path = predict_dataset(dataset, frozen, args.output)
        print(str(path))
    else:
        criteria = json.loads(Path(args.criteria).read_text(encoding="utf-8")) if args.criteria else None
        truth = load_dataset(args.truth, allow_blind_targets=True)
        result = score_predictions(args.predictions, truth, criteria=criteria)
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"output": args.output, "passed": result.get("acceptance", {}).get("passed")}, indent=2))


if __name__ == "__main__":
    _main()
