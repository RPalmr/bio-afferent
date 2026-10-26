"""Tests for the validation dataset schema and provenance checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from validation.io import (
    Dataset,
    DatasetError,
    Trial,
    dataset_hash,
    discover_datasets,
    load_dataset,
    save_dataset,
    verify_manifest,
    write_template,
)
from validation.splits import assert_blind_inputs_only, assert_no_leakage


def _metadata(split: str = "calibration", dataset_id: str = "demo") -> dict:
    return {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "split": split,
        "source": {"citation": "Example et al. (2020)", "doi": "10.0000/example", "figure": "1A"},
        "license": "CC BY 4.0",
        "species": "cat",
        "muscle": "soleus",
        "preparation": "in_vivo",
        "normalization": {"length": "L/L0", "velocity": "L0/s", "force": "N"},
        "digitization": {
            "method": "WebPlotDigitizer",
            "digitizer": "A. Person",
            "date": "2026-01-01",
            "independent_repeats": 2,
            "error": {"ia_rate_hz": 1.0, "time_s": 0.005},
        },
        "conditions": ["ramp_0.1"],
    }


def _dataset(split: str = "calibration", with_targets: bool = True, dataset_id: str = "demo") -> Dataset:
    time = np.linspace(0.0, 0.01, 11)
    trial = Trial(
        trial_id="t1",
        condition="ramp_0.1",
        time_s=time,
        inputs={"length_norm": np.ones_like(time), "velocity_norm_s": np.zeros_like(time)},
        targets={"ia_rate_hz": np.full_like(time, 10.0)} if with_targets else {},
        spikes={"ia": np.array([0.001, 0.005])} if with_targets else {},
    )
    return Dataset(metadata=_metadata(split, dataset_id=dataset_id), trials=(trial,))


def test_roundtrip_hash_and_manifest(tmp_path: Path):
    dataset = _dataset()
    root = save_dataset(dataset, tmp_path / "demo")
    loaded = load_dataset(root)
    assert loaded.dataset_id == "demo"
    assert loaded.trials[0].trial_id == "t1"
    assert dataset_hash(loaded) == dataset_hash(dataset)
    assert verify_manifest(root)


def test_hash_changes_with_data(tmp_path: Path):
    dataset = _dataset()
    before = dataset_hash(dataset)
    trial = dataset.trials[0]
    changed = Trial(
        trial_id=trial.trial_id,
        condition=trial.condition,
        time_s=trial.time_s,
        inputs=trial.inputs,
        targets={"ia_rate_hz": trial.targets["ia_rate_hz"] + 1.0},
        spikes=trial.spikes,
    )
    after = dataset_hash(Dataset(metadata=dataset.metadata, trials=(changed,)))
    assert before != after


def test_blind_targets_are_sealed(tmp_path: Path):
    dataset = _dataset(split="blind", with_targets=True)
    root = save_dataset(dataset, tmp_path / "blind")
    with pytest.raises(DatasetError, match="target values"):
        load_dataset(root)


def test_blind_inputs_only_loads(tmp_path: Path):
    dataset = _dataset(split="blind", with_targets=False)
    root = save_dataset(dataset, tmp_path / "blind")
    loaded = load_dataset(root)
    assert_blind_inputs_only(loaded)
    assert not loaded.trials[0].targets


def test_missing_provenance_is_rejected(tmp_path: Path):
    dataset = _dataset()
    metadata = dict(dataset.metadata)
    del metadata["digitization"]
    broken = Dataset(metadata=metadata, trials=dataset.trials)
    with pytest.raises(DatasetError, match="digitization"):
        save_dataset(broken, tmp_path / "broken")


def test_condition_must_be_declared(tmp_path: Path):
    dataset = _dataset()
    metadata = dict(dataset.metadata)
    metadata["conditions"] = ["other"]
    broken = Dataset(metadata=metadata, trials=dataset.trials)
    with pytest.raises(DatasetError, match="not in metadata.conditions"):
        save_dataset(broken, tmp_path / "broken")


def test_template_is_created(tmp_path: Path):
    root = write_template(tmp_path / "template")
    assert (root / "metadata.template.json").exists()
    assert (root / "traces.template.csv").exists()
    payload = json.loads((root / "metadata.template.json").read_text())
    assert payload["schema_version"] == "1.0"


def test_discover_and_leakage(tmp_path: Path):
    save_dataset(_dataset(dataset_id="a"), tmp_path / "a")
    save_dataset(_dataset(split="validation", dataset_id="b"), tmp_path / "b")
    datasets = discover_datasets(tmp_path)
    with pytest.raises(DatasetError, match="appears in both"):
        assert_no_leakage(datasets)
