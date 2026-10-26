"""Tests for the freeze/predict/score blind-validation workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bioafferent import SpindleConfig
from validation.blind import (
    freeze_model,
    load_frozen,
    load_predictions,
    predict_dataset,
    score_predictions,
)
from validation.io import Dataset, Trial, load_dataset, save_dataset


def _metadata(split: str, dataset_id: str) -> dict:
    return {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "split": split,
        "source": {"citation": "x", "doi": "10.0/x"},
        "license": "CC BY 4.0",
        "species": "cat",
        "muscle": "soleus",
        "preparation": "in_vivo",
        "normalization": {"length": "L/L0", "velocity": "L0/s", "force": "N"},
        "digitization": {
            "method": "x",
            "digitizer": "x",
            "date": "2026-01-01",
            "independent_repeats": 2,
            "error": {"ia_rate_hz": 1.0},
        },
        "conditions": ["ramp"],
    }


def _trial(*, targets: bool) -> Trial:
    time = np.linspace(0.0, 0.2, 201)
    length = 1.0 + 0.05 * np.sin(2.0 * np.pi * time)
    velocity = 0.05 * 2.0 * np.pi * np.cos(2.0 * np.pi * time)
    return Trial(
        trial_id="t1",
        condition="ramp",
        time_s=time,
        inputs={"length_norm": length, "velocity_norm_s": velocity},
        targets={"ia_rate_hz": np.full_like(time, 12.0)} if targets else {},
    )


def test_freeze_predict_score_roundtrip(tmp_path: Path):
    frozen_path = tmp_path / "frozen.json"
    payload = freeze_model(SpindleConfig(), frozen_path, model="spindle")
    assert payload["config_sha256"]
    frozen = load_frozen(frozen_path)

    blind = Dataset(metadata=_metadata("blind", "blind"), trials=(_trial(targets=False),))
    blind_root = save_dataset(blind, tmp_path / "blind")
    loaded_blind = load_dataset(blind_root)
    prediction_path = predict_dataset(loaded_blind, frozen, tmp_path / "predictions")
    predictions = load_predictions(prediction_path)
    assert "t1" in predictions
    assert predictions["t1"]["ia_rate_hz"].shape == (201,)

    truth = Dataset(metadata=_metadata("blind", "blind"), trials=(_trial(targets=True),))
    truth_root = save_dataset(truth, tmp_path / "truth")
    loaded_truth = load_dataset(truth_root, allow_blind_targets=True)
    result = score_predictions(prediction_path, loaded_truth, criteria={"r2": (">=", -10.0)})
    assert "aggregate" in result
    assert result["acceptance"]["passed"] is True


def test_frozen_model_tamper_detected(tmp_path: Path):
    path = tmp_path / "frozen.json"
    freeze_model(SpindleConfig(), path)
    payload = json.loads(path.read_text())
    payload["config"]["occlusion"] = 1.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen(path)


def test_predict_refuses_outcomes(tmp_path: Path):
    path = tmp_path / "frozen.json"
    frozen = freeze_model(SpindleConfig(), path)
    dataset = Dataset(metadata=_metadata("validation", "with-targets"), trials=(_trial(targets=True),))
    with pytest.raises(ValueError, match="contains outcomes"):
        predict_dataset(dataset, frozen, tmp_path / "predictions")
