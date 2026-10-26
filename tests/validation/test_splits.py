"""Tests for condition-level split and leakage enforcement."""

from __future__ import annotations

import numpy as np
import pytest

from validation.io import Dataset, DatasetError, Trial
from validation.splits import (
    assert_blind_inputs_only,
    assert_calibration_has_targets,
    assert_no_leakage,
    group_conditions_by_split,
    plan_from_datasets,
    trials_for_split,
)


def _trial(trial_id: str, condition: str, *, targets: bool = True) -> Trial:
    time = np.linspace(0.0, 0.01, 11)
    return Trial(
        trial_id=trial_id,
        condition=condition,
        time_s=time,
        inputs={"length_norm": np.ones_like(time)},
        targets={"ia_rate_hz": np.ones_like(time)} if targets else {},
    )


def _dataset(split: str, condition: str, *, targets: bool = True) -> Dataset:
    metadata = {
        "schema_version": "1.0",
        "dataset_id": f"{split}-{condition}",
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
        "conditions": [condition],
    }
    return Dataset(metadata=metadata, trials=(_trial("t1", condition, targets=targets),))


def test_plan_and_grouping():
    datasets = (_dataset("calibration", "ramp"), _dataset("validation", "step"))
    plan = plan_from_datasets(datasets)
    assert plan.split_for("ramp") == "calibration"
    assert plan.split_for("step") == "validation"
    grouped = group_conditions_by_split(datasets)
    assert grouped["calibration"] == ("ramp",)
    assert len(trials_for_split(datasets, "validation")) == 1


def test_leakage_detected():
    datasets = (_dataset("calibration", "ramp"), _dataset("validation", "ramp"))
    with pytest.raises(DatasetError, match="appears in both"):
        assert_no_leakage(datasets)


def test_blind_inputs_only():
    dataset = _dataset("blind", "future", targets=False)
    assert_blind_inputs_only(dataset)
    with pytest.raises(DatasetError, match="not a blind dataset"):
        assert_blind_inputs_only(_dataset("validation", "step"))


def test_calibration_requires_targets():
    with pytest.raises(DatasetError, match="no targets"):
        assert_calibration_has_targets((_dataset("calibration", "ramp", targets=False),))
