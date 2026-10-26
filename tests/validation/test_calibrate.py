"""Tests for parameter flattening, calibration, and held-out evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bioafferent import SpindleConfig
from validation.calibrate import (
    apply_params,
    default_bounds,
    evaluate_model,
    fit_model,
    flatten_config,
    load_fit,
    save_fit,
)
from validation.io import Trial
from validation.reference import SpindleReference


def _trial() -> Trial:
    time = np.linspace(0.0, 0.1, 101)
    length = np.ones_like(time)
    velocity = np.zeros_like(time)
    reference = SpindleReference(SpindleConfig(dtype="float64"))
    state = reference.init_state(())
    _, out = reference.rollout(state, length, velocity, 0.0, 0.0, 0.001)
    return Trial(
        trial_id="t1",
        condition="rest",
        time_s=time,
        inputs={"length_norm": length, "velocity_norm_s": velocity},
        targets={"ia_rate_hz": out["ia_rate"], "ii_rate_hz": out["ii_rate"]},
    )


def test_flatten_apply_roundtrip():
    config = SpindleConfig()
    fields = ("baseline_ia", "stiffness_rest", "ia_weights")
    vector = flatten_config(config, fields)
    rebuilt = apply_params(config, fields, vector)
    assert rebuilt == config
    assert len(default_bounds(config, fields)) == vector.size


def test_fit_recovers_baseline():
    trial = _trial()
    result = fit_model(
        "spindle",
        SpindleConfig(baseline_ia=0.0),
        [trial],
        fields=("baseline_ia",),
        max_nfev=100,
    )
    assert result.success
    np.testing.assert_allclose(result.config["baseline_ia"], 5.0, atol=1e-3)


def test_evaluate_and_save(tmp_path: Path):
    trial = _trial()
    result = evaluate_model("spindle", SpindleConfig(), [trial])
    assert result["aggregate"]["r2"]["mean"] > 0.99
    result = fit_model(
        "spindle",
        SpindleConfig(baseline_ia=0.0),
        [trial],
        fields=("baseline_ia",),
        max_nfev=100,
    )
    path = save_fit(result, tmp_path / "fit.json")
    payload = load_fit(path)
    assert payload["model"] == "spindle"
    assert payload["fields"] == ["baseline_ia"]
