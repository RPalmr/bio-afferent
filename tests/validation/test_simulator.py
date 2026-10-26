"""Tests for MuJoCo/MJX adapter semantics and CPU/MJX parity."""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")

from validation.simulator import (  # noqa: E402
    SLIDE_MUSCLE_XML,
    actuator_semantics,
    run_simulator_checks,
    validate_mjx_cpu_parity,
    validate_slide_muscle_adapter,
)


def test_slide_muscle_semantics():
    report = validate_slide_muscle_adapter()
    assert report["passed"], report
    assert report["raw_force_n"] < 0.0
    assert report["adapted_force_n"] > 0.0


def test_actuator_classification():
    model = mujoco.MjModel.from_xml_string(SLIDE_MUSCLE_XML)
    semantics = actuator_semantics(model)
    assert len(semantics) == 1
    assert semantics[0]["is_muscle"] is True
    assert any("force_mode='magnitude'" in warning for warning in semantics[0]["warnings"])


def test_mjx_cpu_parity():
    pytest.importorskip("mujoco.mjx", reason="mujoco-mjx not installed")
    report = validate_mjx_cpu_parity()
    assert report["passed"], report


def test_full_simulator_suite():
    report = run_simulator_checks()
    assert report["available"] is True
    assert report["passed"] is True
