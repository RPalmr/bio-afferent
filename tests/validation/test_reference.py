"""Verification tests: JAX implementation vs independent NumPy reference."""

from __future__ import annotations

import pytest

from bioafferent import GTOConfig, SpindleConfig
from validation.verification import (
    verify_dt_convergence,
    verify_integrators,
    verify_precision,
    verify_reference_parity,
)


@pytest.mark.parametrize("integrator", ["exponential", "rk2", "euler"])
def test_reference_parity_all_integrators(integrator: str):
    config = SpindleConfig(
        dtype="float64",
        integrator=integrator,
        enable_thixotropy=True,
        thixotropy_scale=0.2,
        thixotropy_tau=0.3,
    )
    report = verify_reference_parity(config)
    assert report["passed"], report


def test_gto_reference_parity():
    report = verify_reference_parity(gto_config=GTOConfig(dtype="float64"))
    assert report["passed"], report


def test_dt_convergence():
    report = verify_dt_convergence()
    assert report["passed"], report
    errors = report["errors"]
    assert errors[-1]["ia_rate"] <= errors[0]["ia_rate"]


def test_precision_and_integrators():
    assert verify_precision()["passed"]
    assert verify_integrators()["passed"]
