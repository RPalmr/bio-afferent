"""Code verification: JAX implementation vs NumPy reference and numerics.

Verification answers "are we solving the intended equations correctly?".
Validation (does the model match experimental data?) lives in
``validation.calibrate`` and ``validation.blind``. Keeping the two separate is
required by the V&V protocol.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bioafferent.models.gto import GolgiTendonOrgan
from bioafferent.models.gto_parameters import GTOConfig
from bioafferent.models.spindle import MuscleSpindle
from bioafferent.models.spindle_parameters import SpindleConfig
from validation.reference import GTOReference, SpindleReference


@contextlib.contextmanager
def _x64(enabled: bool) -> Iterator[None]:
    previous = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def ramp_inputs(dt: float = 0.001, duration_s: float = 2.7) -> dict[str, np.ndarray]:
    """Vannucci-style ramp-and-hold input at an arbitrary timestep."""
    n = int(round(duration_s / dt))
    t = np.arange(n) * dt
    length = np.where(t < 0.5, 0.95, 0.95 + 0.11 * np.clip(t - 0.5, 0.0, 1.1))
    velocity = np.where((t >= 0.5) & (t <= 1.6), 0.11, 0.0)
    return {
        "length": length,
        "velocity": velocity,
        "gamma_dynamic": np.full(n, 40.0),
        "gamma_static": np.full(n, 30.0),
    }


def gto_inputs(dt: float = 0.001, duration_s: float = 2.0) -> dict[str, np.ndarray]:
    n = int(round(duration_s / dt))
    t = np.arange(n) * dt
    force = np.where(t < 0.5, 0.0, np.where(t < 1.0, 50.0, np.where(t < 1.5, 120.0, 20.0)))
    return {"force": force}


def _jax_spindle(config: SpindleConfig, inputs: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    model = MuscleSpindle(config)
    state = model.init_state(())
    _, out = model.rollout(
        state,
        jnp.asarray(inputs["length"]),
        jnp.asarray(inputs["velocity"]),
        jnp.asarray(inputs.get("gamma_dynamic", 0.0)),
        jnp.asarray(inputs.get("gamma_static", 0.0)),
        dt,
    )
    names = (
        "ia_rate",
        "ii_rate",
        "tension",
        "activation_dynamic",
        "activation_static",
    )
    return {name: np.asarray(getattr(out, name)) for name in names}


def _reference_spindle(config: SpindleConfig, inputs: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    model = SpindleReference(config)
    state = model.init_state(())
    _, out = model.rollout(
        state,
        inputs["length"],
        inputs["velocity"],
        inputs.get("gamma_dynamic", 0.0),
        inputs.get("gamma_static", 0.0),
        dt,
    )
    return out


def _jax_gto(config: GTOConfig, inputs: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    model = GolgiTendonOrgan(config)
    state = model.init_state(())
    _, out = model.rollout(state, jnp.asarray(inputs["force"]), dt)
    return {"ib_rate": np.asarray(out.ib_rate), "force_filtered": np.asarray(out.force_filtered)}


def _reference_gto(config: GTOConfig, inputs: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    model = GTOReference(config)
    state = model.init_state(())
    _, out = model.rollout(state, inputs["force"], dt)
    return out


def _max_error(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, float]:
    return {key: float(np.max(np.abs(np.asarray(a[key]) - np.asarray(b[key])))) for key in a if key in b}


def verify_reference_parity(
    spindle_config: SpindleConfig | None = None,
    gto_config: GTOConfig | None = None,
    *,
    dt: float = 0.001,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Compare JAX and NumPy implementations in float64."""
    spindle_config = spindle_config or SpindleConfig(dtype="float64")
    gto_config = gto_config or GTOConfig(dtype="float64")
    with _x64(True):
        spindle_jax = _jax_spindle(spindle_config, ramp_inputs(dt), dt)
        spindle_ref = _reference_spindle(spindle_config, ramp_inputs(dt), dt)
        gto_jax = _jax_gto(gto_config, gto_inputs(dt), dt)
        gto_ref = _reference_gto(gto_config, gto_inputs(dt), dt)
    spindle_errors = _max_error(spindle_jax, spindle_ref)
    gto_errors = _max_error(gto_jax, gto_ref)
    return {
        "spindle_max_abs_error": spindle_errors,
        "gto_max_abs_error": gto_errors,
        "atol": atol,
        "passed": all(value <= atol for value in spindle_errors.values())
        and all(value <= atol for value in gto_errors.values()),
    }


def verify_dt_convergence(
    config: SpindleConfig | None = None,
    *,
    dts: tuple[float, ...] = (1e-3, 5e-4, 2.5e-4),
    duration_s: float = 2.7,
) -> dict[str, Any]:
    """Check that halving dt reduces the trajectory difference."""
    if len(dts) < 2:
        raise ValueError("Need at least two timesteps.")
    config = config or SpindleConfig()
    traces = []
    for dt in dts:
        with _x64(True):
            traces.append(_jax_spindle(config, ramp_inputs(dt, duration_s), dt))
    errors = []
    for index in range(len(dts) - 1):
        coarse, fine = traces[index], traces[index + 1]
        ratio = dts[index] / dts[index + 1]
        stride = int(round(ratio))
        if not np.isclose(ratio, stride):
            raise ValueError("dts must be refined by an integer factor.")
        errors.append(
            {
                "ia_rate": float(
                    np.max(np.abs(coarse["ia_rate"] - fine["ia_rate"][::stride][: coarse["ia_rate"].size]))
                ),
                "ii_rate": float(
                    np.max(np.abs(coarse["ii_rate"] - fine["ii_rate"][::stride][: coarse["ii_rate"].size]))
                ),
            }
        )
    monotone = all(
        errors[i + 1]["ia_rate"] <= errors[i]["ia_rate"] and errors[i + 1]["ii_rate"] <= errors[i]["ii_rate"]
        for i in range(len(errors) - 1)
    )
    return {"dts": list(dts), "errors": errors, "monotone": monotone, "passed": monotone}


def verify_precision(
    *,
    dt: float = 0.001,
    atol: float = 0.5,
) -> dict[str, Any]:
    """Compare float32 and float64 trajectories (rates in Hz)."""
    inputs = ramp_inputs(dt)
    with _x64(False):
        f32 = _jax_spindle(SpindleConfig(dtype="float32"), inputs, dt)
    with _x64(True):
        f64 = _jax_spindle(SpindleConfig(dtype="float64"), inputs, dt)
    errors = {
        key: float(np.max(np.abs(f32[key].astype(float) - f64[key].astype(float)))) for key in ("ia_rate", "ii_rate")
    }
    return {"max_abs_error_hz": errors, "atol": atol, "passed": all(v <= atol for v in errors.values())}


def verify_integrators(
    *,
    dt: float = 1e-4,
    duration_s: float = 2.7,
    atol: float = 1.0,
) -> dict[str, Any]:
    """Compare exponential, RK2, and Euler at a small timestep (rates in Hz)."""
    inputs = ramp_inputs(dt, duration_s)
    results = {}
    for integrator in ("exponential", "rk2", "euler"):
        config = SpindleConfig(integrator=integrator, dtype="float64")
        with _x64(True):
            results[integrator] = _jax_spindle(config, inputs, dt)
    reference = results["exponential"]
    errors = {
        name: {key: float(np.max(np.abs(trace[key] - reference[key]))) for key in ("ia_rate", "ii_rate")}
        for name, trace in results.items()
        if name != "exponential"
    }
    passed = all(v <= atol for e in errors.values() for v in e.values())
    return {"max_abs_error_hz": errors, "atol": atol, "passed": passed}


def run_verification_suite() -> dict[str, Any]:
    """Run every code-verification check and return a JSON-serialisable report."""
    report = {
        "reference_parity": verify_reference_parity(),
        "dt_convergence": verify_dt_convergence(),
        "precision": verify_precision(),
        "integrators": verify_integrators(),
    }
    report["passed"] = all(
        check["passed"] for check in report.values() if isinstance(check, dict) and "passed" in check
    )
    return report


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run bioafferent code-verification checks.")
    parser.add_argument("--json", action="store_true", help="Emit the full JSON report.")
    args = parser.parse_args()
    report = run_verification_suite()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"verification: {'PASS' if report['passed'] else 'FAIL'}")
        for name, check in report.items():
            if isinstance(check, dict) and "passed" in check:
                print(f"  {name}: {'pass' if check['passed'] else 'fail'}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    _main()
