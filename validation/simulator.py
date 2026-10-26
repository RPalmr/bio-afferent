"""MuJoCo/MJX semantics checks for the afferent adapters.

The adapters assume that ``actuator_length``, ``actuator_velocity``, and
``actuator_force`` describe a muscle-tendon unit. That assumption is only true
for muscle actuators (and, with caveats, tendon transmissions). These checks
make the assumption explicit and verify CPU/MJX parity on a minimal muscle
model.

The force-sign check is important: MuJoCo muscle actuators report negative
force in the pulling direction, while the GTO expects a non-negative tendon
force. The adapter therefore defaults to ``force_mode="magnitude"``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from bioafferent.wrappers.mujoco import extract_mujoco_mechanics

try:
    import mujoco  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    mujoco = None  # type: ignore[assignment]

SLIDE_MUSCLE_XML = """\
<mujoco model="slide_muscle">
  <option timestep="0.001"/>
  <worldbody>
    <body name="fascicle" pos="0 0 0">
      <joint name="stretch" type="slide" axis="1 0 0" range="0 1"/>
      <geom type="sphere" size="0.05" mass="0.1"/>
    </body>
  </worldbody>
  <tendon>
    <fixed name="tendon">
      <joint joint="stretch" coef="1"/>
    </fixed>
  </tendon>
  <actuator>
    <muscle name="muscle" tendon="tendon" lengthrange="0.0 1.0" force="100" scale="1"/>
  </actuator>
</mujoco>
"""


def _require_mujoco() -> Any:
    if mujoco is None:
        raise ImportError("MuJoCo is required: pip install 'bioafferent[mujoco]'.")
    return mujoco


def actuator_semantics(model: Any) -> list[dict[str, Any]]:
    """Classify each actuator's length/force semantics."""
    mj = _require_mujoco()
    report: list[dict[str, Any]] = []
    for index in range(model.nu):
        dyntype = int(model.actuator_dyntype[index])
        trntype = int(model.actuator_trntype[index])
        is_muscle = dyntype == int(mj.mjtDyn.mjDYN_MUSCLE)
        warnings: list[str] = []
        if not is_muscle:
            warnings.append("not a muscle actuator: actuator_length is transmission length, not muscle length")
        if trntype not in (int(mj.mjtTrn.mjTRN_JOINT), int(mj.mjtTrn.mjTRN_TENDON)):
            warnings.append("transmission is not joint/tendon; muscle-length semantics are unsupported")
        if is_muscle:
            warnings.append("muscle forces are signed; use force_mode='magnitude' for GTO input")
        report.append(
            {
                "id": index,
                "name": mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, index) or f"actuator_{index}",
                "dyntype": dyntype,
                "trntype": trntype,
                "is_muscle": is_muscle,
                "warnings": warnings,
            }
        )
    return report


def validate_slide_muscle_adapter() -> dict[str, Any]:
    """Exercise the CPU adapter on a one-muscle slide-joint model."""
    mj = _require_mujoco()
    model = mj.MjModel.from_xml_string(SLIDE_MUSCLE_XML)
    data = mj.MjData(model)
    data.qpos[0] = 0.2
    data.qvel[0] = 0.5
    mj.mj_forward(model, data)

    semantics = actuator_semantics(model)
    mech = extract_mujoco_mechanics(data, rest_lengths=[0.2], force_mode="magnitude")
    checks = {
        "muscle_actuator_detected": bool(semantics and semantics[0]["is_muscle"]),
        "length_normalised": bool(np.isclose(float(mech.length[0]), 1.0, atol=1e-5)),
        "velocity_normalised": bool(np.isclose(float(mech.velocity[0]), 2.5, atol=1e-5)),
        "zero_force_at_rest": bool(np.isclose(float(mech.force[0]), 0.0, atol=1e-5)),
    }

    data.ctrl[0] = 1.0
    for _ in range(5):
        mj.mj_step(model, data)
    raw_force = float(data.actuator_force[0])
    mech_active = extract_mujoco_mechanics(data, rest_lengths=[0.2], force_mode="magnitude")
    checks["muscle_force_is_signed"] = raw_force < 0.0
    checks["magnitude_maps_to_positive"] = float(mech_active.force[0]) > 0.0
    checks["magnitude_matches_abs"] = bool(np.isclose(float(mech_active.force[0]), abs(raw_force), atol=1e-5))
    return {
        "semantics": semantics,
        "raw_force_n": raw_force,
        "adapted_force_n": float(mech_active.force[0]),
        "checks": checks,
        "passed": all(checks.values()),
    }


def validate_mjx_cpu_parity() -> dict[str, Any]:
    """Compare MJX and CPU actuator mechanics at the same state."""
    mj = _require_mujoco()
    try:
        import jax.numpy as jnp
        import mujoco.mjx as mjx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("MJX parity requires 'bioafferent[mjx]'.") from exc

    from bioafferent.wrappers.mjx import extract_mjx_mechanics

    model = mj.MjModel.from_xml_string(SLIDE_MUSCLE_XML)
    data = mj.MjData(model)
    data.qpos[0] = 0.2
    data.qvel[0] = 0.5
    mj.mj_forward(model, data)
    cpu = extract_mujoco_mechanics(data, rest_lengths=[0.2], force_mode="magnitude")

    mx = mjx.put_model(model)
    dx = mjx.make_data(mx).replace(qpos=jnp.asarray([0.2]), qvel=jnp.asarray([0.5]))
    dx = mjx.forward(mx, dx)
    mjx_mech = extract_mjx_mechanics(dx, rest_lengths=jnp.asarray([0.2]), force_mode="magnitude")

    errors = {
        "length": float(np.max(np.abs(np.asarray(cpu.length) - np.asarray(mjx_mech.length)))),
        "velocity": float(np.max(np.abs(np.asarray(cpu.velocity) - np.asarray(mjx_mech.velocity)))),
        "force": float(np.max(np.abs(np.asarray(cpu.force) - np.asarray(mjx_mech.force)))),
    }
    atol = 1e-4
    return {"max_abs_error": errors, "atol": atol, "passed": all(value <= atol for value in errors.values())}


def run_simulator_checks() -> dict[str, Any]:
    """Run every available simulator check and return a JSON report."""
    report: dict[str, Any] = {}
    if mujoco is None:
        return {"available": False, "passed": False, "reason": "mujoco not installed"}
    report["cpu_adapter"] = validate_slide_muscle_adapter()
    try:
        report["mjx_parity"] = validate_mjx_cpu_parity()
    except ImportError as exc:
        report["mjx_parity"] = {"available": False, "passed": False, "reason": str(exc)}
    report["available"] = True
    report["passed"] = all(check.get("passed", False) for check in report.values() if isinstance(check, dict))
    return report


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run MuJoCo/MJX adapter semantics checks.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_simulator_checks()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"simulator checks: {'PASS' if report.get('passed') else 'FAIL'}")
        for name, check in report.items():
            if isinstance(check, dict) and "passed" in check:
                print(f"  {name}: {'pass' if check['passed'] else 'fail'}")
    raise SystemExit(0 if report.get("passed") else 1)


if __name__ == "__main__":
    _main()
