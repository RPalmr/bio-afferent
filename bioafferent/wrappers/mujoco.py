"""MuJoCo (CPU, NumPy-side) adapter: actuator mechanics to MechanicalState.

Assumptions (documented, checked at runtime where possible):

- Muscles are MuJoCo actuators with position transmission (tendon/muscle
  length available as ``data.actuator_length``). ``actuator_velocity`` and
  ``actuator_force`` supply stretch velocity and tendon force.
- ``length`` is normalized by the per-actuator rest length measured (or
  supplied) at model load: ``L = actuator_length / rest_length``.
- Units are the model's native units; convert to SI (m, m/s, N) *before*
  constructing the adapter if the model uses other scales.
- Only fixed numeric id lists are supported; equality/branching structures
  without actuator arrays raise :class:`UnsupportedBackendError`.

The core physiological models never see MuJoCo objects: this module converts
to plain JAX arrays at the boundary (one host-side conversion per step; the
sensory model itself stays JAX-native).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from bioafferent.core.exceptions import ConfigurationError, UnsupportedBackendError
from bioafferent.wrappers.base import MechanicalState

try:
    import mujoco  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - mujoco is an optional dependency
    mujoco = None  # type: ignore[assignment]


def _require_mujoco() -> None:
    if mujoco is None:
        raise UnsupportedBackendError(
            "mujoco is not installed. Install the 'mujoco' extra: "
            "pip install bioafferent[mujoco]."
        )


@dataclass
class MuJoCoMuscleSet:
    """Identifies which actuators are treated as muscles.

    Attributes:
        actuator_ids: MuJoCo actuator ids in muscle order.
        rest_lengths: per-actuator rest length in model length units, used for
            L/L0 normalization. If None, measured from the data at bind time
            (qpos0 forward pass) — document which you used.
        length_scale: multiply model lengths by this to get meters.
        velocity_scale: multiply model velocities by this to get m/s.
        force_scale: multiply model forces by this to get newton.
    """

    actuator_ids: tuple[int, ...]
    rest_lengths: tuple[float, ...] | None = None
    length_scale: float = 1.0
    velocity_scale: float = 1.0
    force_scale: float = 1.0

    def __post_init__(self) -> None:
        if len(self.actuator_ids) == 0:
            raise ConfigurationError("actuator_ids must be non-empty.")
        if self.rest_lengths is not None and len(self.rest_lengths) != len(
            self.actuator_ids
        ):
            raise ConfigurationError("rest_lengths must match actuator_ids in size.")
        if any(v <= 0.0 for v in (self.length_scale, self.velocity_scale)):
            raise ConfigurationError("length/velocity scales must be positive.")
        if self.force_scale <= 0.0:
            raise ConfigurationError("force_scale must be positive.")


def extract_mujoco_mechanics(
    model: Any, data: Any, muscles: MuJoCoMuscleSet
) -> MechanicalState:
    """Read actuator length/velocity/force into a :class:`MechanicalState`.

    Works with real ``mujoco.MjModel``/``MjData`` and with duck-typed test
    doubles exposing ``actuator_length``/``actuator_velocity``/``actuator_force``
    arrays. Rest lengths default to the current ``actuator_length`` snapshot
    (document this calibration choice when used).
    """
    _require_mujoco()
    ids = np.asarray(muscles.actuator_ids, dtype=int)
    try:
        alen = np.asarray(data.actuator_length)
        avel = np.asarray(data.actuator_velocity)
        afor = np.asarray(data.actuator_force)
    except AttributeError as exc:
        raise UnsupportedBackendError(
            "MuJoCo data lacks actuator_length/_velocity/_force arrays; "
            "this model structure is unsupported."
        ) from exc
    if alen.ndim != 1 or avel.shape != alen.shape or afor.shape != alen.shape:
        raise UnsupportedBackendError(
            "Unexpected MuJoCo actuator array shapes; unsupported configuration."
        )
    if int(np.max(ids)) >= alen.shape[0] or int(np.min(ids)) < 0:
        raise ConfigurationError(
            f"actuator_ids {tuple(ids)} out of range for nu={alen.shape[0]}."
        )
    if muscles.rest_lengths is None:
        rest = alen[ids].astype(float)
    else:
        rest = np.asarray(muscles.rest_lengths, dtype=float)
    if bool(np.any(rest <= 0.0)):
        raise ConfigurationError("rest_lengths must be strictly positive.")
    length = (alen[ids] * muscles.length_scale) / rest
    velocity = (avel[ids] * muscles.velocity_scale) / rest
    force = np.maximum(afor[ids] * muscles.force_scale, 0.0)
    return MechanicalState(
        length=jnp.asarray(length, dtype=jnp.float32),
        velocity=jnp.asarray(velocity, dtype=jnp.float32),
        force=jnp.asarray(force, dtype=jnp.float32),
    )
