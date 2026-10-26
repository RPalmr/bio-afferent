"""MuJoCo (CPU, NumPy-side) adapter: actuator mechanics to MechanicalState.

Zero-config usage models every actuator as a muscle, measuring rest lengths
from the current data snapshot::

    from bioafferent.wrappers.mujoco import extract_mujoco_mechanics
    mech = extract_mujoco_mechanics(data)

Pass ``actuator_ids`` to select a subset and ``rest_lengths`` to fix the L/L0
normalization instead of measuring it. Assumptions (checked at runtime):

- Position-transmission actuators expose ``actuator_length/_velocity/_force``.
- Units are the model's native units; convert to SI (m, m/s, N) with the
  ``*_scale`` factors when the model uses other scales.

The core models never see MuJoCo objects: this module converts to plain JAX
arrays at the boundary (one host-side conversion per step).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from bioafferent.core.exceptions import ConfigurationError, UnsupportedBackendError
from bioafferent.wrappers.base import MechanicalState

try:
    import mujoco  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - mujoco is an optional dependency
    mujoco = None  # type: ignore[assignment]


def extract_mujoco_mechanics(
    data: Any,
    actuator_ids: Any = None,
    rest_lengths: Any = None,
    length_scale: float = 1.0,
    velocity_scale: float = 1.0,
    force_scale: float = 1.0,
    force_mode: str = "magnitude",
) -> MechanicalState:
    """Read actuator length/velocity/force into a :class:`MechanicalState`.

    Works with real ``mujoco.MjData`` and duck-typed test doubles. Raises
    :class:`UnsupportedBackendError` when mujoco is missing or the data lacks
    actuator arrays.

    ``force_mode`` controls how signed actuator forces are mapped to the
    non-negative tendon force expected by the GTO:

    - ``"magnitude"`` (default): ``abs(force)``. MuJoCo muscle actuators report
      negative force in the pulling direction, so magnitude is the tendon force.
    - ``"positive"``: ``max(force, 0)`` for sources that already report a
      non-negative tendon force.
    """
    if mujoco is None:
        raise UnsupportedBackendError(
            "mujoco is not installed. Install the 'mujoco' extra: pip install bioafferent[mujoco]."
        )
    try:
        alen = np.asarray(data.actuator_length)
        avel = np.asarray(data.actuator_velocity)
        afor = np.asarray(data.actuator_force)
    except AttributeError as exc:
        raise UnsupportedBackendError(
            "MuJoCo data lacks actuator_length/_velocity/_force arrays; this model structure is unsupported."
        ) from exc
    if alen.ndim != 1 or avel.shape != alen.shape or afor.shape != alen.shape:
        raise UnsupportedBackendError("Unexpected MuJoCo actuator array shapes; unsupported configuration.")
    ids = np.arange(alen.shape[0]) if actuator_ids is None else np.asarray(actuator_ids, dtype=int)
    if ids.size == 0 or int(ids.max()) >= alen.shape[0] or int(ids.min()) < 0:
        raise ConfigurationError(f"actuator_ids out of range for nu={alen.shape[0]}.")
    rest = alen[ids].astype(float) if rest_lengths is None else np.asarray(rest_lengths, dtype=float)
    if rest.shape != ids.shape or bool(np.any(rest <= 0.0)):
        raise ConfigurationError("rest_lengths must be positive with one entry per muscle.")
    if min(length_scale, velocity_scale, force_scale) <= 0.0:
        raise ConfigurationError("scale factors must be positive.")
    if force_mode not in ("magnitude", "positive"):
        raise ConfigurationError(f"force_mode must be 'magnitude' or 'positive', got {force_mode!r}.")
    raw_force = afor[ids] * force_scale
    force = np.abs(raw_force) if force_mode == "magnitude" else np.maximum(raw_force, 0.0)
    return MechanicalState(
        length=jnp.asarray(alen[ids] * length_scale / rest, dtype=jnp.float32),
        velocity=jnp.asarray(avel[ids] * velocity_scale / rest, dtype=jnp.float32),
        force=jnp.asarray(force, dtype=jnp.float32),
    )
