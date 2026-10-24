"""MJX (JAX-native) path: mechanics to afferents without host round-trips.

Preferred architecture::

    MJX state -> JAX mechanical arrays -> JAX sensory model -> JAX observation

Zero-config usage reads every actuator, normalizing by the current snapshot::

    from bioafferent.wrappers.mjx import extract_mjx_mechanics
    mech = extract_mjx_mechanics(mjx_data)

`mjx_afferent_step` then advances spindle + GTO states in one JIT-compilable
function. No NumPy conversion, no device transfer.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from bioafferent.core.exceptions import ConfigurationError, UnsupportedBackendError
from bioafferent.core.state import GTOState, SpindleState
from bioafferent.models.gto import GolgiTendonOrgan
from bioafferent.models.gto_parameters import GTOConfig
from bioafferent.models.spindle import MuscleSpindle
from bioafferent.models.spindle_parameters import SpindleConfig
from bioafferent.wrappers.base import MechanicalState

try:
    import mujoco.mjx as _mjx  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - mujoco-mjx is an optional dependency
    _mjx = None  # type: ignore[assignment]


def mjx_available() -> bool:
    """True when ``mujoco.mjx`` can be imported in this environment."""
    return _mjx is not None


def _require_mjx() -> None:
    if _mjx is None:
        raise UnsupportedBackendError(
            "mujoco-mjx is not installed. Install the 'mjx' extra: pip install bioafferent[mjx]."
        )


def mjx_rest_lengths(
    model: Any,
    actuator_ids: Any = None,
    length_scale: float = 1.0,
) -> jax.Array:
    """Actuator lengths at the model reference configuration.

    Runs MJX forward kinematics at ``model.qpos0`` with zero velocity and
    control, then returns the actuator lengths (multiplied by
    ``length_scale``). Pass the result as ``rest_lengths`` to
    :func:`extract_mjx_mechanics`, or pass ``model`` directly, to remove the
    pose dependence of the default L/L0 normalization.

    This is a host-side setup helper; construct the values before ``jit``.
    """
    _require_mjx()
    assert _mjx is not None
    data = _mjx.forward(model, _mjx.make_data(model))
    alen = jnp.asarray(data.actuator_length)
    ids = (
        jnp.arange(alen.shape[0], dtype=jnp.int32)
        if actuator_ids is None
        else jnp.asarray(actuator_ids, dtype=jnp.int32)
    )
    if ids.ndim != 1:
        raise ConfigurationError("actuator_ids must be 1-D.")
    if ids.size == 0 or int(ids.min()) < 0 or int(ids.max()) >= alen.shape[0]:
        raise ConfigurationError(f"actuator_ids out of range for nu={alen.shape[0]}.")
    return jnp.take(alen, ids, axis=0) * length_scale


def extract_mjx_mechanics(
    data: Any,
    actuator_ids: Any = None,
    rest_lengths: Any = None,
    length_scale: float = 1.0,
    velocity_scale: float = 1.0,
    force_scale: float = 1.0,
    model: Any = None,
) -> MechanicalState:
    """Pure-JAX extraction of per-muscle mechanics from MJX ``Data``.

    Defaults model every actuator with rest lengths from the current snapshot.
    Pass ``model`` (recommended) to use reference-configuration rest lengths
    instead, or pass explicit ``rest_lengths``. Works on real MJX Data pytrees
    and duck-typed test doubles. Raises :class:`UnsupportedBackendError` when
    the actuator arrays are absent.
    """
    try:
        alen = jnp.asarray(data.actuator_length)
        avel = jnp.asarray(data.actuator_velocity)
        afor = jnp.asarray(data.actuator_force)
    except AttributeError as exc:
        raise UnsupportedBackendError(
            "MJX data lacks actuator_length/_velocity/_force arrays; this model structure is unsupported."
        ) from exc
    ids = jnp.arange(alen.shape[0]) if actuator_ids is None else jnp.asarray(actuator_ids, dtype=jnp.int32)
    if rest_lengths is None:
        rest = mjx_rest_lengths(model, ids, length_scale) if model is not None else alen[ids]
    else:
        rest = jnp.asarray(rest_lengths)
    return MechanicalState(
        length=jnp.take(alen, ids, axis=0) * length_scale / rest,
        velocity=jnp.take(avel, ids, axis=0) * velocity_scale / rest,
        force=jnp.maximum(jnp.take(afor, ids, axis=0) * force_scale, 0.0),
    )


class MJXAfferents(NamedTuple):
    """Stacked afferent readout for the MJX path."""

    ia_rate: jax.Array
    ii_rate: jax.Array
    ib_rate: jax.Array


def mjx_afferent_step(
    spindle_state: SpindleState,
    gto_state: GTOState,
    mechanics: MechanicalState,
    spindle: MuscleSpindle,
    gto: GolgiTendonOrgan,
    gamma_dynamic: float = 0.0,
    gamma_static: float = 0.0,
    dt: float = 0.001,
) -> tuple[SpindleState, GTOState, MJXAfferents]:
    """Single JIT-compilable spindle+GTO advance on JAX-native mechanics."""
    new_spindle, sp = spindle.step(
        spindle_state,
        mechanics.length,
        mechanics.velocity,
        gamma_dynamic,
        gamma_static,
        dt,
    )
    new_gto, go = gto.step(gto_state, mechanics.force, dt)
    return (
        new_spindle,
        new_gto,
        MJXAfferents(ia_rate=sp.ia_rate, ii_rate=sp.ii_rate, ib_rate=go.ib_rate),
    )


def default_mjx_models(
    spindle_config: SpindleConfig | None = None,
    gto_config: GTOConfig | None = None,
) -> tuple[MuscleSpindle, GolgiTendonOrgan]:
    """Construct default model objects for the MJX path (host side)."""
    return MuscleSpindle(spindle_config), GolgiTendonOrgan(gto_config)
