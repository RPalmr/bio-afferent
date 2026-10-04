"""MJX (JAX-native) path: mechanics to afferents without host round-trips.

Preferred architecture::

    MJX state -> JAX mechanical arrays -> JAX sensory model -> JAX observation

`extract_mjx_mechanics` reads ``actuator_length``/``actuator_velocity``/
/// ``actuator_force`` from an MJX ``Data`` pytree (or duck-typed equivalent)
using pure JAX ops, and `mjx_afferent_step` advances spindle + GTO states in
one JIT-compilable function. No NumPy conversion, no device transfer.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from bioafferent.core.exceptions import UnsupportedBackendError
from bioafferent.core.state import GTOState, SpindleState
from bioafferent.models.gto import GolgiTendonOrgan
from bioafferent.models.gto_parameters import GTOConfig
from bioafferent.models.spindle import MuscleSpindle
from bioafferent.models.spindle_parameters import SpindleConfig
from bioafferent.wrappers.base import MechanicalState

try:
    import mujoco.mjx as _mjx  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - mujoco is an optional dependency
    _mjx = None  # type: ignore[assignment]


def mjx_available() -> bool:
    """True when ``mujoco.mjx`` can be imported in this environment."""
    return _mjx is not None


class MJXMuscleConfig(NamedTuple):
    """Static muscle indexing for the MJX path (all leaves static/constant)."""

    actuator_ids: jax.Array  # int array, shape (muscles,)
    rest_lengths: jax.Array  # float array, shape (muscles,), model length units
    length_scale: float = 1.0
    velocity_scale: float = 1.0
    force_scale: float = 1.0


def extract_mjx_mechanics(data: Any, cfg: MJXMuscleConfig) -> MechanicalState:
    """Pure-JAX extraction of per-muscle mechanics from MJX ``Data``.

    Reads ``data.actuator_length`` etc. with ``jnp.take``; works on real MJX
    Data pytrees and on duck-typed test doubles. Raises
    :class:`UnsupportedBackendError` if the arrays are absent.
    """
    try:
        alen = jnp.asarray(data.actuator_length)
        avel = jnp.asarray(data.actuator_velocity)
        afor = jnp.asarray(data.actuator_force)
    except AttributeError as exc:
        raise UnsupportedBackendError(
            "MJX data lacks actuator_length/_velocity/_force arrays; "
            "this model structure is unsupported."
        ) from exc
    ids = jnp.asarray(cfg.actuator_ids, dtype=jnp.int32)
    rest = jnp.asarray(cfg.rest_lengths)
    length = jnp.take(alen, ids, axis=0) * cfg.length_scale / rest
    velocity = jnp.take(avel, ids, axis=0) * cfg.velocity_scale / rest
    force = jnp.maximum(jnp.take(afor, ids, axis=0) * cfg.force_scale, 0.0)
    return MechanicalState(length=length, velocity=velocity, force=force)


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
