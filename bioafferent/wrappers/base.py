"""Adapter interfaces separating wrappers from core physiology.

Dependency direction: ``wrappers -> models -> core``. Wrappers convert
simulator-specific mechanics into :class:`MechanicalState`; core models never
import wrappers or simulators.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

import jax


class MechanicalState(NamedTuple):
    """Simulator-agnostic per-muscle mechanics. Each leaf shape ``(muscles,)``."""

    length: jax.Array  # fascicle length, L/L0 (dimensionless)
    velocity: jax.Array  # stretch velocity, L0/s
    force: jax.Array  # muscle-tendon force, N


@runtime_checkable
class MechanicalAdapter(Protocol):
    """Extract :class:`MechanicalState` from an environment transition."""

    def extract(self, env: object, obs: object, info: dict) -> MechanicalState:
        """Return per-muscle mechanics for the current env state."""
        ...
