"""Explicit immutable simulation state (JAX pytree compatible).

All state objects are registered JAX pytrees, so they pass transparently
through ``jax.jit``, ``jax.vmap`` and ``jax.lax.scan``. States are treated as
immutable: every ``step`` returns a new state object.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class

from bioafferent.utils.typing import FloatArray


@register_pytree_node_class
class SpindleState:
    """State of one `MuscleSpindle` step.

    Attributes:
        tension: per-fiber tension ``(..., 3)`` with fiber order
            ``(bag1, bag2, chain)``. Units: normalized tension (dimensionless).
        activation_dynamic: dynamic fusimotor activation in ``[0, 1]``,
            shape ``(...)``.
        activation_static: static fusimotor activation in ``[0, 1]``,
            shape ``(...)``.
    """

    tension: FloatArray
    activation_dynamic: FloatArray
    activation_static: FloatArray

    def __init__(
        self,
        tension: Array,
        activation_dynamic: Array,
        activation_static: Array,
    ) -> None:
        self.tension = tension
        self.activation_dynamic = activation_dynamic
        self.activation_static = activation_static

    def replace(self, **kwargs: Array) -> SpindleState:
        return SpindleState(
            tension=kwargs.get("tension", self.tension),
            activation_dynamic=kwargs.get(
                "activation_dynamic", self.activation_dynamic
            ),
            activation_static=kwargs.get("activation_static", self.activation_static),
        )

    def tree_flatten(self) -> tuple[tuple[Array, ...], None]:
        return ((self.tension, self.activation_dynamic, self.activation_static), None)

    @classmethod
    def tree_unflatten(cls, aux: None, children: tuple[Array, ...]) -> SpindleState:
        tension, act_d, act_s = children
        return cls(tension=tension, activation_dynamic=act_d, activation_static=act_s)


@register_pytree_node_class
class GTOState:
    """State of one `GolgiTendonOrgan` step.

    Attributes:
        force_filtered: low-pass filtered tendon force in newton, shape ``(...)``.
    """

    force_filtered: FloatArray

    def __init__(self, force_filtered: Array) -> None:
        self.force_filtered = force_filtered

    def replace(self, **kwargs: Array) -> GTOState:
        return GTOState(
            force_filtered=kwargs.get("force_filtered", self.force_filtered)
        )

    def tree_flatten(self) -> tuple[tuple[Array, ...], None]:
        return ((self.force_filtered,), None)

    @classmethod
    def tree_unflatten(cls, aux: None, children: tuple[Array, ...]) -> GTOState:
        return cls(force_filtered=children[0])


@register_pytree_node_class
class EncoderState:
    """State of one `AfferentEncoder` step.

    Attributes:
        residual: time to next spike (gamma-renewal) or remaining absolute
            refractory time (Poisson modes) in seconds, shape ``(...)``.
            Zeros when no countdown is active.
    """

    residual: FloatArray

    def __init__(self, residual: Array) -> None:
        self.residual = residual

    def replace(self, **kwargs: Array) -> EncoderState:
        return EncoderState(residual=kwargs.get("residual", self.residual))

    def tree_flatten(self) -> tuple[tuple[Array, ...], None]:
        return ((self.residual,), None)

    @classmethod
    def tree_unflatten(cls, aux: None, children: tuple[Array, ...]) -> EncoderState:
        return cls(residual=children[0])


def zeros_state_helper(shape: tuple[int, ...], dtype: Any) -> Array:
    """Allocate a zero array for state construction (host-side helper)."""
    return jnp.zeros(shape, dtype=dtype)
