"""Style-free input validation helpers used by models and wrappers."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from bioafferent.core.exceptions import ConfigurationError, ShapeError


def is_tracer(value: Array) -> bool:
    """True when ``value`` is a JAX tracer (i.e. inside jit/vmap/scan)."""
    import jax

    return isinstance(value, jax.core.Tracer)


def check_finite(name: str, value: Array) -> Array:
    """Raise if any element is NaN or infinite. Returns the value unchanged.

    Data-dependent checks run eagerly; under JIT tracing the check is skipped
    (shapes remain statically validated) and invalid inputs propagate through
    the documented clipping behavior instead of raising.
    """
    import jax

    if is_tracer(value):
        return value
    bad = jnp.logical_not(jnp.all(jnp.isfinite(value)))
    if bool(jax.device_get(bad)):
        raise ValueError(
            f"{name} must be finite, got non-finite values. "
            f"Check upstream mechanics/extraction for NaN or inf."
        )
    return value


def check_positive(name: str, value: float) -> float:
    if not value > 0.0:
        raise ConfigurationError(f"{name} must be strictly positive, got {value}.")
    return value


def check_closed_interval(name: str, value: float, lo: float, hi: float) -> float:
    if not lo <= value <= hi:
        raise ConfigurationError(f"{name} must lie in [{lo}, {hi}], got {value}.")
    return value


def check_broadcastable(name: str, value: Array, ref_shape: tuple[int, ...]) -> Array:
    """Validate that `value` broadcasts against reference batch shape."""
    try:
        jnp.broadcast_shapes(value.shape, ref_shape)
    except ValueError as exc:
        raise ShapeError(
            f"{name} with shape {tuple(value.shape)} is not broadcast-compatible "
            f"with state batch shape {tuple(ref_shape)}. Expected scalar, "
            f"{tuple(ref_shape)}, or a leading subset."
        ) from exc
    return value


def check_dtype_support(dtype_name: str, context: str) -> None:
    """Raise if float64 is requested without JAX x64 enabled.

    JAX silently truncates float64 to float32 unless 64-bit mode is enabled;
    fail loudly instead so precision expectations are explicit.
    """
    import jax

    if dtype_name == "float64" and not jax.config.read("jax_enable_x64"):
        raise ConfigurationError(
            f"{context}: dtype='float64' requires JAX 64-bit mode, which is "
            "disabled. Enable it before constructing models, e.g. "
            "`jax.config.update('jax_enable_x64', True)` or "
            "`with jax.experimental.enable_x64():`. "
            "float32 (default) needs no extra setup."
        )


def as_array(value: object, name: str, dtype: object) -> Array:
    try:
        arr = jnp.asarray(value, dtype=dtype)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ShapeError(f"{name} could not be converted to an array: {exc}") from exc
    return arr
