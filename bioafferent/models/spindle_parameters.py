"""Spindle configuration and per-fiber parameters.

Scientific record: the implemented dynamics follow the *structure* of
Mileusnic et al. (2006) — three intrafusal fiber types (bag1, bag2, chain)
with polar/sensory tension dynamics, dynamic/static fusimotor drives, and
partial occlusion of convergent inputs onto the Ia ending — reduced to
first-order tension dynamics per Vannucci et al. (2017). The numeric
coefficients below are documented engineering defaults calibrated for
qualitative agreement with published response features (see
``docs/physiology.md``); they are NOT the copyrighted Table 1 values of
Mileusnic et al. All parameters are user-overridable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from bioafferent.core.exceptions import ConfigurationError, ParameterError

N_FIBERS = 3
FIBER_NAMES = ("bag1", "bag2", "chain")


def _triplet(values: tuple[float, float, float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ParameterError(
            f"{name} must be 3 finite values (bag1, bag2, chain), got {values}."
        )
    return arr


@dataclass(frozen=True)
class SpindleConfig:
    """Immutable spindle configuration.

    Units: length dimensionless (L/L0), velocity in L0/s, gamma rates in Hz,
    firing rates in Hz, time constants in s. ``dtype`` selects compute precision
    ("float32" default, "float64" supported).
    """

    # --- fusimotor transduction (Hill + low-pass) ---
    hill_half_rate_dynamic: float = 60.0  # Hz; half-activation gamma_d rate
    hill_half_rate_static: float = 60.0  # Hz; half-activation gamma_s rate
    hill_exponent: float = 2.0  # Hill cooperativity exponent
    # Rise to 90% of step response: 343 ms (dynamic) / 471 ms (static),
    # after Vannucci et al. 2017 (tau = t90 / ln(10)).
    tau_activation_dynamic: float = 0.343 / 2.302585093
    tau_activation_static: float = 0.471 / 2.302585093
    chain_static_scale: float = 0.829  # chain static-activation scaling

    # --- per-fiber mechanics, order (bag1, bag2, chain) ---
    # bag1 is velocity-specialized (low stiffness, high damping); bag2/chain
    # are length-specialized. prestretch_gain sets polar-preload per unit
    # fusimotor activation (dimensionless L0), the source of gamma-driven
    # resting discharge at short muscle lengths.
    stiffness_rest: tuple[float, float, float] = (8.0, 15.0, 10.0)
    stiffness_gain: tuple[float, float, float] = (2.5, 2.0, 1.5)
    damping_rest: tuple[float, float, float] = (40.0, 4.0, 3.0)
    damping_gain: tuple[float, float, float] = (6.0, 2.0, 1.5)
    tau_fiber_rest: tuple[float, float, float] = (0.060, 0.120, 0.150)
    tau_fiber_gain: tuple[float, float, float] = (2.0, 1.5, 1.0)
    length_threshold: tuple[float, float, float] = (0.98, 0.97, 0.97)
    prestretch_gain: tuple[float, float, float] = (0.06, 0.08, 0.06)

    # --- sensory transduction ---
    receptor_gain: tuple[float, float, float] = (13.0, 14.0, 10.0)  # Hz/unit
    receptor_exponent: tuple[float, float, float] = (0.7, 0.6, 0.6)
    receptor_threshold: tuple[float, float, float] = (0.5, 0.4, 0.4)

    # --- afferent combination ---
    ia_weights: tuple[float, float, float] = (1.0, 0.6, 0.4)
    occlusion: float = 0.005  # partial-occlusion saturation constant
    baseline_ia: float = 5.0  # Hz
    baseline_ii: float = 5.0  # Hz
    max_rate: float = 300.0  # Hz hard cap

    # --- thixotropy / history effects (Blum et al. 2017) ---
    enable_thixotropy: bool = False
    thixotropy_tau: float = 0.5  # history decay time constant in seconds
    thixotropy_scale: float = 0.1  # history effect magnitude (dimensionless)

    integrator: str = "exponential"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.dtype not in ("float32", "float64"):
            raise ConfigurationError(
                f"dtype must be 'float32' or 'float64', got {self.dtype!r}."
            )
        if not self.thixotropy_tau > 0.0:
            raise ConfigurationError("thixotropy_tau must be strictly positive.")
        if not self.thixotropy_scale >= 0.0:
            raise ConfigurationError("thixotropy_scale must be non-negative.")
        if self.integrator not in ("euler", "rk2", "exponential"):
            raise ConfigurationError(
                f"Unknown integrator {self.integrator!r}. "
                "Choose 'euler', 'rk2', or 'exponential'."
            )
        for name in (
            "hill_half_rate_dynamic",
            "hill_half_rate_static",
            "tau_activation_dynamic",
            "tau_activation_static",
            "max_rate",
        ):
            if getattr(self, name) <= 0.0:
                raise ConfigurationError(
                    f"{name} must be strictly positive, got {getattr(self, name)}."
                )
        if not self.hill_exponent > 0.0:
            raise ConfigurationError("hill_exponent must be positive.")
        if not 0.0 <= self.chain_static_scale <= 1.0:
            raise ConfigurationError("chain_static_scale must lie in [0, 1].")
        if not self.occlusion >= 0.0:
            raise ConfigurationError("occlusion must be non-negative.")
        for name in (
            "stiffness_rest",
            "stiffness_gain",
            "damping_rest",
            "damping_gain",
            "tau_fiber_rest",
            "tau_fiber_gain",
            "length_threshold",
            "receptor_gain",
            "receptor_exponent",
            "receptor_threshold",
            "prestretch_gain",
            "ia_weights",
        ):
            vals = _triplet(getattr(self, name), name)
            if name in (
                "stiffness_rest",
                "damping_rest",
                "tau_fiber_rest",
                "receptor_gain",
                "receptor_exponent",
            ):
                if not bool(np.all(vals > 0.0)):
                    raise ParameterError(f"{name} must be strictly positive.")
            if name in (
                "stiffness_gain",
                "damping_gain",
                "tau_fiber_gain",
                "ia_weights",
                "prestretch_gain",
            ):
                if not bool(np.all(vals >= 0.0)):
                    raise ParameterError(f"{name} must be non-negative.")
        if not self.baseline_ia >= 0.0 or not self.baseline_ii >= 0.0:
            raise ConfigurationError("baselines must be non-negative.")

    def fiber_array(self, name: str) -> jnp.ndarray:
        """Return per-fiber parameter as a ``(3,)`` JAX array in config dtype."""
        return jnp.asarray(_triplet(getattr(self, name), name), dtype=self.numpy_dtype)

    @property
    def numpy_dtype(self):  # type: ignore[no-untyped-def]
        return np.float32 if self.dtype == "float32" else np.float64


class SpindleOutput(NamedTuple):
    """Per-step spindle readout. All arrays share batch shape ``(...)``."""

    ia_rate: jnp.ndarray  # Hz, primary afferent
    ii_rate: jnp.ndarray  # Hz, secondary afferent
    tension: jnp.ndarray  # (..., 3) per-fiber tension
    activation_dynamic: jnp.ndarray  # [0, 1]
    activation_static: jnp.ndarray  # [0, 1]
