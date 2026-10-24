"""Golgi tendon organ configuration.

Scientific record: the implemented transducer combines the static
properties reported by Houk & Simon (threshold, steep low-force
sensitivity, saturating Ib response) with first-order force dynamics in
the style of Lin & Crago (2002). Numeric defaults are documented
engineering choices (see ``docs/physiology.md``), not fitted constants
from the original papers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bioafferent.core.exceptions import ConfigurationError


@dataclass(frozen=True)
class GTOConfig:
    """Immutable GTO configuration. Force in newton, rates in Hz, time in s."""

    tau_force: float = 0.050  # s; first-order force-filter time constant
    threshold: float = 1.0  # N; force threshold below which Ib is baseline
    gain: float = 4.0  # Hz/N; low-force slope
    saturation_force: float = 60.0  # N; scale of saturating asymptote
    baseline: float = 2.0  # Hz; resting Ib discharge
    max_rate: float = 250.0  # Hz hard cap
    integrator: str = "exponential"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.dtype not in ("float32", "float64"):
            raise ConfigurationError(f"dtype must be 'float32' or 'float64', got {self.dtype!r}.")
        if self.integrator not in ("euler", "rk2", "exponential"):
            raise ConfigurationError(f"Unknown integrator {self.integrator!r}.")
        for name in ("tau_force", "saturation_force", "max_rate"):
            if getattr(self, name) <= 0.0:
                raise ConfigurationError(f"{name} must be strictly positive, got {getattr(self, name)}.")
        if self.threshold < 0.0:
            raise ConfigurationError("threshold must be non-negative.")
        if self.gain < 0.0:
            raise ConfigurationError("gain must be non-negative.")
        if self.baseline < 0.0:
            raise ConfigurationError("baseline must be non-negative.")

    @property
    def numpy_dtype(self):  # type: ignore[no-untyped-def]
        return np.float32 if self.dtype == "float32" else np.float64
