"""SI unit conventions and conversion helpers.

Internal units:
- muscle/fascicle length: meters (m); normalized length: dimensionless (L/L0)
- velocity: meters per second (m/s); normalized: L0/s
- force: newton (N)
- firing rate: hertz (Hz, spikes/s)
- time / timestep / spike time: seconds (s)
- fusimotor drive: Hz for gamma rates, dimensionless [0, 1] for activations

Physics engines may use different scales; convert *before* calling models.
"""

from __future__ import annotations

MM_PER_M = 1000.0
MS_PER_S = 1000.0


def mm_to_m(x: float) -> float:
    """Millimeters to meters."""
    return x / MM_PER_M


def m_to_mm(x: float) -> float:
    """Meters to millimeters."""
    return x * MM_PER_M


def ms_to_s(x: float) -> float:
    """Milliseconds to seconds."""
    return x / MS_PER_S


def normalize_length(length_m: float, rest_length_m: float) -> float:
    """Absolute length (m) to dimensionless stretch ratio L/L0."""
    if rest_length_m <= 0.0:
        raise ValueError(f"rest_length_m must be > 0, got {rest_length_m}.")
    return length_m / rest_length_m
