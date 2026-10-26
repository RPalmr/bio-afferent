"""Per-field parameter provenance for validation and reporting.

The distinction matters for scientific claims:

- ``PUBLISHED``: the numeric value (or a direct derivation such as
  ``t90 / ln(10)``) is taken from the cited source.
- ``ENGINEERING``: the equation structure follows a source, but the coefficient
  is this library's calibration choice.
- ``NUMERICAL``: implementation setting with no physiological claim.

Every configuration field must have an entry. The validation harness uses this
registry to generate the parameter table required by ``validation/protocol.md``
and to prevent an engineering default from being described as published.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any

from bioafferent.models.gto_parameters import GTOConfig
from bioafferent.models.spindle_parameters import SpindleConfig


class ProvenanceKind(str, Enum):
    PUBLISHED = "published"
    ENGINEERING = "engineering"
    NUMERICAL = "numerical"


@dataclass(frozen=True)
class FieldProvenance:
    kind: ProvenanceKind
    source: str
    note: str = ""


SPINDLE_PROVENANCE: dict[str, FieldProvenance] = {
    "hill_half_rate_dynamic": FieldProvenance(
        ProvenanceKind.ENGINEERING,
        "Mileusnic et al. 2006 (Hill structure); value is a calibration choice",
    ),
    "hill_half_rate_static": FieldProvenance(
        ProvenanceKind.ENGINEERING,
        "Mileusnic et al. 2006 (Hill structure); value is a calibration choice",
    ),
    "hill_exponent": FieldProvenance(
        ProvenanceKind.ENGINEERING,
        "Mileusnic et al. 2006 (Hill structure); value is a calibration choice",
    ),
    "tau_activation_dynamic": FieldProvenance(
        ProvenanceKind.PUBLISHED,
        "Vannucci et al. 2017: 90% rise in 343 ms; tau = t90 / ln(10)",
    ),
    "tau_activation_static": FieldProvenance(
        ProvenanceKind.PUBLISHED,
        "Vannucci et al. 2017: 90% rise in 471 ms; tau = t90 / ln(10)",
    ),
    "chain_static_scale": FieldProvenance(
        ProvenanceKind.ENGINEERING,
        "Engineering calibration of static drive onto chain fibers",
    ),
    "stiffness_rest": FieldProvenance(ProvenanceKind.ENGINEERING, "Engineering calibration"),
    "stiffness_gain": FieldProvenance(ProvenanceKind.ENGINEERING, "Engineering calibration"),
    "damping_rest": FieldProvenance(ProvenanceKind.ENGINEERING, "Engineering calibration"),
    "damping_gain": FieldProvenance(ProvenanceKind.ENGINEERING, "Engineering calibration"),
    "tau_fiber_rest": FieldProvenance(ProvenanceKind.ENGINEERING, "Engineering calibration"),
    "tau_fiber_gain": FieldProvenance(ProvenanceKind.ENGINEERING, "Engineering calibration"),
    "length_threshold": FieldProvenance(ProvenanceKind.ENGINEERING, "Engineering calibration"),
    "prestretch_gain": FieldProvenance(ProvenanceKind.ENGINEERING, "Engineering adaptation"),
    "receptor_gain": FieldProvenance(ProvenanceKind.ENGINEERING, "Engineering adaptation"),
    "receptor_exponent": FieldProvenance(ProvenanceKind.ENGINEERING, "Engineering adaptation"),
    "receptor_threshold": FieldProvenance(ProvenanceKind.ENGINEERING, "Engineering adaptation"),
    "ia_weights": FieldProvenance(ProvenanceKind.ENGINEERING, "Engineering calibration"),
    "occlusion": FieldProvenance(ProvenanceKind.ENGINEERING, "Engineering adaptation"),
    "baseline_ia": FieldProvenance(ProvenanceKind.ENGINEERING, "Engineering calibration"),
    "baseline_ii": FieldProvenance(ProvenanceKind.ENGINEERING, "Engineering calibration"),
    "max_rate": FieldProvenance(ProvenanceKind.NUMERICAL, "Hard output cap"),
    "enable_thixotropy": FieldProvenance(
        ProvenanceKind.ENGINEERING,
        "Feature flag; the implemented history term is not the full Blum et al. 2017 model",
    ),
    "thixotropy_tau": FieldProvenance(
        ProvenanceKind.ENGINEERING,
        "Engineering approximation; validate before attributing to Blum et al. 2017",
    ),
    "thixotropy_scale": FieldProvenance(
        ProvenanceKind.ENGINEERING,
        "Engineering approximation; validate before attributing to Blum et al. 2017",
    ),
    "integrator": FieldProvenance(ProvenanceKind.NUMERICAL, "ODE solver selection"),
    "dtype": FieldProvenance(ProvenanceKind.NUMERICAL, "Compute precision"),
}

GTO_PROVENANCE: dict[str, FieldProvenance] = {
    "tau_force": FieldProvenance(
        ProvenanceKind.ENGINEERING,
        "First-order dynamics in the style of Lin & Crago 2002; value is a calibration choice",
    ),
    "threshold": FieldProvenance(
        ProvenanceKind.ENGINEERING,
        "Threshold property after Houk & Simon 1967; value is a calibration choice",
    ),
    "gain": FieldProvenance(ProvenanceKind.ENGINEERING, "Engineering calibration"),
    "saturation_force": FieldProvenance(
        ProvenanceKind.ENGINEERING,
        "Saturating property after Houk & Simon 1967; value is a calibration choice",
    ),
    "baseline": FieldProvenance(ProvenanceKind.ENGINEERING, "Engineering calibration"),
    "max_rate": FieldProvenance(ProvenanceKind.NUMERICAL, "Hard output cap"),
    "integrator": FieldProvenance(ProvenanceKind.NUMERICAL, "ODE solver selection"),
    "dtype": FieldProvenance(ProvenanceKind.NUMERICAL, "Compute precision"),
}


def provenance_for(config: SpindleConfig | GTOConfig) -> dict[str, FieldProvenance]:
    """Return the provenance registry for a configuration instance."""
    if isinstance(config, SpindleConfig):
        return SPINDLE_PROVENANCE
    if isinstance(config, GTOConfig):
        return GTO_PROVENANCE
    raise TypeError(f"Unsupported config type {type(config).__name__}.")


def check_provenance_complete(config: SpindleConfig | GTOConfig) -> None:
    """Raise if a field is missing from (or extra in) the registry."""
    registry = provenance_for(config)
    names = {item.name for item in fields(config)}
    missing = names - set(registry)
    extra = set(registry) - names
    if missing or extra:
        raise ValueError(f"Provenance registry mismatch: missing={sorted(missing)}, extra={sorted(extra)}.")


def parameter_table(config: SpindleConfig | GTOConfig) -> list[dict[str, Any]]:
    """Return a JSON-serialisable provenance table for reports."""
    check_provenance_complete(config)
    registry = provenance_for(config)
    rows: list[dict[str, Any]] = []
    for item in fields(config):
        provenance = registry[item.name]
        rows.append(
            {
                "field": item.name,
                "value": getattr(config, item.name),
                "kind": provenance.kind.value,
                "source": provenance.source,
                "note": provenance.note,
            }
        )
    return rows
