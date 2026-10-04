"""Bioafferent: JAX proprioceptive sensory feedback (spindle Ia/II, GTO Ib)."""

from bioafferent._version import __version__
from bioafferent.core.exceptions import (
    BioafferentError,
    ConfigurationError,
    InvalidStateError,
    NumericalStabilityError,
    ParameterError,
    PySpindleError,
    ShapeError,
    UnsupportedBackendError,
)
from bioafferent.core.integration import IntegrationConfig
from bioafferent.core.state import EncoderState, GTOState, SpindleState
from bioafferent.models.encoding import AfferentEncoder, EncoderConfig
from bioafferent.models.gto import GolgiTendonOrgan
from bioafferent.models.gto_parameters import GTOConfig
from bioafferent.models.spindle import MuscleSpindle
from bioafferent.models.spindle_parameters import SpindleConfig

__all__ = [
    "AfferentEncoder",
    "BioafferentError",
    "ConfigurationError",
    "EncoderConfig",
    "EncoderState",
    "GTOConfig",
    "GTOState",
    "GolgiTendonOrgan",
    "IntegrationConfig",
    "InvalidStateError",
    "MuscleSpindle",
    "NumericalStabilityError",
    "ParameterError",
    "PySpindleError",
    "ShapeError",
    "SpindleConfig",
    "SpindleState",
    "UnsupportedBackendError",
    "__version__",
]
