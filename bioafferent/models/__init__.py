"""Physiological models: spindle, GTO, spike encoding and decoding."""

from bioafferent.models.decoding import decode_rates
from bioafferent.models.encoding import (
    AfferentEncoder,
    EncoderConfig,
    EncoderOutput,
    population_encode,
)
from bioafferent.models.gto import GolgiTendonOrgan, GTOOutput
from bioafferent.models.gto_parameters import GTOConfig
from bioafferent.models.spindle import MuscleSpindle
from bioafferent.models.spindle_parameters import (
    FIBER_NAMES,
    N_FIBERS,
    SpindleConfig,
    SpindleOutput,
)

__all__ = [
    "FIBER_NAMES",
    "N_FIBERS",
    "AfferentEncoder",
    "EncoderConfig",
    "EncoderOutput",
    "GTOConfig",
    "GTOOutput",
    "GolgiTendonOrgan",
    "MuscleSpindle",
    "SpindleConfig",
    "SpindleOutput",
    "decode_rates",
    "population_encode",
]
