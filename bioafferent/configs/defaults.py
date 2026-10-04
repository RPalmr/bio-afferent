"""Default configuration factories."""

from __future__ import annotations

from bioafferent.core.integration import IntegrationConfig
from bioafferent.models.encoding import EncoderConfig
from bioafferent.models.gto_parameters import GTOConfig
from bioafferent.models.spindle_parameters import SpindleConfig

__all__ = [
    "EncoderConfig",
    "GTOConfig",
    "IntegrationConfig",
    "SpindleConfig",
    "default_configs",
]


def default_configs() -> dict:
    """Return a dict of default configs: spindle, gto, encoder, integration."""
    return {
        "spindle": SpindleConfig(),
        "gto": GTOConfig(),
        "encoder": EncoderConfig(),
        "integration": IntegrationConfig(),
    }
