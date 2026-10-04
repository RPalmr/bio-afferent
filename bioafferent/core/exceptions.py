"""Domain-specific exceptions for Bioafferent."""

from __future__ import annotations


class BioafferentError(Exception):
    """Base class for all Bioafferent errors."""


#: Backwards-compatible alias for the name used in the design specification.
PySpindleError = BioafferentError


class ConfigurationError(BioafferentError):
    """Invalid model or integration configuration value."""


class ParameterError(BioafferentError):
    """Invalid physiological parameter value."""


class NumericalStabilityError(BioafferentError):
    """Integration produced non-finite or exploding state."""


class InvalidStateError(BioafferentError):
    """State object is inconsistent with the model or input shapes."""


class ShapeError(BioafferentError):
    """Array shape incompatible with the documented shape convention."""


class UnsupportedBackendError(BioafferentError):
    """Requested backend / optional integration is unavailable."""
