"""Library logging: quiet by default, diagnostics opt-in."""

from __future__ import annotations

import logging

logger = logging.getLogger("bioafferent")
logger.addHandler(logging.NullHandler())


def set_level(level: int | str) -> None:
    """Set the bioafferent logger verbosity, e.g. ``set_level("WARNING")``."""
    logger.setLevel(level)
