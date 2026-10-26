"""Research-grade validation harness for bioafferent.

This package is intentionally outside the installable ``bioafferent`` package:
it carries heavier optional dependencies (SciPy, pandas, MuJoCo) and is meant
to be run from a checkout by researchers, not imported by library users.

The modules implement the verification/validation protocol in
``validation/protocol.md``:

- :mod:`validation.io` -- digitised-trace schema, provenance, hashing;
- :mod:`validation.splits` -- condition-level calibration/validation/blind splits;
- :mod:`validation.metrics` -- rate, spike-train, and decoding metrics;
- :mod:`validation.reference` -- independent NumPy reference implementation;
- :mod:`validation.verification` -- code verification against the reference;
- :mod:`validation.calibrate` -- parameter fitting and held-out evaluation;
- :mod:`validation.blind` -- freeze/predict/score blind-validation workflow;
- :mod:`validation.simulator` -- MuJoCo/MJX semantics and parity checks.

Nothing in this package asserts that the model is validated. It provides the
machinery to make that claim falsifiable once published traces are digitised
and the pre-registered acceptance criteria are met.
"""

from __future__ import annotations

__all__ = [
    "blind",
    "calibrate",
    "io",
    "metrics",
    "reference",
    "simulator",
    "splits",
    "verification",
]
