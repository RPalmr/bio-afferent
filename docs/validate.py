"""Calibration/validation harness for the default parameter set.

Regenerates reference traces and checks qualitative calibration targets.
Usage: ``python docs/validate.py``
"""

from __future__ import annotations

import numpy as np

from bioafferent import GolgiTendonOrgan, MuscleSpindle

DT = 0.001


def _ramp_and_hold():
    n_base, n_ramp, n_hold = 500, 1100, 1100
    base = np.full(n_base, 0.95)
    ramp = 0.95 + 0.11 * DT * np.arange(1, n_ramp + 1)
    hold = np.full(n_hold, float(ramp[-1]))
    length = np.concatenate([base, ramp, hold])
    velocity = np.concatenate([[0.0], np.diff(length) / DT])
    return length, velocity


def check_spindle_targets():
    L, V = _ramp_and_hold()
    _, _, n_hold = 500, 1100, 1100
    sp = MuscleSpindle()
    ss = sp.init_state(())
    ia, ii = [], []
    for length, vel in zip(L, V, strict=True):
        ss, out = sp.step(ss, float(length), float(vel), 70.0, 70.0, DT)
        ia.append(float(out.ia_rate))
        ii.append(float(out.ii_rate))
    ia = np.array(ia)
    targets = {
        "ia_peak_above_hold_mean": bool(np.max(ia) > np.mean(ia[-n_hold:]) + 5.0),
        "ii_follows_length": bool(np.mean(np.array(ii)[500:]) > np.mean(np.array(ii)[:500]) + 0.5),
    }
    return targets


def check_gto_monotonic():
    forces = np.linspace(0.0, 200.0, 101)
    gto = GolgiTendonOrgan()
    ibs = []
    for f in forces:
        _, out = gto.step(gto.init_state(()), float(f), DT)
        ibs.append(float(out.ib_rate))
    ibs = np.array(ibs)
    targets = {
        "gto_monotonic": bool(np.all(np.diff(ibs) >= -1e-3)),
    }
    return targets


if __name__ == "__main__":
    print("Spindle targets:", check_spindle_targets())
    print("GTO targets:", check_gto_monotonic())
