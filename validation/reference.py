"""Independent NumPy reference implementation of the current model equations.

This is *code verification*, not scientific validation: it re-implements the
same equations as the JAX library so that implementation bugs (shape handling,
in-place updates, integrator dispatch, thixotropy bookkeeping) are detectable
without trusting the JAX code. It does not reproduce Mileusnic et al. (2006)
or Vannucci et al. (2017) unless their published parameters are supplied.

The reference is deliberately written in plain loops and NumPy so it is easy to
audit against the equations in ``docs/physiology.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bioafferent.models.gto_parameters import GTOConfig
from bioafferent.models.spindle_parameters import SpindleConfig


@dataclass
class ReferenceSpindleState:
    tension: np.ndarray
    activation_dynamic: np.ndarray
    activation_static: np.ndarray
    history: np.ndarray


@dataclass
class ReferenceGTOState:
    force_filtered: np.ndarray


def _triplet(value: tuple[float, float, float] | list[float]) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def hill_activation(rate_hz: np.ndarray, half_rate: float, exponent: float) -> np.ndarray:
    rate = np.maximum(np.asarray(rate_hz, dtype=np.float64), 0.0)
    num = np.power(rate, exponent)
    den = num + np.power(np.asarray(half_rate, dtype=np.float64), exponent)
    return num / np.maximum(den, 1e-12)


def first_order_targets(
    tensions: np.ndarray,
    length: np.ndarray,
    velocity: np.ndarray,
    activation: np.ndarray,
    stiffness_rest: np.ndarray,
    stiffness_gain: np.ndarray,
    damping_rest: np.ndarray,
    damping_gain: np.ndarray,
    tau_rest: np.ndarray,
    tau_gain: np.ndarray,
    length_threshold: np.ndarray,
    prestretch_gain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    del tensions  # kept for signature parity with the JAX implementation
    stretch = np.maximum(length[..., None] - length_threshold + prestretch_gain * activation, 0.0)
    stiff = stiffness_rest * (1.0 + stiffness_gain * activation)
    damp = damping_rest * (1.0 + damping_gain * activation)
    tau = tau_rest / (1.0 + tau_gain * activation)
    target = stiff * stretch + damp * velocity[..., None]
    return np.maximum(target, 0.0), tau


def receptor_potentials(
    tensions: np.ndarray,
    gain: np.ndarray,
    exponent: np.ndarray,
    threshold: np.ndarray,
) -> np.ndarray:
    drive = np.maximum(tensions - threshold, 0.0)
    return gain * np.power(drive + 1e-12, exponent)


def combine_afferents(
    potentials: np.ndarray,
    ia_weights: np.ndarray,
    occlusion: float,
    baseline_ia: float,
    baseline_ii: float,
    max_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    p_bag1, p_bag2, p_chain = potentials[..., 0], potentials[..., 1], potentials[..., 2]
    weighted = ia_weights[0] * p_bag1 + ia_weights[1] * p_bag2 + ia_weights[2] * p_chain
    ia = weighted / (1.0 + occlusion * weighted)
    ii = p_bag2 + p_chain
    return (
        np.clip(ia + baseline_ia, 0.0, max_rate),
        np.clip(ii + baseline_ii, 0.0, max_rate),
    )


def gto_transduction(
    force_filtered: np.ndarray,
    threshold: float,
    gain: float,
    saturation_force: float,
    baseline: float,
    max_rate: float,
) -> np.ndarray:
    x = np.maximum(force_filtered - threshold, 0.0)
    rate = baseline + gain * x / (1.0 + x / max(saturation_force, 1e-12))
    return np.clip(rate, 0.0, max_rate)


def _exponential_step(target: np.ndarray, tau: np.ndarray, state: np.ndarray, dt: float) -> np.ndarray:
    tau_safe = np.maximum(tau, 1e-9)
    alpha = 1.0 - np.exp(-dt / tau_safe)
    return state + (target - state) * alpha


def _euler_step(target: np.ndarray, tau: np.ndarray, state: np.ndarray, dt: float) -> np.ndarray:
    return state + dt * (target - state) / np.maximum(tau, 1e-9)


def _rk2_step(target: np.ndarray, tau: np.ndarray, state: np.ndarray, dt: float) -> np.ndarray:
    k1 = (target - state) / np.maximum(tau, 1e-9)
    midpoint = state + 0.5 * dt * k1
    k2 = (target - midpoint) / np.maximum(tau, 1e-9)
    return state + dt * k2


class SpindleReference:
    """NumPy implementation of :class:`bioafferent.models.spindle.MuscleSpindle`."""

    def __init__(self, config: SpindleConfig | None = None) -> None:
        self.config = config or SpindleConfig(dtype="float64")
        c = self.config
        self._k_rest = _triplet(c.stiffness_rest)
        self._k_gain = _triplet(c.stiffness_gain)
        self._b_rest = _triplet(c.damping_rest)
        self._b_gain = _triplet(c.damping_gain)
        self._tau_rest = _triplet(c.tau_fiber_rest)
        self._tau_gain = _triplet(c.tau_fiber_gain)
        self._l_thr = _triplet(c.length_threshold)
        self._k_pre = _triplet(c.prestretch_gain)
        self._g = _triplet(c.receptor_gain)
        self._p = _triplet(c.receptor_exponent)
        self._t0 = _triplet(c.receptor_threshold)
        self._w = _triplet(c.ia_weights)

    def init_state(self, shape: tuple[int, ...] | int = ()) -> ReferenceSpindleState:
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        zeros = np.zeros(shape, dtype=np.float64)
        return ReferenceSpindleState(
            tension=np.zeros(shape + (3,), dtype=np.float64),
            activation_dynamic=zeros.copy(),
            activation_static=zeros.copy(),
            history=zeros.copy(),
        )

    def step(
        self,
        state: ReferenceSpindleState,
        length: Any,
        velocity: Any,
        gamma_dynamic: Any = 0.0,
        gamma_static: Any = 0.0,
        dt: float = 0.001,
    ) -> tuple[ReferenceSpindleState, dict[str, np.ndarray]]:
        if not dt > 0.0:
            raise ValueError("dt must be positive.")
        c = self.config
        L = np.asarray(length, dtype=np.float64)
        V = np.asarray(velocity, dtype=np.float64)
        gd = np.asarray(gamma_dynamic, dtype=np.float64)
        gs = np.asarray(gamma_static, dtype=np.float64)

        a_d_inf = hill_activation(gd, c.hill_half_rate_dynamic, c.hill_exponent)
        a_s_inf = hill_activation(gs, c.hill_half_rate_static, c.hill_exponent)
        alpha_d = 1.0 - np.exp(-dt / c.tau_activation_dynamic)
        alpha_s = 1.0 - np.exp(-dt / c.tau_activation_static)
        a_d = state.activation_dynamic + (a_d_inf - state.activation_dynamic) * alpha_d
        a_s = state.activation_static + (a_s_inf - state.activation_static) * alpha_s
        a_d = np.clip(a_d, 0.0, 1.0)
        a_s = np.clip(a_s, 0.0, 1.0)
        activation = np.stack([a_d, a_s, a_s * c.chain_static_scale], axis=-1)

        target, tau = first_order_targets(
            state.tension,
            L,
            V,
            activation,
            self._k_rest,
            self._k_gain,
            self._b_rest,
            self._b_gain,
            self._tau_rest,
            self._tau_gain,
            self._l_thr,
            self._k_pre,
        )
        if c.integrator == "exponential":
            tension = _exponential_step(target, tau, state.tension, dt)
        elif c.integrator == "euler":
            tension = _euler_step(target, tau, state.tension, dt)
        elif c.integrator == "rk2":
            tension = _rk2_step(target, tau, state.tension, dt)
        else:
            raise ValueError(f"Unknown integrator {c.integrator!r}.")
        tension = np.maximum(tension, 0.0)

        if c.enable_thixotropy:
            tension_change = np.sum(tension - state.tension, axis=-1) / max(dt, 1e-9)
            alpha_hist = 1.0 - np.exp(-dt / c.thixotropy_tau)
            new_history = state.history + (tension_change - state.history) * alpha_hist
            tension = tension + c.thixotropy_scale * new_history * np.ones_like(tension)
        else:
            new_history = state.history

        potentials = receptor_potentials(tension, self._g, self._p, self._t0)
        ia_rate, ii_rate = combine_afferents(
            potentials,
            self._w,
            c.occlusion,
            c.baseline_ia,
            c.baseline_ii,
            c.max_rate,
        )
        new_state = ReferenceSpindleState(
            tension=tension,
            activation_dynamic=a_d,
            activation_static=a_s,
            history=new_history,
        )
        return new_state, {
            "ia_rate": ia_rate,
            "ii_rate": ii_rate,
            "tension": tension,
            "activation_dynamic": a_d,
            "activation_static": a_s,
        }

    def rollout(
        self,
        state: ReferenceSpindleState,
        length: np.ndarray,
        velocity: np.ndarray,
        gamma_dynamic: np.ndarray | float = 0.0,
        gamma_static: np.ndarray | float = 0.0,
        dt: float = 0.001,
    ) -> tuple[ReferenceSpindleState, dict[str, np.ndarray]]:
        L = np.asarray(length, dtype=np.float64)
        V = np.asarray(velocity, dtype=np.float64)
        gd = np.broadcast_to(np.asarray(gamma_dynamic, dtype=np.float64), L.shape)
        gs = np.broadcast_to(np.asarray(gamma_static, dtype=np.float64), L.shape)
        outputs: dict[str, list[np.ndarray]] = {}
        for index in range(L.shape[0]):
            state, out = self.step(state, L[index], V[index], gd[index], gs[index], dt)
            for key, value in out.items():
                outputs.setdefault(key, []).append(value)
        return state, {key: np.stack(values, axis=0) for key, values in outputs.items()}


class GTOReference:
    """NumPy implementation of :class:`bioafferent.models.gto.GolgiTendonOrgan`."""

    def __init__(self, config: GTOConfig | None = None) -> None:
        self.config = config or GTOConfig(dtype="float64")

    def init_state(self, shape: tuple[int, ...] | int = ()) -> ReferenceGTOState:
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        return ReferenceGTOState(force_filtered=np.zeros(shape, dtype=np.float64))

    def step(
        self,
        state: ReferenceGTOState,
        force: Any,
        dt: float = 0.001,
    ) -> tuple[ReferenceGTOState, dict[str, np.ndarray]]:
        if not dt > 0.0:
            raise ValueError("dt must be positive.")
        c = self.config
        F = np.asarray(force, dtype=np.float64)
        if c.integrator == "exponential":
            target = F
            tau = np.full_like(state.force_filtered, c.tau_force, dtype=np.float64)
            filtered = _exponential_step(target, tau, state.force_filtered, dt)
        elif c.integrator == "euler":
            filtered = state.force_filtered + dt * (F - state.force_filtered) / c.tau_force
        elif c.integrator == "rk2":
            k1 = (F - state.force_filtered) / c.tau_force
            k2 = (F - (state.force_filtered + 0.5 * dt * k1)) / c.tau_force
            filtered = state.force_filtered + dt * k2
        else:
            raise ValueError(f"Unknown integrator {c.integrator!r}.")
        ib = gto_transduction(filtered, c.threshold, c.gain, c.saturation_force, c.baseline, c.max_rate)
        return ReferenceGTOState(force_filtered=filtered), {"ib_rate": ib, "force_filtered": filtered}

    def rollout(
        self,
        state: ReferenceGTOState,
        force: np.ndarray,
        dt: float = 0.001,
    ) -> tuple[ReferenceGTOState, dict[str, np.ndarray]]:
        F = np.asarray(force, dtype=np.float64)
        outputs: dict[str, list[np.ndarray]] = {}
        for index in range(F.shape[0]):
            state, out = self.step(state, F[index], dt)
            for key, value in out.items():
                outputs.setdefault(key, []).append(value)
        return state, {key: np.stack(values, axis=0) for key, values in outputs.items()}
