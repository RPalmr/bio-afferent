"""Beginner-friendly facade: one object for Ia/II/Ib (+ spikes) feedback.

Experts keep the functional API (``MuscleSpindle.step`` etc. with explicit
states); newcomers can ignore states, keys, and wiring entirely::

    from bioafferent import Afferents
    aff = Afferents(n_muscles=2)
    out = aff.step(length=1.03, velocity=0.02, force=15.0)
    out.ia_rate, out.ii_rate, out.ib_rate  # plain JAX arrays
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bioafferent.core.state import EncoderState, GTOState, SpindleState
from bioafferent.models.encoding import AfferentEncoder, EncoderConfig
from bioafferent.models.gto import GolgiTendonOrgan
from bioafferent.models.gto_parameters import GTOConfig
from bioafferent.models.spindle import MuscleSpindle
from bioafferent.models.spindle_parameters import SpindleConfig


class AfferentRates(NamedTuple):
    """One step of afferent feedback. Arrays share shape ``(n_muscles,)``."""

    ia_rate: jax.Array  # Hz
    ii_rate: jax.Array  # Hz
    ib_rate: jax.Array  # Hz
    spikes: jax.Array | None = None  # bool, present only when encoding


class Afferents:
    """Stateful Ia/II/Ib model for ``n_muscles`` muscles.

    Args:
        n_muscles: number of muscles (int).
        spindle_config / gto_config: model configs (defaults if None).
        encoder_config: if given, ``step`` also returns spike trains.
        seed: PRNG seed for spike encoding (int).
    """

    def __init__(
        self,
        n_muscles: int = 1,
        spindle_config: SpindleConfig | None = None,
        gto_config: GTOConfig | None = None,
        encoder_config: EncoderConfig | None = None,
        seed: int = 0,
    ) -> None:
        if not isinstance(n_muscles, int) or n_muscles < 1:
            raise ValueError(f"n_muscles must be a positive int, got {n_muscles}.")
        self.n_muscles = n_muscles
        self.spindle = MuscleSpindle(spindle_config)
        self.gto = GolgiTendonOrgan(gto_config)
        self.encoder = AfferentEncoder(encoder_config) if encoder_config else None
        self._seed = seed
        self._key = jax.random.PRNGKey(seed)
        self.reset()

    def reset(self) -> None:
        """Return all afferent states to rest (keeps the random stream)."""
        n = (self.n_muscles,)
        self._spindle_state: SpindleState = self.spindle.init_state(n)
        self._gto_state: GTOState = self.gto.init_state(n)
        self._enc_state: EncoderState | None = (
            self.encoder.init_state((3 * self.n_muscles,)) if self.encoder else None
        )

    def reseed(self, seed: int) -> None:
        """Restart the spike-encoding random stream from ``seed``."""
        self._seed = seed
        self._key = jax.random.PRNGKey(seed)

    def step(
        self,
        length: object = 1.0,
        velocity: object = 0.0,
        force: object = 0.0,
        gamma_dynamic: float = 0.0,
        gamma_static: float = 0.0,
        dt: float = 0.001,
    ) -> AfferentRates:
        """Advance all muscles one step; returns rates in Hz (and spikes).

        Called with no arguments, the muscle rests at ``L0`` with no drive,
        so all afferents report baseline discharge.
        """
        self._spindle_state, sp = self.spindle.step(
            self._spindle_state,
            length,
            velocity,
            gamma_dynamic,
            gamma_static,
            dt,
        )
        self._gto_state, go = self.gto.step(self._gto_state, force, dt)
        spikes: jax.Array | None = None
        if self.encoder is not None and self._enc_state is not None:
            rates = jnp.concatenate([sp.ia_rate, sp.ii_rate, go.ib_rate])
            self._key, self._enc_state, enc = self.encoder.step(
                self._key, self._enc_state, rates, dt
            )
            spikes = enc.spikes
        return AfferentRates(
            ia_rate=sp.ia_rate,
            ii_rate=sp.ii_rate,
            ib_rate=go.ib_rate,
            spikes=spikes,
        )
