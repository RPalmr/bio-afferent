"""Gymnasium wrapper injecting proprioceptive afferent observations.

Conceptual flow: ``env -> adapter.extract -> spindle/GTO -> [encoder] ->
augmented observation``. MuJoCo-specific extraction lives in
`bioafferent.wrappers.mujoco`, never here.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bioafferent.core.exceptions import ConfigurationError, UnsupportedBackendError
from bioafferent.core.state import EncoderState, GTOState, SpindleState
from bioafferent.models.encoding import AfferentEncoder, EncoderConfig
from bioafferent.models.gto import GolgiTendonOrgan
from bioafferent.models.gto_parameters import GTOConfig
from bioafferent.models.spindle import MuscleSpindle
from bioafferent.models.spindle_parameters import SpindleConfig
from bioafferent.wrappers.base import MechanicalAdapter, MechanicalState

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - gymnasium is an optional dependency
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

_Base = gym.Wrapper if gym is not None else object


class DictInfoAdapter:
    """Adapter reading per-muscle mechanics from the step ``info`` dict.

    Expects ``info["muscle_length"]``, ``info["muscle_velocity"]`` and
    ``info["muscle_force"]`` (array-likes of shape ``(n_muscles,)``). Intended
    for testing and for envs that already compute muscle kinematics.
    """

    def __init__(self, n_muscles: int) -> None:
        self.n_muscles = n_muscles

    def extract(self, env: object, obs: object, info: dict) -> MechanicalState:
        try:
            L = jnp.asarray(info["muscle_length"], dtype=jnp.float32)
            V = jnp.asarray(info["muscle_velocity"], dtype=jnp.float32)
            F = jnp.asarray(info["muscle_force"], dtype=jnp.float32)
        except KeyError as exc:
            raise ConfigurationError(
                "DictInfoAdapter requires info keys 'muscle_length', "
                f"'muscle_velocity', 'muscle_force'; missing {exc}."
            ) from exc
        for name, arr in (
            ("muscle_length", L),
            ("muscle_velocity", V),
            ("muscle_force", F),
        ):
            if tuple(arr.shape) != (self.n_muscles,):
                raise ConfigurationError(
                    f"info[{name!r}] must have shape ({self.n_muscles},), "
                    f"got {tuple(arr.shape)}."
                )
        return MechanicalState(length=L, velocity=V, force=F)


class JointAdapter:
    """Adapter mapping joint positions/velocities to agonist/antagonist muscles.

    Each ``(pos, vel)`` index pair in ``joints`` produces two muscles whose
    lengths vary oppositely about rest length::

        L_agonist = 1 - r * q        V_agonist = -r * qd
        L_antagonist = 1 + r * q     V_antagonist = +r * qd

    with dimensionless moment arm ``r``. Tendon force is not observable from
    kinematics, so it defaults to a constant ``rest_force`` (Ib then reports
    baseline tone) unless ``force_idx`` names per-muscle effort observations.
    This is a demo-grade default with explicit assumptions — envs with real
    musculature should supply a dedicated adapter.
    """

    def __init__(
        self,
        joints: list[tuple[int, int]],
        moment_arm: float = 0.05,
        rest_force: float = 10.0,
        force_idx: list[int] | None = None,
    ) -> None:
        if not joints:
            raise ConfigurationError(
                "joints must be a non-empty list of (pos_idx, vel_idx) pairs."
            )
        if not moment_arm > 0.0:
            raise ConfigurationError("moment_arm must be positive.")
        if force_idx is not None and len(force_idx) != 2 * len(joints):
            raise ConfigurationError(
                "force_idx must name one entry per muscle "
                f"({2 * len(joints)}), got {len(force_idx)}."
            )
        self.joints = list(joints)
        self.moment_arm = moment_arm
        self.rest_force = rest_force
        self.force_idx = list(force_idx) if force_idx is not None else None
        self.n_muscles = 2 * len(joints)

    def extract(self, env: object, obs: object, info: dict) -> MechanicalState:
        try:
            arr = np.asarray(obs, dtype=np.float32).ravel()
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                "JointAdapter needs a flat array observation, got "
                f"{type(obs).__name__}."
            ) from exc
        lengths, velocities = [], []
        for pos_i, vel_i in self.joints:
            if pos_i >= arr.size or vel_i >= arr.size:
                raise ConfigurationError(
                    f"joint index ({pos_i}, {vel_i}) out of range for "
                    f"observation of size {arr.size}."
                )
            q, qd = float(arr[pos_i]), float(arr[vel_i])
            lengths += [1.0 - self.moment_arm * q, 1.0 + self.moment_arm * q]
            velocities += [-self.moment_arm * qd, self.moment_arm * qd]
        if self.force_idx is None:
            force = np.full(self.n_muscles, self.rest_force, np.float32)
        else:
            force = np.asarray([arr[i] for i in self.force_idx], dtype=np.float32)
            force = np.maximum(force, 0.0)
        return MechanicalState(
            length=jnp.asarray(np.array(lengths, np.float32)),
            velocity=jnp.asarray(np.array(velocities, np.float32)),
            force=jnp.asarray(force),
        )


def wrap(
    env: Any,
    joints: list[tuple[int, int]],
    moment_arm: float = 0.05,
    rest_force: float = 10.0,
    force_idx: list[int] | None = None,
    observation_mode: str = "concat",
    seed: int | None = None,
    spindle_config: SpindleConfig | None = None,
    gto_config: GTOConfig | None = None,
    encoder_config: EncoderConfig | None = None,
    dt: float = 0.001,
    gamma_dynamic: float = 0.0,
    gamma_static: float = 0.0,
) -> Any:
    """One-line afferent augmentation for joint-space Gymnasium envs.

    Example::

        import gymnasium as gym
        from bioafferent.wrappers.gymnasium import wrap
        env = wrap(gym.make("CartPole-v1"), joints=[(2, 3)])

    ``joints`` lists ``(pos_idx, vel_idx)`` observation indices; each joint
    yields an agonist/antagonist muscle pair (see :class:`JointAdapter`).
    Remaining arguments mirror :class:`AfferentWrapper`.
    """
    adapter = JointAdapter(
        joints=joints,
        moment_arm=moment_arm,
        rest_force=rest_force,
        force_idx=force_idx,
    )
    return AfferentWrapper(
        env,
        adapter,
        adapter.n_muscles,
        spindle_config=spindle_config,
        gto_config=gto_config,
        encoder_config=encoder_config,
        dt=dt,
        gamma_dynamic=gamma_dynamic,
        gamma_static=gamma_static,
        observation_mode=observation_mode,
        seed=seed,
    )


class AfferentWrapper(_Base):  # type: ignore[valid-type, misc]
    """Gymnasium wrapper adding Ia/II/Ib ([+ spikes]) to observations.

    Preserves the wrapped env's action space, reset/step semantics,
    termination/truncation, info dict, metadata, and seeding; only the
    observation is augmented.

    Args:
        env: the environment to wrap.
        adapter: maps ``(env, obs, info)`` to :class:`MechanicalState`.
        n_muscles: number of muscles modelled.
        spindle_config / gto_config: model configs (defaults used if None).
        encoder_config: if given, binary spike trains are appended too.
        dt: afferent integration timestep in seconds.
        gamma_dynamic / gamma_static: constant fusimotor rates in Hz.
        observation_mode: ``"concat"`` appends to Box obs; ``"dict"`` returns
            ``{"obs": ..., "ia_rate": ..., "ii_rate": ..., "ib_rate": ...}``.
        seed: seed for the encoder PRNG stream (int or None).
    """

    def __init__(
        self,
        env: Any,
        adapter: MechanicalAdapter,
        n_muscles: int,
        spindle_config: SpindleConfig | None = None,
        gto_config: GTOConfig | None = None,
        encoder_config: EncoderConfig | None = None,
        dt: float = 0.001,
        gamma_dynamic: float = 0.0,
        gamma_static: float = 0.0,
        observation_mode: str = "concat",
        seed: int | None = None,
    ) -> None:
        if gym is None:
            raise UnsupportedBackendError(
                "gymnasium is not installed. Install the 'gymnasium' extra: "
                "pip install bioafferent[gymnasium]."
            )
        if observation_mode not in ("concat", "dict"):
            raise ConfigurationError(
                f"observation_mode must be 'concat' or 'dict', "
                f"got {observation_mode!r}."
            )
        if not dt > 0.0:
            raise ConfigurationError(f"dt must be positive, got {dt}.")
        super().__init__(env)
        self.adapter = adapter
        self.n_muscles = n_muscles
        self.dt = dt
        self.gamma_dynamic = gamma_dynamic
        self.gamma_static = gamma_static
        self.observation_mode = observation_mode
        self.spindle = MuscleSpindle(spindle_config)
        self.gto = GolgiTendonOrgan(gto_config)
        self.encoder = AfferentEncoder(encoder_config) if encoder_config else None
        self._key = jax.random.PRNGKey(seed if seed is not None else 0)
        self._spindle_state: SpindleState = self.spindle.init_state((n_muscles,))
        self._gto_state: GTOState = self.gto.init_state((n_muscles,))
        self._enc_state: EncoderState | None = (
            self.encoder.init_state((3 * n_muscles,)) if self.encoder else None
        )
        self.observation_space = self._build_observation_space(env)

    def _build_observation_space(self, env: Any) -> Any:
        base_space = env.observation_space
        n = self.n_muscles
        if self.observation_mode == "dict" or not isinstance(base_space, spaces.Box):
            if self.observation_mode == "concat":
                raise ConfigurationError(
                    "observation_mode='concat' requires a Box observation space; "
                    f"got {type(base_space).__name__}. Use mode='dict'."
                )
            if isinstance(base_space, spaces.Dict):
                spaces_dict = dict(base_space.spaces)
            else:
                spaces_dict = {"obs": base_space}
            spaces_dict.update(
                {
                    "ia_rate": spaces.Box(0.0, np.inf, (n,), np.float32),
                    "ii_rate": spaces.Box(0.0, np.inf, (n,), np.float32),
                    "ib_rate": spaces.Box(0.0, np.inf, (n,), np.float32),
                }
            )
            if self.encoder:
                spaces_dict["spikes"] = spaces.MultiBinary(3 * n)
            return spaces.Dict(spaces_dict)
        extra = 3 * n + (3 * n if self.encoder else 0)
        low = np.concatenate(
            [np.asarray(base_space.low).ravel(), np.zeros(extra, dtype=np.float32)]
        )
        high = np.concatenate(
            [
                np.asarray(base_space.high).ravel(),
                np.full(extra, np.inf, dtype=np.float32),
            ]
        )
        return spaces.Box(low, high, dtype=np.float32)

    def _augment(self, obs: Any, mech: MechanicalState) -> Any:
        self._spindle_state, sp = self.spindle.step(
            self._spindle_state,
            mech.length,
            mech.velocity,
            self.gamma_dynamic,
            self.gamma_static,
            self.dt,
        )
        self._gto_state, go = self.gto.step(self._gto_state, mech.force, self.dt)
        ia = np.asarray(jax.device_get(sp.ia_rate), dtype=np.float32)
        ii = np.asarray(jax.device_get(sp.ii_rate), dtype=np.float32)
        ib = np.asarray(jax.device_get(go.ib_rate), dtype=np.float32)
        spikes: np.ndarray | None = None
        if self.encoder is not None and self._enc_state is not None:
            rates = jnp.concatenate([sp.ia_rate, sp.ii_rate, go.ib_rate])
            self._key, self._enc_state, enc = self.encoder.step(
                self._key, self._enc_state, rates, self.dt
            )
            spikes = np.asarray(jax.device_get(enc.spikes), dtype=np.int8)
        if isinstance(self.observation_space, spaces.Dict):
            out = dict(obs) if isinstance(obs, dict) else {"obs": obs}
            out.update({"ia_rate": ia, "ii_rate": ii, "ib_rate": ib})
            if spikes is not None:
                out["spikes"] = spikes
            return out
        flat = np.asarray(obs, dtype=np.float32).ravel()
        parts = [flat, ia, ii, ib]
        if spikes is not None:
            parts.append(spikes.astype(np.float32))
        return np.concatenate(parts).astype(np.float32)

    def reset(self, **kwargs: Any) -> Any:
        if kwargs.get("seed") is not None:
            self._key = jax.random.PRNGKey(int(kwargs["seed"]))
        obs, info = super().reset(**kwargs)
        self._spindle_state = self.spindle.init_state((self.n_muscles,))
        self._gto_state = self.gto.init_state((self.n_muscles,))
        if self.encoder is not None:
            self._enc_state = self.encoder.init_state((3 * self.n_muscles,))
        mech = self.adapter.extract(self.env, obs, info)
        return self._augment(obs, mech), info

    def step(self, action: Any) -> Any:
        obs, reward, terminated, truncated, info = super().step(action)
        mech = self.adapter.extract(self.env, obs, info)
        return self._augment(obs, mech), reward, terminated, truncated, info
