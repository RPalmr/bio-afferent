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

from bioafferent.afferents import AfferentRates, Afferents
from bioafferent.core.exceptions import ConfigurationError, UnsupportedBackendError
from bioafferent.models.encoding import EncoderConfig
from bioafferent.models.gto_parameters import GTOConfig
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
                f"DictInfoAdapter requires info keys 'muscle_length', 'muscle_velocity', 'muscle_force'; missing {exc}."
            ) from exc
        for name, arr in (
            ("muscle_length", L),
            ("muscle_velocity", V),
            ("muscle_force", F),
        ):
            if tuple(arr.shape) != (self.n_muscles,):
                raise ConfigurationError(f"info[{name!r}] must have shape ({self.n_muscles},), got {tuple(arr.shape)}.")
        return MechanicalState(length=L, velocity=V, force=F)


class _AutoJoints:
    """Derive agonist/antagonist muscles from a flat joint observation.

    Each ``(pos, vel)`` pair yields two muscles with
    ``L = 1 ∓ r·q``. Tendon force defaults to a constant ``rest_force``
    (kinematics alone cannot observe it). Only used through :func:`wrap`.
    """

    def __init__(
        self,
        joints: list[tuple[int, int]],
        moment_arm: float = 0.05,
        rest_force: float = 10.0,
    ) -> None:
        self.joints = list(joints)
        self.moment_arm = moment_arm
        self.rest_force = rest_force
        self.n_muscles = 2 * len(joints)

    def extract(self, env: object, obs: object, info: dict) -> MechanicalState:
        arr = np.asarray(obs, dtype=np.float32).ravel()
        lengths, velocities = [], []
        for pos_i, vel_i in self.joints:
            q, qd = float(arr[pos_i]), float(arr[vel_i])
            lengths += [1.0 - self.moment_arm * q, 1.0 + self.moment_arm * q]
            velocities += [-self.moment_arm * qd, self.moment_arm * qd]
        return MechanicalState(
            length=jnp.asarray(np.array(lengths, np.float32)),
            velocity=jnp.asarray(np.array(velocities, np.float32)),
            force=jnp.full((self.n_muscles,), self.rest_force, np.float32),
        )


def _infer_joints(observation_space: Any) -> list[tuple[int, int]]:
    size = int(np.asarray(observation_space.shape).prod())
    if size < 2:
        raise ConfigurationError(
            "Automatic joint inference needs a flat Box observation with "
            f"size >= 2, got shape {observation_space.shape}. Pass "
            "explicit joints=[(pos_idx, vel_idx), ...] or use DictInfoAdapter."
        )
    return [(i, i + 1) for i in range(0, 2 * (size // 2), 2)]


def wrap(
    env: Any,
    joints: list[tuple[int, int]] | None = None,
    dt: float = 0.001,
    gamma_dynamic: float = 0.0,
    gamma_static: float = 0.0,
    observation_mode: str = "dict",
    seed: int | None = None,
    moment_arm: float = 0.05,
    rest_force: float = 10.0,
    spindle_config: SpindleConfig | None = None,
    gto_config: GTOConfig | None = None,
    encoder_config: EncoderConfig | None = None,
) -> Any:
    """Augment a joint-space Gymnasium env with Ia/II/Ib observations.

    Minimal usage does everything behind the scenes::

        import gymnasium as gym
        from bioafferent.wrappers.gymnasium import wrap
        env = wrap(gym.make("CartPole-v1"))

    With ``joints=None`` (default), consecutive observation entries are paired
    as ``(position, velocity)`` joints; pass explicit ``[(pos_idx, vel_idx)]``
    pairs to override. Each joint yields one agonist/antagonist muscle pair.
    """
    if gym is None:
        raise UnsupportedBackendError(
            "gymnasium is not installed. Install the 'gymnasium' extra: pip install bioafferent[gymnasium]."
        )
    pairs = list(joints) if joints is not None else _infer_joints(env.observation_space)
    adapter = _AutoJoints(pairs, moment_arm, rest_force)
    return AfferentWrapper(
        env,
        adapter,
        adapter.n_muscles,
        spindle_config,
        gto_config,
        encoder_config,
        dt,
        gamma_dynamic,
        gamma_static,
        observation_mode,
        seed,
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
        observation_mode: str = "dict",
        seed: int | None = None,
    ) -> None:
        if gym is None:
            raise UnsupportedBackendError(
                "gymnasium is not installed. Install the 'gymnasium' extra: pip install bioafferent[gymnasium]."
            )
        if observation_mode not in ("concat", "dict"):
            raise ConfigurationError(f"observation_mode must be 'concat' or 'dict', got {observation_mode!r}.")
        if not dt > 0.0:
            raise ConfigurationError(f"dt must be positive, got {dt}.")
        super().__init__(env)
        self.adapter = adapter
        self.n_muscles = n_muscles
        self.dt = dt
        self.gamma_dynamic = gamma_dynamic
        self.gamma_static = gamma_static
        self.observation_mode = observation_mode
        self.afferents = Afferents(
            n_muscles,
            spindle_config=spindle_config,
            gto_config=gto_config,
            encoder_config=encoder_config,
            seed=seed if seed is not None else 0,
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
            if self.afferents.encoder:
                spaces_dict["spikes"] = spaces.MultiBinary(3 * n)
            return spaces.Dict(spaces_dict)
        extra = 3 * n + (3 * n if self.afferents.encoder else 0)
        low = np.concatenate([np.asarray(base_space.low).ravel(), np.zeros(extra, dtype=np.float32)])
        high = np.concatenate(
            [
                np.asarray(base_space.high).ravel(),
                np.full(extra, np.inf, dtype=np.float32),
            ]
        )
        return spaces.Box(low, high, dtype=np.float32)

    def _augment(self, obs: Any, mech: MechanicalState) -> Any:
        out: AfferentRates = self.afferents.step(
            mech.length,
            mech.velocity,
            mech.force,
            gamma_dynamic=self.gamma_dynamic,
            gamma_static=self.gamma_static,
            dt=self.dt,
        )
        ia = np.asarray(jax.device_get(out.ia_rate), dtype=np.float32)
        ii = np.asarray(jax.device_get(out.ii_rate), dtype=np.float32)
        ib = np.asarray(jax.device_get(out.ib_rate), dtype=np.float32)
        spikes: np.ndarray | None = (
            np.asarray(jax.device_get(out.spikes), dtype=np.int8) if out.spikes is not None else None
        )
        if isinstance(self.observation_space, spaces.Dict):
            obs_out = dict(obs) if isinstance(obs, dict) else {"obs": obs}
            obs_out.update({"ia_rate": ia, "ii_rate": ii, "ib_rate": ib})
            if spikes is not None:
                obs_out["spikes"] = spikes
            return obs_out
        flat = np.asarray(obs, dtype=np.float32).ravel()
        parts = [flat, ia, ii, ib]
        if spikes is not None:
            parts.append(spikes.astype(np.float32))
        return np.concatenate(parts).astype(np.float32)

    def reset(self, **kwargs: Any) -> Any:
        if kwargs.get("seed") is not None:
            self.afferents.reseed(int(kwargs["seed"]))
        obs, info = super().reset(**kwargs)
        self.afferents.reset()
        mech = self.adapter.extract(self.env, obs, info)
        return self._augment(obs, mech), info

    def step(self, action: Any) -> Any:
        obs, reward, terminated, truncated, info = super().step(action)
        mech = self.adapter.extract(self.env, obs, info)
        return self._augment(obs, mech), reward, terminated, truncated, info
