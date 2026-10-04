"""Gymnasium example: wrap a toy muscle env with afferent observations."""

import numpy as np

from bioafferent.models.encoding import EncoderConfig
from bioafferent.wrappers.gymnasium import AfferentWrapper, DictInfoAdapter

try:
    import gymnasium as gym
except ImportError as exc:
    raise SystemExit(
        "gymnasium not installed; pip install bioafferent[gymnasium]"
    ) from exc


class ToyMuscleEnv(gym.Env):
    def __init__(self, n_muscles=2):
        self.n = n_muscles
        self.observation_space = gym.spaces.Box(-1, 1, (3,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (1,), np.float32)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.zeros(3, np.float32), self._info()

    def step(self, action):
        self._t += 1
        return np.zeros(3, np.float32), 0.0, self._t >= 4, False, self._info()

    def _info(self):
        return {
            "muscle_length": np.full(self.n, 1.0 + 0.0005 * self._t),
            "muscle_velocity": np.full(self.n, 0.02),
            "muscle_force": np.full(self.n, 12.0),
        }


env = AfferentWrapper(
    ToyMuscleEnv(2),
    DictInfoAdapter(2),
    2,
    encoder_config=EncoderConfig(mode="poisson"),
    observation_mode="dict",
    seed=0,
)
obs, _ = env.reset()
print("reset keys:", sorted(obs))
for _ in range(3):
    obs, _, term, _, _ = env.step(np.zeros(1, np.float32))
    print(
        "Ia:",
        obs["ia_rate"],
        "II:",
        obs["ii_rate"],
        "Ib:",
        obs["ib_rate"],
        "spikes:",
        obs["spikes"].sum(),
    )
print("done, terminated:", term)
