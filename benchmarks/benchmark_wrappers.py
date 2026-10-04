"""Benchmark: Gymnasium wrapper overhead per env step (requires gymnasium)."""

import numpy as np

from bioafferent.wrappers.gymnasium import AfferentWrapper, DictInfoAdapter

try:
    import gymnasium as gym

    class _Env(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(-1, 1, (4,), np.float32)
            self.action_space = gym.spaces.Box(-1, 1, (2,), np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(4, np.float32), self._info()

        def step(self, a):
            return np.zeros(4, np.float32), 0.0, False, False, self._info()

        def _info(self):
            return {
                "muscle_length": np.full(3, 1.02),
                "muscle_velocity": np.full(3, 0.02),
                "muscle_force": np.full(3, 15.0),
            }

    env = AfferentWrapper(_Env(), DictInfoAdapter(3), 3)
    env.reset()
    action = np.zeros(2, np.float32)
    env.step(action)  # warmup through wrapper
    import time

    def thunk():
        return env.step(action)

    t0, n = time.perf_counter(), 200
    for _ in range(n):
        thunk()
    print(f"[wrapper] {(time.perf_counter() - t0) / n * 1e3:.3f} ms/step (n={n})")
except ImportError:
    print("[wrapper] gymnasium not installed; skipping")
