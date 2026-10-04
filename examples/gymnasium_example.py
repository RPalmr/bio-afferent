"""Gymnasium example: zero-config afferent augmentation on CartPole."""

import gymnasium as gym

from bioafferent import wrap

env = wrap(gym.make("CartPole-v1"), gamma_static=40.0, dt=0.02)
obs, _ = env.reset(seed=0)
print("keys:", sorted(obs))
for _ in range(30):
    obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
    print(f"Ia={obs['ia_rate'].round(1)} II={obs['ii_rate'].round(1)}")
    if terminated or truncated:
        break
