"""Gymnasium example: one-line afferent augmentation on CartPole.

The pole joint (obs index 2 = angle, 3 = angular velocity) drives one
agonist/antagonist muscle pair; the wrapper appends Ia/II/Ib firing rates
to the observation.
"""

import gymnasium as gym

from bioafferent.wrappers.gymnasium import wrap

env = wrap(
    gym.make("CartPole-v1"),
    joints=[(2, 3)],
    gamma_static=40.0,
    dt=0.02,  # match CartPole's 20 ms control timestep
)
obs, _ = env.reset(seed=0)
print("obs shape:", obs.shape, "(4 cartpole + 6 afferent)")
for _ in range(30):
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    ia, ii, ib = obs[4:6], obs[6:8], obs[8:10]
    print(f"Ia={ia.round(1)} II={ii.round(1)} Ib={ib.round(1)} reward={reward}")
    if terminated or truncated:
        break
