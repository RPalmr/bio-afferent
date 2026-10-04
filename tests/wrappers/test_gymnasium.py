"""Wrapper tests: Gymnasium AfferentWrapper (skipped if gymnasium missing)."""

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium", reason="gymnasium optional dependency")

from bioafferent.models.encoding import EncoderConfig  # noqa: E402
from bioafferent.wrappers.gymnasium import (  # noqa: E402
    AfferentWrapper,
    DictInfoAdapter,
    wrap,
)


class _MuscleEnv(gymnasium.Env):
    """Minimal muscle env: obs in R^2, info carries muscle mechanics."""

    def __init__(self, n_muscles=2):
        self.n = n_muscles
        self.observation_space = gymnasium.spaces.Box(-1, 1, (2,), np.float32)
        self.action_space = gymnasium.spaces.Box(-1, 1, (1,), np.float32)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.zeros(2, np.float32), self._info()

    def step(self, action):
        self._t += 1
        obs = np.ones(2, np.float32) * 0.1 * self._t
        term = self._t >= 5
        return obs, 1.0, term, False, self._info()

    def _info(self):
        return {
            "muscle_length": np.full(self.n, 1.0 + 0.001 * self._t),
            "muscle_velocity": np.full(self.n, 0.05),
            "muscle_force": np.full(self.n, 10.0),
        }


def _wrap(**kw):
    return AfferentWrapper(_MuscleEnv(2), DictInfoAdapter(2), 2, **kw)


def test_concat_observation_shape():
    env = _wrap(observation_mode="concat")
    obs, _ = env.reset()
    assert obs.shape == (2 + 6,)
    obs, _, _, _, _ = env.step(np.zeros(1, np.float32))
    assert obs.shape == (2 + 6,)
    assert np.isfinite(obs).all()


def test_dict_observation_keys():
    env = _wrap(observation_mode="dict")
    obs, _ = env.reset()
    assert set(obs) >= {"obs", "ia_rate", "ii_rate", "ib_rate"}
    assert obs["ia_rate"].shape == (2,)


def test_action_space_preserved():
    env = _wrap()
    assert env.action_space.shape == (1,)


def test_terminated_truncated_and_info():
    env = _wrap()
    env.reset()
    seen_term = False
    for _ in range(6):
        _, _, term, trunc, info = env.step(np.zeros(1, np.float32))
        seen_term |= term
        assert isinstance(info, dict) and "muscle_length" in info
    assert seen_term and not trunc


def test_deterministic_reset():
    a, _ = _wrap(seed=7).reset()
    b, _ = _wrap(seed=7).reset()
    assert set(a) == set(b)
    for key in a:
        np.testing.assert_array_equal(np.asarray(a[key]), np.asarray(b[key]))


def test_reset_seed_kwarg_reseeds_encoder():
    env = _wrap(encoder_config=EncoderConfig(mode="poisson"))
    o1, _ = env.reset(seed=123)
    o2, _ = env.reset(seed=123)
    for key in o1:
        np.testing.assert_array_equal(np.asarray(o1[key]), np.asarray(o2[key]))


def test_spike_observation_appended():
    env = _wrap(encoder_config=EncoderConfig(mode="poisson"), observation_mode="concat")
    obs, _ = env.reset()
    assert obs.shape == (2 + 6 + 6,)
    env2 = AfferentWrapper(
        _MuscleEnv(2),
        DictInfoAdapter(2),
        2,
        encoder_config=EncoderConfig(mode="poisson"),
        observation_mode="dict",
    )
    obs2, _ = env2.reset()
    assert obs2["spikes"].shape == (6,)


def test_gymnasium_passthrough_check():
    from gymnasium.utils.env_checker import check_env

    check_env(_MuscleEnv(2), skip_render_check=True)


def test_wrap_zero_config_cartpole():
    env = wrap(gymnasium.make("CartPole-v1"), seed=0)
    obs, _ = env.reset(seed=0)
    # 2 auto joints -> 4 muscles; dict mode is the default
    assert set(obs) >= {"obs", "ia_rate", "ii_rate", "ib_rate"}
    assert obs["ia_rate"].shape == (4,)
    assert np.isfinite(obs["ia_rate"]).all()
    obs, _, term, trunc, _ = env.step(0)
    assert obs["ii_rate"].shape == (4,)
    assert not term and not trunc


def test_wrap_explicit_joints():
    env = wrap(gymnasium.make("CartPole-v1"), joints=[(2, 3)])
    obs, _ = env.reset()
    assert obs["ia_rate"].shape == (2,)
