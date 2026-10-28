import numpy as np
import pytest

from autonomous_agent.data import ReplayBuffer, collect, random_policy
from autonomous_agent.env import Dynamics, EnvConfig, PointMassEnv, step_dynamics
from autonomous_agent.world_model import EnsembleWorldModel, ModelConfig


def test_dynamics_matches_hand_calculation():
    s = np.array([0.0, 0.0, 1.0, 0.0])
    nxt = step_dynamics(s, np.array([1.0, 0.0]), Dynamics(gain=1.0, drag=0.5, dt=0.1), arena=2.0)
    # v' = 1 + 0.1*(1 - 0.5*1) = 1.05 ; x' = 0.105
    np.testing.assert_allclose(nxt, [0.105, 0.0, 1.05, 0.0])


def test_actions_are_clipped_and_walls_stop_motion():
    s = np.array([1.99, 0.0, 5.0, 0.0])
    nxt = step_dynamics(s, np.array([10.0, 0.0]), Dynamics(), arena=2.0)
    assert nxt[0] == 2.0 and nxt[2] == 0.0


def test_reset_is_seeded_and_set_dynamics_changes_transition():
    env = PointMassEnv(EnvConfig(), seed=1)
    s1, i1 = env.reset(seed=3)
    s2, i2 = env.reset(seed=3)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(i1["goal"], i2["goal"])
    a = np.array([1.0, 1.0])
    before = env.step(a)[0]
    env.reset(seed=3)
    env.set_dynamics(drag=2.0, gain=0.2)
    after = env.step(a)[0]
    assert not np.allclose(before, after)


def test_episode_terminates_and_buffer_tracks_interactions():
    env = PointMassEnv(EnvConfig(max_steps=20), seed=0)
    buf = collect(env, random_policy(np.random.default_rng(0)), 3, ReplayBuffer())
    assert len(buf) <= 60 and len(buf) >= 3
    assert set(np.unique(buf.arrays()["episode"])) == {0, 1, 2}


def test_model_beats_trivial_baseline_on_tiny_problem():
    env = PointMassEnv(EnvConfig(), seed=0)
    d = collect(env, random_policy(np.random.default_rng(0)), 20, ReplayBuffer()).arrays()
    m = EnsembleWorldModel(config=ModelConfig(n_members=2, epochs=30, hidden=64), seed=0)
    m.fit(d["state"], d["action"], d["next_state"])
    assert m.mse(d["state"], d["action"], d["next_state"]) < np.mean((d["state"] - d["next_state"]) ** 2)
    roll = m.rollout(d["state"][0], d["action"][:5])
    assert roll.shape == (2, 1, 6, 4)
    np.testing.assert_allclose(roll[:, 0, 0], np.repeat(d["state"][:1], 2, 0))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        EnsembleWorldModel().predict(np.zeros((1, 4)), np.zeros((1, 2)))
