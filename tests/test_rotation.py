import numpy as np

from autonomous_agent.agent import AgentConfig, WorldModelAgent, run_episode
from autonomous_agent.env import Dynamics, EnvConfig, PointMassEnv, effective_action, step_dynamics
from autonomous_agent.planner import PlannerConfig
from autonomous_agent.world_model import EnsembleWorldModel, ModelConfig


def test_zero_rotation_preserves_dynamics():
    rng = np.random.default_rng(0)
    s = np.column_stack([rng.uniform(-2, 2, (200, 2)), rng.uniform(-3, 3, (200, 2))])
    a = rng.uniform(-1.5, 1.5, (200, 2))
    d0 = Dynamics()
    # Hand-written pre-rotation transition (the original step_dynamics).
    ac = np.clip(a, -1, 1)
    v = s[:, 2:] + d0.dt * (d0.gain * ac - d0.drag * s[:, 2:])
    p = s[:, :2] + d0.dt * v
    hit = np.abs(p) > 2.0
    expected = np.concatenate([np.clip(p, -2, 2), np.where(hit, 0, v)], 1)
    np.testing.assert_array_equal(step_dynamics(s, a, Dynamics(rotation_deg=0.0), 2.0), expected)


def test_rotation_60_known_input():
    eff = effective_action(np.array([1.0, 0.0]), Dynamics(rotation_deg=60))
    np.testing.assert_allclose(eff, [0.5, np.sqrt(3) / 2], atol=1e-12)
    # From rest, one step: v' = dt * gain * R a.
    nxt = step_dynamics(np.zeros(4), np.array([1.0, 0.0]), Dynamics(rotation_deg=60, drag=0.5), 2.0)
    np.testing.assert_allclose(nxt[2:], [0.05, 0.1 * np.sqrt(3) / 2], atol=1e-12)


def test_rotation_90_known_input_and_clip_before_rotate():
    eff = effective_action(np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 0.0]]), Dynamics(rotation_deg=90))
    np.testing.assert_allclose(eff, [[0, 1], [-1, 0], [0, 1]], atol=1e-12)  # issued 3.0 is clipped to 1 first


def test_collisions_unchanged_under_rotation():
    # Pushing +x at the right wall; with 90 deg rotation the push is +y instead, so no collision.
    s = np.array([1.99, 0.0, 5.0, 0.0])
    hit = step_dynamics(s, np.array([1.0, 0.0]), Dynamics(), 2.0)
    assert hit[0] == 2.0 and hit[2] == 0.0
    rot = step_dynamics(s, np.array([1.0, 0.0]), Dynamics(rotation_deg=90), 2.0)
    assert rot[0] == 2.0 and rot[2] == 0.0 and rot[3] > 0  # velocity carried it into the wall; push is now +y


def test_rotation_persists_after_shift_across_episodes():
    rng = np.random.default_rng(0)
    s, a = rng.normal(size=(200, 4)), rng.uniform(-1, 1, (200, 2))
    m = EnsembleWorldModel(config=ModelConfig(n_members=2, hidden=16, epochs=1), seed=0)
    m.fit(s, a, s + 0.01)
    agent = WorldModelAgent(m, AgentConfig(planner=PlannerConfig(horizon=3, n_candidates=16, n_elites=4,
                                                                 iterations=1)), seed=0)
    cfg = EnvConfig(max_steps=5)
    rots = []
    shift_step = None
    for ep in range(4):
        if ep == 1:
            shift_step = agent.global_step
        r = run_episode(PointMassEnv(cfg, seed=ep), agent, seed=ep, episode=ep, shift_at_step=shift_step,
                        shift={"rotation_deg": 60.0}, record=True)
        rots.append({rec["dynamics_true"]["rotation_deg"] for rec in r["records"]})
        if ep >= 1:  # effective action really is the rotated issued action
            rec = r["records"][0]
            np.testing.assert_allclose(rec["effective_action"],
                                       effective_action(np.array(rec["action"]), Dynamics(rotation_deg=60)))
    assert rots == [{0.0}, {60.0}, {60.0}, {60.0}]
