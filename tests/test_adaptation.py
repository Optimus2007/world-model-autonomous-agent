from dataclasses import replace

import numpy as np

from autonomous_agent.agent import AdaptConfig, AgentConfig, WorldModelAgent
from autonomous_agent.planner import Plan, PlannerConfig
from autonomous_agent.world_model import EnsembleWorldModel, ModelConfig


def _model(seed=0):
    rng = np.random.default_rng(seed)
    s, a = rng.normal(size=(300, 4)), rng.uniform(-1, 1, (300, 2))
    m = EnsembleWorldModel(config=ModelConfig(n_members=2, hidden=16, epochs=2), seed=seed)
    m.fit(s, a, s + 0.05 * rng.normal(size=s.shape))
    return m


def _agent(adapt: AdaptConfig, mode="oracle"):
    agent = WorldModelAgent(_model(), AgentConfig(planner=PlannerConfig(horizon=2), adapt=adapt, detect_mode=mode))
    agent.calibrate(np.full(50, 1e-3) * np.exp(np.random.default_rng(0).normal(size=50)),
                    np.full(50, 1e-5) * np.exp(np.random.default_rng(1).normal(size=50)))
    return agent


def _observe(agent, marker, shifted):
    s = np.full(4, float(marker))
    plan = Plan(action=np.zeros(2), actions=np.zeros((2, 2)), predicted_states=np.zeros((2, 3, 4)),
                predicted_cost=0.0, predicted_disagreement=0.0, candidate_states=None, candidate_costs=None,
                n_model_calls=0)
    return agent.observe(s, np.zeros(2), s + 0.01, 0.0, False, 0, plan, shift_happened=shifted)


def test_first_finetune_after_exactly_min_new_then_every_until_max():
    agent = _agent(AdaptConfig(min_new=10, every=20, max_new=50, epochs=1))
    for _ in range(5):
        _observe(agent, 0, shifted=False)
    for _ in range(80):
        _observe(agent, 0, shifted=True)
    assert agent.alarms == [5]
    fts = [e for e in agent.events if e["type"] == "finetune"]
    # post-alarm rows counted from the alarm step (inclusive): 10 -> step 14, 30 -> 34, 50 -> 54, then stop
    assert [e["step"] for e in fts] == [14, 34, 54]
    assert [e["n_fresh"] for e in fts] == [10, 30, 50]
    assert not agent.adapting


def test_replay_uses_only_data_held_before_the_alarm():
    agent = _agent(AdaptConfig(min_new=30, every=1000, max_new=30, epochs=1, replay_ratio=0.5))
    agent.prior_data = {k: np.full((40, d), 100.0) for k, d in (("state", 4), ("action", 2), ("next_state", 4))}
    for _ in range(6):
        _observe(agent, -100, shifted=False)          # agent's own pre-alarm transitions
    for _ in range(30):
        _observe(agent, 7, shifted=True)              # post-alarm transitions (never "old")
    ft = [e for e in agent.events if e["type"] == "finetune"]
    assert len(ft) == 1 and ft[0]["n_fresh"] == 30 and ft[0]["n_replay"] == 30
    old = agent.replay_sample(30)
    assert set(np.unique(old["state"])) <= {100.0, -100.0}
    assert agent.pre_alarm_end == 6


def test_replay_ratio_sizes_and_no_rng_use_at_zero():
    agent = _agent(AdaptConfig(replay_ratio=0.25))
    agent.prior_data = {k: np.zeros((500, d)) for k, d in (("state", 4), ("action", 2), ("next_state", 4))}
    assert len(agent.replay_sample(30)["state"]) == 10          # 10 / (30 + 10) = 0.25
    agent.cfg = replace(agent.cfg, adapt=AdaptConfig(replay_ratio=0.0))
    state = agent.rng.bit_generator.state
    assert agent.replay_sample(30) is None
    assert agent.rng.bit_generator.state == state              # ratio 0 reproduces earlier runs exactly


def test_checkpoint_consistent_after_finetune(tmp_path):
    m = _model()
    norm = (m.in_norm.mean.copy(), m.out_norm.std.copy())
    rng = np.random.default_rng(3)
    s, a = rng.normal(size=(50, 4)), rng.uniform(-1, 1, (50, 2))
    m.finetune(s, a, s + 0.2, epochs=2)
    np.testing.assert_array_equal(m.in_norm.mean, norm[0])     # normalisers fixed during fine-tuning
    np.testing.assert_array_equal(m.out_norm.std, norm[1])
    m.save(str(tmp_path / "ft.pt"))
    m2 = EnsembleWorldModel.load(str(tmp_path / "ft.pt"))
    np.testing.assert_allclose(m.predict_members(s, a), m2.predict_members(s, a), rtol=0, atol=0)
