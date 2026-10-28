import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from autonomous_agent.agent import AdaptConfig, AgentConfig, WorldModelAgent
from autonomous_agent.config import load_config
from autonomous_agent.env import Dynamics, effective_action
from autonomous_agent.planner import Plan, PlannerConfig
from autonomous_agent.schedule import episode_phase, phase_dynamics
from autonomous_agent.world_model import EnsembleWorldModel, ModelConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_reversion import run_condition  # noqa: E402


def _model():
    rng = np.random.default_rng(0)
    s, a = rng.normal(size=(300, 4)), rng.uniform(-1, 1, (300, 2))
    m = EnsembleWorldModel(config=ModelConfig(n_members=2, hidden=16, epochs=2), seed=0)
    m.fit(s, a, s + 0.05 * rng.normal(size=s.shape))
    return m, {"state": s, "action": a, "next_state": s + 0.01}


def test_phase_schedule():
    assert [episode_phase(e, 8, 20) for e in (0, 7, 8, 19, 20, 29)] == [0, 0, 1, 1, 2, 2]
    assert [episode_phase(e, 8) for e in (7, 8, 29)] == [0, 1, 1]          # no reversion configured
    shift, revert = {"rotation_deg": 60.0}, {"rotation_deg": 0.0}
    assert [phase_dynamics(p, shift, revert) for p in (0, 1, 2)] == [None, shift, revert]


def _small_cfg(shift_ep=2, revert_ep=4):
    cfg = load_config(ROOT / "configs" / "reversion60.yaml")
    cfg["env"] = replace(cfg["env"], max_steps=5)
    cfg["agent"] = replace(cfg["agent"], planner=PlannerConfig(horizon=3, n_candidates=16, n_elites=4, iterations=1))
    cfg["shift_experiment"] = {**cfg["shift_experiment"], "shift_episode": shift_ep, "revert_episode": revert_ep}
    return cfg


def test_runner_applies_change_and_reversion_and_logs_actions():
    cfg = _small_cfg()
    model, data = _model()
    cal = {"residual": {"mu_log": -7.0, "sd_log": 1.0}, "disagreement": {"mu_log": -11.0, "sd_log": 1.0}}
    heldout = {"old_dynamics": data, "new_dynamics": data}
    spec = {"name": "oracle_m50_r0.25", "mode": "oracle", "min_new": 3, "replay_ratio": 0.25}
    run = run_condition(cfg, spec, model, data, cal, heldout, list(range(500, 506)), 0, record=True)
    by_ep = {}
    for r in run["records"]:
        by_ep.setdefault(r["episode"], set()).add(r["dynamics_true"]["rotation_deg"])
        R = Dynamics(rotation_deg=r["dynamics_true"]["rotation_deg"])
        np.testing.assert_allclose(r["effective_action"], effective_action(np.array(r["action"]), R), atol=1e-12)
        assert r["phase"] == episode_phase(r["episode"], 2, 4)
    # constant within an episode; 0 -> 60 at episode 2; back to 0 at episode 4
    assert by_ep == {0: {0.0}, 1: {0.0}, 2: {60.0}, 3: {60.0}, 4: {0.0}, 5: {0.0}}
    assert [c["episode"] for c in run["changes"]] == [2, 4]
    starts = {r["episode"]: r["global_step"] for r in run["records"] if r["t"] == 0}
    assert [c["step"] for c in run["changes"]] == [starts[2], starts[4]]
    assert run["alarms"] == [starts[2], starts[4]]                   # oracle: exactly at each change
    assert set(run["phase_end_heldout"]) == {"original", "changed", "reverted"}


def _observe(agent, marker):
    s = np.full(4, float(marker))
    plan = Plan(action=np.zeros(2), actions=np.zeros((2, 2)), predicted_states=np.zeros((2, 3, 4)),
                predicted_cost=0.0, predicted_disagreement=0.0, candidate_states=None, candidate_costs=None,
                n_model_calls=0)
    agent.observe(s, np.zeros(2), s + 0.01, 0.0, False, 0, plan)


def test_replay_pool_across_change_and_reversion():
    m, _ = _model()
    agent = WorldModelAgent(m, AgentConfig(adapt=AdaptConfig(min_new=1000, replay_ratio=0.5), detect_mode="none"))
    agent.prior_data = {k: np.full((20, d), 100.0) for k, d in (("state", 4), ("action", 2), ("next_state", 4))}
    for _ in range(5):
        _observe(agent, -1)        # original dynamics, online
    agent.trigger_adaptation()     # change
    assert agent.pre_alarm_end == 5
    for _ in range(7):
        _observe(agent, 60)        # changed dynamics
    old = agent.replay_sample(200)
    assert set(np.unique(old["state"])) == {100.0, -1.0}            # nothing collected after the change
    assert len(agent.post_alarm) == 7
    agent.trigger_adaptation()     # reversion: restarts the adaptation in progress
    assert agent.post_alarm == [] and agent.next_adapt_at == 1000 and agent.pre_alarm_end == 12
    for _ in range(4):
        _observe(agent, 0)         # reverted dynamics
    old = agent.replay_sample(200)
    assert set(np.unique(old["state"])) == {100.0, -1.0, 60.0}       # changed-phase data is now "old"
    assert len(agent.alarms) == 2


def test_detector_oracle_mode_unchanged():
    """The pre-existing single-change oracle (detect_mode='oracle' inside observe) still alarms once."""
    m, _ = _model()
    agent = WorldModelAgent(m, AgentConfig(adapt=AdaptConfig(min_new=1000), detect_mode="oracle"))
    agent.calibrate(np.full(10, 1e-3), np.full(10, 1e-5))
    plan = Plan(action=np.zeros(2), actions=np.zeros((2, 2)), predicted_states=np.zeros((2, 3, 4)),
                predicted_cost=0.0, predicted_disagreement=0.0, candidate_states=None, candidate_costs=None,
                n_model_calls=0)
    for t in range(6):
        agent.observe(np.zeros(4), np.zeros(2), np.zeros(4), 0.0, False, 0, plan, shift_happened=t >= 2)
    assert agent.alarms == [2]
    assert agent.events == [{"step": 2, "type": "alarm", "mode": "oracle"}]
