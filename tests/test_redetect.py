import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from autonomous_agent.agent import AdaptConfig, AgentConfig, WorldModelAgent
from autonomous_agent.config import load_config
from autonomous_agent.planner import Plan, PlannerConfig
from autonomous_agent.schedule import segment_index
from autonomous_agent.world_model import EnsembleWorldModel, ModelConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_redetect  # noqa: E402
import run_reversion  # noqa: E402

PLAN = Plan(action=np.zeros(2), actions=np.zeros((2, 2)), predicted_states=np.zeros((2, 3, 4)),
            predicted_cost=0.0, predicted_disagreement=0.0, candidate_states=None, candidate_costs=None,
            n_model_calls=0)


def _agent(policy, max_redetections=2):
    rng = np.random.default_rng(0)
    s, a = rng.normal(size=(200, 4)), rng.uniform(-1, 1, (200, 2))
    m = EnsembleWorldModel(config=ModelConfig(n_members=2, hidden=16, epochs=2), seed=0)
    m.fit(s, a, s)
    adapt = AdaptConfig(min_new=3, every=3, max_new=30, epochs=1, on_alarm_during_adapt=policy,
                        redetect_drift=1.5, redetect_threshold=5.0, max_redetections=max_redetections)
    agent = WorldModelAgent(m, AgentConfig(adapt=adapt, detect_mode="residual"))
    agent.calibrate(np.exp(rng.normal(np.log(1e-4), 1.0, 200)), np.exp(rng.normal(np.log(1e-6), 1.0, 200)))
    return agent


def _step(agent, err):
    """One transition whose residual is err**2 (the plan predicts zeros)."""
    return agent.observe(np.zeros(4), np.zeros(2), np.full(4, err), 0.0, False, 0, PLAN)


def test_segment_schedule():
    assert [segment_index(e, [8, 11]) for e in (0, 7, 8, 10, 11, 29)] == [0, 0, 1, 1, 2, 2]
    assert [segment_index(e, []) for e in (0, 29)] == [0, 0]


def test_states_normal_collecting_adapting_normal():
    agent = _agent("restart")
    assert agent.state == "NORMAL"
    while not agent.alarms:
        _step(agent, 0.5)                       # residual 0.25 >> calibration: main detector fires
    assert agent.state == "COLLECTING" and agent.adapting
    for _ in range(3):
        _step(agent, 0.01)                      # back to the calibrated level: no re-detection
    assert agent.state == "ADAPTING"            # first fine-tune after min_new = 3 rows
    while agent.adapting:
        _step(agent, 0.01)
    assert agent.state == "NORMAL" and agent.alarms == agent.alarms[:1]
    assert len([e for e in agent.events if e.get("type") == "finetune"]) == 10   # rows 3, 6, ..., 30


def test_redetection_disarmed_while_collecting():
    agent = _agent("restart")
    while not agent.alarms:
        _step(agent, 0.5)
    _step(agent, 1.0)                          # 2nd row of min_new 3; high residual before the first
                                               # fine-tune is expected, not a new change
    assert len(agent.alarms) == 1 and agent.state == "COLLECTING" and agent.redetect_det.S == 0.0


def test_restart_opens_new_window_with_old_rows_in_replay_pool():
    agent = _agent("restart")
    while not agent.alarms:
        _step(agent, 0.5)
    for _ in range(3):
        _step(agent, 0.01)
    assert agent.state == "ADAPTING"
    rows_before = len(agent.buffer)
    while len(agent.alarms) == 1:
        _step(agent, 1.0)                      # persistent new rise while adapting
    ev = [e for e in agent.events if e.get("type") == "alarm"]
    assert ev[-1]["mode"] == "redetect" and agent.state == "COLLECTING"
    assert agent.pre_alarm_end == len(agent.buffer) - 1 >= rows_before   # earlier window rows are now "old"
    assert agent.post_alarm == [len(agent.buffer) - 1] and agent.next_adapt_at == 3
    assert agent.n_redetections == 1


def test_restarts_are_capped_then_counter_resets_on_new_chain():
    agent = _agent("restart", max_redetections=1)
    while not agent.alarms:
        _step(agent, 0.5)
    for _ in range(200):                       # sustained high residual: at most one restart in this chain
        _step(agent, 1.0 if agent.state == "ADAPTING" else 0.01)
        if not agent.adapting:
            break
    modes = [e["mode"] for e in agent.events if e.get("type") == "alarm"]
    assert modes == ["residual", "redetect"]
    assert agent.ignored_redetections                # later firings logged, not acted on
    assert not agent.adapting and agent.state == "NORMAL"
    n_ft = len([e for e in agent.events if e.get("type") == "finetune"])
    assert n_ft <= 2 * 10                            # bounded: (1 + max_redetections) windows
    while len(agent.alarms) < 3:
        _step(agent, 0.5)                            # a fresh alarm from NORMAL starts a new chain
    assert agent.n_redetections == 0


def test_ignore_policy_logs_but_does_not_act():
    agent = _agent("ignore")
    while not agent.alarms:
        _step(agent, 0.5)
    for _ in range(3):
        _step(agent, 0.01)
    for _ in range(20):
        _step(agent, 1.0)
    assert len(agent.alarms) == 1 and agent.ignored_redetections and agent.state == "ADAPTING"


def test_redetect_runner_reproduces_reversion_runner_with_ignore_policy():
    cfg = load_config(ROOT / "configs" / "redetect60.yaml")
    cfg["env"] = replace(cfg["env"], max_steps=8)
    cfg["agent"] = replace(cfg["agent"], planner=PlannerConfig(horizon=3, n_candidates=16, n_elites=4, iterations=1))
    rng = np.random.default_rng(1)
    s, a = rng.normal(size=(300, 4)), rng.uniform(-1, 1, (300, 2))
    model = EnsembleWorldModel(config=ModelConfig(n_members=2, hidden=16, epochs=2), seed=0)
    model.fit(s, a, s + 0.05 * rng.normal(size=s.shape))
    pre = {"state": s, "action": a, "next_state": s + 0.01}
    cal = {"residual": {"mu_log": -9.0, "sd_log": 1.0}, "disagreement": {"mu_log": -11.0, "sd_log": 1.0}}
    held = {"old_dynamics": pre, "new_dynamics": pre}
    seeds = list(range(700, 706))
    old_cfg = {**cfg, "shift_experiment": {**cfg["shift_experiment"], "shift_episode": 2, "revert_episode": 4,
                                           "revert": {"rotation_deg": 0.0}}}
    spec = {"name": "x", "mode": "residual", "min_new": 4, "replay_ratio": 0.25}
    old = run_reversion.run_condition(old_cfg, spec, model, pre, cal, held, seeds, 0)
    scen = {"segments": ["original", "changed", "reverted"],
            "changes": [{"episode": 2, "dynamics": {"rotation_deg": 60.0}}, {"episode": 4, "dynamics": {"rotation_deg": 0.0}}]}
    new = run_redetect.run_condition(cfg, scen, {**spec, "policy": "ignore"}, model, pre, cal, held, seeds, 0)
    assert new["residual"] == old["residual"] and new["alarms"] == old["alarms"]
    assert [e["return"] for e in new["episodes"]] == [e["return"] for e in old["episodes"]]
