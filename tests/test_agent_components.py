import numpy as np

from autonomous_agent.agent import CusumDetector, DetectorConfig
from autonomous_agent.env import Dynamics, step_dynamics
from autonomous_agent.planner import CEMPlanner, PlannerConfig
from autonomous_agent.world_model import EnsembleWorldModel, ModelConfig, apply_wall_rule, is_wall_transition


def test_wall_rule_matches_simulator():
    rng = np.random.default_rng(0)
    dyn = Dynamics(gain=1.0, drag=0.5, dt=0.1)
    s = np.column_stack([rng.uniform(-2, 2, (500, 2)), rng.uniform(-6, 6, (500, 2))])
    a = rng.uniform(-1, 1, (500, 2))
    free = s.copy()  # unclipped semi-implicit Euler step
    free[:, 2:] = s[:, 2:] + dyn.dt * (dyn.gain * a - dyn.drag * s[:, 2:])
    free[:, :2] = s[:, :2] + dyn.dt * free[:, 2:]
    real = np.array([step_dynamics(si, ai, dyn, arena=2.0) for si, ai in zip(s, a)])
    np.testing.assert_allclose(apply_wall_rule(free, 2.0), real, atol=1e-12)
    assert is_wall_transition(real, 2.0).any() and not is_wall_transition(real, 2.0).all()


def test_cusum_quiet_on_reference_and_alarms_on_shift():
    # False alarms are rare but not impossible on in-distribution data, so test a rate, not one draw.
    det = CusumDetector(DetectorConfig(drift=0.5, threshold=10.0))
    alarms = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        det.calibrate(np.exp(rng.normal(-9, 1, 1000)))
        for x in np.exp(rng.normal(-9, 1, 1000)):
            if det.update(x):
                alarms += 1
                det.reset()
    assert alarms <= 2  # 20,000 reference steps
    det.reset()
    shifted = np.exp(rng.normal(-7, 1, 100))  # +2 reference sd in log space
    first = next(i for i, x in enumerate(shifted) if det.update(x))
    assert first < 20


def test_model_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    s, a = rng.normal(size=(300, 4)), rng.normal(size=(300, 2))
    m = EnsembleWorldModel(config=ModelConfig(n_members=2, hidden=16, epochs=2), seed=0)
    m.fit(s, a, s + 0.1 * rng.normal(size=s.shape))
    m.save(str(tmp_path / "m.pt"))
    m2 = EnsembleWorldModel.load(str(tmp_path / "m.pt"))
    np.testing.assert_allclose(m.predict_members(s[:5], a[:5]), m2.predict_members(s[:5], a[:5]))


def test_cem_with_true_dynamics_moves_toward_goal():
    dyn = Dynamics()
    step = lambda st, ac: step_dynamics(st, ac, dyn, 2.0)
    planner = CEMPlanner(step, 1, PlannerConfig(horizon=10, n_candidates=64, n_elites=8, iterations=3), seed=0)
    state, goal = np.array([-1.0, -1.0, 0.0, 0.0]), np.array([1.0, 1.0])
    for _ in range(30):
        state = step(state, planner.plan(state, goal).action)
    assert np.linalg.norm(state[:2] - goal) < 0.5
