"""Cross-Entropy Method (CEM) model-predictive control.

Each candidate action sequence is rolled out by every ensemble member
(PETS-style "trajectory sampling"); its score is the task cost averaged over
members. An optional uncertainty term adds  risk_weight * disagreement
(pessimism: avoid states the model is unsure about) or subtracts
explore_weight * disagreement (optimism: seek them). Both default to 0,
which gives the plain model-based controller.

Only the first action of the best plan is executed; the agent replans after
every real observation (receding horizon).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .env import task_cost

# (states (E, N, obs), actions (E, N, act)) -> next states (E, N, obs)
StepFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class PlannerConfig:
    horizon: int = 15
    n_candidates: int = 256
    n_elites: int = 25
    iterations: int = 4
    init_std: float = 0.6
    min_std: float = 0.05
    action_cost: float = 0.01
    risk_weight: float = 0.0      # > 0: penalise ensemble disagreement along the plan
    explore_weight: float = 0.0   # > 0: reward ensemble disagreement along the plan
    n_logged_candidates: int = 12  # sampled candidates kept for inspection / dashboard


@dataclass
class Plan:
    action: np.ndarray                 # first action, to execute now
    actions: np.ndarray                # (H, act) best sequence
    predicted_states: np.ndarray       # (E, H+1, obs) best sequence under each member
    predicted_cost: float              # task cost of best sequence (member mean)
    predicted_disagreement: float      # summed disagreement along the best sequence
    candidate_states: np.ndarray       # (K, H+1, obs) member-mean rollouts of K sampled candidates
    candidate_costs: np.ndarray        # (K,)
    n_model_calls: int                 # batched step_fn calls (compute accounting)


class CEMPlanner:
    def __init__(self, step_fn: StepFn, n_members: int, config: PlannerConfig | None = None, seed: int = 0):
        self.step_fn = step_fn
        self.E = n_members
        self.cfg = config or PlannerConfig()
        self.rng = np.random.default_rng(seed)
        self._mean: np.ndarray | None = None

    def reset(self) -> None:
        self._mean = None  # forget the warm start at episode boundaries

    def _rollout(self, state: np.ndarray, seqs: np.ndarray):
        """seqs (N, H, act) -> states (E, N, H+1, obs), cost (N,), disagreement (N,)."""
        N, H, _ = seqs.shape
        cur = np.broadcast_to(state, (self.E, N, state.shape[-1])).copy()
        traj = [cur]
        for h in range(H):
            cur = self.step_fn(cur, np.broadcast_to(seqs[:, h], (self.E, N, seqs.shape[-1])))
            traj.append(cur)
        traj = np.stack(traj, 2)                                    # (E, N, H+1, obs)
        return traj

    def score(self, traj: np.ndarray, seqs: np.ndarray, goal: np.ndarray):
        step_costs = task_cost(traj[:, :, 1:], seqs[None], goal, self.cfg.action_cost)  # (E, N, H)
        cost = step_costs.sum(-1).mean(0)                                                # (N,)
        disagreement = traj[:, :, 1:].var(0).mean(-1).sum(-1) if self.E > 1 else np.zeros(len(seqs))
        total = cost + (self.cfg.risk_weight - self.cfg.explore_weight) * disagreement
        return total, cost, disagreement

    def plan(self, state: np.ndarray, goal: np.ndarray) -> Plan:
        c = self.cfg
        H, act_dim = c.horizon, 2
        mean = np.zeros((H, act_dim)) if self._mean is None else self._mean
        std = np.full((H, act_dim), c.init_std)
        calls = 0
        logged_states = logged_costs = None
        for it in range(c.iterations):
            seqs = np.clip(mean + std * self.rng.standard_normal((c.n_candidates, H, act_dim)), -1, 1)
            seqs[0] = np.clip(mean, -1, 1)  # always evaluate the current mean
            traj = self._rollout(state, seqs)
            calls += H
            total, cost, dis = self.score(traj, seqs, goal)
            elite = np.argsort(total)[: c.n_elites]
            mean = seqs[elite].mean(0)
            std = np.maximum(seqs[elite].std(0), c.min_std)
            if it == 0:  # keep a sample of the initial, diverse candidates for inspection
                k = self.rng.choice(c.n_candidates, size=min(c.n_logged_candidates, c.n_candidates), replace=False)
                logged_states, logged_costs = traj[:, k].mean(0), total[k]
        best_seq = np.clip(mean, -1, 1)
        traj = self._rollout(state, best_seq[None])
        calls += H
        total, cost, dis = self.score(traj, best_seq[None], goal)
        # Warm start: shift the plan by one step for the next decision.
        self._mean = np.vstack([best_seq[1:], best_seq[-1:]])
        return Plan(action=best_seq[0], actions=best_seq, predicted_states=traj[:, 0],
                    predicted_cost=float(cost[0]), predicted_disagreement=float(dis[0]),
                    candidate_states=logged_states, candidate_costs=logged_costs, n_model_calls=calls)
