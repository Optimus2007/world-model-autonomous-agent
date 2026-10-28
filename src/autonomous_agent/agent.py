"""Closed-loop world-model agent:

observe -> plan with the learned model (predict + uncertainty) -> act in the real env ->
compare prediction with the real outcome -> store -> (maybe) detect a dynamics change ->
(maybe) update the model -> repeat.

Online prediction error ("residual") is always computed with the model as it was *before*
the transition was observed, so it is a genuine out-of-sample error. Training-time error is
reported separately by EnsembleWorldModel.fit / finetune.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .data import ReplayBuffer
from .env import PointMassEnv
from .planner import CEMPlanner, Plan, PlannerConfig
from .world_model import EnsembleWorldModel

DetectMode = Literal["none", "disagreement", "residual", "oracle"]


@dataclass
class DetectorConfig:
    """One-sided CUSUM (Page 1954) on a standardised log signal.

    z_t = (log(x_t + eps) - mu_ref) / sd_ref, S_t = max(0, S_{t-1} + z_t - drift);
    alarm when S_t > threshold. mu_ref, sd_ref come from a calibration set the model
    was not trained on. `drift` sets the smallest persistent shift (in reference sd)
    the detector is tuned for; `threshold` trades delay against false alarms.
    """
    drift: float = 0.5
    threshold: float = 10.0
    eps: float = 1e-10


class CusumDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.mu = self.sd = None
        self.S = 0.0

    def calibrate(self, values: np.ndarray) -> dict:
        logv = np.log(np.asarray(values) + self.cfg.eps)
        self.mu, self.sd = float(logv.mean()), float(logv.std() + 1e-12)
        self.S = 0.0
        return {"mu_log": self.mu, "sd_log": self.sd, "n": int(len(values))}

    def update(self, x: float) -> bool:
        z = (np.log(x + self.cfg.eps) - self.mu) / self.sd
        self.S = max(0.0, self.S + z - self.cfg.drift)
        return self.S > self.cfg.threshold

    def reset(self) -> None:
        self.S = 0.0


@dataclass
class AdaptConfig:
    min_new: int = 100      # fresh post-alarm transitions before the first fine-tune
    every: int = 100        # fine-tune again after this many more fresh transitions
    max_new: int = 500      # stop adapting after this many fresh transitions
    epochs: int = 40
    lr: float = 1e-3
    # Fraction of each fine-tune set drawn from data the agent held *before* the alarm (pretraining
    # transitions + its own pre-alarm transitions). 0 = fine-tune on fresh post-alarm data only.
    replay_ratio: float = 0.0


@dataclass
class AgentConfig:
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    adapt: AdaptConfig = field(default_factory=AdaptConfig)
    detect_mode: DetectMode = "none"
    refit_every_episode: bool = False  # online learning without change detection (Step 2 learning curve)
    refit_epochs: int = 30


class WorldModelAgent:
    def __init__(self, model: EnsembleWorldModel, config: AgentConfig, seed: int = 0):
        self.model, self.cfg = model, config
        self.planner = CEMPlanner(model.step_members, len(model.members), config.planner, seed=seed)
        self.buffer = ReplayBuffer()
        self.residual_det = CusumDetector(config.detector)
        self.disagree_det = CusumDetector(config.detector)
        self.rng = np.random.default_rng(seed)
        self.adapting = False
        self.post_alarm: list[int] = []      # buffer rows collected since the last alarm
        self.next_adapt_at = 0
        self.alarms: list[int] = []          # global step of each alarm
        self.events: list[dict] = []
        self.global_step = 0
        self.prior_data: dict | None = None   # transitions held before going online (e.g. pretraining set)
        self.pre_alarm_end = 0                # buffer rows [0, pre_alarm_end) predate the latest alarm
        # Optional held-out sets {name: arrays}, used only to *measure* error around each fine-tune.
        self.eval_sets: dict | None = None

    # ---------- calibration (no model updates, data never used for fitting) ----------
    def calibrate(self, residuals: np.ndarray, disagreements: np.ndarray) -> dict:
        return {"residual": self.residual_det.calibrate(residuals),
                "disagreement": self.disagree_det.calibrate(disagreements)}

    # ---------- one decision ----------
    def act(self, state: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, Plan, float]:
        t0 = time.perf_counter()
        plan = self.planner.plan(state, goal)
        return plan.action, plan, (time.perf_counter() - t0) * 1e3

    def observe(self, state, action, next_state, reward, done, episode, plan: Plan,
                shift_happened: bool = False) -> dict:
        """Score the prediction made for this action, store it, run detection / adaptation."""
        member_next = plan.predicted_states[:, 1]                   # (E, obs): prediction for executed action
        pred = member_next.mean(0)
        residual = float(np.mean((pred - next_state) ** 2))         # out-of-sample: model not yet trained on it
        disagreement = float(member_next.var(0).mean())
        self.buffer.add(state, action, next_state, reward, done, episode)
        row = len(self.buffer) - 1

        res_S = dis_S = None
        alarm = False
        if self.residual_det.mu is not None:
            r_alarm = self.residual_det.update(residual)
            d_alarm = self.disagree_det.update(disagreement)
            res_S, dis_S = self.residual_det.S, self.disagree_det.S
            mode = self.cfg.detect_mode
            alarm = ((mode == "residual" and r_alarm) or (mode == "disagreement" and d_alarm)
                     or (mode == "oracle" and shift_happened and not self.alarms))
        if alarm and not self.adapting:  # an alarm during an ongoing adaptation is ignored
            self._start_adaptation(pre_alarm_end=row, mode=self.cfg.detect_mode)
        update = None
        if self.adapting:
            self.post_alarm.append(row)
            if len(self.post_alarm) >= self.next_adapt_at:
                update = self._adapt()
        self.global_step += 1
        return {"predicted_next": pred, "residual": residual, "disagreement": disagreement,
                "cusum_residual": res_S, "cusum_disagreement": dis_S, "alarm": bool(alarm), "update": update}

    def _start_adaptation(self, pre_alarm_end: int, mode: str) -> None:
        self.alarms.append(self.global_step)
        self.events.append({"step": self.global_step, "type": "alarm", "mode": mode})
        self.adapting, self.post_alarm = True, []
        self.next_adapt_at = self.cfg.adapt.min_new
        self.pre_alarm_end = pre_alarm_end

    def trigger_adaptation(self, mode: str = "oracle") -> None:
        """Start adaptation before the next step, restarting any adaptation in progress. Used only by
        oracle reference conditions that are told when the dynamics change (not a deployable method)."""
        self._start_adaptation(pre_alarm_end=len(self.buffer), mode=mode)

    def replay_sample(self, n_new: int) -> dict[str, np.ndarray] | None:
        """Old transitions to mix into a fine-tune of n_new fresh ones, so that old ones make up
        replay_ratio of the set. Drawn only from data held before the latest alarm."""
        r = self.cfg.adapt.replay_ratio
        if r <= 0:
            return None
        if not 0 <= r < 1:
            raise ValueError("replay_ratio must be in [0, 1)")
        d = self.buffer.arrays()
        keys = ("state", "action", "next_state")
        pool = {k: d[k][: self.pre_alarm_end] for k in keys}
        if self.prior_data is not None:
            pool = {k: np.concatenate([self.prior_data[k], pool[k]]) for k in keys}
        n_old = min(int(round(n_new * r / (1 - r))), len(pool["state"]))
        idx = self.rng.choice(len(pool["state"]), size=n_old, replace=False)
        return {k: v[idx] for k, v in pool.items()}

    def _eval_errors(self) -> dict | None:
        if not self.eval_sets:
            return None
        return {k: self.model.mse(v["state"], v["action"], v["next_state"]) for k, v in self.eval_sets.items()}

    def _adapt(self) -> dict:
        a = self.cfg.adapt
        d = self.buffer.arrays()
        rows = np.array(self.post_alarm)
        s, act, nxt = d["state"][rows], d["action"][rows], d["next_state"][rows]
        old = self.replay_sample(len(rows))
        n_replay = 0
        if old is not None:
            n_replay = len(old["state"])
            s, act, nxt = (np.concatenate([s, old["state"]]), np.concatenate([act, old["action"]]),
                           np.concatenate([nxt, old["next_state"]]))
        before = self._eval_errors()
        t0 = time.perf_counter()
        info = self.model.finetune(s, act, nxt, epochs=a.epochs, lr=a.lr, rng=self.rng)
        info.update({"step": self.global_step, "type": "finetune", "n_fresh": int(len(rows)),
                     "n_replay": n_replay, "seconds": time.perf_counter() - t0})
        if before is not None:
            info.update({"heldout_before": before, "heldout_after": self._eval_errors()})
        self.events.append(info)
        self.next_adapt_at += a.every
        if len(rows) >= a.max_new:
            self.adapting = False
        self.residual_det.reset()
        self.disagree_det.reset()
        return info

    def end_episode(self) -> dict | None:
        self.planner.reset()
        if self.cfg.refit_every_episode:
            d = self.buffer.arrays()
            t0 = time.perf_counter()
            info = self.model.fit(d["state"], d["action"], d["next_state"], epochs=self.cfg.refit_epochs, rng=self.rng)
            info.update({"step": self.global_step, "type": "refit", "seconds": time.perf_counter() - t0})
            self.events.append(info)
            return info
        return None


def run_episode(env: PointMassEnv, agent: WorldModelAgent, seed: int, episode: int,
                shift_at_step: int | None = None, shift: dict | None = None,
                record: bool = False) -> dict:
    """Run one episode. If shift_at_step (a global agent step) falls inside it, the env dynamics
    change at that step. Returns summary plus optional per-step records."""
    state, info = env.reset(seed=seed)
    goal = info["goal"]
    ret, done, steps, records, plan_ms = 0.0, False, 0, [], []
    trace = {k: [] for k in ("global_step", "residual", "disagreement", "cusum_residual",
                             "cusum_disagreement", "alarm", "shifted")}
    while not done:
        shifted = shift_at_step is not None and agent.global_step >= shift_at_step
        if shifted:
            env.set_dynamics(**shift)  # idempotent; applies to every step from the shift onward
        action, plan, ms = agent.act(state, goal)
        next_state, reward, terminated, truncated, info = env.step(action)
        obs = agent.observe(state, action, next_state, reward, terminated, episode, plan, shift_happened=shifted)
        plan_ms.append(ms)
        for k, v in (("global_step", agent.global_step - 1), ("residual", obs["residual"]),
                     ("disagreement", obs["disagreement"]), ("cusum_residual", obs["cusum_residual"]),
                     ("cusum_disagreement", obs["cusum_disagreement"]), ("alarm", obs["alarm"]),
                     ("shifted", shifted)):
            trace[k].append(v)
        if record:
            records.append({
                "global_step": agent.global_step - 1, "episode": episode, "t": steps,
                "state": state.tolist(), "goal": goal.tolist(), "action": action.tolist(),
                "plan_actions": plan.actions.tolist(),
                "plan_states_members": plan.predicted_states[:, :, :2].round(4).tolist(),
                "candidate_states": plan.candidate_states[:, :, :2].round(4).tolist(),
                "candidate_costs": plan.candidate_costs.round(4).tolist(),
                "predicted_cost": plan.predicted_cost, "plan_disagreement": plan.predicted_disagreement,
                "predicted_next": obs["predicted_next"].tolist(), "next_state": next_state.tolist(),
                "residual": obs["residual"], "disagreement": obs["disagreement"],
                "cusum_residual": obs["cusum_residual"], "cusum_disagreement": obs["cusum_disagreement"],
                "alarm": obs["alarm"], "update": obs["update"], "reward": reward,
                "effective_action": info["effective_action"].tolist(),
                "dynamics_true": {"gain": env.dynamics.gain, "drag": env.dynamics.drag,
                                  "rotation_deg": env.dynamics.rotation_deg}, "shifted": shifted,
                "plan_ms": ms,
            })
        ret += reward
        steps += 1
        state, done = next_state, terminated or truncated
    refit = agent.end_episode()
    return {"episode": episode, "seed": seed, "return": ret, "steps": steps, "success": bool(info["success"]),
            "final_distance": info["distance"], "mean_plan_ms": float(np.mean(plan_ms)),
            "refit": refit, "records": records, "trace": trace}


def run_controller_episode(env: PointMassEnv, controller, seed: int) -> dict:
    state, info = env.reset(seed=seed)
    goal = info["goal"]
    controller.reset()
    ret, done, steps, t_ms = 0.0, False, 0, []
    while not done:
        t0 = time.perf_counter()
        action = controller.act(state, goal)
        t_ms.append((time.perf_counter() - t0) * 1e3)
        state, reward, terminated, truncated, info = env.step(action)
        ret += reward
        steps += 1
        done = terminated or truncated
    return {"seed": seed, "return": ret, "steps": steps, "success": bool(info["success"]),
            "final_distance": info["distance"], "mean_plan_ms": float(np.mean(t_ms))}
