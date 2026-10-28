"""Step 2: model-based planning vs. controllers that use no learned model.

Conditions, all evaluated on the same eval episodes (same reset seeds, same noise streams):
  random              uniform random actions
  pd                  PD controller; gains tuned on disjoint tuning episodes
  cem_true_dynamics   CEM with the real simulator as its model (planner upper reference)
  cem_learned         CEM with the learned wall-aware ensemble (pretrained on random data)
  cem_learned_m1      CEM with the M1-style model (no wall rule), same data: Step-1 ablation
  cem_online          learning curve: 1 random episode, then plan / store / refit each episode;
                      the final model is evaluated like the others

Usage: python scripts/run_planning.py --config configs/agent.yaml --out results/s2_planning
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autonomous_agent.agent import WorldModelAgent, run_controller_episode, run_episode
from autonomous_agent.config import load_config
from autonomous_agent.controllers import PDController, RandomController
from autonomous_agent.data import ReplayBuffer, collect, random_policy
from autonomous_agent.env import PointMassEnv, step_dynamics
from autonomous_agent.planner import CEMPlanner
from autonomous_agent.world_model import EnsembleWorldModel


def summarize(eps: list[dict]) -> dict:
    succ = np.array([e["success"] for e in eps], float)
    ret = np.array([e["return"] for e in eps])
    steps = np.array([e["steps"] for e in eps])
    return {"success_rate": float(succ.mean()), "return_mean": float(ret.mean()), "return_std": float(ret.std()),
            "steps_mean": float(steps.mean()), "decision_ms_mean": float(np.mean([e["mean_plan_ms"] for e in eps])),
            "n": len(eps)}


def eval_agent(env_cfg, agent, seeds):
    out = []
    for i, s in enumerate(seeds):
        r = run_episode(PointMassEnv(env_cfg, seed=s), agent, seed=s, episode=10_000 + i)
        r.pop("records")
        out.append(r)
    return out


class TrueDynamicsAgent(WorldModelAgent):
    """CEM over the real simulator (no noise in imagination). Reuses the agent loop."""
    def __init__(self, env_cfg, agent_cfg, seed):
        self.env_cfg = env_cfg
        dyn = replace(env_cfg.dynamics, process_noise=0.0)
        step = lambda s, a: step_dynamics(s, a, dyn, env_cfg.arena)
        fake_model = type("M", (), {"members": [None], "step_members": staticmethod(step)})()
        super().__init__(fake_model, agent_cfg, seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/agent.yaml")
    p.add_argument("--out", default="results/s2_planning")
    args = p.parse_args()
    cfg = load_config(args.config)
    ex, env_cfg = cfg["experiment"], cfg["env"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eval_seeds = [ex["eval_seed_base"] + i for i in range(ex["n_eval_episodes"])]
    tune_seeds = [ex["tune_seed_base"] + i for i in range(ex["n_tune_episodes"])]
    t_start = time.time()
    results: dict = {"config": cfg["raw"], "conditions": {}, "online_curves": {}}

    # ---- PD gain tuning on disjoint episodes (same 9-config budget regardless of outcome) ----
    grid = list(itertools.product([1.0, 2.0, 4.0], [0.5, 1.0, 2.0]))
    tune = {f"{kp},{kd}": summarize([run_controller_episode(PointMassEnv(env_cfg, seed=s), PDController(kp, kd), s)
                                     for s in tune_seeds]) for kp, kd in grid}
    best = max(tune, key=lambda k: (tune[k]["success_rate"], tune[k]["return_mean"]))
    kp, kd = map(float, best.split(","))
    results["pd_tuning"] = {"grid": tune, "chosen": {"kp": kp, "kd": kd}}
    print(f"PD tuned on {len(tune_seeds)} episodes: kp={kp} kd={kd}")

    def record(name, eps):
        results["conditions"].setdefault(name, []).extend(eps)

    record("random", [run_controller_episode(PointMassEnv(env_cfg, seed=s), RandomController(s), s) for s in eval_seeds])
    record("pd", [run_controller_episode(PointMassEnv(env_cfg, seed=s), PDController(kp, kd), s) for s in eval_seeds])

    for aseed in ex["agent_seeds"]:
        # Pretraining data: random policy, seeds disjoint from eval and tuning.
        pre = collect(PointMassEnv(env_cfg, seed=aseed), random_policy(np.random.default_rng(aseed)),
                      cfg["data"]["pretrain_episodes"], ReplayBuffer(), seed=1_000 * (aseed + 1)).arrays()
        n_pre = len(pre["state"])

        record("cem_true_dynamics", eval_agent(env_cfg, TrueDynamicsAgent(env_cfg, cfg["agent"], aseed), eval_seeds))

        m = EnsembleWorldModel(config=cfg["model"], seed=aseed)
        fit = m.fit(pre["state"], pre["action"], pre["next_state"])
        eps = eval_agent(env_cfg, WorldModelAgent(m, cfg["agent"], aseed), eval_seeds)
        for e in eps:
            e["pretrain_interactions"] = n_pre
        record("cem_learned", eps)

        m1 = EnsembleWorldModel(config=replace(cfg["model"], wall_clip_predictions=False, train_exclude_walls=False),
                                seed=aseed)
        m1.fit(pre["state"], pre["action"], pre["next_state"])
        record("cem_learned_m1", eval_agent(env_cfg, WorldModelAgent(m1, cfg["agent"], aseed), eval_seeds))

        # ---- online learning curve: plan / act / store / refit after every episode ----
        init = collect(PointMassEnv(env_cfg, seed=aseed), random_policy(np.random.default_rng(aseed)),
                       ex["online_initial_random"], ReplayBuffer(), seed=2_000 * (aseed + 1)).arrays()
        mo = EnsembleWorldModel(config=cfg["model"], seed=aseed)
        mo.fit(init["state"], init["action"], init["next_state"])
        agent = WorldModelAgent(mo, replace(cfg["agent"], refit_every_episode=True), aseed)
        for i in range(len(init["state"])):
            agent.buffer.add(*(init[k][i] for k in ("state", "action", "next_state", "reward", "done", "episode")))
        curve = []
        for ep in range(ex["online_episodes"]):
            s = 3_000 * (aseed + 1) + ep
            r = run_episode(PointMassEnv(env_cfg, seed=s), agent, seed=s, episode=ep + 1)
            r.pop("records")
            curve.append({"episode": ep, "interactions_so_far": len(agent.buffer), "success": r["success"],
                          "return": r["return"], "steps": r["steps"],
                          "refit_holdout_mse": r["refit"]["holdout_mse"], "refit_seconds": r["refit"]["seconds"]})
        results["online_curves"][str(aseed)] = curve
        agent.cfg = replace(agent.cfg, refit_every_episode=False)
        eps = eval_agent(env_cfg, agent, eval_seeds)
        for e in eps:
            e["online_interactions"] = curve[-1]["interactions_so_far"]
        record("cem_online", eps)
        print(f"agent seed {aseed} done ({time.time() - t_start:.0f}s); pretrain fit {fit}")

    results["summary"] = {k: summarize(v) for k, v in results["conditions"].items()}
    results["wall_seconds"] = time.time() - t_start
    (out / "metrics.json").write_text(json.dumps(results, indent=2, default=float))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for aseed, curve in results["online_curves"].items():
        x = [c["interactions_so_far"] for c in curve]
        ax[0].plot(x, [c["return"] for c in curve], marker="o", label=f"agent seed {aseed}")
        ax[1].plot(x, [c["steps"] for c in curve], marker="o", label=f"agent seed {aseed}")
    pd_ret = results["summary"]["pd"]["return_mean"]
    ax[0].axhline(pd_ret, color="k", ls="--", label="PD (eval mean, different episodes)")
    ax[0].set(xlabel="real environment interactions", ylabel="episode return", title="Online MBRL: return per episode")
    ax[1].set(xlabel="real environment interactions", ylabel="steps to goal (100 = failed)", title="Online MBRL: episode length")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "online_learning_curve.png", dpi=120)
    print(json.dumps(results["summary"], indent=1))
    print(f"total {results['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
