"""Step 1: does applying the simulator's exact wall rule improve the world model?

Variants (same training data, same held-out test data, same model seeds):
  original         M1 model: trained on all transitions, raw predictions
  posthoc_clip     same trained weights as `original`, wall rule applied to predictions
  free_motion_rule trained on free-motion transitions only, wall rule applied to predictions

Usage: python scripts/compare_wall_handling.py --out results/s1_walls
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from autonomous_agent.config import load_config
from autonomous_agent.data import ReplayBuffer, collect, random_policy
from autonomous_agent.env import PointMassEnv
from autonomous_agent.evaluate import multi_step_report
from autonomous_agent.world_model import EnsembleWorldModel, is_wall_transition


def make_data(env_cfg, n_episodes, seed):
    env = PointMassEnv(env_cfg, seed=seed)
    return collect(env, random_policy(np.random.default_rng(seed)), n_episodes, ReplayBuffer(), seed=seed).arrays()


def split_mse(model, d, arena):
    mean, _ = model.predict(d["state"], d["action"])
    err = np.mean((mean - d["next_state"]) ** 2, axis=1)
    wall = is_wall_transition(d["next_state"], arena)
    return {"overall": float(err.mean()), "wall_contact": float(err[wall].mean()),
            "free_motion": float(err[~wall].mean()), "n_wall": int(wall.sum()), "n_free": int((~wall).sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default="results/s1_walls")
    p.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env_cfg, seed = cfg["env"], cfg["seed"]
    arena = env_cfg.arena
    train = make_data(env_cfg, cfg["data"]["train_episodes"], seed)
    test = make_data(env_cfg, cfg["data"]["test_episodes"], seed + 10_000)  # same seeds as M1

    results = {}
    for ms in args.model_seeds:
        base = replace(cfg["model"], arena=arena)
        orig = EnsembleWorldModel(config=base, seed=ms)
        orig.fit(train["state"], train["action"], train["next_state"])
        clip = copy.deepcopy(orig)
        clip.config = replace(base, wall_clip_predictions=True)
        free = EnsembleWorldModel(config=replace(base, wall_clip_predictions=True, train_exclude_walls=True), seed=ms)
        free.fit(train["state"], train["action"], train["next_state"])
        for name, m in [("original", orig), ("posthoc_clip", clip), ("free_motion_rule", free)]:
            r = split_mse(m, test, arena)
            r["multi_step"] = {k: v["mse_mean"] for k, v in multi_step_report(m, test, seed=seed).items()}
            results.setdefault(name, []).append(r)
            print(f"model_seed={ms} {name:17s} overall={r['overall']:.2e} wall={r['wall_contact']:.2e} "
                  f"free={r['free_motion']:.2e} h20={r['multi_step']['h20']:.2e}")

    summary = {}
    for name, rs in results.items():
        summary[name] = {k: {"mean": float(np.mean([r[k] for r in rs])), "std": float(np.std([r[k] for r in rs]))}
                         for k in ("overall", "wall_contact", "free_motion")}
        summary[name].update({f"rollout_{h}": {"mean": float(np.mean([r["multi_step"][h] for r in rs])),
                                               "std": float(np.std([r["multi_step"][h] for r in rs]))}
                              for h in rs[0]["multi_step"]})
    (out / "metrics.json").write_text(json.dumps({"per_seed": results, "summary": summary,
                                                  "n_wall": results["original"][0]["n_wall"],
                                                  "n_free": results["original"][0]["n_free"]}, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
