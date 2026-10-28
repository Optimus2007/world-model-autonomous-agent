"""Milestone 1: collect random-policy data, train the ensemble world model, evaluate it.

Usage:  python scripts/train_model.py --config configs/default.yaml --out results/m1
Writes metrics.json, figures and the collected data into --out.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autonomous_agent.config import load_config
from autonomous_agent.data import ReplayBuffer, collect, random_policy
from autonomous_agent.env import PointMassEnv
from autonomous_agent.evaluate import multi_step_report, one_step_report
from autonomous_agent.world_model import EnsembleWorldModel

TEST_SEED_OFFSET = 10_000  # test episodes never share a reset seed with training episodes


def make_data(env_cfg, n_episodes, seed):
    env = PointMassEnv(env_cfg, seed=seed)
    return collect(env, random_policy(np.random.default_rng(seed)), n_episodes, ReplayBuffer(), seed=seed)


def plot_rollouts(model, data, out: Path, dt: float, n_plots: int = 3, H: int = 30):
    ep = data["episode"]
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    for ax, e in zip(axes, np.unique(ep)[:n_plots]):
        rows = np.flatnonzero(ep == e)[:H]
        roll = model.rollout(data["state"][rows[0]], data["action"][rows])[:, 0]  # (E, H+1, obs)
        true = np.vstack([data["state"][rows[0]], data["next_state"][rows]])
        for k, r in enumerate(roll):
            ax.plot(r[:, 0], r[:, 1], color="tab:orange", alpha=0.35, lw=1,
                    label="ensemble members" if k == 0 else None)
        ax.plot(roll.mean(0)[:, 0], roll.mean(0)[:, 1], color="tab:red", ls="--", lw=2, label="ensemble mean")
        ax.plot(true[:, 0], true[:, 1], color="tab:blue", lw=2, label="real trajectory")
        ax.scatter(*true[0, :2], color="k", zorder=5, label="start")
        ax.set_title(f"Held-out episode {e}: {H}-step open-loop rollout")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "rollouts.png", dpi=120)
    plt.close(fig)


def plot_uncertainty(model, sets: dict, out: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for (name, data), c in zip(sets.items(), ["tab:blue", "tab:red"]):
        mean, u = model.predict(data["state"], data["action"])
        err = np.mean((mean - data["next_state"]) ** 2, axis=1)
        ax.scatter(u, err, s=4, alpha=0.3, color=c, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ensemble disagreement u(s,a)  [estimated]")
    ax.set_ylabel("actual one-step squared error  [measured]")
    ax.set_title("Does disagreement track real error?")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "uncertainty_vs_error.png", dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default="results/m1")
    args = p.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed, env_cfg = cfg["seed"], cfg["env"]
    dt = env_cfg.dynamics.dt

    train = make_data(env_cfg, cfg["data"]["train_episodes"], seed)
    test = make_data(env_cfg, cfg["data"]["test_episodes"], seed + TEST_SEED_OFFSET)
    shifted_cfg = replace(env_cfg, dynamics=replace(env_cfg.dynamics, **cfg["shift"]))
    shifted = make_data(shifted_cfg, cfg["data"]["test_episodes"], seed + 2 * TEST_SEED_OFFSET)
    train.save(str(out / "train_data.npz"))
    print(f"collected train={len(train)} test={len(test)} shifted={len(shifted)} transitions")

    model = EnsembleWorldModel(config=cfg["model"], seed=seed)
    t0 = time.perf_counter()
    d = train.arrays()
    fit = model.fit(d["state"], d["action"], d["next_state"])
    fit["fit_seconds"] = time.perf_counter() - t0
    print("fit:", json.dumps(fit, indent=1))

    metrics = {
        "config": {"seed": seed, "env": asdict(env_cfg), "model": asdict(cfg["model"]),
                   "data": cfg["data"], "shift": cfg["shift"]},
        "fit": fit,
        "test_one_step": one_step_report(model, test.arrays(), dt),
        "test_multi_step": multi_step_report(model, test.arrays(), seed=seed),
        "shifted_one_step": one_step_report(model, shifted.arrays(), dt),
        "shifted_multi_step": multi_step_report(model, shifted.arrays(), seed=seed),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in metrics if k != "config"}, indent=1))

    plot_rollouts(model, test.arrays(), out, dt)
    plot_uncertainty(model, {"held-out (same dynamics)": test.arrays(),
                             f"shifted dynamics {cfg['shift']}": shifted.arrays()}, out)
    print(f"wrote {out}/metrics.json, rollouts.png, uncertainty_vs_error.png")


if __name__ == "__main__":
    main()
