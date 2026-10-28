"""Step 3: detecting and adapting to a mid-run dynamics change.

Per seed:
  1. Pretrain the ensemble on random-policy data (pre-shift dynamics). Checkpoint saved.
  2. Calibration: run planner episodes with the frozen model (seeds disjoint from everything else).
     Their residuals / disagreements set each detector's reference distribution. These transitions
     are never used to fit any model, so the reference is out-of-sample.
  3. Online runs, all starting from a copy of the same pretrained model, same episode seeds:
       none          no detection, no adaptation
       disagreement  CUSUM on ensemble disagreement -> fine-tune on fresh transitions
       residual      CUSUM on observed prediction residual -> fine-tune on fresh transitions
       oracle        adaptation triggered exactly at the true shift (reference only, not deployable)
       no_shift      `none` agent, dynamics never change: false-alarm control
     plus the PD controller on the same episodes (no model, no adaptation).
  The dynamics change at the first step of episode `shift_episode` and stay changed.

Detection quality is measured offline on the `none` and `no_shift` traces (no adaptation, so the
signal is not altered by the detector's own actions): CUSUM with reset after each alarm.

Usage: python scripts/run_shift.py --config configs/agent.yaml --out results/s3_shift
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autonomous_agent.agent import CusumDetector, WorldModelAgent, run_controller_episode, run_episode
from autonomous_agent.config import load_config
from autonomous_agent.controllers import PDController
from autonomous_agent.data import ReplayBuffer, collect, random_policy
from autonomous_agent.env import PointMassEnv
from autonomous_agent.evaluate import multi_step_report
from autonomous_agent.world_model import EnsembleWorldModel

CONDITIONS = ["none", "disagreement", "residual", "oracle", "no_shift"]


def offline_alarms(values, cal, det_cfg):
    """Alarm steps (indices) of a CUSUM run over `values`, resetting after each alarm."""
    d = CusumDetector(det_cfg)
    d.mu, d.sd = cal["mu_log"], cal["sd_log"]
    alarms = []
    for i, v in enumerate(values):
        if d.update(v):
            alarms.append(i)
            d.reset()
    return alarms


def run_condition(cond, model, cal_stats, cfg, ep_seeds, shift_ep, aseed, record_demo=False,
                  prior_data=None, eval_sets=None):
    mode = "none" if cond == "no_shift" else cond
    agent = WorldModelAgent(copy.deepcopy(model), replace(cfg["agent"], detect_mode=mode), seed=aseed)
    agent.prior_data, agent.eval_sets = prior_data, eval_sets
    agent.residual_det.mu, agent.residual_det.sd = cal_stats["residual"]["mu_log"], cal_stats["residual"]["sd_log"]
    agent.disagree_det.mu, agent.disagree_det.sd = (cal_stats["disagreement"]["mu_log"],
                                                    cal_stats["disagreement"]["sd_log"])
    episodes, trace, shift_step, records = [], {}, None, []
    for ep, s in enumerate(ep_seeds):
        if ep == shift_ep and cond != "no_shift":
            shift_step = agent.global_step
        r = run_episode(PointMassEnv(cfg["env"], seed=s), agent, seed=s, episode=ep,
                        shift_at_step=shift_step, shift=cfg["shift"], record=record_demo)
        records += r.pop("records")
        for k, v in r.pop("trace").items():
            trace.setdefault(k, []).extend(v)
        r["post_shift"] = shift_step is not None
        episodes.append(r)
    return {"episodes": episodes, "trace": trace, "shift_step": shift_step, "alarms": agent.alarms,
            "events": agent.events, "records": records, "model": agent.model}


def recovery(post_eps, streak=3):
    """First post-shift episode k that starts `streak` consecutive successes; real steps spent before it."""
    succ = [e["success"] for e in post_eps]
    for k in range(len(succ) - streak + 1):
        if all(succ[k: k + streak]):
            return {"episodes_to_recover": k, "steps_to_recover": int(sum(e["steps"] for e in post_eps[:k]))}
    return {"episodes_to_recover": None, "steps_to_recover": None}


def run_pd(env_cfg, ep_seeds, shift_ep, shift, gains_pre, gains_post):
    eps = []
    for ep, s in enumerate(ep_seeds):
        env = PointMassEnv(env_cfg, seed=s)
        post = ep >= shift_ep
        if post:
            env.set_dynamics(**shift)
        eps.append({**run_controller_episode(env, PDController(*(gains_post if post else gains_pre)), s),
                    "episode": ep, "post_shift": post})
    return {"episodes": eps}


def tune_pd_shifted(env_cfg, shift, seeds):
    """Oracle reference: pick PD gains on tuning episodes that already have the shifted dynamics."""
    best, best_key = None, None
    for kp, kd in itertools.product([1.0, 2.0, 4.0], [0.5, 1.0, 2.0]):
        eps = []
        for s in seeds:
            env = PointMassEnv(env_cfg, seed=s)
            env.set_dynamics(**shift)
            eps.append(run_controller_episode(env, PDController(kp, kd), s))
        key = (np.mean([e["success"] for e in eps]), np.mean([e["return"] for e in eps]))
        if best_key is None or key > best_key:
            best, best_key = (kp, kd), key
    return best, {"success": float(best_key[0]), "return": float(best_key[1])}


def window_means(res, shift_step, windows=((0, 100), (100, 300), (300, 10**9))):
    res = np.asarray(res)
    out = {"pre_shift": float(res[:shift_step].mean())} if shift_step else {}
    for a, b in windows:
        seg = res[shift_step + a: shift_step + b] if shift_step is not None else []
        if len(seg):
            out[f"post_{a}_{b if b < 10**9 else 'end'}"] = float(seg.mean())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/agent.yaml")
    p.add_argument("--out", default="results/s3_shift")
    args = p.parse_args()
    cfg = load_config(args.config)
    sx, env_cfg = cfg["shift_experiment"], cfg["env"]
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_runs = {}

    for aseed in sx["seeds"]:
        pre = collect(PointMassEnv(env_cfg, seed=aseed), random_policy(np.random.default_rng(aseed)),
                      cfg["data"]["pretrain_episodes"], ReplayBuffer(), seed=1_000 * (aseed + 1)).arrays()
        model = EnsembleWorldModel(config=cfg["model"], seed=aseed)
        fit = model.fit(pre["state"], pre["action"], pre["next_state"])
        model.save(str(out / "checkpoints" / f"pretrained_seed{aseed}.pt"))

        # Calibration on planner episodes the model never trains on.
        cal_agent = WorldModelAgent(copy.deepcopy(model), cfg["agent"], seed=aseed)
        res, dis = [], []
        for i in range(sx["calib_episodes"]):
            s = 7_000 * (aseed + 1) + i
            tr = run_episode(PointMassEnv(env_cfg, seed=s), cal_agent, seed=s, episode=-1 - i)["trace"]
            res += tr["residual"]
            dis += tr["disagreement"]
        cal_stats = cal_agent.calibrate(np.array(res), np.array(dis))

        ep_seeds = [4_000 * (aseed + 1) + i for i in range(sx["episodes"])]
        runs = {c: run_condition(c, model, cal_stats, cfg, ep_seeds, sx["shift_episode"], aseed,
                                 record_demo=(aseed == sx["seeds"][0] and c == "residual"))
                for c in CONDITIONS}

        # PD on the same episodes, with the same shift point (episode index). Gains default to the
        # Step-2 tuned values (kp=4, kd=2), chosen before any shift experiment.
        gains = tuple(sx.get("pd_gains", [4.0, 2.0]))
        runs["pd"] = run_pd(env_cfg, ep_seeds, sx["shift_episode"], cfg["shift"], gains, gains)
        if sx.get("pd_retune_oracle"):
            g_post, g_info = tune_pd_shifted(env_cfg, cfg["shift"], [60_000 + i for i in range(20)])
            runs["pd_retuned_oracle"] = run_pd(env_cfg, ep_seeds, sx["shift_episode"], cfg["shift"], gains, g_post)
            runs["pd_retuned_oracle"]["gains_post"] = {"kp": g_post[0], "kd": g_post[1], "tuning": g_info}

        # Held-out model quality: fresh random-policy trajectories under each dynamics (seeds disjoint
        # from every other range). One-step and open-loop multi-step error of each final model.
        heldout = {}
        for name, dyn_change in (("pre_shift", {}), ("post_shift", cfg["shift"])):
            env = PointMassEnv(env_cfg, seed=aseed)
            env.set_dynamics(**dyn_change)
            heldout[name] = collect(env, random_policy(np.random.default_rng(90_000 + aseed)), 10,
                                    ReplayBuffer(), seed=9_000 * (aseed + 1)).arrays()
        models = {"pretrained": model, **{c: runs[c].pop("model") for c in CONDITIONS}}
        model_eval = {}
        for c, m in models.items():
            model_eval[c] = {k: {"one_step_mse": m.mse(d["state"], d["action"], d["next_state"]),
                                 "multi_step": multi_step_report(m, d, horizons=(1, 5, 10), seed=aseed)}
                             for k, d in heldout.items()}
            if c != "pretrained":
                m.save(str(out / "checkpoints" / f"final_{c}_seed{aseed}.pt"))

        if aseed == sx["seeds"][0]:
            comparison = {c: {"episodes": [{k: e[k] for k in ("episode", "return", "steps", "success", "post_shift")}
                                           for e in runs[c]["episodes"]],
                              "residual": runs[c]["trace"]["residual"] if "trace" in runs[c] else None,
                              "alarms": runs[c].get("alarms")}
                          for c in runs}
            demo = {"description": "residual-detection agent, real per-step log", "seed": aseed,
                    "shift_step": runs["residual"]["shift_step"], "shift": cfg["shift"],
                    "calibration": cal_stats, "events": runs["residual"]["events"],
                    "episodes": [{k: v for k, v in e.items()} for e in runs["residual"]["episodes"]],
                    "comparison": comparison, "model_eval": model_eval,
                    "config": cfg["raw"], "steps": runs["residual"]["records"]}
            (out / "demo_run.json").write_text(json.dumps(demo, default=float))
        for c in runs:
            runs[c].pop("records", None)
        all_runs[str(aseed)] = {"pretrain_fit": fit, "calibration": cal_stats, "runs": runs,
                                "model_eval": model_eval}
        print(f"seed {aseed} done ({time.time() - t0:.0f}s)")

    # ---------------- analysis ----------------
    det_cfg = cfg["agent"].detector
    analysis = {"per_seed": {}}
    for aseed, R in all_runs.items():
        runs, cal = R["runs"], R["calibration"]
        shift_step = runs["none"]["shift_step"]
        a = {}
        for sig in ("residual", "disagreement"):
            alarms = offline_alarms(runs["none"]["trace"][sig], cal[sig], det_cfg)
            ctrl = offline_alarms(runs["no_shift"]["trace"][sig], cal[sig], det_cfg)
            post = [x for x in alarms if x >= shift_step]
            a[f"offline_{sig}"] = {
                "false_alarms_pre_shift": len([x for x in alarms if x < shift_step]), "pre_shift_steps": shift_step,
                "false_alarms_no_shift_control": len(ctrl),
                "no_shift_control_steps": len(runs["no_shift"]["trace"][sig]),
                "detection_delay_steps": (post[0] - shift_step) if post else None}
        for c in runs:
            run = runs[c]
            pre_eps = [e for e in run["episodes"] if not e["post_shift"]]
            post_eps = [e for e in run["episodes"] if e["post_shift"]]
            a[c] = {"pre_shift_success": float(np.mean([e["success"] for e in pre_eps])),
                    "pre_shift_return": float(np.mean([e["return"] for e in pre_eps])),
                    "post_shift_success": float(np.mean([e["success"] for e in post_eps])) if post_eps else None,
                    "post_shift_return": float(np.mean([e["return"] for e in post_eps])) if post_eps else None,
                    "post_shift_steps": float(np.mean([e["steps"] for e in post_eps])) if post_eps else None,
                    "first5_post_shift_return": float(np.mean([e["return"] for e in post_eps[:5]])) if post_eps else None,
                    "last5_post_shift_return": float(np.mean([e["return"] for e in post_eps[-5:]])) if post_eps else None,
                    "decision_ms": float(np.mean([e["mean_plan_ms"] for e in run["episodes"]])),
                    **(recovery(post_eps) if post_eps else {})}
            if "alarms" in run and c != "no_shift":
                fts = [e for e in run["events"] if e.get("type") == "finetune"]
                a[c].update({
                    "online_alarms": run["alarms"],
                    "online_false_alarms": len([x for x in run["alarms"] if x < shift_step]),
                    "online_detection_delay": next((x - shift_step for x in run["alarms"] if x >= shift_step), None),
                    "n_finetunes": len(fts), "finetune_seconds_total": float(sum(e["seconds"] for e in fts)),
                    "residual_windows": window_means(run["trace"]["residual"], shift_step)})
        for c, ev in R.get("model_eval", {}).items():
            for k, v in ev.items():
                a.setdefault("model_eval", {}).setdefault(c, {})[k] = {
                    "h1": v["one_step_mse"], **{h: v["multi_step"][h]["mse_mean"] for h in ("h5", "h10") if h in v["multi_step"]}}
        analysis["per_seed"][aseed] = a

    def agg(path):
        vals = []
        for a in analysis["per_seed"].values():
            v = a
            for k in path:
                v = v.get(k) if isinstance(v, dict) else None
            vals.append(v)
        num = [v for v in vals if v is not None]
        return {"values": vals, "mean": float(np.mean(num)) if num else None,
                "std": float(np.std(num)) if num else None}

    summary = {}
    for sig in ("residual", "disagreement"):
        for k in ("false_alarms_pre_shift", "false_alarms_no_shift_control", "detection_delay_steps"):
            summary[f"offline_{sig}.{k}"] = agg([f"offline_{sig}", k])
    conds = list(next(iter(all_runs.values()))["runs"])
    for c in conds:
        for k in ("pre_shift_success", "pre_shift_return", "post_shift_success", "post_shift_return",
                  "post_shift_steps", "first5_post_shift_return", "last5_post_shift_return",
                  "episodes_to_recover", "steps_to_recover", "decision_ms"):
            summary[f"{c}.{k}"] = agg([c, k])
        if c in ("none", "disagreement", "residual", "oracle"):
            for k in ("online_detection_delay", "online_false_alarms", "n_finetunes", "finetune_seconds_total"):
                summary[f"{c}.{k}"] = agg([c, k])
            for w in ("pre_shift", "post_0_100", "post_100_300", "post_300_end"):
                summary[f"{c}.residual.{w}"] = agg([c, "residual_windows", w])
    for c in ("pretrained", "none", "disagreement", "residual", "oracle", "no_shift"):
        for data in ("pre_shift", "post_shift"):
            for h in ("h1", "h5", "h10"):
                summary[f"model_eval.{c}.{data}.{h}"] = agg(["model_eval", c, data, h])
    analysis["summary"] = summary
    analysis["wall_seconds"] = time.time() - t0
    (out / "metrics.json").write_text(json.dumps({"config": cfg["raw"], "analysis": analysis, "runs": all_runs},
                                                 default=float))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    # ---------------- figure: seed 0 traces ----------------
    s0 = all_runs[str(sx["seeds"][0])]["runs"]
    shift_step = s0["none"]["shift_step"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for c, col in zip(["none", "disagreement", "residual", "oracle"], ["0.5", "tab:purple", "tab:red", "tab:green"]):
        r = np.asarray(s0[c]["trace"]["residual"])
        k = 15
        sm = np.convolve(r, np.ones(k) / k, mode="valid")
        axes[0].semilogy(np.arange(len(sm)) + k - 1, sm, color=col, label=f"{c}")
        for al in s0[c]["alarms"]:
            axes[0].axvline(al, color=col, ls=":", lw=1)
    axes[0].set_ylabel("observed prediction residual\n(15-step moving mean)")
    axes[0].set_title("Seed 0: online prediction error; dotted = detector alarm; black = true shift")
    axes[1].plot(s0["none"]["trace"]["cusum_residual"], color="tab:red", label="CUSUM on residual")
    axes[1].plot(s0["none"]["trace"]["cusum_disagreement"], color="tab:purple", label="CUSUM on disagreement")
    axes[1].axhline(det_cfg.threshold, color="k", ls="--", lw=1, label="threshold")
    axes[1].set_ylabel("CUSUM statistic\n(no-adaptation run)")
    axes[1].set_yscale("symlog")
    axes[2].semilogy(np.convolve(s0["none"]["trace"]["disagreement"], np.ones(15) / 15, mode="valid"),
                     color="tab:purple")
    axes[2].set_ylabel("ensemble disagreement\n(15-step moving mean)")
    axes[2].set_xlabel("agent step (real environment interactions)")
    for ax in axes:
        ax.axvline(shift_step, color="k", lw=1.5)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "shift_traces_seed0.png", dpi=120)
    # ---------------- figure: per-episode task performance across seeds ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    styles = {"none": ("0.5", "-"), "disagreement": ("tab:purple", "-"), "residual": ("tab:red", "-"),
              "oracle": ("tab:green", "--"), "pd": ("k", "-"), "pd_retuned_oracle": ("k", ":"),
              "no_shift": ("tab:blue", ":")}
    for c in conds:
        succ = np.array([[e["success"] for e in R["runs"][c]["episodes"]] for R in all_runs.values()], float)
        ret = np.array([[e["return"] for e in R["runs"][c]["episodes"]] for R in all_runs.values()])
        col, ls = styles.get(c, ("0.3", "-"))
        axes[0].plot(succ.mean(0), color=col, ls=ls, marker="o", ms=3, label=c)
        axes[1].plot(ret.mean(0), color=col, ls=ls, marker="o", ms=3, label=c)
    for ax, yl in zip(axes, ["success rate across seeds", "mean episode return across seeds"]):
        ax.axvline(sx["shift_episode"] - 0.5, color="k", lw=1)
        ax.set(xlabel=f"episode (dynamics changed from episode {sx['shift_episode']} on)", ylabel=yl)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Shift {cfg['shift']} at episode {sx['shift_episode']}, {len(all_runs)} seeds")
    fig.tight_layout()
    fig.savefig(out / "recovery_by_episode.png", dpi=120)
    print(json.dumps(summary, indent=1))
    print(f"total {analysis['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
