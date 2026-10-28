"""Dynamics reversion: original (episodes < shift_episode) -> changed (until revert_episode) -> original.

Question: when the original dynamics return, does replay-based adaptation (which forgets less) recover
control better than fresh-only adaptation, enough to justify its cost on the changed dynamics?

Per agent seed, pretraining, detector calibration and held-out sets are exactly those of
run_adapt_sweep.py / run_shift.py (same code path), so episodes before the reversion reproduce
results/s5_adapt_eval60 for the same adaptation settings. Nothing here is tuned: the adaptation
settings are the s5 selection (min_new 50) with and without its replay ratio.

Conditions (config `reversion.conditions`): no adaptation; residual-triggered adaptation (fresh-only,
replay); and oracle references that are told the exact episode of every change and start adaptation
then. Oracles are not deployable methods.

Prediction errors:
  - online residual: one-step error of the agent's model on each real transition, scored BEFORE the
    model trains on it;
  - held-out: random-policy transitions under the original and the changed dynamics (seeds no agent
    trains on), measured at the end of each phase and immediately before/after every fine-tune.

Usage: python scripts/run_reversion.py --config configs/reversion60.yaml --out results/s6_reversion --workers 5
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_adapt_sweep import prepare  # noqa: E402
from run_shift import recovery  # noqa: E402

from autonomous_agent.agent import WorldModelAgent, run_episode  # noqa: E402
from autonomous_agent.config import load_config  # noqa: E402
from autonomous_agent.env import PointMassEnv  # noqa: E402
from autonomous_agent.schedule import episode_phase, phase_dynamics  # noqa: E402

PHASES = ("original", "changed", "reverted")


def run_condition(cfg, spec, model, pre, cal_stats, heldout, ep_seeds, aseed, record=False):
    sx = cfg["shift_experiment"]
    shift_ep, revert_ep = sx["shift_episode"], sx["revert_episode"]
    oracle = spec["mode"] == "oracle"
    adapt = replace(cfg["agent"].adapt, min_new=spec.get("min_new", cfg["agent"].adapt.min_new),
                    replay_ratio=spec.get("replay_ratio", 0.0))
    mode = {"none": "none", "residual": "residual", "oracle": "none"}[spec["mode"]]
    agent = WorldModelAgent(copy.deepcopy(model), replace(cfg["agent"], adapt=adapt, detect_mode=mode), seed=aseed)
    agent.residual_det.mu, agent.residual_det.sd = cal_stats["residual"]["mu_log"], cal_stats["residual"]["sd_log"]
    agent.disagree_det.mu, agent.disagree_det.sd = (cal_stats["disagreement"]["mu_log"],
                                                    cal_stats["disagreement"]["sd_log"])
    agent.prior_data, agent.eval_sets = pre, heldout
    err = lambda: {k: agent.model.mse(v["state"], v["action"], v["next_state"]) for k, v in heldout.items()}
    episodes, records, residual, phase_of_step = [], [], [], []
    changes, phase_end_errors = [], {}
    for ep, s in enumerate(ep_seeds):
        phase = episode_phase(ep, shift_ep, revert_ep)
        if ep in (shift_ep, revert_ep):
            changes.append({"episode": ep, "step": agent.global_step, "to": phase_dynamics(phase, cfg["shift"], sx["revert"]),
                            "adapting_at_change": agent.adapting, "post_alarm_rows_at_change": len(agent.post_alarm)})
            if oracle:
                agent.trigger_adaptation("oracle")
        dyn = phase_dynamics(phase, cfg["shift"], sx["revert"])
        r = run_episode(PointMassEnv(cfg["env"], seed=s), agent, seed=s, episode=ep,
                        shift_at_step=agent.global_step if dyn else None, shift=dyn, record=record)
        for rec in r["records"]:
            rec["phase"] = phase
        records += r.pop("records")
        tr = r.pop("trace")
        residual += tr["residual"]
        phase_of_step += [phase] * len(tr["residual"])
        r.update({"phase": phase, "post_shift": phase == 1})
        r.pop("refit", None)
        episodes.append(r)
        if ep == len(ep_seeds) - 1 or episode_phase(ep + 1, shift_ep, revert_ep) != phase:
            phase_end_errors[PHASES[phase]] = err()
    return {"episodes": episodes, "records": records, "residual": residual, "phase_of_step": phase_of_step,
            "changes": changes, "phase_end_heldout": phase_end_errors, "alarms": agent.alarms,
            "events": agent.events, "adapt": asdict(adapt)}


def run_metrics(run):
    eps, changes = run["episodes"], run["changes"]
    fts = [e for e in run["events"] if e.get("type") == "finetune"]
    out = {"n_finetunes": len(fts), "finetune_seconds_total": float(sum(e["seconds"] for e in fts)),
           "decision_ms": float(np.mean([e["mean_plan_ms"] for e in eps]))}
    res, ph = np.asarray(run["residual"]), np.asarray(run["phase_of_step"])
    for p, name in enumerate(PHASES):
        pe = [e for e in eps if e["phase"] == p]
        out[f"{name}.success"] = float(np.mean([e["success"] for e in pe]))
        out[f"{name}.return"] = float(np.mean([e["return"] for e in pe]))
        out[f"{name}.steps"] = float(np.mean([e["steps"] for e in pe]))
        out[f"{name}.online_residual"] = float(res[ph == p].mean())
        for k, v in run["phase_end_heldout"][name].items():
            out[f"{name}.end_heldout.{k}"] = v
    bounds = [c["step"] for c in changes] + [10 ** 9]
    for i, c in enumerate(changes):
        name = PHASES[i + 1]
        lo, hi = bounds[i], bounds[i + 1]
        pe = [e for e in eps if e["phase"] == i + 1]
        k = next((j for j, e in enumerate(pe) if e["success"]), None)
        in_win = [x for x in run["alarms"] if lo <= x < hi]
        ft_win = [e for e in fts if lo <= e["step"] < hi]
        out.update({
            f"{name}.first_episode_success": float(pe[0]["success"]),
            f"{name}.first_episode_return": float(pe[0]["return"]),
            f"{name}.detection_delay": (in_win[0] - lo) if in_win else None,
            f"{name}.first_finetune_delay": (ft_win[0]["step"] - lo) if ft_win else None,
            f"{name}.steps_to_first_success": int(sum(e["steps"] for e in pe[: k + 1])) if k is not None else None,
            f"{name}.episodes_to_recover": recovery(pe)["episodes_to_recover"],
            f"{name}.n_finetunes": len(ft_win),
            f"{name}.finetune_seconds": float(sum(e["seconds"] for e in ft_win)),
            f"{name}.adapting_at_change": float(c["adapting_at_change"]),
            f"{name}.residual_first100": float(res[lo: lo + 100].mean()),
        })
    return out


def run_seed(job):
    cfg_path, aseed, record_name = job
    import torch
    torch.set_num_threads(1)
    cfg = load_config(cfg_path)
    model, pre, cal_stats, heldout, ep_seeds = prepare(cfg, aseed)
    out, demo = {}, None
    for spec in cfg["raw"]["reversion"]["conditions"]:
        run = run_condition(cfg, spec, model, pre, cal_stats, heldout, ep_seeds, aseed, record=spec["name"] == record_name)
        met = run_metrics(run)
        out[spec["name"]] = {"metrics": met, "episodes": [{k: e[k] for k in ("episode", "phase", "return", "steps",
                                                                             "success", "post_shift")} for e in run["episodes"]],
                             "residual": run["residual"], "alarms": run["alarms"], "changes": run["changes"],
                             "phase_end_heldout": run["phase_end_heldout"], "adapt": run["adapt"],
                             "finetunes": [{k: e.get(k) for k in ("step", "n_fresh", "n_replay", "seconds",
                                                                  "heldout_before", "heldout_after")}
                                           for e in run["events"] if e.get("type") == "finetune"]}
        if spec["name"] == record_name:
            demo = {"run": run, "cal_stats": cal_stats}
        print(f"seed {aseed} {spec['name']}: success orig/changed/reverted "
              f"{met['original.success']:.2f}/{met['changed.success']:.2f}/{met['reverted.success']:.2f}", flush=True)
    init = {k: model.mse(v["state"], v["action"], v["next_state"]) for k, v in heldout.items()}
    return aseed, {"conditions": out, "pretrained_heldout": init}, demo


def aggregate(per_seed):
    names = list(next(iter(per_seed.values()))["conditions"])
    keys = list(next(iter(per_seed.values()))["conditions"][names[0]]["metrics"])
    summary = {}
    for c in names:
        for k in keys:
            vals = [p["conditions"][c]["metrics"][k] for p in per_seed.values()]
            num = [v for v in vals if v is not None]
            summary[f"{c}.{k}"] = {"values": vals, "mean": float(np.mean(num)) if num else None,
                                   "std": float(np.std(num)) if num else None}
    return summary


def plot(cfg, per_seed, summary, names, seeds, out):
    """Success / return per episode across seeds, and held-out error at the end of each phase."""
    sx = cfg["shift_experiment"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    styles = {"none": ("0.5", "-"), "adapt_m50_r0": ("tab:blue", "-"), "adapt_m50_r0.25": ("tab:red", "-"),
              "oracle_m50_r0": ("tab:blue", ":"), "oracle_m50_r0.25": ("tab:red", ":")}
    for c in names:
        succ = np.array([[e["success"] for e in per_seed[s]["conditions"][c]["episodes"]] for s in per_seed], float)
        ret = np.array([[e["return"] for e in per_seed[s]["conditions"][c]["episodes"]] for s in per_seed])
        col, ls = styles.get(c, ("k", "-"))
        axes[0].plot(succ.mean(0), color=col, ls=ls, marker="o", ms=3, label=c)
        axes[1].plot(ret.mean(0), color=col, ls=ls, marker="o", ms=3, label=c)
    for ax, yl in zip(axes[:2], ["success rate across seeds", "mean episode return across seeds"]):
        ax.axvspan(sx["shift_episode"] - 0.5, sx["revert_episode"] - 0.5, color="orange", alpha=0.12)
        ax.set(xlabel=f"episode (rotated dynamics shaded: {sx['shift_episode']}-{sx['revert_episode'] - 1})", ylabel=yl)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    # held-out error at phase ends: one panel row per dynamics, conditions as grouped bars
    ax = axes[2]
    x = np.arange(len(PHASES))
    w = 0.8 / len(names)
    for i, c in enumerate(names):
        col, ls = styles.get(c, ("k", "-"))
        for d, mk in (("old_dynamics", "o"), ("new_dynamics", "^")):
            v = [summary[f"{c}.{ph}.end_heldout.{d}"]["mean"] for ph in PHASES]
            ax.scatter(x + (i - len(names) / 2 + 0.5) * w, v, color=col, marker=mk, s=40,
                       facecolors="none" if ls == ":" else col,
                       label=f"{c}" if d == "old_dynamics" else None)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"end of {p}" for p in PHASES])
    ax.set_ylabel("held-out one-step MSE")
    ax.set_title("circles = original dynamics, triangles = rotated; hollow = oracle", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    fig.suptitle(f"Reversion: {cfg['shift']} at episode {sx['shift_episode']}, back to {sx['revert']} at "
                 f"episode {sx['revert_episode']}; {len(seeds)} seeds")
    fig.tight_layout()
    fig.savefig(out / "reversion_by_episode.png", dpi=120)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", help="override seeds (pilot only)")
    p.add_argument("--replot", action="store_true", help="only redraw the figure from an existing --out")
    args = p.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    if args.replot:
        m = json.loads((out / "metrics.json").read_text())
        plot(cfg, m["per_seed"], json.loads((out / "summary.json").read_text()), m["conditions"], m["seeds"], out)
        return
    if out.exists():
        raise SystemExit(f"{out} exists; results are never overwritten")
    out.mkdir(parents=True)
    sx, rv = cfg["shift_experiment"], cfg["raw"]["reversion"]
    seeds = args.seeds or sx["seeds"]
    t0 = time.time()
    jobs = [(args.config, a, rv["record_condition"] if a == seeds[0] else None) for a in seeds]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(run_seed, jobs)
    else:
        results = [run_seed(j) for j in jobs]
    per_seed = {str(a): r for a, r, _ in results}
    summary = aggregate(per_seed)
    names = [c["name"] for c in rv["conditions"]]

    demo = next(d for _, _, d in results if d)
    run = demo["run"]
    s0 = per_seed[str(seeds[0])]["conditions"]
    rec_met = s0[rv["record_condition"]]["metrics"]
    demo_json = {
        "description": f"{rv['record_condition']}: real per-step log of a run with a change and a reversion",
        "seed": seeds[0], "shift_step": run["changes"][0]["step"], "shift": cfg["shift"],
        "calibration": demo["cal_stats"], "events": run["events"], "episodes": run["episodes"],
        "condition": rv["record_condition"], "adapt_settings": run["adapt"],
        "changes": [{**c, "detection_delay": rec_met[f"{PHASES[i + 1]}.detection_delay"],
                     "first_finetune_delay": rec_met[f"{PHASES[i + 1]}.first_finetune_delay"],
                     "first_episode_success": rec_met[f"{PHASES[i + 1]}.first_episode_success"]}
                    for i, c in enumerate(run["changes"])],
        "comparison": {c: {"episodes": s0[c]["episodes"], "residual": s0[c]["residual"], "alarms": s0[c]["alarms"]}
                       for c in names},
        "phase_end_heldout": {c: s0[c]["phase_end_heldout"] for c in names},
        "heldout_initial": {k: {"h1": v} for k, v in per_seed[str(seeds[0])]["pretrained_heldout"].items()},
        "command": " ".join(["python scripts/run_reversion.py"] + sys.argv[1:]),
        "config": cfg["raw"], "steps": run["records"]}
    (out / "demo_run.json").write_text(json.dumps(demo_json, default=float))
    (out / "metrics.json").write_text(json.dumps({"config": cfg["raw"], "seeds": seeds, "conditions": names,
                                                  "wall_seconds": time.time() - t0, "workers": args.workers,
                                                  "per_seed": per_seed}, default=float))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    plot(cfg, per_seed, summary, names, seeds, out)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
