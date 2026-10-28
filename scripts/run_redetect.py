"""A dynamics change during an active adaptation window: ignore it (legacy) or restart the window.

Scenarios (config `redetect.scenarios`) are piecewise-constant schedules over episodes:
  persistent     0 deg -> 60 deg at episode 8 (control: any restart here happens without a new change)
  reversion      0 -> 60 (ep 8) -> 0 (ep 20), the s6 schedule
  second_change  0 -> 60 (ep 8) -> -30 (ep 11), a second change inside the first change's window

Per agent seed, pretraining, calibration, episode seeds and held-out sets are those of run_reversion.py
(same `prepare`), so the "ignore" conditions of the reversion scenario reproduce results/s6_reversion.
Extra held-out sets (e.g. -30 deg) use the same collection seeds under the other dynamics.

Phases:
  --phase tune  seeds 100-102; restart conditions over the re-detection grid, plus the ignore baselines,
                on the `tune.scenarios`. Selection rule, fixed before running:
                  1. false restarts = restarts in the persistent scenario, summed over seeds and both replay
                     ratios. Keep the settings with the fewest false restarts, allowing one extra
                     (i.e. min + 1), so one noisy seed cannot alone decide the choice;
                  2. among those, the highest mean whole-run return in second_change (mean over seeds and
                     both ratios);
                  3. ties: the larger drift, then the larger threshold (more conservative).
                Writes selection.json. Nothing is evaluated on seeds 0-4 in this phase.
  --phase eval  seeds 0-4; all conditions on all scenarios, with the re-detection settings fixed in the
                config (`redetect.redetect_drift / redetect_threshold`) from the tuning selection.

Usage:
  python scripts/run_redetect.py --phase tune --config configs/redetect60.yaml --out results/s7_redetect_tune --workers 7
  python scripts/run_redetect.py --phase eval --config configs/redetect60.yaml --out results/s7_redetect --workers 7
"""
from __future__ import annotations

import argparse
import copy
import itertools
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
from autonomous_agent.data import ReplayBuffer, collect, random_policy  # noqa: E402
from autonomous_agent.env import PointMassEnv  # noqa: E402
from autonomous_agent.schedule import segment_index  # noqa: E402


def dyn_label(dyn: dict | None) -> str:
    dyn = dyn or {}
    assert set(dyn) <= {"rotation_deg", "gain", "drag"}, "held-out labels cover rotation, gain and drag changes"
    extra = "".join(f"_{k}{dyn[k]:g}" for k in ("gain", "drag") if k in dyn)   # rotation-only labels unchanged
    return f"rot{dyn.get('rotation_deg', 0.0):g}{extra}"


def heldout_sets(cfg, aseed, base, scenario):
    """{label: arrays} for every dynamics in the schedule; 0 and 60 deg reuse prepare()'s sets."""
    out = {dyn_label(None): base["old_dynamics"], dyn_label(cfg["shift"]): base["new_dynamics"]}
    for ch in scenario["changes"]:
        lab = dyn_label(ch["dynamics"])
        if lab not in out:
            env = PointMassEnv(cfg["env"], seed=aseed)
            env.set_dynamics(**ch["dynamics"])
            out[lab] = collect(env, random_policy(np.random.default_rng(90_000 + aseed)), 10,
                               ReplayBuffer(), seed=9_000 * (aseed + 1)).arrays()
    return out


def run_condition(cfg, scenario, spec, model, pre, cal_stats, heldout, ep_seeds, aseed, record=False):
    rd = cfg["raw"]["redetect"]
    change_eps = [c["episode"] for c in scenario["changes"]]
    oracle = spec["mode"] == "oracle"
    adapt = replace(cfg["agent"].adapt, min_new=spec.get("min_new", cfg["agent"].adapt.min_new),
                    max_new=spec.get("max_new", cfg["agent"].adapt.max_new),
                    replay_ratio=spec.get("replay_ratio", 0.0),
                    on_alarm_during_adapt=spec.get("policy", "ignore"),
                    redetect_drift=spec.get("redetect_drift", rd["redetect_drift"] or 1.5),
                    redetect_threshold=spec.get("redetect_threshold", rd["redetect_threshold"] or 10.0),
                    max_redetections=rd["max_redetections"])
    mode = {"none": "none", "residual": "residual", "oracle": "none"}[spec["mode"]]
    agent = WorldModelAgent(copy.deepcopy(model), replace(cfg["agent"], adapt=adapt, detect_mode=mode), seed=aseed)
    agent.residual_det.mu, agent.residual_det.sd = cal_stats["residual"]["mu_log"], cal_stats["residual"]["sd_log"]
    agent.disagree_det.mu, agent.disagree_det.sd = (cal_stats["disagreement"]["mu_log"],
                                                    cal_stats["disagreement"]["sd_log"])
    agent.set_redetect_reference()
    agent.prior_data, agent.eval_sets = pre, heldout
    err = lambda: {k: agent.model.mse(v["state"], v["action"], v["next_state"]) for k, v in heldout.items()}
    names = scenario["segments"]
    episodes, records, residual, seg_of_step, changes, seg_end = [], [], [], [], [], {}
    for ep, s in enumerate(ep_seeds):
        seg = segment_index(ep, change_eps)
        if ep in change_eps:
            changes.append({"episode": ep, "step": agent.global_step, "to": scenario["changes"][seg - 1]["dynamics"],
                            "adapting_at_change": agent.adapting, "state_at_change": agent.state,
                            "post_alarm_rows_at_change": len(agent.post_alarm)})
            if oracle:
                agent.trigger_adaptation("oracle")
        dyn = scenario["changes"][seg - 1]["dynamics"] if seg else None
        r = run_episode(PointMassEnv(cfg["env"], seed=s), agent, seed=s, episode=ep,
                        shift_at_step=agent.global_step if dyn else None, shift=dyn, record=record)
        for rec in r["records"]:
            rec["phase"] = seg
        records += r.pop("records")
        tr = r.pop("trace")
        residual += tr["residual"]
        seg_of_step += [seg] * len(tr["residual"])
        r.update({"phase": seg, "post_shift": seg > 0})
        r.pop("refit", None)
        episodes.append(r)
        if ep == len(ep_seeds) - 1 or segment_index(ep + 1, change_eps) != seg:
            seg_end[names[seg]] = err()
    return {"episodes": episodes, "records": records, "residual": residual, "phase_of_step": seg_of_step,
            "changes": changes, "phase_end_heldout": seg_end, "alarms": agent.alarms, "events": agent.events,
            "ignored_redetections": agent.ignored_redetections, "adapt": asdict(adapt)}


def run_metrics(run, names):
    eps, changes = run["episodes"], run["changes"]
    fts = [e for e in run["events"] if e.get("type") == "finetune"]
    alarms = [e for e in run["events"] if e.get("type") == "alarm"]
    out = {"n_finetunes": len(fts), "finetune_seconds_total": float(sum(e["seconds"] for e in fts)),
           "n_restarts": sum(a["mode"] == "redetect" for a in alarms),
           "n_ignored_redetections": len(run["ignored_redetections"]),
           "return": float(np.mean([e["return"] for e in eps])),
           "success": float(np.mean([e["success"] for e in eps])),
           "decision_ms": float(np.mean([e["mean_plan_ms"] for e in eps]))}
    res, ph = np.asarray(run["residual"]), np.asarray(run["phase_of_step"])
    for p, name in enumerate(names):
        pe = [e for e in eps if e["phase"] == p]
        out[f"{name}.success"] = float(np.mean([e["success"] for e in pe]))
        out[f"{name}.return"] = float(np.mean([e["return"] for e in pe]))
        out[f"{name}.online_residual"] = float(res[ph == p].mean())
        for k, v in run["phase_end_heldout"][name].items():
            out[f"{name}.end_heldout.{k}"] = v
    bounds = [c["step"] for c in changes] + [10 ** 9]
    for i, c in enumerate(changes):
        name, lo, hi = names[i + 1], bounds[i], bounds[i + 1]
        pe = [e for e in eps if e["phase"] == i + 1]
        k = next((j for j, e in enumerate(pe) if e["success"]), None)
        a_win = [a for a in alarms if lo <= a["step"] < hi]
        ft_win = [e for e in fts if lo <= e["step"] < hi]
        clean = [e for e in ft_win if e.get("window_start") is not None and e["window_start"] >= lo]
        out.update({
            f"{name}.first_episode_success": float(pe[0]["success"]),
            f"{name}.first_episode_return": float(pe[0]["return"]),
            f"{name}.detection_delay": (a_win[0]["step"] - lo) if a_win else None,
            f"{name}.detected": float(bool(a_win)),
            f"{name}.detected_by_restart": float(bool(a_win) and a_win[0]["mode"] == "redetect"),
            f"{name}.first_finetune_delay": (ft_win[0]["step"] - lo) if ft_win else None,
            # first fine-tune whose window opened after this change, i.e. not mixing pre-change rows
            f"{name}.clean_finetune_delay": (clean[0]["step"] - lo) if clean else None,
            f"{name}.steps_to_first_success": int(sum(e["steps"] for e in pe[: k + 1])) if k is not None else None,
            f"{name}.episodes_to_recover": recovery(pe)["episodes_to_recover"],
            f"{name}.n_finetunes": len(ft_win),
            f"{name}.finetune_seconds": float(sum(e["seconds"] for e in ft_win)),
            f"{name}.n_restarts": sum(a["mode"] == "redetect" for a in a_win),
            f"{name}.adapting_at_change": float(c["adapting_at_change"]),
            f"{name}.residual_first100": float(res[lo: lo + 100].mean()),
        })
    return out


_PREP: dict = {}


def run_job(job):
    """One (agent seed, scenario, condition) run. prepare() is cached per worker process."""
    cfg_path, aseed, scen_name, spec, record = job
    import torch
    torch.set_num_threads(1)
    cfg = load_config(cfg_path)
    if aseed not in _PREP:
        _PREP[aseed] = prepare(cfg, aseed)
    model, pre, cal_stats, base_heldout, ep_seeds = _PREP[aseed]
    scen = cfg["raw"]["redetect"]["scenarios"][scen_name]
    heldout = heldout_sets(cfg, aseed, base_heldout, scen)
    run = run_condition(cfg, scen, spec, model, pre, cal_stats, heldout, ep_seeds, aseed, record=record)
    met = run_metrics(run, scen["segments"])
    res = {"metrics": met,
           "episodes": [{k: e[k] for k in ("episode", "phase", "return", "steps", "success", "post_shift")}
                        for e in run["episodes"]],
           "residual": run["residual"], "alarms": run["alarms"], "changes": run["changes"],
           "alarm_events": [e for e in run["events"] if e.get("type") == "alarm"],
           "ignored_redetections": run["ignored_redetections"],
           "phase_end_heldout": run["phase_end_heldout"], "adapt": run["adapt"],
           "finetunes": [{k: e.get(k) for k in ("step", "n_fresh", "n_replay", "seconds", "window_start",
                                                "heldout_before", "heldout_after")}
                         for e in run["events"] if e.get("type") == "finetune"]}
    print(f"seed {aseed} {scen_name} {spec['name']}: return {met['return']:.1f}, restarts {met['n_restarts']}, "
          f"finetunes {met['n_finetunes']}", flush=True)
    demo = None
    if record:
        init = {k: model.mse(v["state"], v["action"], v["next_state"]) for k, v in heldout.items()}
        demo = {"run": run, "cal_stats": cal_stats, "heldout_initial": init}
    return aseed, scen_name, spec["name"], res, demo


def aggregate(per_seed, scen, names):
    summary = {}
    for c in names:
        keys = list(per_seed[next(iter(per_seed))][scen][c]["metrics"])
        for k in keys:
            vals = [per_seed[s][scen][c]["metrics"][k] for s in per_seed]
            num = [v for v in vals if v is not None]
            summary[f"{c}.{k}"] = {"values": vals, "mean": float(np.mean(num)) if num else None,
                                   "std": float(np.std(num)) if num else None}
    return summary


def plot(scen_name, scen, per_seed, names, seeds, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    styles = {"none": ("0.5", "-"), "fresh_ignore": ("tab:blue", "--"), "fresh_restart": ("tab:blue", "-"),
              "replay_ignore": ("tab:red", "--"), "replay_restart": ("tab:red", "-"),
              "oracle_fresh": ("tab:blue", ":"), "oracle_replay": ("tab:red", ":"),
              "fresh_ignore_short": ("tab:cyan", "-."), "replay_ignore_short": ("tab:orange", "-.")}
    for c in names:
        ret = np.array([[e["return"] for e in per_seed[s][scen_name][c]["episodes"]] for s in per_seed])
        col, ls = styles.get(c, ("k", "-"))
        axes[0].plot(ret.mean(0), color=col, ls=ls, marker="o", ms=3, label=c)
        cum = np.array([np.cumsum([e["return"] for e in per_seed[s][scen_name][c]["episodes"]]) for s in per_seed])
        axes[1].plot(cum.mean(0), color=col, ls=ls, label=c)
    edges = [c["episode"] for c in scen["changes"]]
    for ax, yl in zip(axes, ["mean episode return across seeds", "mean cumulative return"]):
        for e in edges:
            ax.axvline(e - 0.5, color="orange", lw=1.2)
        ax.set(xlabel="episode (orange lines: dynamics changes)", ylabel=yl)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    sched = ", ".join(f"ep {c['episode']}: {c['dynamics']}" for c in scen["changes"])
    fig.suptitle(f"{scen_name}: {sched}; {len(seeds)} seeds")
    fig.tight_layout()
    fig.savefig(out / f"{scen_name}_by_episode.png", dpi=120)
    plt.close(fig)


def tune_conditions(rd):
    conds = [c for c in rd["conditions"] if c["mode"] == "residual" and c["policy"] == "ignore"]
    g = rd["tune"]["grid"]
    for d, t in itertools.product(g["redetect_drift"], g["redetect_threshold"]):
        for base in (c for c in rd["conditions"] if c["mode"] == "residual" and c["policy"] == "restart"):
            conds.append({**base, "name": f"{base['name']}_d{d:g}_t{t:g}", "redetect_drift": d,
                          "redetect_threshold": t, "grid": [d, t]})
    return conds


def select(per_seed, rd):
    """Selection rule from the module docstring."""
    g = rd["tune"]["grid"]
    rows = []
    for d, t in itertools.product(g["redetect_drift"], g["redetect_threshold"]):
        names = [n for n in per_seed[next(iter(per_seed))]["persistent"] if n.endswith(f"_d{d:g}_t{t:g}")]
        false = sum(per_seed[s]["persistent"][n]["metrics"]["n_restarts"] for s in per_seed for n in names)
        ret2 = float(np.mean([per_seed[s]["second_change"][n]["metrics"]["return"] for s in per_seed for n in names]))
        rows.append({"redetect_drift": d, "redetect_threshold": t, "false_restarts": int(false),
                     "second_change_return": ret2})
    lo = min(r["false_restarts"] for r in rows)
    ok = [r for r in rows if r["false_restarts"] <= lo + 1]
    best = max(ok, key=lambda r: (r["second_change_return"], r["redetect_drift"], r["redetect_threshold"]))
    return {"rule": "fewest false restarts (+1 allowed), then highest second_change return, then larger drift/threshold",
            "grid": rows, "selected": {k: best[k] for k in ("redetect_drift", "redetect_threshold")}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("tune", "eval"), required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", help="override seeds (pilot only)")
    p.add_argument("--scenarios", nargs="+", help="run only these scenarios")
    p.add_argument("--conditions", nargs="+", help="run only these conditions (eval phase)")
    args = p.parse_args()
    cfg = load_config(args.config)
    rd = cfg["raw"]["redetect"]
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; results are never overwritten")
    if args.phase == "tune":
        seeds, scens, conds = rd["tune"]["seeds"], rd["tune"]["scenarios"], tune_conditions(rd)
    else:
        if rd["redetect_drift"] is None or rd["redetect_threshold"] is None:
            raise SystemExit("set redetect.redetect_drift / redetect_threshold from the tuning selection first")
        seeds, scens, conds = cfg["shift_experiment"]["seeds"], list(rd["scenarios"]), rd["conditions"]
    seeds, scens = args.seeds or seeds, args.scenarios or scens
    if args.conditions:
        conds = [c for c in conds if c["name"] in args.conditions]
        assert len(conds) == len(args.conditions), "unknown condition name"
    if args.phase == "eval":
        assert not set(seeds) & set(rd["tune"]["seeds"]), "evaluation seeds overlap tuning seeds"
    out.mkdir(parents=True)
    t0 = time.time()
    recs = {(r["scenario"], r["condition"]) for r in rd["record"]}
    jobs = [(args.config, a, sc, spec, args.phase == "eval" and a == seeds[0] and (sc, spec["name"]) in recs)
            for a in seeds for sc in scens for spec in conds]
    if args.workers > 1:
        with Pool(args.workers, maxtasksperchild=None) as pool:
            results = pool.map(run_job, jobs, chunksize=1)
    else:
        results = [run_job(j) for j in jobs]
    per_seed, demos = {}, {}
    for a, sc, name, res, d in results:
        per_seed.setdefault(str(a), {}).setdefault(sc, {})[name] = res
        if d:
            demos[(sc, name)] = d
    names = [c["name"] for c in conds]
    (out / "metrics.json").write_text(json.dumps({"phase": args.phase, "config": cfg["raw"], "seeds": seeds,
                                                  "scenarios": scens, "conditions": names,
                                                  "wall_seconds": time.time() - t0, "workers": args.workers,
                                                  "command": " ".join(["python scripts/run_redetect.py"] + sys.argv[1:]),
                                                  "per_seed": per_seed}, default=float))
    if args.phase == "tune":
        sel = select(per_seed, rd)
        (out / "selection.json").write_text(json.dumps(sel, indent=1))
        print(json.dumps(sel, indent=1))
    else:
        for sc in scens:
            d = out / sc
            d.mkdir()
            (d / "summary.json").write_text(json.dumps(aggregate(per_seed, sc, names), indent=1, default=float))
            plot(sc, rd["scenarios"][sc], per_seed, names, seeds, out)
        for (sc, name), d in demos.items():
            write_demo(cfg, rd, sc, name, d, per_seed, seeds, names, out / sc)
    print(f"total {time.time() - t0:.0f}s")


def write_demo(cfg, rd, scen_name, cond_name, demo, per_seed, seeds, names, out):
    """Dashboard log in the s6 format, for a recorded scenario/condition of the first seed."""
    run, rec = demo["run"], {"scenario": scen_name, "condition": cond_name}
    scen = rd["scenarios"][scen_name]
    s0 = per_seed[str(seeds[0])][scen_name]
    met = s0[cond_name]["metrics"]
    segs = scen["segments"]
    demo_json = {
        "description": f"{rec['condition']} in scenario {rec['scenario']}: real per-step log",
        "seed": seeds[0], "shift_step": run["changes"][0]["step"], "shift": scen["changes"][0]["dynamics"],
        "segments": segs, "calibration": demo["cal_stats"], "events": run["events"], "episodes": run["episodes"],
        "condition": rec["condition"], "adapt_settings": run["adapt"],
        "changes": [{**c, "detection_delay": met[f"{segs[i + 1]}.detection_delay"],
                     "first_finetune_delay": met[f"{segs[i + 1]}.first_finetune_delay"],
                     "first_episode_success": met[f"{segs[i + 1]}.first_episode_success"]}
                    for i, c in enumerate(run["changes"])],
        "comparison": {c: {"episodes": s0[c]["episodes"], "residual": s0[c]["residual"], "alarms": s0[c]["alarms"]}
                       for c in names},
        "phase_end_heldout": {c: s0[c]["phase_end_heldout"] for c in names},
        "heldout_initial": {k: {"h1": v} for k, v in demo["heldout_initial"].items()},
        "command": " ".join(["python scripts/run_redetect.py"] + sys.argv[1:]),
        "config": cfg["raw"], "steps": run["records"]}
    (out / "demo_run.json").write_text(json.dumps(demo_json, default=float))


if __name__ == "__main__":
    main()
