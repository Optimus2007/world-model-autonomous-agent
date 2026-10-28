"""Adaptation speed vs. forgetting: sweep adapt.min_new and adapt.replay_ratio, select on tuning seeds,
evaluate the selection on held-out seeds.

Everything except the adaptation settings is taken from the config unchanged (environment, reward,
shift and its timing, planner, model, detector and its calibration procedure). Per agent seed the
pretraining, calibration and held-out data use exactly the code path of run_shift.py, so evaluation
seeds reproduce the pretrained models of results/s4_rotation*.

Phases
  tune  run every (min_new, replay_ratio) pair plus a no-adaptation control on `tune_seeds`, then apply
        the selection rule below and write selected.json.
  eval  run on `eval_seeds`: no adaptation, the previous setting (min_new 100, ratio 0), the 2x2
        factorial {previous, selected min_new} x {0, selected ratio}, and every grid replay ratio at the
        selected min_new.

Selection rule (fixed before any sweep result was seen):
  1. highest mean post-change success rate over tuning seeds;
  2. among those, configurations whose mean post-change return is within `return_tolerance` of the best;
  3. among those, lowest mean held-out one-step error on the ORIGINAL dynamics (forgetting);
  4. remaining ties: smaller min_new, then smaller replay ratio.

Replay data for a fine-tune comes only from what the agent held before the alarm (its pretraining set
and its own pre-alarm transitions; see WorldModelAgent.replay_sample). Held-out sets use reset seeds
that no agent ever trains on and are used only for measurement.

Usage:
  python scripts/run_adapt_sweep.py --phase tune --config configs/adapt60.yaml --out results/s5_adapt_tune60
  python scripts/run_adapt_sweep.py --phase eval --config configs/adapt60.yaml --out results/s5_adapt_eval60 \
      --selected results/s5_adapt_tune60/selected.json
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_shift import recovery, run_condition  # noqa: E402

from autonomous_agent.agent import AdaptConfig, WorldModelAgent, run_episode  # noqa: E402
from autonomous_agent.config import load_config  # noqa: E402
from autonomous_agent.data import ReplayBuffer, collect, random_policy  # noqa: E402
from autonomous_agent.env import PointMassEnv  # noqa: E402
from autonomous_agent.evaluate import multi_step_report  # noqa: E402
from autonomous_agent.world_model import EnsembleWorldModel  # noqa: E402


def reset_seeds(aseed, cfg):
    """Every environment reset seed an agent seed touches (pretraining, calibration, episodes, held-out)."""
    sx = cfg["shift_experiment"]
    return (set(range(1_000 * (aseed + 1), 1_000 * (aseed + 1) + cfg["data"]["pretrain_episodes"]))
            | set(range(7_000 * (aseed + 1), 7_000 * (aseed + 1) + sx["calib_episodes"]))
            | set(range(4_000 * (aseed + 1), 4_000 * (aseed + 1) + sx["episodes"]))
            | set(range(9_000 * (aseed + 1), 9_000 * (aseed + 1) + 10)))


def prepare(cfg, aseed):
    """Pretrain, calibrate and build held-out sets exactly as run_shift.py does."""
    env_cfg, sx = cfg["env"], cfg["shift_experiment"]
    pre = collect(PointMassEnv(env_cfg, seed=aseed), random_policy(np.random.default_rng(aseed)),
                  cfg["data"]["pretrain_episodes"], ReplayBuffer(), seed=1_000 * (aseed + 1)).arrays()
    model = EnsembleWorldModel(config=cfg["model"], seed=aseed)
    model.fit(pre["state"], pre["action"], pre["next_state"])
    cal_agent = WorldModelAgent(copy.deepcopy(model), cfg["agent"], seed=aseed)
    res, dis = [], []
    for i in range(sx["calib_episodes"]):
        s = 7_000 * (aseed + 1) + i
        tr = run_episode(PointMassEnv(env_cfg, seed=s), cal_agent, seed=s, episode=-1 - i)["trace"]
        res += tr["residual"]
        dis += tr["disagreement"]
    cal_stats = cal_agent.calibrate(np.array(res), np.array(dis))
    heldout = {}
    for name, dyn_change in (("old_dynamics", {}), ("new_dynamics", cfg["shift"])):
        env = PointMassEnv(env_cfg, seed=aseed)
        env.set_dynamics(**dyn_change)
        heldout[name] = collect(env, random_policy(np.random.default_rng(90_000 + aseed)), 10,
                                ReplayBuffer(), seed=9_000 * (aseed + 1)).arrays()
    ep_seeds = [4_000 * (aseed + 1) + i for i in range(sx["episodes"])]
    return model, pre, cal_stats, heldout, ep_seeds


def cond_name(min_new, ratio):
    return "none" if min_new is None else f"adapt_m{min_new}_r{ratio:g}"


def run_metrics(run, heldout, seed):
    eps, shift = run["episodes"], run["shift_step"]
    pre = [e for e in eps if not e["post_shift"]]
    post = [e for e in eps if e["post_shift"]]
    fts = [e for e in run["events"] if e.get("type") == "finetune"]
    first_ft = next((e for e in fts if e["step"] >= shift), None)
    k = next((i for i, e in enumerate(post) if e["success"]), None)
    m = run["model"]
    final = {n: {"h1": m.mse(d["state"], d["action"], d["next_state"]),
                 "h10": multi_step_report(m, d, horizons=(10,), seed=seed)["h10"]["mse_mean"]}
             for n, d in heldout.items()}
    return {
        "pre_shift_success": float(np.mean([e["success"] for e in pre])),
        "post_shift_success": float(np.mean([e["success"] for e in post])),
        "post_shift_return": float(np.mean([e["return"] for e in post])),
        "first_post_success": float(post[0]["success"]), "first_post_return": float(post[0]["return"]),
        "return_after_first_post": float(np.mean([e["return"] for e in post[1:]])),
        "online_detection_delay": next((x - shift for x in run["alarms"] if x >= shift), None),
        "online_false_alarms": len([x for x in run["alarms"] if x < shift]),
        "first_adapt_delay": (first_ft["step"] - shift) if first_ft else None,
        "episodes_to_first_success": k,
        "steps_to_first_success": int(sum(e["steps"] for e in post[: k + 1])) if k is not None else None,
        **recovery(post),
        "n_finetunes": len(fts), "finetune_seconds_total": float(sum(e["seconds"] for e in fts)),
        "n_replay_total": int(sum(e.get("n_replay", 0) for e in fts)),
        "decision_ms": float(np.mean([e["mean_plan_ms"] for e in eps])),
        "finetunes": [{k2: e.get(k2) for k2 in ("step", "n_fresh", "n_replay", "seconds", "finetune_mse",
                                                 "heldout_before", "heldout_after")} for e in fts],
        "final_heldout": final,
    }


def run_seed(job):
    cfg_path, aseed, conds, record_cond = job
    import torch
    torch.set_num_threads(1)
    cfg = load_config(cfg_path)
    model, pre, cal_stats, heldout, ep_seeds = prepare(cfg, aseed)
    sx = cfg["shift_experiment"]
    out, demo = {}, None
    for min_new, ratio in conds:
        name = cond_name(min_new, ratio)
        if min_new is None:
            c = cfg
        else:
            adapt = replace(cfg["agent"].adapt, min_new=min_new, replay_ratio=ratio)
            c = {**cfg, "agent": replace(cfg["agent"], adapt=adapt)}
        run = run_condition("none" if min_new is None else "residual", model, cal_stats, c, ep_seeds,
                            sx["shift_episode"], aseed, record_demo=(name == record_cond),
                            prior_data=pre, eval_sets=heldout)
        met = run_metrics(run, heldout, aseed)
        met["episodes"] = [{k: e[k] for k in ("episode", "return", "steps", "success", "post_shift")}
                           for e in run["episodes"]]
        met["residual"] = run["trace"]["residual"]
        met["alarms"], met["shift_step"] = run["alarms"], run["shift_step"]
        met["adapt_settings"] = None if min_new is None else asdict(c["agent"].adapt)
        out[name] = met
        if name == record_cond:
            demo = {"run": run, "adapt": asdict(c["agent"].adapt), "cal_stats": cal_stats}
        print(f"seed {aseed} {name}: post success {met['post_shift_success']:.2f}, first post "
              f"{met['first_post_success']:.0f}, old-dyn h1 {met['final_heldout']['old_dynamics']['h1']:.2e}",
              flush=True)
    init = {n: {"h1": model.mse(d["state"], d["action"], d["next_state"])} for n, d in heldout.items()}
    return aseed, {"conditions": out, "pretrained_heldout": init}, demo


METRICS = ("pre_shift_success", "post_shift_success", "post_shift_return", "first_post_success",
           "first_post_return", "return_after_first_post", "online_detection_delay", "online_false_alarms",
           "first_adapt_delay", "episodes_to_first_success", "steps_to_first_success", "episodes_to_recover",
           "steps_to_recover", "n_finetunes", "finetune_seconds_total", "n_replay_total", "decision_ms")


def aggregate(per_seed):
    def agg(vals):
        num = [v for v in vals if v is not None]
        return {"values": vals, "mean": float(np.mean(num)) if num else None,
                "std": float(np.std(num)) if num else None}
    names = list(next(iter(per_seed.values()))["conditions"])
    summary = {}
    for c in names:
        rows = [p["conditions"][c] for p in per_seed.values()]
        for k in METRICS:
            summary[f"{c}.{k}"] = agg([r[k] for r in rows])
        for d, alias in (("old_dynamics", "pre_shift"), ("new_dynamics", "post_shift")):
            for h in ("h1", "h10"):
                summary[f"model_eval.{c}.{alias}.{h}"] = agg([r["final_heldout"][d][h] for r in rows])
    for d, alias in (("old_dynamics", "pre_shift"), ("new_dynamics", "post_shift")):
        summary[f"model_eval.pretrained.{alias}.h1"] = agg([p["pretrained_heldout"][d]["h1"] for p in per_seed.values()])
    return summary


def select(summary, grid, tol):
    """Apply the fixed selection rule; returns (min_new, ratio) and the ranking table."""
    rows = []
    for m, r in grid:
        c = cond_name(m, r)
        rows.append({"min_new": m, "replay_ratio": r, "success": summary[f"{c}.post_shift_success"]["mean"],
                     "return": summary[f"{c}.post_shift_return"]["mean"],
                     "old_h1": summary[f"model_eval.{c}.pre_shift.h1"]["mean"],
                     "new_h1": summary[f"model_eval.{c}.post_shift.h1"]["mean"],
                     "first_post_success": summary[f"{c}.first_post_success"]["mean"]})
    best_s = max(r["success"] for r in rows)
    cand = [r for r in rows if r["success"] >= best_s - 1e-9]
    best_r = max(r["return"] for r in cand)
    cand = [r for r in cand if r["return"] >= best_r - tol]
    pick = min(cand, key=lambda r: (r["old_h1"], r["min_new"], r["replay_ratio"]))
    return (pick["min_new"], pick["replay_ratio"]), {"rows": rows, "candidates": cand, "selected": pick,
                                                     "best_success": best_s, "best_return": best_r}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--phase", choices=["tune", "eval"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--selected", help="selected.json from the tune phase (eval phase)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", help="override seeds (pilot only)")
    p.add_argument("--episodes", type=int, help="override episodes per run (pilot only)")
    args = p.parse_args()
    cfg = load_config(args.config)
    sw = cfg["raw"]["adapt_sweep"]
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; results are never overwritten")
    out.mkdir(parents=True)
    cfg_path = args.config
    if args.episodes:  # pilot: write a shortened copy of the config next to the results
        import yaml
        raw = copy.deepcopy(cfg["raw"])
        raw["shift_experiment"]["episodes"] = args.episodes
        cfg_path = str(out / "pilot_config.yaml")
        Path(cfg_path).write_text(yaml.safe_dump(raw, sort_keys=False))
        cfg = load_config(cfg_path)

    tune_seeds, eval_seeds = sw["tune_seeds"], sw["eval_seeds"]
    overlap = set().union(*(reset_seeds(a, cfg) for a in tune_seeds)) & set().union(
        *(reset_seeds(a, cfg) for a in eval_seeds))
    assert not overlap, f"tuning and evaluation share environment seeds: {sorted(overlap)[:5]}"

    if args.phase == "tune":
        seeds = args.seeds or tune_seeds
        grid = [(m, r) for m in sw["min_new"] for r in sw["replay_ratio"]]
        conds, record = [(None, None)] + grid, None
        selected = None
    else:
        seeds = args.seeds or eval_seeds
        sel = json.loads(Path(args.selected).read_text())["selected"]
        m_s, r_s = sel["min_new"], sel["replay_ratio"]
        # Factorial around the selection, plus every grid replay ratio at the selected min_new so the
        # replay comparison is always measured on held-out seeds, whatever was selected.
        conds = ([(None, None), (100, 0.0), (m_s, 0.0), (100, r_s), (m_s, r_s)]
                 + [(m_s, r) for r in sw["replay_ratio"]])
        conds = list(dict.fromkeys(conds))  # drop duplicates if the selection equals the old setting
        record, selected = cond_name(m_s, r_s), sel
    t0 = time.time()
    jobs = [(cfg_path, a, conds, record if a == seeds[0] else None) for a in seeds]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(run_seed, jobs)
    else:
        results = [run_seed(j) for j in jobs]
    per_seed = {str(a): r for a, r, _ in results}
    summary = aggregate(per_seed)
    result = {"phase": args.phase, "config": cfg["raw"], "seeds": seeds, "conditions": [cond_name(*c) for c in conds],
              "wall_seconds": time.time() - t0, "workers": args.workers, "per_seed": per_seed}
    if args.phase == "tune":
        sel, table = select(summary, grid, sw["return_tolerance"])
        result["selection"] = table
        (out / "selected.json").write_text(json.dumps({"selected": table["selected"], "rule": __doc__.split(
            "Selection rule")[1].split("Replay data")[0].strip(), "table": table}, indent=1, default=float))
        print("selected", sel)
    else:
        result["selected"] = selected
        demo = next(d for _, _, d in results if d)
        run = demo["run"]
        comparison = {c: {"episodes": per_seed[str(seeds[0])]["conditions"][c]["episodes"],
                          "residual": per_seed[str(seeds[0])]["conditions"][c]["residual"],
                          "alarms": per_seed[str(seeds[0])]["conditions"][c]["alarms"]}
                      for c in result["conditions"]}
        rec_met = per_seed[str(seeds[0])]["conditions"][record]
        first_ok = rec_met["episodes_to_first_success"]
        demo_json = {
            "description": f"{record}: residual detection + adaptation, real per-step log", "seed": seeds[0],
            "shift_step": run["shift_step"], "shift": cfg["shift"], "calibration": demo["cal_stats"],
            "events": run["events"], "episodes": run["episodes"], "comparison": comparison,
            "adapt_settings": demo["adapt"], "condition": record,
            "timeline": {"change_step": run["shift_step"],
                         "detection_step": next((x for x in run["alarms"] if x >= run["shift_step"]), None),
                         "first_adapt_step": next((e["step"] for e in run["events"]
                                                   if e.get("type") == "finetune" and e["step"] >= run["shift_step"]), None),
                         "first_successful_post_episode": (cfg["shift_experiment"]["shift_episode"] + first_ok
                                                           if first_ok is not None else None)},
            "heldout_initial": per_seed[str(seeds[0])]["pretrained_heldout"],
            "command": " ".join(["python scripts/run_adapt_sweep.py"] + sys.argv[1:]),
            "config": cfg["raw"], "steps": run["records"]}
        (out / "demo_run.json").write_text(json.dumps(demo_json, default=float))
    (out / "metrics.json").write_text(json.dumps(result, default=float))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
