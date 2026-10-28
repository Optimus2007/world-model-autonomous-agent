# World-model-guided autonomous agent

A small, fully reproducible research codebase. An agent in a 2-D continuous point-mass world:

- learns an ensemble dynamics model from its own experience
- plans with that model (CEM model-predictive control)
- watches its own prediction errors to detect when the environment's dynamics change
- adapts the model on fresh experience, optionally replaying old data to limit forgetting

Every number below was measured by the scripts in this repository and is stored under `results/`.
Nothing in the dashboard is simulated or hand-entered.

## The closed loop

```
observe s ─► plan with the model (CEM over ensemble rollouts) ─► execute the first action
   ▲                                                                   │
   │                                                                   ▼
update model ◄── fine-tune on fresh (+ replayed) data ◄── alarm? ◄── compare the prediction with the real s'
   (only after an alarm)                                (CUSUM on residual)   (residual scored BEFORE training on it)
```

## Architecture

| Module | Role |
|---|---|
| `src/autonomous_agent/env.py` | Point mass: v' = v + dt(gain·R(φ)·a − drag·v), p' = p + dt·v'. The arena walls clip position and zero the normal velocity. Optional velocity noise. Optional actuator rotation R(φ) applied to the issued action (default 0°). Dynamics can be changed mid-run with `set_dynamics`. |
| `world_model.py` | Ensemble of 5 MLPs predicting normalised state deltas, trained on bootstrap resamples; batched member forward pass. The exact simulator wall rule is applied to predictions, and the model is fit on free-motion transitions (Step 1). `finetune()` keeps the normalisers fixed. Checkpoint save/load. |
| `planner.py` | CEM-MPC with PETS-style trajectory sampling (each member rolls out each candidate); executes the first action and replans with a warm start. |
| `agent.py` | The loop above. One-sided CUSUM detectors on the log residual and on the log ensemble disagreement, calibrated on episodes the model never trains on. Adaptation schedule: `min_new`, `every`, `max_new`, `epochs`, `replay_ratio`. Replay draws only from data held before the alarm. |
| `controllers.py` | PD controller (tuned on separate episodes) and a random controller. |
| `schedule.py` | Dynamics schedule over episodes: original → changed → reverted. |
| `evaluate.py` | One-step and multi-step (open-loop) error, trivial baselines, Spearman correlation of uncertainty against error. |
| `config.py` | YAML to dataclasses; unknown keys raise an error. |
| `scripts/` | One script per experiment (see the table below). |
| `dashboard/` | A static page that replays the real per-step logs (`results/<run>/demo_run.json`). |

## Setup

Tested on macOS (Apple M2), Python 3.13.7, torch 2.14.1, numpy 2.5.4, matplotlib 3.11.2,
PyYAML 6.0.3 and pytest 9.1.1. Only CPU is used.

```
python3 -m venv .venv
.venv/bin/pip install torch numpy matplotlib pyyaml pytest
.venv/bin/pip install -e .
.venv/bin/python -m pytest -q          # 23 tests
```

Runtime dependencies are `numpy`, `torch`, `matplotlib` and `pyyaml` (see `pyproject.toml`); `pytest` is
needed only for the tests. The dashboard needs only a browser and Python's built-in `http.server`.

## Experiments

Run these from the project root with the virtual environment active. `run_adapt_sweep.py` and
`run_reversion.py` refuse to overwrite an existing output directory. The earlier scripts write into their
`--out` directory, so give them a new path to keep published results intact.

| Step | Command | Output |
|---|---|---|
| M1: model vs trivial baselines | `python scripts/train_model.py --config configs/default.yaml --out results/m1` | `results/m1` |
| 1: collision handling | `python scripts/compare_wall_handling.py` | `results/s1_walls` |
| 2: planning vs PD / random | `python scripts/run_planning.py --config configs/agent.yaml --out results/s2_planning` | `results/s2_planning` |
| 3: gain/drag change, detection, adaptation | `python scripts/run_shift.py --config configs/agent.yaml --out results/s3_shift` | `results/s3_shift` |
| 4: actuator rotation 60° / 90° | `python scripts/run_shift.py --config configs/rotation60.yaml --out results/s4_rotation60` (and `rotation90`) | `results/s4_rotation60`, `results/s4_rotation90` |
| 5a: tune `min_new` × `replay_ratio` (seeds 100–102) | `python scripts/run_adapt_sweep.py --phase tune --config configs/adapt60.yaml --out results/s5_adapt_tune60 --workers 3` | `results/s5_adapt_tune60` (`selected.json`) |
| 5b: held-out evaluation (seeds 0–4) | `python scripts/run_adapt_sweep.py --phase eval --config configs/adapt60.yaml --out results/s5_adapt_eval60 --selected results/s5_adapt_tune60/selected.json --workers 5` (and `adapt90` → `s5_adapt_eval90`) | `results/s5_adapt_eval60`, `results/s5_adapt_eval90` |
| 6: reversion 0° → 60° → 0° | `python scripts/run_reversion.py --config configs/reversion60.yaml --out results/s6_reversion --workers 5` | `results/s6_reversion` |

Seeds are fixed in the configs. Pretraining, calibration, evaluation, tuning and held-out reset seeds
come from separate ranges, and `run_adapt_sweep.py` asserts that tuning and evaluation share no environment
seed. Three cross-checks reproduce exactly (identical episode returns): Step 5's old setting reproduces
Step 4, and episodes 0–19 of Step 6 reproduce Step 5.

## Results

All values are means over seeds (± sd where shown). "Error" is mean squared one-step prediction error.
Decision times depend on machine load (several runs used parallel workers), so compare them only within
one run.

**M1, world model.** One-step held-out error was 1.40e-3 for the model, against 3.58e-3 for zero-delta and
2.83e-3 for constant-velocity. Ensemble disagreement correlated with error at Spearman 0.46 on held-out data.
Under a dynamics change the members agreed while being wrong: disagreement *fell* while error rose.

**Step 1, walls.** Wall transitions were 11% of the data but 86% of the error. Applying the simulator's exact
wall rule and fitting on free motion cut one-step error from 1.42e-3 to about 1e-8. The dynamics are linear
apart from the walls, so this is easy.

**Step 2, planning.** Tuned PD (kp=4, kd=2) and CEM with the learned model both reached 100% success
(return −31.1 vs −31.8), at 0.003 ms vs about 26 ms per decision. The task did not separate the controllers.

**Step 3, gain 1→0.6, drag 0.5→1.5** (5 seeds):
- The residual CUSUM detected the change in 6.2 ± 1.3 steps, with 0 false alarms in 909 pre-change and
  3,264 no-change control steps.
- The disagreement CUSUM never detected it and raised 26 false alarms in the control runs.
- Adaptation brought the error back to its pre-change level, but every controller, PD included, still
  reached 100% success, so the task showed no benefit.

**Step 4, actuator rotation** (5 seeds):

| | 60° | 90° |
|---|---|---|
| Post-change success: planner without adaptation | 0% | 0% |
| Post-change success: PD with pre-change gains | 0% | 0% |
| Post-change success: residual detection + adaptation | 95.5% | 95.5% |
| Detection delay (steps) | 3.6 | 2.6 |

- PD fails because its loop becomes linearly unstable (spectral radius 1.003 at 60°).
- An oracle PD re-tuned on the changed dynamics reached 100% at 60° but only 31% at 90°.
- The first post-change episode always failed, because adaptation waited for 100 fresh transitions.

**Step 5, adaptation speed vs forgetting.** Selected on tuning seeds 100–102 by a rule fixed in advance:
`min_new` 50, replay ratio 0.25. Held-out seeds 0–4:

| 60° | 1st post-change episode | Post-change return | Error, new dynamics | Error, original dynamics |
|---|---|---|---|---|
| old setting (`min_new` 100, replay 0) | 0/5 | −28.6 | 2.5e-4 | 1.69e-3 |
| `min_new` 50, replay 0 | 5/5 | −27.3 | 2.3e-4 | 1.74e-3 |
| `min_new` 50, replay 0.25 | 5/5 | −28.0 | 6.4e-4 | 7.8e-4 |

- At 90°: fresh-only `min_new` 50 succeeded in the first post-change episode in 5/5 seeds; replay 0.25 in
  3/5 (its first fine-tune leaves more new-dynamics error).
- Replay roughly halves forgetting but raises error on the new dynamics.

**Step 6, reversion** (rotation 0° for episodes 0–7, 60° for 8–19, 0° for 20–29; 5 seeds):

| | No adaptation | Fresh-only (`min_new` 50) | Replay 0.25 (`min_new` 50) | Oracle, fresh | Oracle, replay |
|---|---|---|---|---|---|
| Success: original / changed / reverted | 100 / 0 / 100% | 100 / 100 / 96% | 100 / 100 / 100% | 100 / 100 / 100% | 100 / 100 / 100% |
| Return: changed phase | −90.6 | −31.1 | −31.9 | −30.7 | −31.5 |
| Return: reverted phase | −22.5 | −32.4 | **−25.5** | −26.5 | −24.1 |
| 1st episode after reversion: success / return | 5/5, −27 | 3/5, −78 | 5/5, −35 | 5/5, −64 | 5/5, −36 |
| Error on original dynamics at end of changed phase | 2.8e-4 | 1.67e-3 | 9.4e-4 | 1.63e-3 | 9.5e-4 |
| Whole-run mean return | −51.2 | −30.7 | **−28.7** | −28.6 | −28.1 |

- In paired comparisons, replay beat fresh-only after the reversion in all 5 seeds (+6.9 return on
  average; first reverted episode +27 to +66). It cost −0.8 return in the changed phase, also in all 5 seeds.
- Both adaptive agents were still inside their adaptation window when the reversion came (100% of seeds),
  so the detector's alarm was ignored and fine-tuning continued on mixed data. The oracles restart
  adaptation at each change; they are references, not deployable methods.
- The no-adaptation agent is best in the reverted phase, but only because its model never changed. It
  fails every episode of the changed phase.

**Recommendation from these runs.** If the original dynamics may return, use `min_new` 50 with replay
0.25: it gave the best whole-run return of the deployable methods, at a small cost while the dynamics stay
changed. If a change is known to be permanent, fresh-only `min_new` 50 adapts slightly better and, at 90°,
succeeded more often in the first post-change episode.

## Dashboard

```
python -m http.server 8000                    # from the project root
open "http://localhost:8000/dashboard/?run=s6_reversion"    # add #step=N to jump to a step
```

Runs: `s3_shift`, `s4_rotation60`, `s4_rotation90`, `s5_adapt_eval60`, `s5_adapt_eval90` and
`s6_reversion`. The dashboard shows:
- the real trajectory, each ensemble member's predicted path and the planner's candidates
- issued vs effective (rotated) action and the current true rotation
- observed error vs uncertainty, the CUSUM statistic, alarms and fine-tunes
- markers for the change and the reversion
- held-out original- and changed-dynamics error before and after every fine-tune
- per-episode returns for each condition, and tables of all-seed results read from `summary.json`

Design notes, which follow the Impresario skill's dashboard guidance, are in `dashboard/DIRECTION.md`.

## What is in `results/` and what is not

**Included:**
- `summary.json` and `metrics.json` (per-seed data behind every number)
- plots, run logs, `selected.json`
- the per-step `demo_run.json` replayed by the dashboard

**Not included** (in `.gitignore`):
- **Model checkpoints** (`results/*/checkpoints/*.pt`, about 25 MB). Every run is deterministic on CPU and
  regenerates them; the reproduction checks above match to the last digit.
- **Collected training data** (`*.npz`), regenerated by `train_model.py`.
- `results/m1_repro`, a duplicate of `results/m1` used once to check a refactor.

## Provenance of reused material

- `github.com/Optimus2007/physics-informed-world-models` (MIT): inspected (RK4 step, NaiveMLP, LatentODE,
  PINN, trajectory plots). No code was copied. Its "MLP vs. physics prior" comparison informed the
  trivial-baseline comparison in `evaluate.py`.
- `github.com/Optimus2007/rl-alignment-and-reward-hacking` (MIT): inspected (SB3 PPO, gym wrappers). Not
  used, because this project needs neither PPO nor gym.
- `github.com/Shushant17711/impresario` (MIT): design guidance only (`SKILL.md`, `references/colour.md`)
  for the dashboard. No code was copied.
- Methods, implemented from their published descriptions:
  - ensemble disagreement (Lakshminarayanan et al., 2017)
  - CEM-MPC with trajectory sampling (Chua et al., 2018, PETS)
  - CUSUM (Page, 1954)
  - experience replay against forgetting (a standard continual-learning baseline)

## Limitations

- **A single, simple task family.** Linear dynamics apart from the walls, low noise, a 2-D point mass.
  Changes are a gain/drag change or a pure actuator rotation, which an MLP learns easily.
- **Small samples.** 5 evaluation seeds and 3 tuning seeds. Many differences are within about 1 sd and
  are reported per seed rather than as significance tests. The selection rule's return tolerance (2.0)
  was a judgment call fixed before the sweep.
- **Alarms are ignored during an ongoing adaptation window.** At the reversion this delayed the
  detector-driven agents' response; only the oracles restarted.
- **Held-out errors use random-policy transitions**, which cover a different region of state-action space
  from the planner's own data.
- **Uncertainty.** Ensemble disagreement is not a calibrated uncertainty, and it failed as a change signal
  for the gain/drag change.
- **Timing.** Decision and fine-tune times depend on machine load.
