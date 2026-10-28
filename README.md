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
| `agent.py` | The loop above. One-sided CUSUM detectors on the log residual and on the log ensemble disagreement, calibrated on episodes the model never trains on. Adaptation schedule: `min_new`, `every`, `max_new`, `epochs`, `replay_ratio`. Replay draws only from data held before the alarm. A small adaptation state machine (NORMAL → COLLECTING → ADAPTING → NORMAL) decides what happens when the residual rises again during an open window (`on_alarm_during_adapt`: `ignore` or `restart`; Step 7). |
| `controllers.py` | PD controller (tuned on separate episodes) and a random controller. |
| `schedule.py` | Dynamics schedule over episodes: original → changed → reverted, or any list of changes. |
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

Run these from the project root with the virtual environment active. `run_adapt_sweep.py`,
`run_reversion.py` and `run_redetect.py` refuse to overwrite an existing output directory. The earlier scripts write into their
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
| 7a: tune re-detection (seeds 100–102) | `python scripts/run_redetect.py --phase tune --config configs/redetect60.yaml --out results/s7_redetect_tune --workers 7` | `results/s7_redetect_tune` (`selection.json`) |
| 7b: change during adaptation (seeds 0–4) | `python scripts/run_redetect.py --phase eval --config configs/redetect60.yaml --out results/s7_redetect --workers 7` | `results/s7_redetect` |

Seeds are fixed in the configs. Pretraining, calibration, evaluation, tuning and held-out reset seeds
come from separate ranges, and `run_adapt_sweep.py` asserts that tuning and evaluation share no environment
seed. Three cross-checks reproduce exactly (identical episode returns): Step 5's old setting reproduces
Step 4, episodes 0–19 of Step 6 reproduce Step 5, and Step 7's `ignore` conditions in the reversion
scenario reproduce Step 6.

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

**Step 7, a change during an active adaptation window.** Before Step 7, an alarm raised during an open
window was discarded, and the window kept fine-tuning on a mix of old- and new-dynamics data until it closed.
The agent now runs a small state machine:

```
NORMAL --alarm--> COLLECTING --min_new rows: fine-tune--> ADAPTING --max_new rows--> NORMAL
                                                             |  persistent residual rise ("restart" policy)
                                                             +--> new window: COLLECTING
```

- **Re-detection is armed only in ADAPTING**, i.e. after the window's first fine-tune. Before that the
  model is unchanged and a high residual is expected. This is the cooldown: at least `min_new` rows and
  one fine-tune separate two (re)starts.
- **A separate CUSUM detects the new rise.** It uses the same calibrated residual scale, a larger drift
  and its own threshold, and resets after each fine-tune. The main detector (drift 0.5) fires repeatedly
  during a normal adaptation without any new change: 414 of 816 steps in the s6 seed-0 fresh-only run.
- **Policy on re-detection (`restart`):** close the current window, so its rows become old data (and
  part of the replay pool when `replay_ratio` > 0). Then open a new window that collects only
  post-re-detection rows. A merged window would mix two dynamics, which is what the legacy behaviour did.
- **Retraining is bounded.** At most `max_redetections` (2) restarts happen per chain of windows. Further
  firings are logged in `ignored_redetections` and not acted on. With `ignore`, firings are logged only,
  and the behaviour is bit-identical to Steps 1–6.
- **The re-detection settings were selected on tuning seeds 100–102** with a rule fixed beforehand:
  fewest restarts in a no-second-change control, then best return when a second change occurs. The
  selection was drift 1.5, threshold 10, with 1 false restart in 6 control runs. Drift 1.0 caused 6–9
  false restarts.

Evaluation on seeds 0–4 used three scenarios. "Persistent" (0° → 60° at episode 8) is the control:
any restart there happens without a new change. "Reversion" is the s6 schedule. "Second change"
(0° → 60° at episode 8 → −30° at episode 11) puts the second change inside the first change's window.

| | Fresh, ignore | Fresh, restart | Replay, ignore | Replay, restart | Oracle fresh / replay |
|---|---|---|---|---|---|
| Persistent: whole-run return | −27.5 | −27.5 | −28.0 | −28.1 | −27.3 / −27.9 |
| Persistent: restarts without a new change (total, 5 seeds) | — | 0 | — | 3 (2 seeds) | — |
| Reversion: detection delay after reversion (steps) | 238 | **7.8** | 210, 4/5 seeds never | **41.8**, 1/5 never | 0 |
| Reversion: return after reversion | −32.4 | **−27.6** | −25.5 | **−23.9** | −26.5 / −24.1 |
| Reversion: 1st episode after reversion succeeded | 3/5 | 5/5 | 5/5 | 5/5 | 5/5 / 5/5 |
| Reversion: whole-run return | −30.7 | −29.1 | −28.7 | **−28.2** | −28.6 / −28.1 |
| Second change: detection delay (steps) | 457 | **6.4** | 455, 2/5 seeds never | **8.2** | 0 |
| Second change: first fine-tune on post-change data only (steps) | 506 | 55.4 | 490, 3/5 never | 57.2 | 49 |
| Second change: return after it | −35.8 | **−29.0** | −30.3 | **−27.8** | −29.4 / −27.8 |
| Second change: whole-run return | −34.0 | −29.7 | −30.6 | **−29.0** | −29.8 / −28.8 |
| Second change: fine-tunes (fine-tune seconds) | 7.2 (2.1) | 5.8 (1.4) | 6.2 (2.3) | 5.6 (1.6) | 5.8 / 5.6 |

The held-out one-step MSE at the end of the run, original-dynamics error first:

| | Fresh, ignore | Fresh, restart | Replay, ignore | Replay, restart |
|---|---|---|---|---|
| Reversion: original / 60° | 7.3e-4 / 1.2e-3 | 3.5e-4 / 1.7e-3 | 4.9e-4 / 1.1e-3 | 3.4e-4 / 1.6e-3 |
| Second change: −30° / 60° / original | 5.6e-4 / 3.1e-3 / 9.9e-4 | 2.6e-4 / 3.1e-3 / 6.8e-4 | 5.4e-4 / 2.0e-3 / 5.6e-4 | 3.3e-4 / 2.3e-3 / 4.6e-4 |

- **Paired by seed, restart minus ignore.**
  - Reversion, fresh: +4.8 return after the reversion (5/5 seeds).
  - Reversion, replay: +1.6 (positive in 4 seeds; −0.1 in one).
  - Second change, fresh: +6.7 return after the second change (5/5 seeds).
  - Second change, replay: +2.5 (5/5 seeds).
  - The phase before the second event was unchanged, apart from −0.2 in the seeds with a false restart.
- **Cost.** Restart used fewer fine-tunes than ignore. The ignore agents re-alarm once their window closes
  and open another window. Restart forgets the intermediate dynamics more: higher 60° error after the
  reversion, because the restarted window trains on post-change data only.
- **False restarts:** 0 for fresh-only. For replay, there were 3 in 2 of 5 control seeds (−0.2 return
  in those seeds). One replay seed in the reversion also restarted before the reversion, then
  re-detected the reversion itself.
- **Not every result favours adaptation.** −30° is close to the original dynamics, so the non-adapting
  agent did best after the second change (−27.6). The comparison of interest is restart vs ignore.

**Recommendation from these runs.** Use `on_alarm_during_adapt: restart` with re-detection drift 1.5,
threshold 10 and at most 2 restarts. It improved control after a change during an open window in every
paired comparison except one near-tie, at no cost in the control apart from a few replay false restarts.
For the replay ratio, the Step 6 advice holds: with restart, replay 0.25 gave the best whole-run return in
both multi-change scenarios. Fresh-only remains the choice if changes are known to be permanent and
new-dynamics error matters most.

## Dashboard

```
python -m http.server 8000                    # from the project root
open "http://localhost:8000/dashboard/?run=s7_redetect/second_change"    # add #step=N to jump to a step
```

Runs: `s3_shift`, `s4_rotation60`, `s4_rotation90`, `s5_adapt_eval60`, `s5_adapt_eval90`,
`s6_reversion`, `s7_redetect/second_change` and `s7_redetect/reversion`. The dashboard shows:
- the real trajectory, each ensemble member's predicted path and the planner's candidates
- issued vs effective (rotated) action and the current true rotation
- observed error vs uncertainty, the CUSUM statistic, alarms and fine-tunes
- markers for the change and the reversion or second change, and re-detections that restarted a window
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
- **Re-detection is tuned for this task.** Its drift and threshold were chosen on 3 tuning seeds for
  rotation changes. It stays disarmed until the window's first fine-tune, so a second change within
  `min_new` steps of the first is only caught after that fine-tune. The cap of 2 restarts per chain means
  a third change inside a chain is ignored until the window closes. The `ignore` policy remains the code
  default so that earlier runs reproduce; Step 7's configs set `restart`.
- **Held-out errors use random-policy transitions**, which cover a different region of state-action space
  from the planner's own data.
- **Uncertainty.** Ensemble disagreement is not a calibrated uncertainty, and it failed as a change signal
  for the gain/drag change.
- **Timing.** Decision and fine-tune times depend on machine load.
