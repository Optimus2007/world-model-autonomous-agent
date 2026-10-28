# Audit, research gap and analysis plan

Written on 2026-10-28, after the repository audit and literature review, while the Step 8 runs were in progress.
At the time of writing, the only Step 8 results seen were from a one-seed pilot (seed 0, four scenarios, four
conditions). That pilot was used to check the pipeline and the exact reproduction of `results/s7_redetect`.
Nothing was changed after it.

## 1. Audit summary

- **Repository state.** Branch `main`; local `HEAD` and `origin/main` are both
  `8786b2b0d14d1de21b429b01fab716341f22207c`; the working tree was clean. The test suite gave 30 passed.
  Software: Python 3.13.7, torch 2.14.1, numpy 2.5.4, matplotlib 3.11.2, PyYAML 6.0.3.
- **Adaptation state machine:** `src/autonomous_agent/agent.py`, `WorldModelAgent`.
  - Main residual CUSUM: drift 0.5, threshold 10, on the log residual standardised by calibration statistics.
  - Re-detection CUSUM: drift 1.5, threshold 10. It is armed only in ADAPTING, that is after the window's
    first fine-tune, and is reset after every fine-tune.
  - Window: `min_new` 50, `every` 100, `max_new` 500, 40 epochs, lr 1e-3.
  - `max_redetections` 2.
- **Verified against saved data.** Every number reported for Steps 1 to 7 was recomputed from the saved
  `metrics.json` and `summary.json` files and matched:
  - M1 one-step MSE: model 1.40e-3, constant velocity 2.83e-3, zero delta 3.58e-3. Spearman correlation of
    disagreement with error: 0.461 on stationary test data, 0.292 after the s3 gain/drag shift.
  - Walls: 213 of 2000 test transitions (10.7%) are wall contacts and carry 85% of the squared error. The
    free-motion rule gives 1.2e-8, in a setting without process noise.
  - Planning: CEM-MPC on the learned model returned −31.8 vs −31.1 for tuned PD, both with 100% success;
    26.0 ms vs 0.0032 ms per decision.
  - The s4, s5, s6 and s7 tables quoted in the README.
- **Reproduction.** The `ignore` conditions of `results/s7_redetect` reproduce `results/s6_reversion`
  exactly. The Step 8 configuration reproduces `results/s7_redetect` seed 0 exactly (pilot).
- **Two definitions that must be stated precisely in the report:**
  - Under `ignore`, "detection delay" of a change inside an open window is the delay to the first alarm of
    any kind after the change. That alarm comes from the main detector only after the old window closes.
    It is not a re-detection.
  - Under `restart`, the rows between the true change and the re-detection are assigned to the "old" pool,
    so replay can draw a few post-change transitions as "old" data.

## 2. Research gap (from `related_work.md`)

Per-context model libraries (MBCD, MOLe, Gaussian-process mixtures) already handle recurring dynamics and
changes during learning. Discarding stale data on detected change and guarding detection with a warm-up are
established ideas (ADWIN, DDM, MBCD). The Step 7 restart policy is therefore not a new mechanism.

The remaining gap is narrow and empirical. A widely used simple pipeline (a single ensemble world model,
detector-triggered windowed fine-tuning, alarms suppressed while a window is open) fails in a specific way
when a second change lands inside the window. We want to quantify:

- how large that failure is;
- whether it is caused by alarm suppression or merely by window length;
- what restart costs (false restarts, forgetting, compute);
- how all of this interacts with replay.

## 3. Research questions, hypotheses and competing explanations

**RQ1. When a second change occurs inside an open adaptation window, does restarting the window on
re-detection improve control after that change, compared with ignoring the detection?**
- Outcome: per-seed paired difference in mean episode return over the segment after the second change
  (restart minus ignore), for fresh-only and for replay 0.25 separately.
- H1: the difference is positive in most seeds and scenarios.
- Null: no difference.
- **Competing explanation C1:** restart helps only because it ends the stale window sooner. Test: compare
  restart with `ignore_short` (ignore with `max_new` 150, i.e. two fine-tunes per window, chosen before any
  run as the shortest window with more than one update). If C1 holds, `ignore_short` matches restart in the
  second-change scenarios. Neither policy is expected to win everywhere: `ignore_short` also stops adapting
  earlier after a single change.

**RQ2. What does restart cost?**
- Outcomes:
  - restarts in the persistent (single-change) control: each is a false restart by construction;
  - number of fine-tunes and fine-tune seconds;
  - held-out one-step error on each earlier regime at the end of the run (forgetting);
  - return in the persistent control.
- H2: few false restarts and no measurable return cost in the control, at the price of more forgetting of
  the intermediate regime.

**RQ3. When does replay (ratio 0.25) improve whole-run return?**
- Outcome: per-seed paired difference replay minus fresh-only, under each interruption policy, per scenario.
- H3: positive when an earlier regime recurs (reversion); zero or negative when changes are permanent or lead
  to novel dynamics (persistent, second change).
- Competing explanation C3: replay helps whenever the second change moves the dynamics back towards the
  pretraining regime (−30° is closer to 0° than 60° is), whether or not the regime literally recurs. The
  `large_second` scenario (60° → −60°) partly separates the two explanations.

**RQ4. Is one fixed replay ratio adequate for both permanent and reversible changes?**
- Answered from existing data, with no new runs:
  - s5 grid (ratios 0, 0.25, 0.5) under a permanent change at 60° and 90°;
  - s6 and s7 under reversion.
- H4: no single ratio is best in both.

## 4. Experiments

| ID | Purpose | Seeds | Scenarios | Conditions | Output |
|---|---|---|---|---|---|
| Reused | Steps 1 to 7 | as published | as published | as published | `results/m1` … `results/s7_redetect` |
| E8a | Short-window competitor (C1) on the Step 7 seeds | 0–4 | persistent, reversion, second_change | fresh/replay_ignore_short | `results/s8_short_s0_4` |
| E8b | More evaluation seeds for RQ1 to RQ3 | 5–9 | persistent, reversion, second_change | all 9 | `results/s8_seeds5_9` |
| E8c | Robustness: scenarios never used for tuning | 0–9 | late_second (2nd change at episode 15), large_second (60° → −60°), gain_second (gain 1 → 0.5 at 60°) | none, fresh/replay × ignore/restart/ignore_short | `results/s8_newscen` |

All settings are fixed in `configs/redetect60_ext.yaml`, and nothing is tuned on evaluation seeds. Oracle
conditions are references only and are told the change episodes.

## 5. Analysis rules

- **The unit of analysis is the agent seed.** Time steps and episodes within a run are never treated as
  independent samples.
- **Paired differences** use the same seed. Within a seed, all conditions share the pretrained model,
  calibration, held-out sets and episode reset seeds.
- **Reported per comparison:**
  - per-seed values;
  - mean and standard deviation;
  - the number of seeds with a positive difference;
  - an exact two-sided sign test;
  - a 95% percentile bootstrap interval over seeds (10,000 resamples).

  With 10 seeds these are descriptive. They are not corrected for multiple comparisons, and they do not
  support claims beyond this environment.
- **Time points are kept distinct:** true change step, alarm (re-detection) step, window start, first
  fine-tune on post-change data only ("clean fine-tune"), the window's last fine-tune, and recovery (the
  first of three consecutive successful episodes after the change).
- **Leakage check.** Held-out sets are random-policy transitions from reset seeds disjoint from training,
  calibration and episode seeds. They are only ever passed to `model.mse`. Online residuals are scored
  before the model trains on the transition.
