"""World-model evaluation: one-step error, multi-step error, uncertainty-vs-error."""
from __future__ import annotations

import numpy as np

from .world_model import EnsembleWorldModel, constant_velocity_baseline, zero_delta_baseline


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation without SciPy (ties broken by order; fine for continuous data)."""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def one_step_report(model: EnsembleWorldModel, data: dict[str, np.ndarray], dt: float) -> dict:
    s, a, s2 = data["state"], data["action"], data["next_state"]
    mean, u = model.predict(s, a)
    err = np.mean((mean - s2) ** 2, axis=1)  # per-transition squared error
    return {
        "n": int(len(s)),
        "mse_model": float(err.mean()),
        "mse_zero_delta": float(np.mean((zero_delta_baseline(s, a) - s2) ** 2)),
        "mse_const_velocity": float(np.mean((constant_velocity_baseline(s, a, dt) - s2) ** 2)),
        "mean_disagreement": float(u.mean()),
        "spearman_disagreement_vs_error": spearman(u, err),
        # Fraction of the top-10% largest errors that are also in the top-10% disagreement.
        "top10_overlap": _top_overlap(u, err, 0.1),
    }


def _top_overlap(u: np.ndarray, err: np.ndarray, q: float) -> float:
    k = max(1, int(q * len(u)))
    return len(set(np.argsort(-u)[:k]) & set(np.argsort(-err)[:k])) / k


def episode_slices(data: dict[str, np.ndarray]) -> list[np.ndarray]:
    ep = data["episode"]
    return [np.flatnonzero(ep == e) for e in np.unique(ep)]


def multi_step_report(model: EnsembleWorldModel, data: dict[str, np.ndarray],
                      horizons=(1, 5, 10, 20), starts_per_episode: int = 3,
                      seed: int = 0) -> dict:
    """Open-loop rollout error of the ensemble mean vs. the real trajectory at several horizons."""
    rng = np.random.default_rng(seed)
    H = max(horizons)
    errs = {h: [] for h in horizons}
    for idx in episode_slices(data):
        if len(idx) <= H:
            continue
        for t0 in rng.choice(len(idx) - H, size=min(starts_per_episode, len(idx) - H), replace=False):
            rows = idx[t0: t0 + H]
            pred = model.rollout(data["state"][rows[0]], data["action"][rows])[:, 0].mean(0)  # (H+1, obs)
            true = data["next_state"][rows]                                                 # (H, obs)
            for h in horizons:
                errs[h].append(float(np.mean((pred[h] - true[h - 1]) ** 2)))
    return {f"h{h}": {"mse_mean": float(np.mean(v)), "mse_median": float(np.median(v)), "n": len(v)}
            for h, v in errs.items() if v}
