"""Learned dynamics: an ensemble of MLPs predicting normalised state deltas.

Each member predicts  Δs = s' - s  from (s, a). Inputs and targets are
z-scored with statistics from the training set. Members differ by
initialisation and by bootstrap resampling of the data, which is what
makes their disagreement informative (Lakshminarayanan et al. 2017;
Chua et al. 2018, PETS). Disagreement is NOT a calibrated probability;
`evaluate.py` measures how well it tracks real error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class ModelConfig:
    n_members: int = 5
    hidden: int = 128
    layers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 60
    holdout_frac: float = 0.1
    bootstrap: bool = True
    # Collision handling (see apply_wall_rule). Both off reproduces the M1 model exactly.
    wall_clip_predictions: bool = False   # apply the simulator's wall rule to predictions
    train_exclude_walls: bool = False     # fit only free-motion transitions
    arena: float = 2.0


def is_wall_transition(next_states: np.ndarray, arena: float) -> np.ndarray:
    """True where the simulator clipped the position. The simulator writes exactly +-arena on a hit,
    which free motion reaches with probability zero, so this identifies hits exactly."""
    return (np.abs(next_states[..., :2]) >= arena).any(-1)


def apply_wall_rule(pred: np.ndarray, arena: float) -> np.ndarray:
    """The simulator's collision rule (env.step_dynamics) applied to an *unclipped* free-motion
    prediction: components beyond the wall are clipped and their velocity is set to zero."""
    pos, vel = pred[..., :2], pred[..., 2:]
    hit = np.abs(pos) > arena
    return np.concatenate([np.clip(pos, -arena, arena), np.where(hit, 0.0, vel)], -1)


class _Normaliser:
    def __init__(self, x: np.ndarray):
        self.mean = x.mean(0)
        self.std = x.std(0) + 1e-6

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, z):
        return z * self.std + self.mean


def _mlp(inp: int, out: int, hidden: int, layers: int) -> nn.Sequential:
    mods, d = [], inp
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out))
    return nn.Sequential(*mods)


class EnsembleWorldModel:
    def __init__(self, obs_dim: int = 4, act_dim: int = 2, config: ModelConfig | None = None, seed: int = 0):
        self.config = config or ModelConfig()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.seed = seed
        torch.manual_seed(seed)
        self.members = [_mlp(obs_dim + act_dim, obs_dim, self.config.hidden, self.config.layers)
                        for _ in range(self.config.n_members)]
        self.in_norm: _Normaliser | None = None
        self.out_norm: _Normaliser | None = None
        self.history: list[dict] = []  # one entry per fit() call

    # ---------- training ----------
    def fit(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray,
            epochs: int | None = None, rng: np.random.Generator | None = None) -> dict:
        """Train all members on the given transitions. Returns train / held-out MSE (raw units)."""
        cfg = self.config
        rng = rng or np.random.default_rng(self.seed)
        if cfg.train_exclude_walls:
            keep = ~is_wall_transition(next_states, cfg.arena)
            states, actions, next_states = states[keep], actions[keep], next_states[keep]
        x = np.concatenate([states, actions], 1)
        y = next_states - states
        n = len(x)
        perm = rng.permutation(n)
        n_hold = max(1, int(cfg.holdout_frac * n)) if n >= 20 else 0
        hold, train = perm[:n_hold], perm[n_hold:]

        # Normalisers are refit on each call so they follow the data distribution.
        self.in_norm, self.out_norm = _Normaliser(x[train]), _Normaliser(y[train])
        xt = torch.as_tensor(self.in_norm(x), dtype=torch.float32)
        yt = torch.as_tensor(self.out_norm(y), dtype=torch.float32)

        self._train_members(xt, yt, train, epochs or cfg.epochs, cfg.lr, rng)

        result = {"n_train": int(len(train)), "n_holdout": int(n_hold),
                  "train_mse": self.mse(states[train], actions[train], next_states[train])}
        result["holdout_mse"] = (self.mse(states[hold], actions[hold], next_states[hold])
                                 if n_hold else float("nan"))
        self.history.append(result)
        return result

    def finetune(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray,
                 epochs: int = 30, lr: float | None = None, rng: np.random.Generator | None = None) -> dict:
        """Continue training every member on new transitions (warm start).

        Normalisers are kept fixed so the network inputs keep their meaning.
        """
        if self.in_norm is None:
            raise RuntimeError("finetune() needs a fitted model")
        cfg = self.config
        rng = rng or np.random.default_rng(self.seed + len(self.history))
        if cfg.train_exclude_walls:
            keep = ~is_wall_transition(next_states, cfg.arena)
            states, actions, next_states = states[keep], actions[keep], next_states[keep]
        x = np.concatenate([states, actions], 1)
        xt = torch.as_tensor(self.in_norm(x), dtype=torch.float32)
        yt = torch.as_tensor(self.out_norm(next_states - states), dtype=torch.float32)
        self._train_members(xt, yt, np.arange(len(x)), epochs, lr or cfg.lr, rng)
        result = {"finetune_n": int(len(x)), "finetune_mse": self.mse(states, actions, next_states)}
        self.history.append(result)
        return result

    def _train_members(self, xt, yt, idx_pool, epochs, lr, rng) -> None:
        cfg = self.config
        for m in self.members:
            idx = rng.choice(idx_pool, size=len(idx_pool), replace=True) if cfg.bootstrap else idx_pool
            opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=cfg.weight_decay)
            m.train()
            for _ in range(epochs):
                order = rng.permutation(idx)
                for b in range(0, len(order), cfg.batch_size):
                    j = torch.as_tensor(order[b: b + cfg.batch_size])
                    loss = nn.functional.mse_loss(m(xt[j]), yt[j])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            m.eval()
        self._stack_params()

    def _stack_params(self) -> None:
        """Stack member weights so all members run in one batched matmul per layer."""
        linears = [[l for l in m if isinstance(l, nn.Linear)] for m in self.members]
        self._W = [torch.stack([ls[k].weight.detach().T for ls in linears]) for k in range(len(linears[0]))]
        self._b = [torch.stack([ls[k].bias.detach() for ls in linears])[:, None, :] for k in range(len(linears[0]))]
        f32 = lambda a: torch.as_tensor(a, dtype=torch.float32)
        self._in_mean, self._in_std = f32(self.in_norm.mean), f32(self.in_norm.std)
        self._out_mean, self._out_std = f32(self.out_norm.mean), f32(self.out_norm.std)

    def save(self, path: str) -> None:
        torch.save({"config": self.config.__dict__, "seed": self.seed,
                    "members": [m.state_dict() for m in self.members],
                    "in_norm": (self.in_norm.mean, self.in_norm.std),
                    "out_norm": (self.out_norm.mean, self.out_norm.std)}, path)

    @classmethod
    def load(cls, path: str) -> "EnsembleWorldModel":
        ck = torch.load(path, weights_only=False)
        m = cls(config=ModelConfig(**ck["config"]), seed=ck["seed"])
        for member, sd in zip(m.members, ck["members"]):
            member.load_state_dict(sd)
            member.eval()
        m.in_norm, m.out_norm = _Normaliser.__new__(_Normaliser), _Normaliser.__new__(_Normaliser)
        m.in_norm.mean, m.in_norm.std = ck["in_norm"]
        m.out_norm.mean, m.out_norm.std = ck["out_norm"]
        m._stack_params()
        return m

    # ---------- prediction ----------
    @torch.no_grad()
    def step_members(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Each member e advances its own batch: states (E, N, obs), actions (E, N, act) -> (E, N, obs)."""
        if self.in_norm is None:
            raise RuntimeError("World model has not been fit yet")
        x = torch.as_tensor(np.concatenate([states, actions], -1), dtype=torch.float32)
        h = (x - self._in_mean) / self._in_std
        for k, (W, b) in enumerate(zip(self._W, self._b)):
            h = torch.baddbmm(b, h, W)
            if k < len(self._W) - 1:
                h = nn.functional.silu(h)
        out = np.asarray(states, dtype=np.float64) + (h * self._out_std + self._out_mean).numpy()
        if self.config.wall_clip_predictions:
            out = apply_wall_rule(out, self.config.arena)
        return out

    def predict_members(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Next-state prediction of every member from the same inputs: shape (E, N, obs_dim)."""
        states, actions = np.atleast_2d(states), np.atleast_2d(actions)
        E = len(self.members)
        return self.step_members(np.broadcast_to(states, (E, *states.shape)),
                                 np.broadcast_to(actions, (E, *actions.shape)))

    def predict(self, states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ensemble mean next state (N, obs_dim) and disagreement u(s,a) (N,).

        u = mean over members and state dims of squared deviation from the ensemble mean.
        """
        preds = self.predict_members(states, actions)
        mean = preds.mean(0)
        disagreement = ((preds - mean) ** 2).mean(axis=(0, 2))
        return mean, disagreement

    def mse(self, states, actions, next_states) -> float:
        mean, _ = self.predict(states, actions)
        return float(np.mean((mean - next_states) ** 2))

    def rollout(self, state: np.ndarray, action_seqs: np.ndarray) -> np.ndarray:
        """Open-loop multi-step prediction, each member propagating its own trajectory.

        state (obs,), action_seqs (N, H, act) or (H, act) -> (E, N, H+1, obs).
        """
        seqs = action_seqs[None] if action_seqs.ndim == 2 else action_seqs
        E, (N, H, _) = len(self.members), seqs.shape
        cur = np.broadcast_to(state, (E, N, self.obs_dim)).copy()
        out = [cur]
        for h in range(H):
            cur = self.step_members(cur, np.broadcast_to(seqs[:, h], (E, N, self.act_dim)))
            out.append(cur)
        return np.stack(out, 2)


# ---------- trivial baselines (no learning) ----------
def zero_delta_baseline(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Predict s' = s."""
    return np.asarray(states).copy()


def constant_velocity_baseline(states: np.ndarray, actions: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """Predict p' = p + dt*v, v' = v (ignores the action and drag)."""
    s = np.asarray(states).copy()
    s[..., :2] += dt * s[..., 2:]
    return s
