"""Controllers that do not use a learned world model."""
from __future__ import annotations

import numpy as np


class PDController:
    """a = clip(kp * (goal - pos) - kd * vel). Uses no model, only position error and velocity."""

    def __init__(self, kp: float = 2.0, kd: float = 1.0):
        self.kp, self.kd = kp, kd

    def reset(self) -> None:
        pass

    def act(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.clip(self.kp * (goal - state[:2]) - self.kd * state[2:], -1.0, 1.0)


class RandomController:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-1, 1, 2)
