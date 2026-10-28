"""2-D point-mass navigation environment with configurable, switchable dynamics.

State  s = [x, y, vx, vy]          (float64, shape (4,))
Action a = [ax, ay] in [-1, 1]^2   (clipped)

Semi-implicit Euler with linear drag:
    v' = v + dt * (gain * R(rotation) a - drag * v)
    p' = p + dt * v'
R(rotation) is a fixed actuator rotation (default 0 degrees = identity). It acts on the issued
action after clipping, so the agent's commands keep their units but push in a rotated direction.
It is applied here only; the world model and planner see the issued action and must learn it.
Positions are confined to a square arena; hitting a wall zeroes the normal
velocity component. The goal is part of the task, not of the state, so the
world model only has to learn the physics.

The interface mirrors Gymnasium (reset -> (obs, info), step -> 5-tuple) without
depending on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class Dynamics:
    gain: float = 1.0   # acceleration per unit action
    drag: float = 0.5   # linear velocity damping coefficient
    dt: float = 0.1
    process_noise: float = 0.0  # std of Gaussian noise added to velocity each step (0 = deterministic)
    rotation_deg: float = 0.0   # actuator rotation applied to the issued action (counter-clockwise)


@dataclass(frozen=True)
class EnvConfig:
    arena: float = 2.0           # half-width of the square arena
    max_steps: int = 100
    goal_radius: float = 0.15    # success when within this distance of the goal
    action_cost: float = 0.01
    dynamics: Dynamics = field(default_factory=Dynamics)


def rotation_matrix(deg: float) -> np.ndarray:
    phi = np.deg2rad(deg)
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def effective_action(action: np.ndarray, dyn: Dynamics) -> np.ndarray:
    """Clipped issued action as the actuators apply it: R(rotation) @ a. Batched over leading dims."""
    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    if dyn.rotation_deg == 0.0:
        return action
    return action @ rotation_matrix(dyn.rotation_deg).T


def step_dynamics(state: np.ndarray, action: np.ndarray, dyn: Dynamics, arena: float,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Transition function. Works on a single state (4,) or a batch (N, 4).
    Noise is added only when dyn.process_noise > 0 and an rng is given."""
    state = np.asarray(state, dtype=np.float64)
    action = effective_action(action, dyn)
    pos, vel = state[..., :2], state[..., 2:]
    vel = vel + dyn.dt * (dyn.gain * action - dyn.drag * vel)
    if dyn.process_noise > 0 and rng is not None:
        vel = vel + dyn.process_noise * rng.standard_normal(vel.shape)
    pos = pos + dyn.dt * vel
    hit = np.abs(pos) > arena
    pos = np.clip(pos, -arena, arena)
    vel = np.where(hit, 0.0, vel)
    return np.concatenate([pos, vel], axis=-1)


def task_cost(state: np.ndarray, action: np.ndarray, goal: np.ndarray, action_cost: float) -> np.ndarray:
    """Per-step cost: distance to goal plus a small control penalty. Batched."""
    dist = np.linalg.norm(state[..., :2] - goal, axis=-1)
    return dist + action_cost * np.sum(np.square(action), axis=-1)


class PointMassEnv:
    obs_dim = 4
    act_dim = 2

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        self.config = config or EnvConfig()
        self.dynamics = self.config.dynamics
        self.rng = np.random.default_rng(seed)
        # Separate stream for process noise so noise does not change start/goal sampling.
        self.noise_rng = np.random.default_rng(None if seed is None else seed + 7_919)
        self.state = np.zeros(4)
        self.goal = np.zeros(2)
        self.t = 0

    def set_dynamics(self, **changes: float) -> Dynamics:
        """Change dynamics mid-run, e.g. env.set_dynamics(drag=1.5)."""
        self.dynamics = replace(self.dynamics, **changes)
        return self.dynamics

    def reset(self, seed: int | None = None, goal: np.ndarray | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.noise_rng = np.random.default_rng(seed + 7_919)
        a = self.config.arena * 0.8
        self.state = np.concatenate([self.rng.uniform(-a, a, 2), np.zeros(2)])
        if goal is None:
            # Keep start and goal at least 1.0 apart so episodes are not trivial.
            while True:
                goal = self.rng.uniform(-a, a, 2)
                if np.linalg.norm(goal - self.state[:2]) > 1.0:
                    break
        self.goal = np.asarray(goal, dtype=np.float64)
        self.t = 0
        return self.state.copy(), {"goal": self.goal.copy()}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.state = step_dynamics(self.state, action, self.dynamics, self.config.arena, self.noise_rng)
        self.t += 1
        cost = float(task_cost(self.state, action, self.goal, self.config.action_cost))
        dist = float(np.linalg.norm(self.state[:2] - self.goal))
        terminated = dist < self.config.goal_radius
        truncated = self.t >= self.config.max_steps
        info = {"goal": self.goal.copy(), "distance": dist, "success": terminated,
                "dynamics": self.dynamics, "effective_action": effective_action(action, self.dynamics)}
        return self.state.copy(), -cost, terminated, truncated, info
