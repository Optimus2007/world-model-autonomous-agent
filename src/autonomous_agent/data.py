"""Transition storage and experience collection."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .env import PointMassEnv

Policy = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (state, goal) -> action


class ReplayBuffer:
    """Append-only transition store. Every row is one real environment interaction."""

    fields = ("state", "action", "next_state", "reward", "done", "episode")

    def __init__(self, obs_dim: int = 4, act_dim: int = 2, capacity: int = 200_000):
        self.capacity = capacity
        self.state = np.zeros((capacity, obs_dim))
        self.action = np.zeros((capacity, act_dim))
        self.next_state = np.zeros((capacity, obs_dim))
        self.reward = np.zeros(capacity)
        self.done = np.zeros(capacity, dtype=bool)
        self.episode = np.zeros(capacity, dtype=np.int64)
        self.size = 0

    def add(self, state, action, next_state, reward, done, episode) -> None:
        if self.size >= self.capacity:
            raise RuntimeError("ReplayBuffer is full")
        i = self.size
        self.state[i], self.action[i], self.next_state[i] = state, action, next_state
        self.reward[i], self.done[i], self.episode[i] = reward, done, episode
        self.size += 1

    def __len__(self) -> int:
        return self.size

    def arrays(self) -> dict[str, np.ndarray]:
        return {f: getattr(self, f)[: self.size] for f in self.fields}

    def save(self, path: str) -> None:
        np.savez_compressed(path, **self.arrays())

    @classmethod
    def load(cls, path: str) -> "ReplayBuffer":
        data = np.load(path)
        buf = cls(data["state"].shape[1], data["action"].shape[1], capacity=max(len(data["state"]), 1) * 2)
        for f in cls.fields:
            getattr(buf, f)[: len(data[f])] = data[f]
        buf.size = len(data["state"])
        return buf


def random_policy(rng: np.random.Generator, smooth: float = 0.8) -> Policy:
    """Temporally correlated random actions, so random data covers non-trivial velocities."""
    prev = np.zeros(2)

    def act(state, goal):
        nonlocal prev
        prev = smooth * prev + (1 - smooth) * rng.uniform(-1, 1, 2) * 3.0
        return np.clip(prev, -1, 1)

    return act


def collect(env: PointMassEnv, policy: Policy, n_episodes: int, buffer: ReplayBuffer,
            seed: int = 0, first_episode_id: int = 0) -> ReplayBuffer:
    for ep in range(n_episodes):
        state, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = policy(state, info["goal"])
            next_state, reward, terminated, truncated, info = env.step(action)
            buffer.add(state, action, next_state, reward, terminated, first_episode_id + ep)
            state, done = next_state, terminated or truncated
    return buffer
