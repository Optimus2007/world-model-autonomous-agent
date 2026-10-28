"""Piecewise-constant dynamics schedule over episodes: original -> changed -> (optionally) reverted.

Each episode runs under one set of dynamics. Changes take effect at the first step of an episode and
persist until the next scheduled change.
"""
from __future__ import annotations


def episode_phase(episode: int, shift_episode: int, revert_episode: int | None = None) -> int:
    """0 = original dynamics, 1 = changed, 2 = reverted to the original."""
    if episode < shift_episode:
        return 0
    if revert_episode is None or episode < revert_episode:
        return 1
    return 2


def phase_dynamics(phase: int, shift: dict, revert: dict | None) -> dict | None:
    """Dynamics overrides to apply for a phase (None = environment defaults)."""
    return {0: None, 1: shift, 2: revert}[phase]
