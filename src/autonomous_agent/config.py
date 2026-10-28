"""YAML config -> dataclasses. Unknown keys raise, so typos fail loudly."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import yaml

from .env import Dynamics, EnvConfig
from .world_model import ModelConfig


def _build(cls, d: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**d)


def load_config(path: str | Path) -> dict:
    # Imported here to avoid a circular import (agent imports world_model, env).
    from .agent import AdaptConfig, AgentConfig, DetectorConfig
    from .planner import PlannerConfig

    raw = yaml.safe_load(Path(path).read_text())
    env = dict(raw.get("env", {}))
    env["dynamics"] = _build(Dynamics, env.get("dynamics", {}))
    return {
        "seed": raw.get("seed", 0),
        "env": _build(EnvConfig, env),
        "model": _build(ModelConfig, raw.get("model", {})),
        "agent": AgentConfig(planner=_build(PlannerConfig, raw.get("planner", {})),
                             detector=_build(DetectorConfig, raw.get("detector", {})),
                             adapt=_build(AdaptConfig, raw.get("adapt", {}))),
        "data": raw.get("data", {}),
        "experiment": raw.get("experiment", {}),
        "shift": raw.get("shift", {}),
        "shift_experiment": raw.get("shift_experiment", {}),
        "raw": raw,
    }
