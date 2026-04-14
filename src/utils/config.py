"""src/utils/config.py — YAML config with dot-notation access."""

import yaml
from pathlib import Path


class Config:
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def to_dict(self) -> dict:
        out = {}
        for k, v in vars(self).items():
            out[k] = v.to_dict() if isinstance(v, Config) else v
        return out

    def __repr__(self):
        return f"Config({vars(self)})"


def load_config(path: str = "configs/default.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p) as f:
        return Config(yaml.safe_load(f))
