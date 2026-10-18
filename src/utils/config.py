"""Configuration loading: YAML files with dotted-key overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from src.utils.io import PathLike, read_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIMULATION_CONFIG = REPO_ROOT / "configs" / "simulation.yaml"
DEFAULT_TRAINING_CONFIG = REPO_ROOT / "configs" / "training.yaml"


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `base` with `override` merged in (dicts merge, everything else replaces)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply `a.b.c=value` strings (value parsed as YAML) to a config copy.

    The key path must already exist, so a misspelt override fails loudly
    instead of silently adding an unused key; prefix it with `+` to add a new key.
    """
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form key.path=value")
        key, raw = item.split("=", 1)
        create = key.startswith("+")
        parts = key.lstrip("+").split(".")
        node = out
        for depth, part in enumerate(parts[:-1]):
            if part not in node:
                if not create:
                    raise KeyError(f"override {key!r}: no key {'.'.join(parts[:depth + 1])!r} in the config")
                node[part] = {}
            node = node[part]
            if not isinstance(node, dict):
                raise TypeError(f"override {key!r}: {'.'.join(parts[:depth + 1])!r} is not a mapping")
        if parts[-1] not in node and not create:
            raise KeyError(f"override {key!r}: no key {key!r} in the config (prefix with + to add it)")
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def load_config(path: Optional[PathLike], default: Path, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    cfg = read_yaml(path if path is not None else default)
    return apply_overrides(cfg, overrides)


def load_simulation_config(path: Optional[PathLike] = None, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    return load_config(path, DEFAULT_SIMULATION_CONFIG, overrides)


def load_training_config(path: Optional[PathLike] = None, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    return load_config(path, DEFAULT_TRAINING_CONFIG, overrides)
