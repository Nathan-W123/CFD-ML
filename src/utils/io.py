"""JSON / YAML / compressed-NumPy helpers shared by every stage.

Every JSON file is written the same way: UTF-8, sorted keys, 2-space indent,
finite numbers only (NaN and infinity are refused, not written as bare
tokens that strict parsers reject).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import yaml

PathLike = Union[str, Path]


def to_jsonable(value: Any) -> Any:
    """Convert numpy / torch scalars and arrays, tuples and paths (recursively) to JSON types.

    Non-finite floats raise: an undefined metric must be written as None
    explicitly, never as NaN.
    """
    if hasattr(value, "detach") and hasattr(value, "tolist"):   # torch.Tensor
        return to_jsonable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"cannot write the non-finite number {v} to JSON")
        return v
    if isinstance(value, Path):
        return str(value)
    return value


def dumps_json(doc: Any) -> str:
    return json.dumps(to_jsonable(doc), sort_keys=True, indent=2, allow_nan=False) + "\n"


def write_json(path: PathLike, doc: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(doc), encoding="utf-8")
    return path


def read_json(path: PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_yaml(path: PathLike, doc: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(to_jsonable(doc), sort_keys=False), encoding="utf-8")
    return path


def read_yaml(path: PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    return doc if doc is not None else {}


_JSON_PREFIX = "__json__"


def save_npz(path: PathLike, **arrays: Any) -> Path:
    """Write a compressed .npz atomically and return its path (suffix forced to .npz).

    dict and list arguments are stored as JSON text under a `__json__` key
    prefix and decoded by `load_npz`; string arrays are stored fixed-width;
    object arrays are refused (they would need pickle to load).
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_name(path.name + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, value in arrays.items():
        if isinstance(value, (dict, list, tuple)):
            out[_JSON_PREFIX + key] = np.array(dumps_json(value))
            continue
        arr = np.asarray(value)
        if arr.dtype == object:
            try:
                arr = arr.astype(str)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"array {key!r} has dtype=object and cannot be stored") from exc
        out[key] = arr
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)
    return path


def load_npz(path: PathLike) -> Dict[str, Any]:
    """Read an .npz written by `save_npz` (JSON entries come back as dicts / lists)."""
    out: Dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            arr = data[key]
            if key.startswith(_JSON_PREFIX):
                out[key[len(_JSON_PREFIX):]] = json.loads(str(arr))
            else:
                out[key] = arr
    return out
