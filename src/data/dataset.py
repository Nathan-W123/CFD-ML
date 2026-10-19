"""Feature datasets and rendered-sequence storage, with the split-leakage check.

Two on-disk record types, both compressed .npz written by `src.utils.io.save_npz`
(no pickle; dicts and lists are stored as JSON text):

* `FeatureSet` (features.npz): one row per rendered sequence, the raw
  (12, 14) features, the two binary targets and the provenance needed for
  grouped splitting and sliced evaluation (sim id, group, split, category,
  regime, hold-out tags, mask source) plus a JSON meta dict per row (seed,
  scenario context, render settings, feature-extraction record).
  A target may be NaN = unknown (real footage whose fog is not annotated);
  such a row's category is one of PARTIAL_CATEGORIES (`label_category`).
* rendered sequences (seqs/<split>/<seq_id>.npz, `save_sequence` /
  `load_sequence`): the 12 observation frames, 12 lag frames (none for a
  profile without lag frames, e.g. the far-field one-frame-per-minute one),
  candidate masks (bit-packed), timestamps, labels, the simulation
  parameters, seed and numerical-validity record, the render parameters, and
  optionally the opacity diagnostics. Enough to re-extract features (e.g.
  with detector masks) without re-rendering.

Leakage rules (docs/design.md section 12): a group (synthetic: sim_id; real:
camera id) and an original simulation / video never span two splits, and no
seed or scenario parameter vector appears in two splits. Robustness re-renders
(splits named "robust:<suite>") are re-renders of test_iid simulations and
count as test_iid.

Where this fits: `scripts/generate_dataset.py` and `src.data.pipeline` write
both record types; `scripts/extract_features.py` reads sequences;
`src.training` reads FeatureSets and wraps them in `SequenceTensorDataset`.
Calls into: numpy, torch (Dataset base class), `src.utils.io`.
Start reading at: FeatureSet, then assert_no_group_leakage.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch

from src.constants import CATEGORY_LABELS
from src.data.feature_spec import N_FEATURES, N_STEPS
from src.utils.io import PathLike, load_npz, save_npz, to_jsonable

FEATURESET_FORMAT_VERSION = 1
SEQUENCE_FORMAT_VERSION = 1

_STR_FIELDS = ("seq_ids", "sim_ids", "groups", "splits", "categories", "regimes", "tags", "mask_sources")
# Keys of a scenario-parameter dict that identify a sample rather than describe it.
_ID_KEYS = ("sim_id", "split", "index", "attempt", "seed", "holdout_tags", "category")

# Categories of rows with an unknown (NaN) target, keyed by (smoke, fog) with None = unknown.
PARTIAL_CATEGORIES: Dict[tuple, str] = {
    (1, None): "smoke_fog_unknown",
    (0, None): "no_smoke_fog_unknown",
    (None, 1): "fog_smoke_unknown",
    (None, 0): "no_fog_smoke_unknown",
    (None, None): "unlabelled",
}


def label_category(smoke: float, fog: float) -> str:
    """Category of a (smoke, fog) target pair; NaN = unknown gives one of PARTIAL_CATEGORIES."""
    key = tuple(None if v is None or not np.isfinite(float(v)) else int(float(v) > 0.5) for v in (smoke, fog))
    if None in key:
        return PARTIAL_CATEGORIES[key]
    for name, pair in CATEGORY_LABELS.items():
        if pair == key:
            return name
    raise ValueError(f"no category for labels {key}")       # unreachable: CATEGORY_LABELS covers {0, 1}^2


def parse_label(value) -> float:
    """A manifest label: 0 / 1, or NaN (unknown) for an empty cell, "nan", "unknown", "?" or "-"."""
    text = str(value if value is not None else "").strip().lower()
    if text in ("", "nan", "none", "null", "unknown", "?", "-"):
        return float("nan")
    v = float(text)
    if v not in (0.0, 1.0):
        raise ValueError(f"label {value!r} must be 0, 1 or empty (unknown)")
    return v


def _str_array(values, n: int, name: str) -> np.ndarray:
    arr = np.asarray(values if values is not None else [""] * n)
    if arr.size == 0:
        arr = np.zeros(0, dtype="<U1")
    arr = arr.astype(str)
    if arr.shape != (n,):
        raise ValueError(f"FeatureSet.{name} must have shape ({n},), got {arr.shape}")
    return arr


@dataclass
class FeatureSet:
    X: np.ndarray                  # (N, 12, 14) float32 raw features (closed gates 0)
    y: np.ndarray                  # (N, 2) float32 (smoke, fog) in [0, 1]; NaN = unknown (not annotated)
    seq_ids: np.ndarray            # (N,) str, unique
    sim_ids: np.ndarray            # (N,) str, original simulation (real: video id)
    groups: np.ndarray             # (N,) str, the leakage unit (synthetic: sim_id; real: camera id)
    splits: np.ndarray             # (N,) str, train / val_select / val_calib / test_iid / test_ood / robust:<suite>
    categories: np.ndarray         # (N,) str, src.constants.CATEGORIES (PARTIAL_CATEGORIES with a NaN target)
    regimes: np.ndarray            # (N,) str
    tags: np.ndarray               # (N,) str, ";"-joined hold-out tags ("" for none)
    mask_sources: np.ndarray       # (N,) str, degraded / bbox / detector / provided / fallback_detector
    meta: List[Dict[str, Any]] = field(default_factory=list)   # per row, JSON-able
    source: str = "synthetic"      # "synthetic" | "real"

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=np.float32)
        if self.X.size == 0:
            self.X = self.X.reshape(0, N_STEPS, N_FEATURES)
        if self.X.ndim != 3 or self.X.shape[-1] != N_FEATURES:
            raise ValueError(f"FeatureSet.X must be (N, T, {N_FEATURES}), got {self.X.shape}")
        n = self.X.shape[0]
        self.y = np.asarray(self.y, dtype=np.float32).reshape(n, -1) if n else np.zeros((0, 2), np.float32)
        if self.y.shape[1] != 2:
            raise ValueError(f"FeatureSet.y must be (N, 2), got {self.y.shape}")
        known = ~np.isnan(self.y)
        if np.any(np.isinf(self.y)) or np.any((self.y[known] < 0) | (self.y[known] > 1)):
            raise ValueError("FeatureSet.y must lie in [0, 1], or be NaN for an unknown label")
        for name in _STR_FIELDS:
            setattr(self, name, _str_array(getattr(self, name), n, name))
        self.meta = list(self.meta) if self.meta else [{} for _ in range(n)]
        if len(self.meta) != n:
            raise ValueError(f"FeatureSet.meta has {len(self.meta)} entries for {n} rows")
        if self.source not in ("synthetic", "real"):
            raise ValueError(f"FeatureSet.source must be 'synthetic' or 'real', got {self.source!r}")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def known(self) -> np.ndarray:
        """(N, 2) bool: True where the target is known (not NaN)."""
        return ~np.isnan(self.y)

    # ------------------------------------------------------------------ storage
    def save(self, path: PathLike) -> Path:
        """Compressed .npz; NaN (unknown) targets are stored as NaN and come back as NaN."""
        arrays = {name: getattr(self, name) for name in _STR_FIELDS}
        return save_npz(path, X=self.X, y=self.y, meta=to_jsonable(self.meta),
                        info={"format": "feature_set", "version": FEATURESET_FORMAT_VERSION,
                              "source": self.source, "n": len(self)},
                        **arrays)

    @classmethod
    def load(cls, path: PathLike) -> "FeatureSet":
        data = load_npz(path)
        info = data.get("info", {})
        if int(info.get("version", 1)) > FEATURESET_FORMAT_VERSION:
            raise ValueError(f"{path}: FeatureSet format {info['version']} is newer than this code")
        return cls(X=data["X"], y=data["y"], meta=data.get("meta", []), source=info.get("source", "synthetic"),
                   **{name: data[name] for name in _STR_FIELDS})

    # ---------------------------------------------------------------- selection
    def subset(self, mask) -> "FeatureSet":
        """Rows selected by a boolean mask (N,) or an integer index array."""
        idx = np.asarray(mask)
        if idx.dtype == bool:
            if idx.shape != (len(self),):
                raise ValueError(f"boolean mask of shape {idx.shape} for {len(self)} rows")
            idx = np.flatnonzero(idx)
        idx = idx.astype(np.int64).ravel()
        return FeatureSet(
            X=self.X[idx], y=self.y[idx], meta=[self.meta[i] for i in idx], source=self.source,
            **{name: getattr(self, name)[idx] for name in _STR_FIELDS})

    def split(self, name: Union[str, Iterable[str]]) -> "FeatureSet":
        """Rows of one split (or of several, given an iterable of names)."""
        names = [name] if isinstance(name, str) else list(name)
        return self.subset(np.isin(self.splits, names))


# =============================================================================
# Leakage
# =============================================================================


def _canonical_split(split: str) -> str:
    # Robustness suites re-render test_iid simulations: they belong to test_iid.
    return "test_iid" if split.startswith("robust:") else split


def _param_key(meta: Dict[str, Any]) -> Optional[str]:
    """Canonical text of a row's scenario-parameter vector (None when the row has none)."""
    if isinstance(meta.get("context"), dict) and meta["context"]:
        return json.dumps(to_jsonable(meta["context"]), sort_keys=True)
    if isinstance(meta.get("params"), dict) and meta["params"]:
        params = {k: v for k, v in meta["params"].items() if k not in _ID_KEYS}
        return json.dumps(to_jsonable(params), sort_keys=True)
    return None


def _spanning(keys: Iterable[Optional[Any]], splits: np.ndarray) -> Dict[Any, List[str]]:
    where: Dict[Any, set] = defaultdict(set)
    for key, split in zip(keys, splits):
        if key is not None:
            where[key].add(split)
    return {k: sorted(v) for k, v in where.items() if len(v) > 1}


def assert_no_group_leakage(fs: FeatureSet) -> None:
    """Raise ValueError if a group, sim / video id, seed, parameter vector or sequence id spans two splits."""
    splits = np.array([_canonical_split(s) for s in fs.splits])
    problems = []
    checks = [
        ("group", list(fs.groups)),
        ("sim_id", list(fs.sim_ids)),
        ("seed", [m.get("seed") if isinstance(m.get("seed"), (int, np.integer)) else None for m in fs.meta]),
        ("parameter vector", [_param_key(m) for m in fs.meta]),
    ]
    for what, keys in checks:
        bad = _spanning(keys, splits)
        if bad:
            examples = "; ".join(f"{str(k)[:80]} in {v}" for k, v in list(bad.items())[:5])
            problems.append(f"{len(bad)} {what}(s) span several splits: {examples}")
    seq, counts = np.unique(fs.seq_ids, return_counts=True)
    if np.any(counts > 1):
        problems.append(f"duplicate seq_ids: {list(seq[counts > 1][:5])}")
    if problems:
        raise ValueError("data leakage between splits: " + " | ".join(problems))


# =============================================================================
# Rendered sequences
# =============================================================================


def save_sequence(path: PathLike, rendered, sim, seq_id: str, extra: Optional[Dict[str, Any]] = None) -> Path:
    """Store one rendered camera sequence with everything needed to re-extract and audit it.

    `rendered` is a `src.data.render.RenderedSequence`, `sim` the
    `src.cfd.results.SimulationResult` it was rendered from. A sequence
    without lag frames (`rendered.lag_frames` None: a profile with
    time.flow_lag_s null / 0) stores none, and `load_sequence` returns None
    for them. `extra` keys: "diagnostics" (bool, default False) also stores
    the float16 opacity maps; "render_info" and anything else go into the
    JSON meta.
    """
    extra = dict(extra or {})
    diagnostics = bool(extra.pop("diagnostics", False))
    masks = np.asarray(rendered.masks).astype(bool)
    lag_frames = getattr(rendered, "lag_frames", None)
    lag_ts = getattr(rendered, "lag_timestamps", None)
    if (lag_frames is None) != (lag_ts is None):
        raise ValueError("lag_frames and lag_timestamps must both be given or both be None")
    arrays: Dict[str, Any] = {
        "frames": np.asarray(rendered.frames, dtype=np.uint8),
        "masks_packed": np.packbits(masks, axis=-1),
        "masks_shape": np.asarray(masks.shape, dtype=np.int64),
        "timestamps": np.asarray(rendered.timestamps, dtype=np.float64),
        "labels": np.asarray(sim.labels, dtype=np.int8),
    }
    if lag_frames is not None:
        arrays["lag_frames"] = np.asarray(lag_frames, dtype=np.uint8)
        arrays["lag_timestamps"] = np.asarray(lag_ts, dtype=np.float64)
    if diagnostics:
        if getattr(rendered, "opacity", None) is not None:
            arrays["opacity"] = np.asarray(rendered.opacity, dtype=np.float16)
        for name, value in (getattr(rendered, "component_opacity", None) or {}).items():
            arrays[f"component_opacity__{name}"] = np.asarray(value, dtype=np.float16)
    params = dict(sim.params)
    meta = {
        "format": "rendered_sequence",
        "version": SEQUENCE_FORMAT_VERSION,
        "seq_id": seq_id,
        "sim_id": sim.sim_id,
        "category": sim.category,
        "regime": sim.regime,
        "labels": [int(v) for v in sim.labels],
        "seed": int(sim.seed),
        "split": params.get("split"),
        "params": params,
        "valid": bool(sim.valid),
        "validity": sim.validity,
        "dx": float(sim.dx),
        "mask_source": rendered.mask_source,
        "render_params": rendered.render_params,
        "visibility": rendered.visibility,
        "has_lag_frames": lag_frames is not None,
        "diagnostics_stored": diagnostics,
        "extra": extra,
    }
    return save_npz(path, meta=to_jsonable(meta), **arrays)


def load_sequence(path: PathLike) -> Dict[str, Any]:
    """A stored sequence as a dict: arrays (frames, lag_frames, masks (T, H, W) bool, timestamps,
    lag_timestamps, labels) plus the meta fields (seq_id, sim_id, category, regime, seed, params,
    validity, render_params, mask_source, visibility, extra) and the full `meta` dict; with
    diagnostics also opacity and component_opacity. lag_frames / lag_timestamps are None for a
    sequence stored without lag frames."""
    data = load_npz(path)
    meta = data.pop("meta")
    if int(meta.get("version", 1)) > SEQUENCE_FORMAT_VERSION:
        raise ValueError(f"{path}: sequence format {meta['version']} is newer than this code")
    shape = tuple(int(v) for v in data.pop("masks_shape"))
    masks = np.unpackbits(data.pop("masks_packed"), axis=-1, count=shape[-1]).astype(bool).reshape(shape)
    out: Dict[str, Any] = {
        "frames": data.pop("frames"),
        "lag_frames": data.pop("lag_frames", None),
        "masks": masks,
        "timestamps": data.pop("timestamps"),
        "lag_timestamps": data.pop("lag_timestamps", None),
        "labels": tuple(int(v) for v in data.pop("labels")),
    }
    for key in ("seq_id", "sim_id", "category", "regime", "seed", "split", "params", "valid", "validity",
                "dx", "mask_source", "render_params", "visibility", "extra"):
        out[key] = meta.get(key)
    out["meta"] = meta
    if "opacity" in data:
        out["opacity"] = data.pop("opacity")
    comps = {k[len("component_opacity__"):]: data.pop(k) for k in list(data) if k.startswith("component_opacity__")}
    if comps:
        out["component_opacity"] = comps
    return out


# =============================================================================
# Torch wrapper
# =============================================================================


class SequenceTensorDataset(torch.utils.data.Dataset):
    """(x (12, 14), y (2,)) float32 tensors; normalised on construction when a normaliser is given.

    `X` is a FeatureSet (labels taken from it) or an array (N, 12, 14);
    `y` defaults to zeros for unlabelled data. NaN targets (unknown) are kept
    as NaN; the training loss masks them.
    """

    def __init__(self, X, y=None, normalizer=None):
        if isinstance(X, FeatureSet):
            y = X.y if y is None else y
            X = X.X
        X = np.asarray(X, dtype=np.float32)
        if normalizer is not None:
            X = normalizer.transform(X)
        y = np.zeros((len(X), 2), dtype=np.float32) if y is None else np.asarray(y, dtype=np.float32)
        if len(y) != len(X):
            raise ValueError(f"{len(X)} sequences but {len(y)} labels")
        self.X = torch.from_numpy(np.ascontiguousarray(X))
        self.y = torch.from_numpy(np.ascontiguousarray(y))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, i):
        return self.X[i], self.y[i]
