"""Gated standardisation of the 12 x 14 motion features (fitted on the train split only).

Each feature column is standardised with a mean and standard deviation
computed ONLY over the timesteps where its gate is open (see
`feature_spec.py`): a closed gate is "no measurement", not a measurement of
zero, so it must not pull the statistics towards 0. After standardisation

* closed-gate entries are exactly 0.0 (the training mean of the open values);
  the network tells "missing" from "average" by the two validity flags,
* open entries are clipped to [-clip, clip] (default 5 sd), so one wild
  real-footage measurement cannot dominate the GRU state,
* the validity flags (features 12, 13) are passed through untouched.

A per-feature standard-deviation floor (in the feature's physical units)
keeps a feature that happened to be nearly constant in the training data from
blowing up at inference; the floors are stored in the JSON next to the fitted
statistics. `clip_fraction` reports how much of a sequence hit the clip;
`PhysicsExpert.predict` abstains when it exceeds `abstain.max_clip_fraction`.

Where this fits: `src.training.train` fits it on the train split and
`src.models.physics_gru.save_model_artifacts` writes it as
feature_normalizer.json; `src.inference.predict` loads it.
Calls into: numpy, `feature_spec.gate_masks`, `src.utils.io` (JSON).
Start reading at: FeatureNormalizer.fit.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from src.data.feature_spec import FEATURE_GATES, FEATURE_NAMES, N_FEATURES, gate_masks
from src.utils.io import PathLike, read_json, write_json

FORMAT_VERSION = 1

# Smallest standard deviation used for each feature, in its own units (lengths
# in L = sqrt(H W) of the analysis frame, rates per second). Roughly a tenth
# of the smallest spread seen in practice: it only matters for a feature that
# is (nearly) constant in the training data.
DEFAULT_STD_FLOOR: Dict[str, float] = {
    "flow_u_mean": 1e-3,
    "flow_v_mean": 1e-3,
    "flow_u_std": 1e-3,
    "flow_v_std": 1e-3,
    "flow_divergence": 1e-3,
    "flow_vorticity_abs": 1e-3,
    "region_area": 1e-3,
    "area_growth": 1e-3,
    "centroid_dx": 1e-3,
    "centroid_dz": 1e-3,
    "boundary_roughness": 1e-2,
    "anchor_stability": 1e-2,
    "mask_valid": 1.0,       # flags are never normalised; listed for completeness
    "flow_valid": 1.0,
}

_MIN_COUNT = 2   # open entries needed to estimate a feature's statistics
_NORMALISED = np.array([gate is not None for gate in FEATURE_GATES])


def _as_batch(X) -> np.ndarray:
    """(N, T, 14) float64 view of a batch or a single (T, 14) sequence."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        X = X[None]
    if X.ndim != 3 or X.shape[-1] != N_FEATURES:
        raise ValueError(f"expected features of shape (N, T, {N_FEATURES}) or (T, {N_FEATURES}), got {X.shape}")
    return X


class FeatureNormalizer:
    """Per-feature gated standardisation with a std floor and clipping."""

    def __init__(self, clip: float = 5.0, std_floor: Optional[Mapping[str, float]] = None):
        if not clip > 0:
            raise ValueError("clip must be positive")
        self.clip = float(clip)
        floors = dict(DEFAULT_STD_FLOOR)
        if std_floor:
            unknown = set(std_floor) - set(FEATURE_NAMES)
            if unknown:
                raise KeyError(f"std_floor for unknown features: {sorted(unknown)}")
            floors.update({k: float(v) for k, v in std_floor.items()})
        self.std_floor = np.array([floors[name] for name in FEATURE_NAMES], dtype=np.float64)
        self.mean: Optional[np.ndarray] = None       # (14,), flags 0
        self.std: Optional[np.ndarray] = None        # (14,) used (floored), flags 1
        self.std_raw: Optional[np.ndarray] = None    # (14,) before the floor
        self.count: Optional[np.ndarray] = None      # (14,) open entries the statistics used
        self.n_sequences: int = 0

    # ------------------------------------------------------------------ fitting
    @property
    def fitted(self) -> bool:
        return self.mean is not None

    def fit(self, X) -> "FeatureNormalizer":
        """Fit on raw TRAIN features (N, T, 14); statistics only over open-gate entries."""
        X = _as_batch(X)
        if not np.all(np.isfinite(X)):
            raise ValueError("cannot fit the normaliser on non-finite features")
        gates = gate_masks(X)
        mean = np.zeros(N_FEATURES)
        std_raw = np.ones(N_FEATURES)
        count = np.zeros(N_FEATURES, dtype=np.int64)
        for i in np.flatnonzero(_NORMALISED):
            values = X[..., i][gates[..., i]]
            count[i] = values.size
            if values.size >= _MIN_COUNT:
                mean[i] = values.mean()
                std_raw[i] = values.std()
            else:
                # Never observed open in training: identity-like scaling. Any open
                # value at inference is then out of distribution and shows up in
                # clip_fraction.
                mean[i] = 0.0
                std_raw[i] = 1.0
        std = np.where(_NORMALISED, np.maximum(std_raw, self.std_floor), 1.0)
        self.mean = np.where(_NORMALISED, mean, 0.0)
        self.std = std
        self.std_raw = np.where(_NORMALISED, std_raw, 1.0)
        self.count = count
        self.n_sequences = int(X.shape[0])
        return self

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("FeatureNormalizer is not fitted")

    # ------------------------------------------------------------- transforming
    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def transform(self, X) -> np.ndarray:
        """float32 array of the input's shape: standardised, closed gates 0, clipped, flags untouched."""
        self._check_fitted()
        single = np.asarray(X).ndim == 2
        X = _as_batch(X)
        if not np.all(np.isfinite(X)):
            raise ValueError("non-finite feature values (the extractor never produces them; check the input)")
        gates = gate_masks(X)
        Z = np.clip(self._standardise(X), -self.clip, self.clip)
        Z = np.where(gates, Z, 0.0)
        Z[..., ~_NORMALISED] = X[..., ~_NORMALISED]
        Z = Z.astype(np.float32) + np.float32(0.0)     # + 0 turns -0.0 into 0.0
        return Z[0] if single else Z

    def clip_fraction(self, X) -> np.ndarray:
        """(N,) fraction of the open-gate, non-flag entries of each sequence at or beyond the clip."""
        self._check_fitted()
        X = _as_batch(X)
        open_ = gate_masks(X) & _NORMALISED
        with np.errstate(invalid="ignore"):
            hit = (np.abs(self._standardise(X)) >= self.clip) | ~np.isfinite(X)
        n_open = open_.reshape(len(X), -1).sum(axis=1)
        n_hit = (hit & open_).reshape(len(X), -1).sum(axis=1)
        return np.where(n_open > 0, n_hit / np.maximum(n_open, 1), 0.0)

    # --------------------------------------------------------------- serialising
    def to_dict(self) -> Dict[str, Any]:
        self._check_fitted()
        return {
            "format": "feature_normalizer",
            "version": FORMAT_VERSION,
            "clip": self.clip,
            "feature_names": list(FEATURE_NAMES),
            "gates": [g if g is not None else "none" for g in FEATURE_GATES],
            "normalised": [bool(v) for v in _NORMALISED],
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "std_raw": [float(v) for v in self.std_raw],
            "std_floor": [float(v) for v in self.std_floor],
            "floored": [bool(n and r < f) for n, r, f in zip(_NORMALISED, self.std_raw, self.std_floor)],
            "count": [int(v) for v in self.count],
            "n_sequences": int(self.n_sequences),
            "statistics": "gated: mean and std over open-gate timesteps of the train split only",
        }

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "FeatureNormalizer":
        names = tuple(doc.get("feature_names", ()))
        if names != FEATURE_NAMES:
            raise ValueError(f"normaliser feature layout {names} does not match {FEATURE_NAMES}")
        if int(doc.get("version", FORMAT_VERSION)) > FORMAT_VERSION:
            raise ValueError(f"normaliser format version {doc['version']} is newer than {FORMAT_VERSION}")
        out = cls(clip=float(doc["clip"]), std_floor=dict(zip(FEATURE_NAMES, doc["std_floor"])))
        out.mean = np.asarray(doc["mean"], dtype=np.float64)
        out.std = np.asarray(doc["std"], dtype=np.float64)
        out.std_raw = np.asarray(doc.get("std_raw", doc["std"]), dtype=np.float64)
        out.count = np.asarray(doc.get("count", [0] * N_FEATURES), dtype=np.int64)
        out.n_sequences = int(doc.get("n_sequences", 0))
        if out.mean.shape != (N_FEATURES,) or out.std.shape != (N_FEATURES,) or np.any(out.std <= 0):
            raise ValueError("normaliser statistics must be 14 values with positive std")
        return out

    def save(self, path: PathLike):
        return write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: PathLike) -> "FeatureNormalizer":
        return cls.from_dict(read_json(path))
