"""The detector: linear smoke and fog heads on window features, with fixed operating thresholds.

Each head is a standardiser plus an L2-regularised logistic regression,
stored as plain numbers (detector.json + detector_weights.npz), so inference
needs numpy only (the CLIP embeddings come from scripts/embed_windows.py or
src.detector.features). A head may also have a tile head: the same kind of
model applied to each fine tile (src.detector.features.tile_instances), the
window scored by its highest tile; the head's logit and its tile head's
logit are then combined by a small logistic "stack" (two weights and an
intercept). Thresholds are probabilities on the head's (stacked) output:

* smoke "deploy": fitted on the training cameras' out-of-fold scores so that
  at most a target share (2.5 %, half the 5 % goal) of the real fog windows
  raise a smoke alarm;
* smoke "strict": the same, but at most 5 % of all the smoke-free windows;
* smoke / fog "balanced": the maximum-Youden threshold on training cameras.

predict() falls back to "balanced" for a head without the requested
operating point, so fog decisions always use the fog balanced threshold.
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from src.detector.features import tile_instances, window_features

FORMAT_VERSION = 2


@dataclass
class LinearHead:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    groups: Sequence[str]
    C: float

    def logit(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, np.float64) - self.mean) / self.scale
        return Z @ self.coef + self.intercept

    def proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(X)))


def tile_logit(head: LinearHead, TF: np.ndarray) -> np.ndarray:
    """A tile head's window logit: the highest tile logit, TF (N, K, d) -> (N,)."""
    TF = np.asarray(TF)
    return head.logit(TF.reshape(-1, TF.shape[-1])).reshape(TF.shape[0], TF.shape[1]).max(1)


def fit_tile_head(TF: np.ndarray, y: np.ndarray, C: float) -> LinearHead:
    """A tile head: every tile of a window takes the window's label (the simplest multiple-instance fit)."""
    TF = np.asarray(TF)
    return fit_head(TF.reshape(-1, TF.shape[-1]), np.repeat(np.asarray(y), TF.shape[1]), ("tiles_fine",), C)


def stack_logit(stack: Dict, logits: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack(logits, 1) @ np.asarray(stack["coef"], np.float64) + float(stack["intercept"])


def fit_head(X: np.ndarray, y: np.ndarray, groups: Sequence[str], C: float,
             sample_weight: Optional[np.ndarray] = None) -> LinearHead:
    """Standardise X and fit an L2 logistic regression (sklearn; C = inverse regularisation)."""
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(X, np.float64)
    mean = X.mean(0)
    scale = X.std(0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    clf = LogisticRegression(C=float(C), max_iter=10000)
    with warnings.catch_warnings():                 # scipy's notice about an lbfgs option sklearn passes
        warnings.simplefilter("ignore")
        clf.fit((X - mean) / scale, np.asarray(y, np.int64), sample_weight=sample_weight)
    return LinearHead(mean=mean, scale=scale, coef=clf.coef_[0].astype(np.float64),
                      intercept=float(clf.intercept_[0]), groups=list(groups), C=float(C))


@dataclass
class Detector:
    heads: Dict[str, LinearHead]
    text: Dict[str, np.ndarray]                 # head -> prompt embeddings (P, D)
    thresholds: Dict[str, Dict[str, float]]     # head -> operating point -> probability threshold
    info: Dict = field(default_factory=dict)
    tile_heads: Dict[str, LinearHead] = field(default_factory=dict)   # head -> tile head (fine tiles)
    stacks: Dict[str, Dict] = field(default_factory=dict)             # head -> {"coef": [w_head, w_tile], "intercept"}

    def features(self, head: str, E: np.ndarray, motion: Optional[np.ndarray] = None,
                 fine: Optional[np.ndarray] = None) -> np.ndarray:
        return window_features(E, self.text[head], self.heads[head].groups, motion, fine)

    def logit(self, head: str, E: np.ndarray, motion: Optional[np.ndarray] = None,
              fine: Optional[np.ndarray] = None) -> np.ndarray:
        z = self.heads[head].logit(self.features(head, E, motion, fine))
        if head in self.tile_heads:
            if fine is None:
                raise ValueError(f"head {head!r} has a tile head: it needs the fine (4 x 6 tile) embeddings")
            zt = tile_logit(self.tile_heads[head], tile_instances(fine, self.text[head]))
            z = stack_logit(self.stacks[head], [z, zt])
        return z

    def proba(self, E: np.ndarray, motion: Optional[np.ndarray] = None,
              fine: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """P(smoke), P(fog) for windows with embeddings E ((N, F, 7, D) or (F, 7, D)) and, if a head uses
        them, fine embeddings ((N, 2, 24, D)) and pooled motion features ((N, 28))."""
        return {h: 1.0 / (1.0 + np.exp(-self.logit(h, E, motion, fine))) for h in self.heads}

    def predict(self, E: np.ndarray, motion: Optional[np.ndarray] = None, operating_point: str = "deploy",
                fine: Optional[np.ndarray] = None) -> Dict:
        p = self.proba(E, motion, fine)
        out = {"probs": p, "decisions": {}}
        for h, ph in p.items():
            thr = self.thresholds[h].get(operating_point, self.thresholds[h]["balanced"])
            out["decisions"][h] = (ph >= thr).astype(np.int64)
        return out

    def save(self, directory) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        arrays = {}
        doc = {"format_version": FORMAT_VERSION, "heads": {}, "thresholds": self.thresholds, "info": self.info}
        for h, hd in self.heads.items():
            for k in ("mean", "scale", "coef"):
                arrays[f"{h}.{k}"] = getattr(hd, k)
            arrays[f"{h}.text"] = self.text[h]
            doc["heads"][h] = {"intercept": hd.intercept, "groups": list(hd.groups), "C": hd.C,
                               "n_features": int(len(hd.coef))}
        doc["tile_heads"] = {}
        for h, hd in self.tile_heads.items():
            for k in ("mean", "scale", "coef"):
                arrays[f"tile.{h}.{k}"] = getattr(hd, k)
            doc["tile_heads"][h] = {"intercept": hd.intercept, "C": hd.C, "n_features": int(len(hd.coef)),
                                    "stack": {"coef": [float(c) for c in self.stacks[h]["coef"]],
                                              "intercept": float(self.stacks[h]["intercept"])}}
        np.savez_compressed(d / "detector_weights.npz", **arrays)
        (d / "detector.json").write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory) -> "Detector":
        d = Path(directory)
        doc = json.loads((d / "detector.json").read_text(encoding="utf-8"))
        if int(doc.get("format_version", 1)) > FORMAT_VERSION:
            raise ValueError(f"{d}: detector format {doc['format_version']} is newer than this code")
        w = np.load(d / "detector_weights.npz")
        heads, text = {}, {}
        for h, hd in doc["heads"].items():
            heads[h] = LinearHead(mean=w[f"{h}.mean"], scale=w[f"{h}.scale"], coef=w[f"{h}.coef"],
                                  intercept=float(hd["intercept"]), groups=hd["groups"], C=float(hd["C"]))
            text[h] = w[f"{h}.text"]
        tile_heads, stacks = {}, {}
        for h, hd in doc.get("tile_heads", {}).items():
            tile_heads[h] = LinearHead(mean=w[f"tile.{h}.mean"], scale=w[f"tile.{h}.scale"], coef=w[f"tile.{h}.coef"],
                                       intercept=float(hd["intercept"]), groups=["tiles_fine"], C=float(hd["C"]))
            stacks[h] = hd["stack"]
        return cls(heads=heads, text=text, thresholds=doc["thresholds"], info=doc.get("info", {}),
                   tile_heads=tile_heads, stacks=stacks)
