"""The detector: linear smoke and fog heads on window features, with fixed operating thresholds.

Each head is a standardiser plus an L2-regularised logistic regression,
stored as plain numbers (detector.json + detector_weights.npz), so inference
needs numpy only (the CLIP embeddings come from scripts/embed_windows.py or
src.detector.features). Thresholds are probabilities on the head's output:

* smoke "deploy": fitted on training cameras so that at most a target share
  (5 %) of the real fog windows raise a smoke alarm;
* smoke / fog "balanced": the maximum-Youden threshold on training cameras.
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from src.detector.features import window_features

FORMAT_VERSION = 1


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

    def features(self, head: str, E: np.ndarray, motion: Optional[np.ndarray] = None) -> np.ndarray:
        return window_features(E, self.text[head], self.heads[head].groups, motion)

    def proba(self, E: np.ndarray, motion: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """P(smoke), P(fog) for windows with embeddings E ((N, F, 7, D) or (F, 7, D))."""
        return {h: self.heads[h].proba(self.features(h, E, motion)) for h in self.heads}

    def predict(self, E: np.ndarray, motion: Optional[np.ndarray] = None, operating_point: str = "deploy") -> Dict:
        p = self.proba(E, motion)
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
        return cls(heads=heads, text=text, thresholds=doc["thresholds"], info=doc.get("info", {}))
