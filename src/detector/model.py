"""The detector: linear smoke and fog heads on window features, with fixed operating thresholds.

Each head is a standardiser plus an L2-regularised logistic regression,
stored as plain numbers (detector.json + detector_weights.npz), so inference
needs numpy only (the CLIP embeddings come from scripts/embed_windows.py or
src.detector.features). A head may also have a tile head: the same kind of
model applied to each fine tile (src.detector.features.tile_instances), the
window scored by its highest tile, and a zoom head: the same on the
full-resolution quarters of the window's top tiles (zoom_instances). The
head's logit, its tile and zoom heads' logits and scalar scores (the smoke
prompts' contrast change within the window and against a reference frame of
the same camera 14 min earlier: change_scores, reference_scores) are
combined by a small logistic "stack". A head keeps several stacks, the first
one whose inputs are all available is used (a window without a reference
frame or zoom crops falls back to a stack without them). Thresholds are
probabilities on the head's (stacked) output:

* smoke "deploy": the out-of-fold dev threshold that keeps a target smoke
  recall (v2: the 80 % v1 reached) but never lets more than 2.5 % of the dev
  fog windows alarm (v1: the fog limit alone);
* smoke "strict": at most 5 % of all the dev smoke-free windows alarm;
* smoke / fog "balanced": the maximum-Youden threshold on training cameras;
* smoke "fog_veto" (if present): a smoke alarm is withheld when P(fog) is at
  least this value.

predict() falls back to "balanced" for a head without the requested
operating point, so fog decisions always use the fog balanced threshold.
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from src.detector.features import (change_scores, reference_scores, tile_instances, window_features,
                                   zoom_instances)

FORMAT_VERSION = 3


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


STACK_INPUTS = ("head", "tile", "zoom", "chg_max", "chg_rz", "ref_max", "ref_rz")


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
    stacks: Dict[str, list] = field(default_factory=dict)             # head -> [{"inputs", "coef", "intercept"}, ...]
    zoom_heads: Dict[str, LinearHead] = field(default_factory=dict)   # head -> zoom head (full-resolution crops)

    def features(self, head: str, E: np.ndarray, motion: Optional[np.ndarray] = None,
                 fine: Optional[np.ndarray] = None) -> np.ndarray:
        return window_features(E, self.text[head], self.heads[head].groups, motion, fine)

    def select_zoom_tiles(self, fine: np.ndarray, k: int = 4, head: str = "smoke") -> np.ndarray:
        """(N, k): the fine tiles the tile head ranks highest, the tiles scripts/zoom_windows.py zooms into."""
        TF = tile_instances(fine, self.text[head])
        L = self.tile_heads[head].logit(TF.reshape(-1, TF.shape[-1])).reshape(TF.shape[0], TF.shape[1])
        return np.argsort(-L, axis=1)[:, :k]

    def stack_inputs(self, head: str, E: np.ndarray, motion: Optional[np.ndarray] = None,
                     fine: Optional[np.ndarray] = None, zoom: Optional[np.ndarray] = None,
                     ref: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """The stack inputs that can be computed from what is given: head, tile, zoom, chg_*, ref_*."""
        T = self.text[head]
        out = {"head": self.heads[head].logit(self.features(head, E, motion, fine))}
        if fine is not None and head in self.tile_heads:
            out["tile"] = tile_logit(self.tile_heads[head], tile_instances(fine, T))
            c = change_scores(fine, T)
            out["chg_max"], out["chg_rz"] = c[:, 0], c[:, 1]
            if ref is not None:
                r = reference_scores(fine, ref, T)
                out["ref_max"], out["ref_rz"] = r[:, 0], r[:, 1]
        if zoom is not None and head in self.zoom_heads:
            out["zoom"] = tile_logit(self.zoom_heads[head], zoom_instances(zoom, T))
        return out

    def logit(self, head: str, E: np.ndarray, motion: Optional[np.ndarray] = None,
              fine: Optional[np.ndarray] = None, zoom: Optional[np.ndarray] = None,
              ref: Optional[np.ndarray] = None) -> np.ndarray:
        inputs = self.stack_inputs(head, E, motion, fine, zoom, ref)
        stacks = self.stacks.get(head)
        if not stacks:
            return inputs["head"]
        for st in stacks:
            if all(k in inputs for k in st["inputs"]):
                return stack_logit(st, [inputs[k] for k in st["inputs"]])
        raise ValueError(f"head {head!r}: no stack can be evaluated from {sorted(inputs)} "
                         f"(it needs e.g. {stacks[-1]['inputs']}; a tile head needs the fine 4 x 6 embeddings)")

    def stack_used(self, head: str, available) -> list:
        """The inputs of the stack logit() uses when the given inputs are available."""
        for st in self.stacks.get(head, []):
            if all(k in available for k in st["inputs"]):
                return list(st["inputs"])
        return ["head"]

    def proba(self, E: np.ndarray, motion: Optional[np.ndarray] = None,
              fine: Optional[np.ndarray] = None, zoom: Optional[np.ndarray] = None,
              ref: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """P(smoke), P(fog) for windows with embeddings E ((N, F, 7, D) or (F, 7, D)) and, if a head uses
        them, fine embeddings ((N, 2, 24, D)), zoom embeddings ((N, 2, 4 k, D)), the reference frame's fine
        embeddings ((N, 24, D)) and pooled motion features ((N, 28))."""
        return {h: 1.0 / (1.0 + np.exp(-self.logit(h, E, motion, fine, zoom, ref))) for h in self.heads}

    def predict(self, E: np.ndarray, motion: Optional[np.ndarray] = None, operating_point: str = "deploy",
                fine: Optional[np.ndarray] = None, zoom: Optional[np.ndarray] = None,
                ref: Optional[np.ndarray] = None) -> Dict:
        p = self.proba(E, motion, fine, zoom, ref)
        out = {"probs": p, "decisions": {}}
        for h, ph in p.items():
            thr = self.thresholds[h].get(operating_point, self.thresholds[h]["balanced"])
            dec = ph >= thr
            veto = self.thresholds[h].get("fog_veto")
            if veto is not None and "fog" in p and h != "fog":
                dec &= p["fog"] < veto
            out["decisions"][h] = dec.astype(np.int64)
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
        for kind, heads in (("tile", self.tile_heads), ("zoom", self.zoom_heads)):
            doc[f"{kind}_heads"] = {}
            for h, hd in heads.items():
                for k in ("mean", "scale", "coef"):
                    arrays[f"{kind}.{h}.{k}"] = getattr(hd, k)
                doc[f"{kind}_heads"][h] = {"intercept": hd.intercept, "C": hd.C, "n_features": int(len(hd.coef))}
        doc["stacks"] = {h: [{"inputs": list(st["inputs"]), "coef": [float(c) for c in st["coef"]],
                              "intercept": float(st["intercept"])} for st in sts] for h, sts in self.stacks.items()}
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
        side = {"tile": {}, "zoom": {}}
        for kind in side:
            for h, hd in doc.get(f"{kind}_heads", {}).items():
                side[kind][h] = LinearHead(mean=w[f"{kind}.{h}.mean"], scale=w[f"{kind}.{h}.scale"],
                                           coef=w[f"{kind}.{h}.coef"], intercept=float(hd["intercept"]),
                                           groups=[f"{kind}s"], C=float(hd["C"]))
        stacks = {h: list(sts) for h, sts in doc.get("stacks", {}).items()}
        for h, hd in doc.get("tile_heads", {}).items():           # format 2: one stack stored with the tile head
            if "stack" in hd and h not in stacks:
                stacks[h] = [{"inputs": ["head", "tile"], **hd["stack"]}]
        return cls(heads=heads, text=text, thresholds=doc["thresholds"], info=doc.get("info", {}),
                   tile_heads=side["tile"], stacks=stacks, zoom_heads=side["zoom"])
