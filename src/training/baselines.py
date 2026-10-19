"""Non-neural baselines and feature-level shortcut probes (numpy only).

* Experiment A, `HeuristicBaseline`: a fixed functional form on four
  time-pooled motion cues a fire-watch operator would use: mean vertical
  flow, mean vertical centroid velocity, mean anchor stability of the
  region's base and mean log-area growth (each averaged over the timesteps
  where its gate is open). Per label, logit = b + sum_k w_k * cue_k on cues
  standardised with train statistics; the five coefficients are fitted on
  the train split only by a tiny L2-regularised logistic regression;
  thresholds (balanced / low_false_alarm) come from val_calib.
* `PooledLogisticBaseline`: the same logistic regression on the mean and
  standard deviation over time of all 14 features (28 inputs): what a model
  without temporal order can do.
* `flags_only_probe`: logistic regression on the two validity flags alone
  (per-timestep pattern + their means). If whether a mask / flow exists
  predicts a label (AUROC > audit.flags_only_auc_max), the dataset carries a
  shortcut that has nothing to do with motion and is flagged.
* `univariate_aucs`: folded AUROC of every pooled feature per label (0.5 =
  uninformative, 1 = separates perfectly in either direction).

Where this fits: `src.training.evaluate` runs these next to the GRU and
writes them into evaluation.json ("ablations.A", "shortcut_audit").
Calls into: numpy, `src.training.metrics`, `src.data.feature_spec`.
Start reading at: HeuristicBaseline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.constants import LABEL_NAMES
from src.data.dataset import FeatureSet
from src.data.feature_spec import FEATURE_INDEX, FEATURE_NAMES, FLOW_VALID_INDEX, MASK_VALID_INDEX, gate_masks
from src.training import metrics as M

HEURISTIC_CUES: Tuple[str, ...] = ("flow_v_mean", "centroid_dz", "anchor_stability", "area_growth")
OPERATING_POINTS = ("balanced", "low_false_alarm")


# =============================================================================
# Pooling and a tiny logistic regression
# =============================================================================


def pooled_feature_names() -> List[str]:
    return [f"{n}_mean" for n in FEATURE_NAMES] + [f"{n}_std" for n in FEATURE_NAMES]


def pooled_features(X) -> np.ndarray:
    """(N, 28): per feature, mean and std over the timesteps where its gate is open (0 when none).

    The validity flags (always "open") become the fraction of valid frames and its spread.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        X = X[None]
    g = gate_masks(X).astype(np.float64)
    n_open = g.sum(axis=1)                                      # (N, 14)
    safe = np.maximum(n_open, 1.0)
    mean = (X * g).sum(axis=1) / safe
    var = (((X - mean[:, None, :]) ** 2) * g).sum(axis=1) / safe
    mean = np.where(n_open > 0, mean, 0.0)
    std = np.where(n_open > 1, np.sqrt(np.maximum(var, 0.0)), 0.0)
    return np.concatenate([mean, std], axis=1)


class _LogisticRegression:
    """L2-regularised logistic regression fitted by Newton's method on standardised inputs."""

    def __init__(self, l2: float = 1.0, max_iter: int = 100, tol: float = 1e-10):
        self.l2, self.max_iter, self.tol = float(l2), int(max_iter), float(tol)
        self.mu = self.sd = self.coef = None
        self.converged = False

    def _design(self, X) -> np.ndarray:
        Xs = (np.asarray(X, dtype=np.float64) - self.mu) / self.sd
        return np.concatenate([np.ones((len(Xs), 1)), Xs], axis=1)

    def fit(self, X, y, weights=None) -> "_LogisticRegression":
        X = np.asarray(X, dtype=np.float64)
        t = (np.asarray(y, dtype=np.float64).ravel() > 0.5).astype(np.float64)
        s = np.ones(len(t)) if weights is None else np.asarray(weights, dtype=np.float64)
        s = s * len(s) / s.sum()
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        A = self._design(X)
        reg = np.full(A.shape[1], self.l2)
        reg[0] = 0.0
        w = np.zeros(A.shape[1])
        for _ in range(self.max_iter):
            p = M.sigmoid(A @ w)
            grad = A.T @ (s * (p - t)) + reg * w
            H = (A * (s * p * (1 - p))[:, None]).T @ A + np.diag(reg) + 1e-9 * np.eye(len(w))
            step = np.linalg.solve(H, grad)
            w -= step
            if np.max(np.abs(step)) < self.tol:
                self.converged = True
                break
        self.coef = w
        return self

    def decision_function(self, X) -> np.ndarray:
        return self._design(X) @ self.coef


# =============================================================================
# Baselines with the GRU's decision rules
# =============================================================================


class _PooledBaseline:
    name = "pooled"
    columns: Sequence[int] = ()

    def __init__(self, l2: float = 1.0):
        self.l2 = l2
        self.models: List[_LogisticRegression] = []
        self.thresholds: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.threshold_fit: Dict[str, Any] = {}

    def inputs(self, X) -> np.ndarray:
        return pooled_features(X)[:, list(self.columns)]

    def fit(self, train: FeatureSet) -> "_PooledBaseline":
        """Coefficients from the train split only (each simulation weighted once)."""
        F = self.inputs(train.X)
        w = M.per_group_weights(train.groups)
        self.models = [_LogisticRegression(self.l2).fit(F, train.y[:, j], w) for j in range(len(LABEL_NAMES))]
        return self

    def logits(self, X) -> np.ndarray:
        F = self.inputs(X)
        return np.stack([m.decision_function(F) for m in self.models], axis=1)

    def calibrate(self, val_calib: FeatureSet, cfg: Mapping[str, Any]) -> "_PooledBaseline":
        """Operating-point thresholds on val_calib (same rules as the GRU; no temperature, no abstention)."""
        ops = M.fit_operating_points(self.logits(val_calib.X), val_calib.y, val_calib.categories,
                                     val_calib.groups, cfg)
        self.thresholds, self.threshold_fit = ops["thresholds"], ops["fit"]
        return self

    def predict(self, X, operating_point: str = "balanced") -> Dict[str, np.ndarray]:
        z = self.logits(X)
        thr = np.array([self.thresholds[operating_point][n]["logit"] for n in LABEL_NAMES])
        hard = (z >= thr).astype(np.int64)
        return {"logits": z, "probs": M.sigmoid(z), "hard": hard, "decisions": hard}

    def describe(self) -> Dict[str, Any]:
        names = pooled_feature_names()
        return {
            "inputs": [names[c] for c in self.columns],
            "coefficients": {n: {"intercept": float(m.coef[0]),
                                 **{names[c]: float(v) for c, v in zip(self.columns, m.coef[1:])}}
                             for n, m in zip(LABEL_NAMES, self.models)},
            "input_mean": [float(v) for v in self.models[0].mu] if self.models else None,
            "input_std": [float(v) for v in self.models[0].sd] if self.models else None,
            "converged": [bool(m.converged) for m in self.models],
            "l2": self.l2,
            "thresholds": self.thresholds,
            "threshold_fit_source": "val_calib",
        }


class HeuristicBaseline(_PooledBaseline):
    """Experiment A: logistic form on 4 time-pooled motion cues (coefficients from train only)."""

    name = "A_heuristic"
    columns = tuple(FEATURE_INDEX[c] for c in HEURISTIC_CUES)


class PooledLogisticBaseline(_PooledBaseline):
    """Logistic regression on mean and std over time of all 14 features (no temporal order)."""

    name = "pooled_logistic"
    columns = tuple(range(2 * len(FEATURE_NAMES)))


def _flag_inputs(X) -> np.ndarray:
    """Summaries of the validity-flag pattern: valid fractions, first-frame flags, mask on/off switches."""
    X = np.asarray(X, dtype=np.float64)
    m, f = X[:, :, MASK_VALID_INDEX], X[:, :, FLOW_VALID_INDEX]
    switches = np.abs(np.diff(m, axis=1)).sum(axis=1)
    return np.stack([m.mean(axis=1), f.mean(axis=1), m[:, 0], f[:, 0], switches], axis=1)


def flags_only_probe(fs: FeatureSet, threshold: float = 0.6, train_split: str = "train",
                     eval_splits: Sequence[str] = ("val_select", "val_calib", "test_iid", "test_ood"),
                     ) -> Dict[str, Any]:
    """Can the validity flags alone predict a label?

    A logistic regression on flag summaries (fraction of valid mask / flow
    frames, first-frame flags, number of mask on/off switches) is fitted on
    the train split and scored by AUROC on the held-out splits pooled (the
    decision) and per split (for reference; small splits are noisy). The
    AUROC is not folded: a trained probe below 0.5 has learned nothing that
    generalises. Flagged when the pooled AUROC exceeds `threshold`.
    """
    train = fs.split(train_split)
    out: Dict[str, Any] = {"threshold": threshold, "train_split": train_split,
                           "inputs": ["mask_valid_fraction", "flow_valid_fraction", "mask_valid_t0",
                                      "flow_valid_t0", "mask_switches"],
                           "pooled": {}, "per_split": {}, "flagged": []}
    held = fs.split(list(eval_splits))
    if len(train) == 0 or len(held) == 0:
        out.update(status="skipped", flag=False)
        return out
    w = M.per_group_weights(train.groups)
    F = _flag_inputs(train.X)
    models = [_LogisticRegression(1.0).fit(F, train.y[:, j], w) for j in range(len(LABEL_NAMES))]

    def score(part: FeatureSet) -> Dict[str, Optional[float]]:
        Fe = _flag_inputs(part.X)
        res = {}
        for j, name in enumerate(LABEL_NAMES):
            auc = M.auroc(models[j].decision_function(Fe), part.y[:, j])
            res[name] = None if auc is None else round(auc, 4)
        return res

    out["pooled"] = {"splits": [s for s in eval_splits if np.any(held.splits == s)], "n": len(held),
                     "auroc": score(held)}
    for name, auc in out["pooled"]["auroc"].items():
        if auc is not None and auc > threshold:
            out["flagged"].append({"label": name, "split": "pooled_held_out", "auc": auc})
    for split in eval_splits:
        ev = fs.split(split)
        if len(ev):
            out["per_split"][split] = {"auroc": score(ev), "n": len(ev)}
    out["flag"] = bool(out["flagged"])
    return out


def univariate_aucs(fs: FeatureSet, split: str = "train", top: Optional[int] = None) -> Dict[str, Any]:
    """Folded AUROC of every pooled feature (mean / std over time) per label, best first."""
    sub = fs.split(split)
    F = pooled_features(sub.X)
    names = pooled_feature_names()
    out: Dict[str, Any] = {"split": split, "n": len(sub), "per_label": {}}
    for j, label in enumerate(LABEL_NAMES):
        res = {}
        for k, name in enumerate(names):
            if np.std(F[:, k]) == 0:
                continue
            auc = M.auroc(F[:, k], sub.y[:, j])
            if auc is not None:
                res[name] = {"folded_auc": round(max(auc, 1 - auc), 4), "direction": "+" if auc >= 0.5 else "-"}
        ranked = sorted(res.items(), key=lambda kv: -kv[1]["folded_auc"])
        out["per_label"][label] = dict(ranked[:top] if top else ranked)
    return out


def _evaluate(baseline: _PooledBaseline, fs: FeatureSet, splits: Sequence[str], n_boot: int, seed: int,
              n_bins: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for split in splits:
        ev = fs.split(split)
        if len(ev) == 0:
            continue
        out[split] = {}
        for op in OPERATING_POINTS:
            p = baseline.predict(ev.X, op)
            doc = M.classification_metrics(ev.y, p["probs"], p["hard"], p["decisions"], ev.categories, n_bins=n_bins)
            if n_boot:
                doc["ci95"] = M.bootstrap_classification_ci(ev.y, p["probs"], p["hard"], p["decisions"],
                                                            ev.categories, ev.groups, n_boot=n_boot, seed=seed,
                                                            n_bins=n_bins)
            out[split][op] = doc
    return out


def run_baselines(fs: FeatureSet, cfg: Mapping[str, Any], splits: Sequence[str] = ("test_iid", "test_ood"),
                  n_boot: int = 0, seed: int = 0) -> Dict[str, Any]:
    """Fit A (heuristic) and the pooled logistic regression on train, thresholds on val_calib, evaluate."""
    train, calib = fs.split("train"), fs.split("val_calib")
    n_bins = int(cfg.get("evaluation", {}).get("calibration_bins", 10))
    out: Dict[str, Any] = {}
    for cls in (HeuristicBaseline, PooledLogisticBaseline):
        b = cls().fit(train).calibrate(calib, cfg)
        out[cls.name] = {"description": (cls.__doc__ or "").strip(), **b.describe(),
                         "results": _evaluate(b, fs, splits, n_boot, seed, n_bins)}
    return out
