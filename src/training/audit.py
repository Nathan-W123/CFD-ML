"""Shortcut audit at the parameter level: can a context or render setting alone predict a label?

Scenario context (wind, terrain, atmosphere, spin-up, haze, camera nuisances)
and render settings are sampled from the same distributions for every class.
Rejection and re-drawing (label support, label visibility) can still make the
ACCEPTED data class-dependent, e.g. smoke sequences surviving only at lower
wind. This module measures, on accepted sequences, how well each scalar
parameter separates each label (AUROC, folded so 0.5 = no information and
1.0 = perfect separation either way) and flags any above a threshold.

Where this fits: `scripts/evaluate_physics.py` writes the result into
evaluation.json ("shortcut_audit.parameters"); the feature-level probes
(flags-only model, univariate feature AUCs) live in baselines.py.
Calls into: numpy only.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


def auroc(score: np.ndarray, label: np.ndarray) -> Optional[float]:
    """Area under the ROC curve (ties averaged); None when only one class is present."""
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label).astype(bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def folded(auc: Optional[float]) -> Optional[float]:
    return None if auc is None else max(auc, 1.0 - auc)


def _columns(records: List[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    """Scalar columns from a list of flat dicts; categoricals become one-hot columns."""
    keys: List[str] = []
    for r in records:
        for k in r:
            if k not in keys:
                keys.append(k)
    cols: Dict[str, np.ndarray] = {}
    for k in keys:
        vals = [r.get(k) for r in records]
        if all(isinstance(v, bool) or v is None for v in vals):
            cols[k] = np.array([float(bool(v)) for v in vals])
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) or v is None for v in vals):
            cols[k] = np.array([np.nan if v is None else float(v) for v in vals])
        elif all(isinstance(v, str) or v is None for v in vals):
            for level in sorted({v for v in vals if v is not None}):
                cols[f"{k}={level}"] = np.array([float(v == level) for v in vals])
    return cols


def parameter_audit(records: List[Mapping[str, Any]], y: np.ndarray, label_names: Iterable[str] = ("smoke", "fog"),
                    threshold: float = 0.6) -> Dict[str, Any]:
    """Folded AUROC of every scalar parameter for every label; flags those above `threshold`."""
    cols = _columns(records)
    y = np.asarray(y)
    out: Dict[str, Any] = {"threshold": threshold, "n": int(len(records)), "per_label": {}, "flagged": []}
    for j, name in enumerate(label_names):
        res = {}
        for k, v in cols.items():
            ok = np.isfinite(v)
            if ok.sum() < 10 or np.nanstd(v) == 0:
                continue
            a = folded(auroc(v[ok], y[ok, j] > 0.5))
            if a is not None:
                res[k] = round(a, 4)
                if a > threshold:
                    out["flagged"].append({"label": name, "parameter": k, "folded_auc": round(a, 4)})
        out["per_label"][name] = dict(sorted(res.items(), key=lambda kv: -kv[1]))
    return out
