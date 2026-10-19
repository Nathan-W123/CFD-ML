"""Classification metrics, grouped bootstrap intervals and the fitted decision rules (numpy only).

Metrics (per label unless noted): precision, recall, F1 at the operating
threshold; AUROC (ties averaged) and AUPRC (average precision, step-wise);
Brier score; expected calibration error with EQUAL-MASS bins (every bin
holds the same number of sequences, so sparse probability ranges do not get
noisy one-sample bins); the 4-class confusion of the two binary decisions
(smoke_only / fog_only / neither / both) and the smoke/fog confusions read
off it; the false-smoke rate on fog_only sequences (the false alarm this
expert exists to reduce); abstention rate and selective accuracy.

Undefined values (a precision with no positive predictions, an AUROC with one
class) are None, never NaN, so they serialise to JSON as null.

Unknown labels: a target may be NaN (real footage whose fog is not
annotated). Every label-dependent function here uses only the rows where
that label is known: per-label metrics, the 4-class confusion (rows with
BOTH labels known), the false-smoke rate (fog_only needs both labels), the
fitted decision rules and the bootstrap. A label with no known rows gets
None metrics and status "no_labels", never 0.

Confidence intervals: `grouped_bootstrap` resamples GROUPS (sim_id for
synthetic data, camera id for real footage) with replacement, because the
rendered variants of one simulation are not independent samples.

Decision rules fitted on val_calib (used by `src.training.train` and the
baselines): `fit_temperature` (per-label temperature, bounded),
`threshold_max_f1`, `threshold_max_false_rate` (the low-false-alarm smoke
threshold) and `fit_abstain_margin` (the logit-space abstain band). They take
optional per-sample weights so counts can be made per simulation rather than
per rendered variant (`per_group_weights`).

Where this fits: `train.py` (calibration), `evaluate.py` and `baselines.py`.
Calls into: numpy only.
Start reading at: classification_metrics.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.constants import CATEGORIES, CATEGORY_LABELS, LABEL_NAMES

ABSTAIN = -1


# =============================================================================
# Small helpers
# =============================================================================


def _bool(y) -> np.ndarray:
    return np.asarray(y).astype(np.float64).ravel() > 0.5


def _weights(w, n: int) -> np.ndarray:
    return np.ones(n) if w is None else np.asarray(w, dtype=np.float64).ravel()


def _ratio(a: float, b: float) -> Optional[float]:
    return None if b <= 0 else float(a / b)


def known_mask(y) -> np.ndarray:
    """True where a target is known (NaN = unknown, e.g. unannotated fog on real footage); y's shape."""
    return ~np.isnan(np.asarray(y, dtype=np.float64))


def _drop_unknown(y, *arrays):
    """(y, *arrays) restricted to the rows where the 1-D target y is known (None arrays pass through)."""
    t = np.asarray(y, dtype=np.float64).ravel()
    k = ~np.isnan(t)
    if k.all():
        return (y, *arrays)
    out = [None if a is None else np.asarray(a).ravel()[k] for a in arrays]
    return (t[k], *out)


def sigmoid(z) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def logit(p, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def per_group_weights(groups) -> np.ndarray:
    """Weights 1 / (rows of the row's group): every group (simulation) counts once in total."""
    groups = np.asarray(groups)
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv]


# =============================================================================
# Threshold metrics
# =============================================================================


def confusion_counts(y_true, y_pred, weights=None) -> Dict[str, float]:
    y_true, y_pred, weights = _drop_unknown(y_true, y_pred, weights)
    t, p = _bool(y_true), _bool(y_pred)
    w = _weights(weights, len(t))
    return {"tp": float(w[t & p].sum()), "fp": float(w[~t & p].sum()),
            "tn": float(w[~t & ~p].sum()), "fn": float(w[t & ~p].sum())}


def precision_recall_f1(y_true, y_pred, weights=None) -> Dict[str, Optional[float]]:
    c = confusion_counts(y_true, y_pred, weights)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    return {"precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn), **c}


# =============================================================================
# Ranking metrics
# =============================================================================


def _average_ranks(score: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged."""
    order = np.argsort(score, kind="mergesort")
    _, inv, counts = np.unique(score[order], return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg = (cum - counts + 1 + cum) / 2.0
    ranks = np.empty(len(score))
    ranks[order] = avg[inv]
    return ranks


def auroc(score, y) -> Optional[float]:
    """Area under the ROC curve (Mann-Whitney, ties count one half); None with one class."""
    y, score = _drop_unknown(y, score)
    s = np.asarray(score, dtype=np.float64).ravel()
    t = _bool(y)
    n_pos, n_neg = int(t.sum()), int((~t).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(s)
    return float((ranks[t].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _descending_counts(score, y) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct thresholds (descending) and cumulative TP / FP when predicting score >= threshold."""
    y, score = _drop_unknown(y, score)
    s = np.asarray(score, dtype=np.float64).ravel()
    t = _bool(y)
    if s.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    order = np.argsort(-s, kind="mergesort")
    s, t = s[order], t[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]        # last index of each distinct value
    tp = np.cumsum(t)[last].astype(np.float64)
    fp = np.cumsum(~t)[last].astype(np.float64)
    return s[last], tp, fp


def roc_curve(score, y) -> Dict[str, np.ndarray]:
    """{"fpr", "tpr", "thresholds"}, starting at (0, 0) with threshold +inf."""
    thr, tp, fp = _descending_counts(score, y)
    n_pos, n_neg = max(tp[-1] if len(tp) else 0, 0), max(fp[-1] if len(fp) else 0, 0)
    fpr = np.r_[0.0, fp / n_neg] if n_neg else np.r_[0.0, np.zeros_like(fp)]
    tpr = np.r_[0.0, tp / n_pos] if n_pos else np.r_[0.0, np.zeros_like(tp)]
    return {"fpr": fpr, "tpr": tpr, "thresholds": np.r_[np.inf, thr]}


def pr_curve(score, y) -> Dict[str, np.ndarray]:
    """{"precision", "recall", "thresholds"} at every distinct threshold (descending)."""
    thr, tp, fp = _descending_counts(score, y)
    n_pos = tp[-1] if len(tp) else 0.0
    precision = tp / np.maximum(tp + fp, 1e-300)
    recall = tp / n_pos if n_pos > 0 else np.zeros_like(tp)
    return {"precision": precision, "recall": recall, "thresholds": thr}


def auprc(score, y) -> Optional[float]:
    """Average precision sum_k (R_k - R_{k-1}) P_k over distinct thresholds; None without positives
    (or without negatives, where it is trivially 1 and says nothing)."""
    y, score = _drop_unknown(y, score)
    t = _bool(y)
    if t.sum() == 0 or (~t).sum() == 0:
        return None
    c = pr_curve(score, y)
    recall_prev = np.r_[0.0, c["recall"][:-1]]
    return float(np.sum((c["recall"] - recall_prev) * c["precision"]))


# =============================================================================
# Calibration metrics
# =============================================================================


def brier_score(prob, y) -> Optional[float]:
    y, prob = _drop_unknown(y, prob)
    p = np.asarray(prob, dtype=np.float64).ravel()
    if p.size == 0:
        return None
    return float(np.mean((p - _bool(y)) ** 2))


def reliability_bins(prob, y, n_bins: int = 10) -> List[Dict[str, float]]:
    """Equal-mass bins (sorted by probability, split into n_bins nearly equal groups); known labels only."""
    y, prob = _drop_unknown(y, prob)
    p = np.asarray(prob, dtype=np.float64).ravel()
    t = _bool(y).astype(np.float64)
    order = np.argsort(p, kind="mergesort")
    out = []
    for idx in np.array_split(order, min(n_bins, max(len(p), 1))):
        if len(idx) == 0:
            continue
        out.append({"p_mean": float(p[idx].mean()), "y_mean": float(t[idx].mean()), "n": int(len(idx)),
                    "p_lo": float(p[idx].min()), "p_hi": float(p[idx].max())})
    return out


def expected_calibration_error(prob, y, n_bins: int = 10) -> Optional[float]:
    """sum_b (n_b / N) |mean(y_b) - mean(p_b)| over equal-mass bins (N = rows with a known label)."""
    y, prob = _drop_unknown(y, prob)
    p = np.asarray(prob).ravel()
    if p.size == 0:
        return None
    bins = reliability_bins(p, y, n_bins)
    return float(sum(b["n"] * abs(b["y_mean"] - b["p_mean"]) for b in bins) / len(p))


# =============================================================================
# Decisions: 4-class confusion, false smoke, abstention
# =============================================================================


def categories_from_decisions(decisions) -> np.ndarray:
    """(N,) category index into src.constants.CATEGORIES, or -1 when either decision abstains."""
    d = np.asarray(decisions).astype(np.int64)
    out = np.full(len(d), ABSTAIN, dtype=np.int64)
    for k, name in enumerate(CATEGORIES):
        s, f = CATEGORY_LABELS[name]
        out[(d[:, 0] == s) & (d[:, 1] == f)] = k
    return out


def confusion_4class(y_true, decisions) -> Dict[str, Any]:
    """Rows: true category; columns: predicted category; abstentions counted separately per row.

    Only rows with BOTH labels known have a true category; the others are
    counted in "n_unknown_labels" and left out.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, len(LABEL_NAMES))
    both = known_mask(y_true).all(axis=1)
    true_cat = categories_from_decisions(y_true[both] > 0.5)
    pred_cat = categories_from_decisions(np.asarray(decisions).reshape(-1, len(LABEL_NAMES))[both])
    m = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype=np.int64)
    abstained = np.zeros(len(CATEGORIES), dtype=np.int64)
    for i in range(len(CATEGORIES)):
        rows = true_cat == i
        abstained[i] = int(np.sum(pred_cat[rows] == ABSTAIN))
        for j in range(len(CATEGORIES)):
            m[i, j] = int(np.sum(pred_cat[rows] == j))
    return {"classes": list(CATEGORIES), "matrix": m.tolist(), "abstained": abstained.tolist(),
            "n": int(len(true_cat)), "n_unknown_labels": int((~both).sum())}


def smoke_fog_confusion(conf: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Rates read off a 4-class confusion: fog_only called smoke(_only) and smoke_only called fog(_only)."""
    m = np.asarray(conf["matrix"], dtype=np.float64)
    ix = {name: k for k, name in enumerate(conf["classes"])}
    fo, so, bo = ix["fog_only"], ix["smoke_only"], ix["both"]
    return {
        "fog_only_as_smoke_only": _ratio(m[fo, so], m[fo].sum()),
        "fog_only_with_smoke": _ratio(m[fo, so] + m[fo, bo], m[fo].sum()),
        "smoke_only_as_fog_only": _ratio(m[so, fo], m[so].sum()),
        "smoke_only_with_fog": _ratio(m[so, fo] + m[so, bo], m[so].sum()),
    }


def false_smoke_rate(smoke_decisions, categories) -> Dict[str, Any]:
    """Share of fog_only sequences with a smoke decision of 1 (among those not abstaining)."""
    d = np.asarray(smoke_decisions).astype(np.int64).ravel()
    fog = np.asarray(categories).astype(str) == "fog_only"
    decided = fog & (d != ABSTAIN)
    n_false = int(np.sum(decided & (d == 1)))
    return {"rate": _ratio(n_false, int(decided.sum())), "n_fog_only": int(fog.sum()),
            "n_false_smoke": n_false, "n_abstained": int(np.sum(fog & (d == ABSTAIN)))}


def abstain_rate(decisions, names: Sequence[str] = LABEL_NAMES) -> Dict[str, Optional[float]]:
    d = np.asarray(decisions).astype(np.int64)
    if len(d) == 0:
        return {**{n: None for n in names}, "any": None}
    out = {n: float(np.mean(d[:, j] == ABSTAIN)) for j, n in enumerate(names)}
    out["any"] = float(np.mean(np.any(d == ABSTAIN, axis=1)))
    return out


def selective_accuracy(y_true, decisions, names: Sequence[str] = LABEL_NAMES) -> Dict[str, Optional[float]]:
    """Accuracy over non-abstained decisions, per label (rows with that label known) and jointly
    (rows with both labels known, both decided and right). coverage_joint is over every row."""
    y = np.asarray(y_true, dtype=np.float64)
    known = known_mask(y)
    t = (y > 0.5).astype(np.int64)
    d = np.asarray(decisions).astype(np.int64)
    out: Dict[str, Optional[float]] = {}
    for j, n in enumerate(names):
        ok = (d[:, j] != ABSTAIN) & known[:, j]
        out[n] = _ratio(float(np.sum(d[ok, j] == t[ok, j])), float(ok.sum()))
    both = np.all(d != ABSTAIN, axis=1) & known.all(axis=1)
    out["joint"] = _ratio(float(np.sum(np.all(d[both] == t[both], axis=1))), float(both.sum()))
    out["coverage_joint"] = float(np.all(d != ABSTAIN, axis=1).mean()) if len(d) else None
    return out


# =============================================================================
# Everything at once
# =============================================================================


LABEL_METRIC_KEYS: Tuple[str, ...] = ("precision", "recall", "f1", "auroc", "auprc", "brier", "ece", "n", "n_pos",
                                      "prevalence", "abstain_rate", "selective_accuracy", "selective_precision",
                                      "selective_recall")


def _label_metrics(t, p, hard, dec, n_bins) -> Dict[str, Any]:
    """Metrics of one label on the rows where it is known; a label with no known row is all None."""
    n_rows = int(len(np.asarray(t).ravel()))
    t, p, hard, dec = _drop_unknown(t, p, hard, dec)
    n_unknown = n_rows - int(len(t))
    if len(t) == 0:
        out: Dict[str, Any] = {k: None for k in LABEL_METRIC_KEYS}
        out.update(n=0, n_pos=None, n_unknown=n_unknown, status="no_labels")
        return out
    prf = precision_recall_f1(t, hard)
    out = {k: prf[k] for k in ("precision", "recall", "f1")}
    out.update({
        "auroc": auroc(p, t), "auprc": auprc(p, t), "brier": brier_score(p, t),
        "ece": expected_calibration_error(p, t, n_bins),
        "n": int(len(t)), "n_pos": int(_bool(t).sum()),
        "prevalence": _ratio(float(_bool(t).sum()), float(len(t))),
        "n_unknown": n_unknown, "status": "evaluated",
    })
    if dec is not None:
        ok = dec != ABSTAIN
        out["abstain_rate"] = _ratio(float((~ok).sum()), float(len(dec)))
        out["selective_accuracy"] = _ratio(float(np.sum(dec[ok] == _bool(t)[ok])), float(ok.sum()))
        sel = precision_recall_f1(_bool(t)[ok], dec[ok] == 1)
        out["selective_precision"], out["selective_recall"] = sel["precision"], sel["recall"]
    return out


def classification_metrics(y, probs, hard, decisions=None, categories=None,
                           names: Sequence[str] = LABEL_NAMES, n_bins: int = 10) -> Dict[str, Any]:
    """Point estimates for one set of predictions.

    y (N, 2) targets (NaN = unknown); probs (N, 2) calibrated probabilities;
    hard (N, 2) 0/1 threshold decisions; decisions (N, 2) with -1 for abstain
    (default: hard); categories (N,) str (for the false-smoke rate; derived
    from y when omitted). Per-label entries use only the rows where that label
    is known ("status": "evaluated" / "no_labels"); the 4-class confusion and
    the false-smoke rate only rows with both labels known.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1, len(names))
    probs = np.asarray(probs, dtype=np.float64).reshape(-1, len(names))
    hard = np.asarray(hard).astype(np.int64).reshape(-1, len(names))
    dec = hard if decisions is None else np.asarray(decisions).astype(np.int64).reshape(-1, len(names))
    known = known_mask(y)
    both = known.all(axis=1)
    if categories is None:
        categories = np.array([CATEGORIES[k] if b else "unknown" for k, b in
                               zip(categories_from_decisions(y > 0.5), both)]) if len(y) else np.array([])
    # A category needs both labels: rows with an unknown label never count as fog_only.
    categories = np.where(both, np.asarray(categories).astype(str), "unknown") if len(y) else np.array([])
    conf = confusion_4class(y, hard)
    conf_sel = confusion_4class(y, dec)
    return {
        "n": int(len(y)),
        "n_known": {**{n: int(known[:, j].sum()) for j, n in enumerate(names)}, "both": int(both.sum())},
        "per_label": {n: _label_metrics(y[:, j], probs[:, j], hard[:, j], dec[:, j], n_bins)
                      for j, n in enumerate(names)},
        "confusion_4class": conf,
        "confusion_4class_selective": conf_sel,
        "smoke_fog_confusion": smoke_fog_confusion(conf),
        "false_smoke_on_fog_only": false_smoke_rate(hard[:, 0], categories),
        "false_smoke_on_fog_only_selective": false_smoke_rate(dec[:, 0], categories),
        "abstain_rate": abstain_rate(dec, names),
        "selective_accuracy": selective_accuracy(y, dec, names),
    }


# Scalars bootstrapped by default: "label.metric" paths into classification_metrics.
CI_METRICS: Tuple[str, ...] = (
    "smoke.precision", "smoke.recall", "smoke.f1", "smoke.auroc", "smoke.auprc", "smoke.brier", "smoke.ece",
    "fog.precision", "fog.recall", "fog.f1", "fog.auroc", "fog.auprc", "fog.brier", "fog.ece",
    "false_smoke_on_fog_only.rate", "abstain_rate.any", "selective_accuracy.joint",
)


def _lookup(doc: Dict[str, Any], path: str) -> Optional[float]:
    head, key = path.split(".", 1)
    node = doc["per_label"].get(head) if head in doc["per_label"] else doc.get(head)
    if not isinstance(node, dict):
        return None
    v = node.get(key)
    return None if v is None else float(v)


def grouped_bootstrap(fn: Callable[[np.ndarray], Dict[str, Optional[float]]], groups, n_boot: int = 1000,
                      seed: int = 0, alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
    """Percentile CIs of every scalar `fn(row_indices)` returns, resampling groups with replacement.

    Returns {name: {"lo", "hi", "n_valid"}}; a statistic undefined in more
    than half of the resamples gets lo = hi = None.
    """
    groups = np.asarray(groups).astype(str)
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="mergesort")
    starts = np.r_[0, np.cumsum(np.bincount(inv, minlength=len(uniq)))]
    rows = [order[starts[g]:starts[g + 1]] for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    samples: Dict[str, List[float]] = {}
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([rows[g] for g in pick]) if len(uniq) else np.zeros(0, np.int64)
        for name, value in fn(idx).items():
            samples.setdefault(name, [])
            if value is not None and np.isfinite(value):
                samples[name].append(float(value))
    out = {}
    for name, vals in samples.items():
        if len(vals) * 2 < n_boot:
            out[name] = {"lo": None, "hi": None, "n_valid": len(vals)}
        else:
            lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            out[name] = {"lo": float(lo), "hi": float(hi), "n_valid": len(vals)}
    return out


def bootstrap_classification_ci(y, probs, hard, decisions, categories, groups, n_boot: int = 1000,
                                seed: int = 0, n_bins: int = 10,
                                metrics: Sequence[str] = CI_METRICS) -> Dict[str, Dict[str, Any]]:
    """95 % grouped-bootstrap intervals of the `metrics` paths of classification_metrics."""
    y, probs, hard, decisions = (np.asarray(a) for a in (y, probs, hard, decisions))
    categories = np.asarray(categories).astype(str)

    def fn(idx):
        doc = classification_metrics(y[idx], probs[idx], hard[idx], decisions[idx], categories[idx], n_bins=n_bins)
        return {m: _lookup(doc, m) for m in metrics}

    return grouped_bootstrap(fn, groups, n_boot=n_boot, seed=seed)


# =============================================================================
# Decision rules fitted on val_calib
# =============================================================================


def _weighted_bce(z, t, w) -> float:
    # log(1 + exp(-z)) for positives, log(1 + exp(z)) for negatives, stably.
    loss = np.logaddexp(0.0, np.where(t, -z, z))
    return float(np.sum(w * loss) / np.sum(w))


def fit_temperature(logits, y, bounds: Tuple[float, float] = (0.5, 5.0), weights=None,
                    tol: float = 1e-6) -> Tuple[float, Dict[str, Any]]:
    """Temperature T minimising the (weighted) NLL of sigmoid(logit / T), searched within `bounds`.

    The NLL is convex in 1 / T, so a golden-section search finds the
    constrained optimum exactly. When it lies at a bound (docs/design.md
    section 11): if val_calib is separable (every positive logit above every
    negative one, so the NLL keeps falling as T -> 0) the fit is not trusted,
    T = 1 and `fallback` is set; otherwise ("optimum_at_upper_bound" /
    "optimum_at_lower_bound": logits far too confident / timid, or
    uninformative) the bound itself is used, with `clipped` set: by
    convexity it has a lower calibration NLL than T = 1.
    """
    y, logits, weights = _drop_unknown(y, logits, weights)
    z = np.asarray(logits, dtype=np.float64).ravel()
    t = _bool(y)
    w = _weights(weights, len(z))
    info: Dict[str, Any] = {"bounds": list(bounds), "fitted_T": None, "at_bound": None, "fallback": False,
                            "reason": None, "nll_T1": _weighted_bce(z, t, w) if len(z) else None}
    if len(z) == 0:
        info.update(fallback=True, reason="no_known_labels")
        return 1.0, info
    if t.all() or (~t).all():
        info.update(fallback=True, reason="single_class")
        return 1.0, info
    lo, hi = 1.0 / bounds[1], 1.0 / bounds[0]           # search a = 1 / T
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = _weighted_bce(c * z, t, w), _weighted_bce(d * z, t, w)
    while b - a > tol:
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = _weighted_bce(c * z, t, w)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = _weighted_bce(d * z, t, w)
    a_opt = 0.5 * (a + b)
    info["fitted_T"] = float(1.0 / a_opt)
    edge = 1e-3 * (hi - lo)
    if a_opt - lo < edge or hi - a_opt < edge:
        side = "upper" if a_opt - lo < edge else "lower"              # upper / lower bound of T
        bound_T = float(bounds[1] if side == "upper" else bounds[0])
        if bool(z[t].min() > z[~t].max()):
            info.update(at_bound=side, fallback=True, bound_T=bound_T, reason="separable", nll=info["nll_T1"])
            return 1.0, info
        info.update(at_bound=side, clipped=True, bound_T=bound_T, reason=f"optimum_at_{side}_bound",
                    nll=_weighted_bce(z / bound_T, t, w))
        return bound_T, info
    info["nll"] = _weighted_bce(a_opt * z, t, w)
    return float(1.0 / a_opt), info


def _candidate_thresholds(score: np.ndarray) -> np.ndarray:
    """Midpoints between consecutive distinct scores, plus one below and one above all of them."""
    u = np.unique(score)
    if len(u) == 0:
        return np.array([0.0])
    return np.r_[u[0] - 1.0, 0.5 * (u[1:] + u[:-1]), u[-1] + 1.0]


def threshold_max_f1(score, y, weights=None) -> Tuple[float, Dict[str, Any]]:
    """Threshold (on the score scale; predict score >= threshold) maximising the (weighted) F1."""
    y, score, weights = _drop_unknown(y, score, weights)
    s = np.asarray(score, dtype=np.float64).ravel()
    t = _bool(y)
    w = _weights(weights, len(s))
    cands = _candidate_thresholds(s)
    order = np.argsort(s)
    s_sorted, pos_w, neg_w = s[order], (w * t)[order], (w * ~t)[order]
    # Weight at or above each candidate: total minus the weight strictly below it.
    below = np.searchsorted(s_sorted, cands, side="left")
    cpos, cneg = np.r_[0.0, np.cumsum(pos_w)], np.r_[0.0, np.cumsum(neg_w)]
    tp = cpos[-1] - cpos[below]
    fp = cneg[-1] - cneg[below]
    fn = cpos[-1] - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-300), 0.0)
    if cpos[-1] <= 0:                    # no positives: F1 is 0 everywhere; predict nothing
        return float(cands[-1]), {"f1": None, "precision": None, "recall": None, "reason": "no_positives"}
    k = int(np.argmax(f1))
    return float(cands[k]), {"f1": float(f1[k]), "precision": _ratio(tp[k], tp[k] + fp[k]),
                             "recall": _ratio(tp[k], tp[k] + fn[k])}


def threshold_max_youden(score, y, weights=None) -> Tuple[float, Dict[str, Any]]:
    """Threshold maximising Youden's J = TPR - FPR (balanced accuracy), weighted.

    Unlike maximum F1, it does not degenerate to "everything positive" when
    the score is weak and the classes are balanced (F1 = 2/3 there for free).
    """
    y, score, weights = _drop_unknown(y, score, weights)
    s = np.asarray(score, dtype=np.float64).ravel()
    t = _bool(y)
    w = _weights(weights, len(s))
    cands = _candidate_thresholds(s)
    if not t.any() or t.all():
        return float(cands[-1] if not t.any() else cands[0]), {"youden_j": None, "reason": "one_class"}
    order = np.argsort(s)
    s_sorted, pos_w, neg_w = s[order], (w * t)[order], (w * ~t)[order]
    below = np.searchsorted(s_sorted, cands, side="left")
    cpos, cneg = np.r_[0.0, np.cumsum(pos_w)], np.r_[0.0, np.cumsum(neg_w)]
    tpr = (cpos[-1] - cpos[below]) / cpos[-1]
    fpr = (cneg[-1] - cneg[below]) / cneg[-1]
    j = tpr - fpr
    k = int(np.argmax(j))
    tp, fp = cpos[-1] - cpos[below][k], cneg[-1] - cneg[below][k]
    return float(cands[k]), {"youden_j": float(j[k]), "tpr": float(tpr[k]), "fpr": float(fpr[k]),
                             "precision": _ratio(tp, tp + fp), "recall": float(tpr[k])}


def threshold_max_false_rate(score, negatives, max_rate: float, weights=None) -> Tuple[float, Dict[str, Any]]:
    """Smallest threshold whose (weighted) rate of `negatives` at or above it is <= max_rate.

    Used for the low-false-alarm smoke threshold: `negatives` marks the
    fog_only sequences; every sequence's score is a candidate location.
    """
    s = np.asarray(score, dtype=np.float64).ravel()
    neg = np.asarray(negatives).astype(bool).ravel()
    w = _weights(weights, len(s))
    cands = _candidate_thresholds(s)
    if not neg.any():
        return float(cands[0]), {"rate": None, "n_negatives": 0}
    ns = s[neg]
    order = np.argsort(ns)
    ns, nw = ns[order], w[neg][order]
    cum = np.r_[0.0, np.cumsum(nw)]
    rate = (cum[-1] - cum[np.searchsorted(ns, cands, side="left")]) / cum[-1]
    ok = np.flatnonzero(rate <= max_rate + 1e-12)
    k = int(ok[0])       # rate is non-increasing in the threshold; the last candidate gives 0
    return float(cands[k]), {"rate": float(rate[k]), "n_negatives": int(neg.sum())}


def fit_abstain_margin(score, y, threshold: float, floor: float = 0.25, target_error: float = 0.05,
                       max_rate: float = 0.3, weights=None) -> Tuple[float, Dict[str, Any]]:
    """Half-width m of the abstain band |score - threshold| < m (score = calibrated logit).

    Starting at `floor`, widen to the smallest m whose confident (non-abstained)
    weighted error is <= target_error while the abstention rate stays <= max_rate.
    When no such m exists, take the m with the lowest confident error among
    those within max_rate (flagged `target_reached = False`). Rows with an
    unknown label are left out; with none known the margin is the floor.
    """
    y, score, weights = _drop_unknown(y, score, weights)
    s = np.asarray(score, dtype=np.float64).ravel()
    t = _bool(y)
    w = _weights(weights, len(s))
    if len(s) == 0:
        return float(floor), {"abstain_rate": None, "confident_error": None, "error_without_abstention": None,
                              "target_reached": False, "floor": float(floor), "target_error": float(target_error),
                              "max_rate": float(max_rate), "reason": "no_known_labels"}
    dist = np.abs(s - threshold)
    err = ((s >= threshold) != t).astype(np.float64)
    order = np.argsort(dist)
    d_sorted, w_sorted, e_sorted = dist[order], w[order], (w * err)[order]
    u = np.unique(d_sorted[d_sorted >= floor])
    cands = np.r_[floor, 0.5 * (u[1:] + u[:-1]) if len(u) > 1 else np.zeros(0)]
    cands = cands[cands >= floor]
    n_abst = np.searchsorted(d_sorted, cands, side="left")       # rows with dist < m abstain
    cw, ce = np.r_[0.0, np.cumsum(w_sorted)], np.r_[0.0, np.cumsum(e_sorted)]
    total_w, total_e = cw[-1], ce[-1]
    rate = cw[n_abst] / max(total_w, 1e-300)
    kept = total_w - cw[n_abst]
    conf_err = np.where(kept > 0, (total_e - ce[n_abst]) / np.maximum(kept, 1e-300), 0.0)
    within = rate <= max_rate + 1e-12
    good = np.flatnonzero(within & (conf_err <= target_error + 1e-12))
    if len(good):
        k, reached = int(good[0]), True
    elif within.any():
        idx = np.flatnonzero(within)
        k, reached = int(idx[np.argmin(conf_err[idx])]), False
    else:
        k, reached = 0, False
    return float(cands[k]), {"abstain_rate": float(rate[k]), "confident_error": float(conf_err[k]),
                             "error_without_abstention": float(total_e / max(total_w, 1e-300)),
                             "target_reached": reached, "floor": float(floor),
                             "target_error": float(target_error), "max_rate": float(max_rate)}


def fit_operating_points(score, y, categories, groups, cfg: Dict[str, Any],
                         names: Sequence[str] = LABEL_NAMES) -> Dict[str, Any]:
    """Per-label thresholds (on the calibrated-logit scale) for every operating point.

    `balanced`: maximum Youden's J per label (thresholds.operating_points.balanced
    = "youden", the default) or maximum F1 ("f1"). `low_false_alarm`: the smoke threshold
    whose false-smoke rate on fog_only sequences is <= the configured maximum
    (never more permissive than the balanced one); fog keeps its balanced
    threshold. Every count is weighted per group (simulation), so a
    simulation rendered three times counts once. Each label uses only the rows
    where it is known (a label with none gets threshold 0 and reason
    "no_known_labels"); the fog_only negatives need both labels known, and
    without any the low-false-alarm smoke threshold equals the balanced one
    (constraint_fitted = False). Returns
    {"thresholds": {op: {label: {"logit", "prob"}}}, "fit": {op: {label: info}}}.
    """
    s = np.asarray(score, dtype=np.float64).reshape(-1, len(names))
    t = np.asarray(y, dtype=np.float64).reshape(-1, len(names))
    groups = np.asarray(groups)
    known = known_mask(t)
    w = per_group_weights(groups)
    max_fsr = float(cfg.get("thresholds", {}).get("operating_points", {})
                    .get("low_false_alarm", {}).get("max_false_smoke_on_fog_only", 0.05))
    thresholds: Dict[str, Dict[str, Dict[str, float]]] = {"balanced": {}, "low_false_alarm": {}}
    fit: Dict[str, Dict[str, Any]] = {"balanced": {}, "low_false_alarm": {}}
    objective = str(cfg.get("thresholds", {}).get("operating_points", {}).get("balanced", "youden"))
    for j, name in enumerate(names):
        k = known[:, j]
        # Weights per group over the known rows (identical to w when every label is known).
        wj = w if k.all() else per_group_weights(groups[k])
        if not k.any():
            thr, info = 0.0, {"reason": "no_known_labels"}
        elif objective == "f1":
            thr, info = threshold_max_f1(s[k, j], t[k, j], wj)
        else:
            thr, info = threshold_max_youden(s[k, j], t[k, j], wj)
        thresholds["balanced"][name] = {"logit": thr, "prob": float(sigmoid(thr))}
        fit["balanced"][name] = {"criterion": "max_f1" if objective == "f1" else "max_youden_j",
                                 "n_known": int(k.sum()), **info}
    fog_only = (np.asarray(categories).astype(str) == "fog_only") & known.all(axis=1)
    for j, name in enumerate(names):
        bal = thresholds["balanced"][name]["logit"]
        if j == 0:
            thr, info = threshold_max_false_rate(s[:, 0], fog_only, max_fsr, w)
            fitted = bool(fog_only.any())
            fit["low_false_alarm"][name] = {"criterion": f"false_smoke_on_fog_only<={max_fsr}", **info,
                                            "constraint_active": bool(fitted and thr > bal),
                                            "constraint_fitted": fitted}
            if not fitted:
                fit["low_false_alarm"][name]["reason"] = ("no fog_only sequences with both labels known: "
                                                          "the threshold equals the balanced one")
                thr = bal
            thr = max(thr, bal)
        else:
            thr = bal
            fit["low_false_alarm"][name] = {"criterion": "same_as_balanced"}
        thresholds["low_false_alarm"][name] = {"logit": float(thr), "prob": float(sigmoid(thr))}
    return {"thresholds": thresholds, "fit": fit}
