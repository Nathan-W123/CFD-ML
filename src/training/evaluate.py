"""Evaluation of the physics expert and assembly of evaluation.json.

`evaluate_model` runs a loaded `PhysicsExpert` (so evaluation applies exactly
the deployed normaliser, temperatures, thresholds and abstention rules) on
the requested splits and operating points and returns, per split and
operating point, the full metric set of `metrics.classification_metrics`
with grouped-bootstrap 95 % intervals (groups = sim_id, camera id for real
data), plus compact metrics per slice: category, regime, hold-out tag, wind
bin (1.5 m/s) and mask source.

`run_evaluation` is the whole synthetic report:

* "synthetic": the exported (val_select-best) seed with CIs and slices on
  test_iid / test_ood, its ROC / PR curves, point estimates on the
  validation splits (marked as such), and mean +- sd over every trained seed;
* "ablations": A (heuristic baseline) and C (this GRU); B and D need real
  labelled footage and are null with the reason; the pooled logistic
  regression is an extra reference;
* "shortcut_audit": flags-only probe, univariate feature AUCs and the
  parameter-level audit (`src.training.audit`), with an overall flag;
* "robustness": every robustness suite against its matched nominal
  re-render; "loro": leave-one-regime-out results if supplied;
* "real": {"status": "pending", ...} until real results are supplied
  (`evaluate_real`, Stage B artifacts on camera-disjoint real test data);
  "synthetic_proxy": optional `run_synthetic_proxy` (pretrained vs scratch
  on a small shifted synthetic domain, flagged not_real_world_evidence);
  "data": counts and generation acceptance;
* figures in the reports directory: 4-class confusion matrices, reliability
  diagrams, ROC / PR curves.

Test splits are only evaluated here, never used for fitting anything.

Where this fits: `scripts/evaluate_physics.py` calls run_evaluation.
Calls into: src.inference.predict (PhysicsExpert), src.training.{metrics,
baselines,audit}, src.data.dataset, matplotlib (Agg) for figures.
Start reading at: run_evaluation, then evaluate_model.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from src.constants import CATEGORIES, LABEL_NAMES
from src.data.dataset import FeatureSet
from src.inference.predict import PhysicsExpert
from src.training import baselines as B
from src.training import metrics as M
from src.utils.io import PathLike, write_json

EVAL_SPLITS = ("test_iid", "test_ood")
VALIDATION_SPLITS = ("val_select", "val_calib")
OPERATING_POINTS = ("balanced", "low_false_alarm")
SLICES = ("category", "regime", "holdout_tag", "wind_bin", "mask_source")
WIND_BIN_MS = 1.5

DISCLAIMER = ("Synthetic evaluation only: these numbers measure how well the motion signal of simulated, "
              "rendered sequences is learned and how robust it is to simulated nuisances. They are not "
              "real-world accuracy and do not show that the model reduces fog-related false alarms on real cameras.")
REAL_PENDING_REASON = ("No labelled real surveillance footage with compatible frame timing (12 observations ~5 s "
                       "apart, each with a frame ~1 s earlier) was available; real-world validation is pending.")
BD_REASON = ("Requires labelled real footage (Stage B: fine-tuning D and the from-scratch comparison B on the "
             "same camera-disjoint real splits); none was available.")


# =============================================================================
# Predictions and slices
# =============================================================================


def predict_featureset(expert: PhysicsExpert, fs: FeatureSet, operating_point: str = "balanced") -> Dict[str, Any]:
    """PhysicsExpert.predict on every row (the stored extraction records feed the timing checks)."""
    if len(fs) == 0:
        z = np.zeros((0, 2))
        return {"probs": z, "hard_decisions": z.astype(int), "decisions": z.astype(int),
                "calibrated_logits": z, "abstain_reasons": []}
    metas = [m.get("feature_meta") if isinstance(m, dict) else None for m in fs.meta]
    return expert.predict(fs.X, operating_point, meta=metas)


def _wind_bin(speed: Optional[float]) -> str:
    if speed is None or not math.isfinite(speed):
        return "unknown"
    lo = math.floor(speed / WIND_BIN_MS) * WIND_BIN_MS
    return f"{lo:.1f}-{lo + WIND_BIN_MS:.1f}"


def slice_values(fs: FeatureSet, name: str) -> np.ndarray:
    """(N,) slice label of every row for a slice name in SLICES."""
    if name == "category":
        return fs.categories
    if name == "regime":
        return fs.regimes
    if name == "mask_source":
        return fs.mask_sources
    if name == "holdout_tag":
        return np.array([t if t else "none" for t in fs.tags])
    if name == "wind_bin":
        out = []
        for m in fs.meta:
            ctx = m.get("context") if isinstance(m, dict) else None
            out.append(_wind_bin(float(ctx["wind_speed_ms"])) if isinstance(ctx, dict) and
                       ctx.get("wind_speed_ms") is not None else "unknown")
        return np.array(out)
    raise KeyError(f"unknown slice {name!r}")


def compact(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """The headline numbers of a classification_metrics dict."""
    out: Dict[str, Any] = {"n": doc["n"]}
    for name in LABEL_NAMES:
        pl = doc["per_label"][name]
        out[name] = {k: pl[k] for k in ("precision", "recall", "f1", "auroc", "auprc", "brier", "ece",
                                        "abstain_rate", "selective_accuracy", "n_pos")}
    out["false_smoke_on_fog_only"] = doc["false_smoke_on_fog_only"]["rate"]
    out["fog_only_with_smoke"] = doc["smoke_fog_confusion"]["fog_only_with_smoke"]
    out["smoke_only_with_fog"] = doc["smoke_fog_confusion"]["smoke_only_with_fog"]
    out["abstain_rate_any"] = doc["abstain_rate"]["any"]
    out["selective_accuracy_joint"] = doc["selective_accuracy"]["joint"]
    return out


def _metrics_for(fs: FeatureSet, pred: Mapping[str, Any], n_bins: int) -> Dict[str, Any]:
    return M.classification_metrics(fs.y, pred["probs"], pred["hard_decisions"], pred["decisions"],
                                    fs.categories, n_bins=n_bins)


def evaluate_model(expert: PhysicsExpert, fs: FeatureSet, splits: Sequence[str] = EVAL_SPLITS,
                   operating_points: Sequence[str] = OPERATING_POINTS, slices: Sequence[str] = SLICES,
                   n_bins: int = 10, n_boot: int = 1000, seed: int = 0) -> Dict[str, Any]:
    """{split: {operating_point: {"overall": metrics (+ "ci95"), "slices": {slice: {value: compact}}}}}."""
    out: Dict[str, Any] = {}
    for split in splits:
        sub = fs.split(split)
        if len(sub) == 0:
            continue
        out[split] = {}
        for op in operating_points:
            pred = predict_featureset(expert, sub, op)
            overall = _metrics_for(sub, pred, n_bins)
            if n_boot:
                overall["ci95"] = M.bootstrap_classification_ci(
                    sub.y, pred["probs"], pred["hard_decisions"], pred["decisions"], sub.categories, sub.groups,
                    n_boot=n_boot, seed=seed, n_bins=n_bins)
            overall["n_groups"] = int(len(np.unique(sub.groups)))
            reasons: Dict[str, int] = {}
            for r in pred["abstain_reasons"]:
                for item in r:
                    reasons[item] = reasons.get(item, 0) + 1
            overall["abstain_reasons"] = reasons
            sl: Dict[str, Any] = {}
            for name in slices:
                vals = slice_values(sub, name)
                sl[name] = {}
                for v in sorted(set(vals)):
                    rows = vals == v
                    part = sub.subset(rows)
                    p = {k: np.asarray(pred[k])[rows] for k in ("probs", "hard_decisions", "decisions")}
                    sl[name][str(v)] = compact(_metrics_for(part, p, n_bins))
            out[split][op] = {"overall": overall, "slices": sl}
    return out


# =============================================================================
# Seeds
# =============================================================================


def _flatten(doc: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for k, v in doc.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            flat.update(_flatten(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None:
            flat[key] = float(v)
    return flat


def aggregate_over_seeds(docs: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{metric path: {"mean", "sd" (ddof 1; 0 for one seed), "n_seeds", "values"}} over per-seed dicts."""
    flats = [_flatten(d) for d in docs]
    keys = sorted({k for f in flats for k in f})
    out = {}
    for k in keys:
        vals = [f[k] for f in flats if k in f]
        out[k] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                  "n_seeds": len(vals), "values": vals}
    return out


def seed_artifact_dirs(artifact_dir: PathLike) -> Dict[int, Path]:
    root = Path(artifact_dir) / "seeds"
    out = {}
    for d in sorted(root.glob("seed_*")) if root.exists() else []:
        if (d / "physics_classifier.pt").exists():
            out[int(d.name.split("_", 1)[1])] = d
    return out


def evaluate_seeds(artifact_dir: PathLike, fs: FeatureSet, splits: Sequence[str] = EVAL_SPLITS,
                   operating_points: Sequence[str] = OPERATING_POINTS, n_bins: int = 10) -> Dict[str, Any]:
    """Point estimates of every trained seed and their mean +- sd (no bootstrap, no slices)."""
    dirs = seed_artifact_dirs(artifact_dir)
    per_seed: Dict[str, Any] = {}
    for seed, d in dirs.items():
        expert = PhysicsExpert.load(d)
        res = evaluate_model(expert, fs, splits, operating_points, slices=(), n_bins=n_bins, n_boot=0)
        per_seed[str(seed)] = {s: {op: compact(r["overall"]) for op, r in v.items()} for s, v in res.items()}
    return {"seeds": sorted(dirs), "per_seed": per_seed,
            "mean_sd": aggregate_over_seeds(list(per_seed.values())) if per_seed else {}}


# =============================================================================
# Shortcut audit and robustness
# =============================================================================


def _parameter_records(fs: FeatureSet) -> List[Dict[str, Any]]:
    """One flat record per row: scenario context, scalar render settings and the mask source."""
    recs = []
    for m, mask_source in zip(fs.meta, fs.mask_sources):
        rec: Dict[str, Any] = {}
        for key in ("context", "render"):
            part = m.get(key) if isinstance(m, dict) else None
            if isinstance(part, dict):
                rec.update({k: v for k, v in part.items() if isinstance(v, (bool, int, float, str)) or v is None})
        recs.append(rec)
        if rec:
            rec["mask_source"] = str(mask_source)
    return recs


def shortcut_audit(fs: FeatureSet, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Feature-level probes (baselines.py) and the parameter-level audit (audit.py), with one overall flag."""
    acfg = cfg.get("audit", {})
    flags = B.flags_only_probe(fs, threshold=float(acfg.get("flags_only_auc_max", 0.6)))
    uni = B.univariate_aucs(fs, "train")
    fit_splits = fs.split(["train", "val_select", "val_calib", "test_iid"])
    recs = _parameter_records(fit_splits)
    if any(recs):
        from src.training.audit import parameter_audit

        params = parameter_audit(recs, fit_splits.y, LABEL_NAMES, threshold=float(acfg.get("parameter_auc_max", 0.6)))
    else:
        params = {"status": "skipped", "reason": "no scenario context / render parameters in the FeatureSet meta"}
    reasons = []
    if flags.get("flag"):
        reasons.append("flags-only probe above threshold: " + ", ".join(
            f"{f['label']}@{f['split']}={f['auc']}" for f in flags["flagged"]))
    if params.get("flagged"):
        reasons.append("parameters predicting a label: " + ", ".join(
            f"{f['label']}:{f['parameter']}={f['folded_auc']}" for f in params["flagged"][:10]))
    return {"flags_only": flags, "univariate_feature_aucs": uni, "parameters": params,
            "flag": bool(reasons), "flag_reasons": reasons}


def robustness_report(expert: PhysicsExpert, robust_fs: FeatureSet, operating_point: str = "balanced",
                      n_bins: int = 10) -> Dict[str, Any]:
    """Compact metrics per robustness suite ("robust:<suite>" splits) and the change against "nominal"."""
    suites = sorted({s.split(":", 1)[1] for s in robust_fs.splits if s.startswith("robust:")})
    out: Dict[str, Any] = {"operating_point": operating_point, "suites": {}}
    for suite in suites:
        sub = robust_fs.split(f"robust:{suite}")
        out["suites"][suite] = compact(_metrics_for(sub, predict_featureset(expert, sub, operating_point), n_bins))
    nominal = out["suites"].get("nominal")
    if nominal:
        nom = _flatten(nominal)
        for suite, doc in out["suites"].items():
            if suite == "nominal":
                continue
            flat = _flatten(doc)
            doc["delta_vs_nominal"] = {k: flat[k] - nom[k] for k in flat if k in nom and not k.endswith(".n_pos")
                                       and k != "n"}
    return out


# =============================================================================
# Figures
# =============================================================================


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_confusion(conf_hard: Mapping[str, Any], conf_sel: Mapping[str, Any], path: PathLike, title: str) -> Path:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, conf, sub in ((axes[0], conf_hard, "thresholded"), (axes[1], conf_sel, "with abstention")):
        m = np.asarray(conf["matrix"], dtype=float)
        cols = list(conf["classes"])
        if sub == "with abstention":
            m = np.concatenate([m, np.asarray(conf["abstained"], dtype=float)[:, None]], axis=1)
            cols = cols + ["abstain"]
        rows = m.sum(axis=1, keepdims=True)
        frac = np.divide(m, rows, out=np.zeros_like(m), where=rows > 0)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{int(m[i, j])}\n{frac[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if frac[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(cols)), cols, rotation=30, fontsize=8)
        ax.set_yticks(range(len(conf["classes"])), conf["classes"], fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(sub, fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_reliability(probs: np.ndarray, y: np.ndarray, path: PathLike, title: str, n_bins: int = 10) -> Path:
    plt = _plt()
    fig, axes = plt.subplots(1, len(LABEL_NAMES), figsize=(4.2 * len(LABEL_NAMES), 4))
    for j, (ax, name) in enumerate(zip(np.atleast_1d(axes), LABEL_NAMES)):
        bins = M.reliability_bins(probs[:, j], y[:, j], n_bins)
        ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--")
        ax.plot([b["p_mean"] for b in bins], [b["y_mean"] for b in bins], "o-", color="C0")
        ece = M.expected_calibration_error(probs[:, j], y[:, j], n_bins)
        ax.set_title(f"{name}: ECE {ece:.3f} (equal-mass bins)" if ece is not None else name, fontsize=9)
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_roc_pr(probs: np.ndarray, y: np.ndarray, path: PathLike, title: str) -> Path:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for j, name in enumerate(LABEL_NAMES):
        roc = M.roc_curve(probs[:, j], y[:, j])
        pr = M.pr_curve(probs[:, j], y[:, j])
        auc, ap = M.auroc(probs[:, j], y[:, j]), M.auprc(probs[:, j], y[:, j])
        axes[0].plot(roc["fpr"], roc["tpr"], color=f"C{j}", label=f"{name} AUROC {auc:.3f}" if auc else name)
        axes[1].plot(np.r_[0, pr["recall"]], np.r_[1, pr["precision"]], color=f"C{j}",
                     label=f"{name} AP {ap:.3f}" if ap else name)
    axes[0].plot([0, 1], [0, 1], color="0.6", lw=1, ls="--")
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right" if ax is axes[0] else "lower left")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def curves(probs: np.ndarray, y: np.ndarray, max_points: int = 200) -> Dict[str, Any]:
    """ROC and PR curves per label (thresholds are calibrated probabilities), thinned to <= max_points."""
    out: Dict[str, Any] = {}
    for j, name in enumerate(LABEL_NAMES):
        roc, pr = M.roc_curve(probs[:, j], y[:, j]), M.pr_curve(probs[:, j], y[:, j])

        def thin(d: Mapping[str, np.ndarray]) -> Dict[str, List[Optional[float]]]:
            n = len(next(iter(d.values())))
            idx = np.unique(np.linspace(0, n - 1, min(n, max_points)).round().astype(int)) if n else []
            return {k: [None if not np.isfinite(v[i]) else round(float(v[i]), 5) for i in idx] for k, v in d.items()}

        out[name] = {"roc": thin(roc), "pr": thin(pr)}
    return out


def write_report_plots(expert: PhysicsExpert, fs: FeatureSet, reports_dir: PathLike,
                       splits: Sequence[str] = EVAL_SPLITS, operating_point: str = "balanced",
                       n_bins: int = 10, tag: str = "") -> List[Path]:
    """Confusion matrices, reliability diagrams and ROC / PR curves per split."""
    out = []
    d = Path(reports_dir)
    for split in splits:
        sub = fs.split(split)
        if len(sub) == 0:
            continue
        pred = predict_featureset(expert, sub, operating_point)
        doc = _metrics_for(sub, pred, n_bins)
        title = f"{split} ({sub.source}, {len(sub)} sequences, {operating_point}){tag}"
        out.append(plot_confusion(doc["confusion_4class"], doc["confusion_4class_selective"],
                                  d / f"confusion_{split}.png", title))
        out.append(plot_reliability(np.asarray(pred["probs"]), sub.y, d / f"reliability_{split}.png", title, n_bins))
        out.append(plot_roc_pr(np.asarray(pred["probs"]), sub.y, d / f"roc_pr_{split}.png", title))
    return out


# =============================================================================
# evaluation.json
# =============================================================================


def data_counts(fs: FeatureSet) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for split in sorted(set(fs.splits)):
        sub = fs.split(split)
        out[split] = {"sequences": len(sub), "simulations": int(len(np.unique(sub.sim_ids))),
                      "groups": int(len(np.unique(sub.groups))),
                      "per_category": {c: {"sequences": int(np.sum(sub.categories == c)),
                                           "simulations": int(len(np.unique(sub.sim_ids[sub.categories == c])))}
                                       for c in CATEGORIES},
                      "mask_sources": {s: int(np.sum(sub.mask_sources == s)) for s in sorted(set(sub.mask_sources))}}
    return out


def build_evaluation(synthetic: Mapping[str, Any], ablation_a: Optional[Mapping[str, Any]],
                     ablation_c: Optional[Mapping[str, Any]], shortcut: Optional[Mapping[str, Any]],
                     robustness: Optional[Mapping[str, Any]] = None, loro: Optional[Mapping[str, Any]] = None,
                     data: Optional[Mapping[str, Any]] = None, real: Optional[Mapping[str, Any]] = None,
                     synthetic_proxy: Optional[Mapping[str, Any]] = None,
                     extra_baselines: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The evaluation.json document (docs/design.md section 11)."""
    real_doc = dict(real) if real else {"status": "pending", "reason": REAL_PENDING_REASON}
    ablations: Dict[str, Any] = {"A": ablation_a, "B": None, "C": ablation_c, "D": None,
                                 "reasons": {"B": BD_REASON, "D": BD_REASON}}
    if real and real.get("status") == "complete":
        ablations["B"] = real.get("B")
        ablations["D"] = real.get("D")
        ablations["reasons"] = {k: v for k, v in ablations["reasons"].items() if ablations[k] is None}
    if extra_baselines:
        ablations["extra_baselines"] = dict(extra_baselines)
    proxy = None
    if synthetic_proxy is not None:
        proxy = {**dict(synthetic_proxy), "not_real_world_evidence": True}
    return {
        "disclaimer": DISCLAIMER,
        "synthetic": dict(synthetic),
        "real": real_doc,
        "ablations": ablations,
        "synthetic_proxy": proxy,
        "shortcut_audit": shortcut,
        "robustness": robustness if robustness is not None else {"status": "not_run"},
        "loro": loro if loro is not None else {"status": "not_run"},
        "data": dict(data or {}),
    }


def evaluate_real(fs_real: FeatureSet, cfg: Mapping[str, Any], artifacts: Mapping[str, PathLike],
                  split: str = "test_iid", n_boot: Optional[int] = None) -> Dict[str, Any]:
    """Real-footage results ("real" section) for the artifacts of experiments B, C (zero-shot) and D.

    `artifacts` maps experiment name -> artifact directory. Camera-disjoint
    splits are required (groups = camera ids).
    """
    from src.data.dataset import assert_no_group_leakage

    if fs_real.source != "real":
        raise ValueError("evaluate_real needs a FeatureSet with source == 'real'")
    assert_no_group_leakage(fs_real)
    n_bins = int(cfg.get("evaluation", {}).get("calibration_bins", 10))
    n_boot = int(cfg.get("evaluation", {}).get("bootstrap_resamples", 1000) if n_boot is None else n_boot)
    out: Dict[str, Any] = {"status": "complete", "split": split}
    counts = {}
    for s in sorted(set(fs_real.splits)):
        sub = fs_real.split(s)
        counts[s] = {"sequences": len(sub), "videos": int(len(np.unique(sub.sim_ids))),
                     "cameras": int(len(np.unique(sub.groups)))}
    out["counts"] = counts
    for name, d in artifacts.items():
        expert = PhysicsExpert.load(d)
        res = evaluate_model(expert, fs_real, (split,), OPERATING_POINTS, slices=("category", "mask_source"),
                             n_bins=n_bins, n_boot=n_boot)
        out[name] = {"artifacts": str(d), "calibration_source": expert.calibration_source,
                     "seeds": evaluate_seeds(d, fs_real, (split,), OPERATING_POINTS, n_bins)["mean_sd"],
                     **res}
    return out


def resplit_by_group(fs: FeatureSet, fractions: Sequence = (("train", 0.5), ("val_select", 0.15),
                                                             ("val_calib", 0.15), ("test_iid", 0.2)),
                     seed: int = 0) -> FeatureSet:
    """A copy of `fs` with new group-disjoint splits, stratified by category (for the synthetic proxy)."""
    out = fs.subset(np.arange(len(fs)))
    new = np.empty(len(fs), dtype=object)
    rng = np.random.default_rng(seed)
    first_cat = {}
    for g, c in zip(fs.groups, fs.categories):
        first_cat.setdefault(g, c)
    names = [n for n, _ in fractions]
    cum = np.cumsum([f for _, f in fractions]) / sum(f for _, f in fractions)
    for c in sorted(set(first_cat.values())):
        groups = sorted(g for g, gc in first_cat.items() if gc == c)
        order = rng.permutation(len(groups))
        for rank, k in enumerate(order):
            split = names[int(np.searchsorted(cum, (rank + 0.5) / len(groups)))]
            new[fs.groups == groups[k]] = split
    out.splits = new.astype(str)
    return out


def run_synthetic_proxy(fs_shifted: FeatureSet, cfg: Mapping[str, Any], init_artifacts: PathLike,
                        out_dir: PathLike, seed: int = 0, n_boot: Optional[int] = None,
                        description: str = "", log: Optional[Callable[[str], None]] = print) -> Dict[str, Any]:
    """Pretrained-vs-scratch on a small SHIFTED synthetic domain, with the Stage B procedure.

    `fs_shifted` (e.g. detector-mask re-renders, or the held-out recipes) is
    re-split by group; D-like = fine-tune the pretrained artifacts, B-like =
    same data from scratch, C = pretrained zero-shot; all evaluated on the
    proxy's test split. Stored under "synthetic_proxy" with
    not_real_world_evidence = true: it is not a substitute for real footage.
    Use data that no reported test number depends on (e.g. separately
    generated shifted simulations): the proxy TRAINS on part of `fs_shifted`.
    """
    from src.training.train import run_finetune

    n_bins = int(cfg.get("evaluation", {}).get("calibration_bins", 10))
    n_boot = int(cfg.get("evaluation", {}).get("bootstrap_resamples", 1000) if n_boot is None else n_boot)
    proxy = resplit_by_group(fs_shifted, seed=seed)
    out = Path(out_dir)
    res: Dict[str, Any] = {"not_real_world_evidence": True, "description": description,
                           "counts": data_counts(proxy)}
    runs = {"D_finetuned": run_finetune(proxy, cfg, init_artifacts, out / "D", from_scratch=False, log=log),
            "B_scratch": run_finetune(proxy, cfg, init_artifacts, out / "B", from_scratch=True, log=log)}
    dirs = {"C_zero_shot": Path(init_artifacts), "D_finetuned": out / "D", "B_scratch": out / "B"}
    for name, d in dirs.items():
        ev = evaluate_model(PhysicsExpert.load(d), proxy, ("test_iid",), OPERATING_POINTS, (), n_bins, n_boot, seed)
        res[name] = {"artifacts": str(d),
                     "test": {s: {op: {**compact(r["overall"]), "ci95": r["overall"].get("ci95")}
                                  for op, r in v.items()} for s, v in ev.items()}}
        if name in runs:
            seeds = evaluate_seeds(d, proxy, ("test_iid",), OPERATING_POINTS, n_bins)
            res[name]["mean_sd_over_seeds"] = seeds["mean_sd"]
            res[name]["lr"] = runs[name]["seeds"][str(runs[name]["exported_seed"])]["settings"]["lr"]
    return res


def run_evaluation(fs: FeatureSet, cfg: Mapping[str, Any], artifact_dir: PathLike, reports_dir: PathLike,
                   robustness_fs: Optional[FeatureSet] = None, loro: Optional[Mapping[str, Any]] = None,
                   data_summary: Optional[Mapping[str, Any]] = None, real: Optional[Mapping[str, Any]] = None,
                   synthetic_proxy: Optional[Mapping[str, Any]] = None, n_boot: Optional[int] = None,
                   seed: int = 0, out_path: Optional[PathLike] = None,
                   log: Optional[Callable[[str], None]] = print) -> Dict[str, Any]:
    """Evaluate the exported model, every seed, the baselines and the audits; write evaluation.json.

    `data_summary` is the dataset generator's summary.json (acceptance rates).
    Returns the document (also written to `out_path`, default
    <artifact_dir>/evaluation.json); figures go to `reports_dir`.
    """
    ecfg = cfg.get("evaluation", {})
    n_bins = int(ecfg.get("calibration_bins", 10))
    n_boot = int(ecfg.get("bootstrap_resamples", 1000) if n_boot is None else n_boot)
    expert = PhysicsExpert.load(artifact_dir)
    fit = expert.config.get("fit", {}) or {}

    def say(msg):
        if log:
            log(msg)

    say("evaluating the exported model on the test splits")
    test = evaluate_model(expert, fs, EVAL_SPLITS, OPERATING_POINTS, SLICES, n_bins, n_boot, seed)
    test_curves = {}
    for split in EVAL_SPLITS:
        sub = fs.split(split)
        if len(sub):
            test_curves[split] = curves(np.asarray(predict_featureset(expert, sub, "balanced")["probs"]), sub.y)
    say("validation splits (reference only: used for selection / calibration)")
    val = evaluate_model(expert, fs, VALIDATION_SPLITS, OPERATING_POINTS, (), n_bins, 0, seed)
    say("per-seed evaluation")
    seeds = evaluate_seeds(artifact_dir, fs, EVAL_SPLITS, OPERATING_POINTS, n_bins)
    synthetic = {
        "model": "C: CFD-pretrained GRU (synthetic data only)",
        "exported_seed": fit.get("exported_seed"),
        "exported_seed_selection": fit.get("selection"),
        "calibration_source": expert.calibration_source,
        "temperatures": expert.config.get("temperatures"),
        "thresholds": expert.config.get("thresholds"),
        "abstain_margins": (expert.config.get("abstain") or {}).get("margins"),
        "calibration": {k: v for k, v in (expert.config.get("calibration") or {}).items()
                        if k in ("fit_source", "n_sequences", "n_groups", "insufficient_calibration_data",
                                 "insufficient_categories", "temperature_fallback", "temperature_clipped",
                                 "prevalence")},
        "test": test,
        "curves": test_curves,
        "validation_reference": {s: {op: compact(r["overall"]) for op, r in v.items()} for s, v in val.items()},
        "seeds": seeds,
        "bootstrap": {"resamples": n_boot, "groups": "sim_id", "interval": 0.95},
    }
    say("baselines")
    base = B.run_baselines(fs, cfg, EVAL_SPLITS, n_boot=n_boot, seed=seed)
    ablation_c = {
        "description": synthetic["model"],
        "exported_seed": fit.get("exported_seed"),
        "test": {s: {op: compact(r["overall"]) for op, r in v.items()} for s, v in test.items()},
        "mean_sd_over_seeds": seeds["mean_sd"],
    }
    ablation_a = dict(base["A_heuristic"])
    ablation_a["test"] = {s: {op: compact(r) for op, r in v.items()} for s, v in ablation_a.pop("results").items()}
    extra = dict(base["pooled_logistic"])
    extra["test"] = {s: {op: compact(r) for op, r in v.items()} for s, v in extra.pop("results").items()}
    say("shortcut audit")
    audit = shortcut_audit(fs, cfg)
    rob = None
    if robustness_fs is not None and len(robustness_fs):
        say("robustness suites")
        rob = robustness_report(expert, robustness_fs, "balanced", n_bins)
    data = {"counts": data_counts(fs), "source": fs.source,
            "acceptance": (data_summary or {}).get("acceptance"),
            "generation": {k: v for k, v in (data_summary or {}).items() if k != "acceptance"} or None}
    doc = build_evaluation(synthetic, ablation_a, ablation_c, audit, rob, loro, data, real, synthetic_proxy,
                           extra_baselines={"pooled_logistic": extra})
    say("figures")
    figs = write_report_plots(expert, fs, reports_dir, EVAL_SPLITS, "balanced", n_bins)
    doc["figures"] = [str(p) for p in figs]
    out_path = Path(out_path) if out_path is not None else Path(artifact_dir) / "evaluation.json"
    write_json(out_path, doc)
    say(f"wrote {out_path}")
    return doc
