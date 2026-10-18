"""Data-quality report for a generated dataset (run before training).

    python scripts/qa_dataset.py --data data/synthetic_debug --out reports/qa_debug.json

Reports, per category: gate validity rates and per-feature medians over open
gates; per label: the univariate AUROC of every feature (time-averaged over
open gates) and of every context / render parameter (folded, 0.5 = no
information), flagging anything above the audit threshold; and a physical
consistency check of the optical flow: on a sample of sequences, the mean
optical flow inside the candidate region against the mean CFD velocity of the
same region (projected to the camera view), as a correlation.
"""

import _bootstrap  # noqa: F401

import argparse
import sys
from pathlib import Path

import numpy as np

from src.constants import CATEGORIES, LABEL_NAMES
from src.data.dataset import FeatureSet
from src.data.feature_spec import FEATURE_NAMES, FLOW_VALID_INDEX, MASK_VALID_INDEX, gate_masks
from src.training.audit import auroc, folded, parameter_audit
from src.utils.io import write_json


def time_means(X):
    g = gate_masks(X)
    num = (X * g).sum(axis=1)
    den = g.sum(axis=1)
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def flow_consistency(base, fs, sel, tm, n_max=200):
    """Measured optical flow vs the aerosol-weighted CFD velocity of the same simulation.

    Units differ (the features are in region sizes per second), so this
    reports the sign agreement and the correlation across sequences.
    """
    from src.cfd.results import SimulationResult
    from src.data.feature_spec import FEATURE_INDEX

    idx = np.flatnonzero(sel)[:n_max]
    u_cfd, w_cfd, u_flow, w_flow = [], [], [], []
    cache = {}
    for i in idx:
        meta = fs.meta[i]
        path = base / meta["sim_path"]
        if path not in cache:
            sim = SimulationResult.load(path)
            weight = sim.smoke * 0.0076 + sim.lwc * 0.14 + sim.tracer * 0.004
            weight = weight / max(float(weight.max()), 1e-12)
            weight = np.where(weight > 0.05, weight, 0.0)
            if sim.ux is None or weight.sum() == 0:
                cache[path] = None
            else:
                cache[path] = (float((sim.ux * weight).sum() / weight.sum()),
                               float((sim.uz * weight).sum() / weight.sum()))
        if cache[path] is None:
            continue
        fu, fw = tm[i, FEATURE_INDEX["flow_u_mean"]], tm[i, FEATURE_INDEX["flow_v_mean"]]
        if not (np.isfinite(fu) and np.isfinite(fw)):
            continue
        u_cfd.append(cache[path][0]); w_cfd.append(cache[path][1]); u_flow.append(fu); w_flow.append(fw)
    out = {"n": len(u_cfd)}
    for name, a, b in (("horizontal", u_cfd, u_flow), ("vertical", w_cfd, w_flow)):
        a, b = np.asarray(a), np.asarray(b)
        strong = np.abs(a) > (0.5 if name == "horizontal" else 0.3)
        out[name] = {
            "correlation": float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else None,
            "sign_agreement_where_cfd_moves": float(np.mean(np.sign(a[strong]) == np.sign(b[strong]))) if strong.any() else None,
            "n_moving": int(strong.sum()),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args(argv)
    fs = FeatureSet.load(Path(args.data) / "features.npz")
    sel = fs.splits == args.split
    X, y = fs.X[sel], fs.y[sel]
    cats = fs.categories[sel]
    meta = [m for m, s in zip(fs.meta, sel) if s]
    report = {"split": args.split, "n": int(sel.sum()), "per_category": {}, "feature_auc": {}, "flags": []}
    tm = time_means(X)
    for c in CATEGORIES:
        m = cats == c
        if not m.any():
            continue
        report["per_category"][c] = {
            "n": int(m.sum()),
            "mask_valid_rate": float(X[m][..., MASK_VALID_INDEX].mean()),
            "flow_valid_rate": float(X[m][..., FLOW_VALID_INDEX].mean()),
            "feature_median": {FEATURE_NAMES[i]: float(np.nanmedian(tm[m, i])) for i in range(12)},
        }
    for j, label in enumerate(LABEL_NAMES):
        res = {}
        for i, name in enumerate(FEATURE_NAMES):
            v = tm[:, i] if i < 12 else X[..., i].mean(axis=1)
            ok = np.isfinite(v)
            a = auroc(v[ok], y[ok, j] > 0.5)
            res[name] = None if a is None else round(a, 4)
            if i >= 12 and a is not None and folded(a) > args.threshold:
                report["flags"].append(f"validity flag {name} predicts {label} (AUC {a:.3f})")
        report["feature_auc"][label] = res
    records = [{**m.get("context", {}), **{k: v for k, v in m.get("render", {}).items()
                                            if not k.startswith("render_view")},
                "mask_source": ms} for m, ms in zip(meta, fs.mask_sources[sel])]
    report["parameter_audit"] = parameter_audit(records, y, LABEL_NAMES, args.threshold)
    for f in report["parameter_audit"]["flagged"]:
        report["flags"].append(f"parameter {f['parameter']} predicts {f['label']} (folded AUC {f['folded_auc']})")
    report["flow_consistency"] = flow_consistency(Path(args.data), fs, sel, tm)
    write_json(args.out, report)
    print(f"n={report['n']}")
    print("  flow vs CFD velocity:", report["flow_consistency"])
    for c, r in report["per_category"].items():
        print(f"  {c:10s} n={r['n']:4d} mask_valid={r['mask_valid_rate']:.2f} flow_valid={r['flow_valid_rate']:.2f}")
    for label in LABEL_NAMES:
        top = sorted(((abs(v - 0.5), k, v) for k, v in report["feature_auc"][label].items() if v is not None), reverse=True)[:6]
        print(f"  {label}: " + ", ".join(f"{k}={v:.2f}" for _, k, v in top))
    print("flags:", report["flags"] or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
