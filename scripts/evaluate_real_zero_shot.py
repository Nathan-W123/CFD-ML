"""Zero-shot check of the exported model on labelled real footage (no fine-tuning).

    python scripts/extract_features.py --real-manifest data/real/<name>/real.csv --artifacts artifacts \\
        --out data/real/<name>/features.npz
    python scripts/evaluate_real_zero_shot.py --features data/real/<name>/features.npz --artifacts artifacts \\
        --name <name> --description "..." --out reports/real_zero_shot_<name>.json

Applies the deployed decision rules (PhysicsExpert.predict with the extractor's
timing record, both operating points) to every extracted window and reports,
per window, the inputs that were measurable, the probabilities, the decisions
and why the model abstained; and, per operating point, what it decided on
smoke-labelled and smoke-free windows (and fog-labelled / fog-free ones).

This is NOT Stage B: nothing is trained or recalibrated on the real data, the
thresholds are still the synthetic val_calib ones, and evaluation.json's
"real" status is left alone. With a handful of videos the numbers are a
sanity check of the pipeline and of the model's behaviour on real video, not
an accuracy estimate; the report says so and lists the windows.
"""

import _bootstrap  # noqa: F401

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from src.constants import LABEL_NAMES
from src.data.dataset import FeatureSet
from src.inference.predict import PhysicsExpert
from src.training import metrics as M
from src.utils.io import write_json

OPS = ("balanced", "low_false_alarm")
DECISION_TEXT = {1: "yes", 0: "no", -1: "abstain"}


def summarize(fs: FeatureSet, out: dict) -> dict:
    """Per label: what the model decided on positive and negative windows (decided = not abstained)."""
    dec = np.asarray(out["decisions"])
    res = {"abstained_any": int((dec == -1).any(axis=1).sum()),
           "abstain_reasons": dict(Counter(r for rs in out["abstain_reasons"] for r in rs))}
    for j, name in enumerate(LABEL_NAMES):
        pos, d = fs.y[:, j] > 0.5, dec[:, j]
        res[name] = {
            "positive_windows": int(pos.sum()), "negative_windows": int((~pos).sum()),
            "positives_decided": int((pos & (d != -1)).sum()), "detected": int((pos & (d == 1)).sum()),
            "missed": int((pos & (d == 0)).sum()),
            "negatives_decided": int((~pos & (d != -1)).sum()), "false_alarms": int((~pos & (d == 1)).sum()),
            "correct_rejections": int((~pos & (d == 0)).sum()),
        }
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", required=True, help="real FeatureSet from extract_features.py --real-manifest")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--name", required=True)
    ap.add_argument("--description", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    fs = FeatureSet.load(args.features)
    if fs.source != "real":
        raise SystemExit(f"{args.features} is not a real-footage FeatureSet (source = {fs.source!r})")
    expert = PhysicsExpert.load(args.artifacts)
    metas = [m.get("feature_meta") or {} for m in fs.meta]
    preds = {op: expert.predict(fs.X, operating_point=op, meta=metas) for op in OPS}
    probs = np.asarray(preds["balanced"]["probs"])

    windows = []
    for i in range(len(fs)):
        m = fs.meta[i]
        windows.append({
            "window": str(fs.seq_ids[i]), "video": str(fs.sim_ids[i]), "camera": str(fs.groups[i]),
            "labels": {n: int(fs.y[i, j] > 0.5) for j, n in enumerate(LABEL_NAMES)},
            "mask_source": str(fs.mask_sources[i]),
            "valid_mask_frames": int(fs.X[i, :, 12].sum()), "valid_flow_frames": int(fs.X[i, :, 13].sum()),
            "stabilization": (m.get("stabilization") or {}).get("reason") or
                             ("applied" if (m.get("stabilization") or {}).get("applied") else None),
            "probs": {n: float(probs[i, j]) for j, n in enumerate(LABEL_NAMES)},
            "decisions": {op: {n: int(preds[op]["decisions"][i][j]) for j, n in enumerate(LABEL_NAMES)}
                          for op in OPS},
            "abstain_reasons": {op: list(preds[op]["abstain_reasons"][i]) for op in OPS},
        })
    ranking = {}
    for j, name in enumerate(LABEL_NAMES):
        pos = fs.y[:, j] > 0.5
        ranking[name] = {"auroc_windows": M.auroc(probs[:, j], pos) if 0 < pos.sum() < len(pos) else None,
                         "n_pos": int(pos.sum()), "n_neg": int((~pos).sum()),
                         "cameras_pos": int(len(set(fs.groups[pos]))), "cameras_neg": int(len(set(fs.groups[~pos])))}
    doc = {
        "status": "zero_shot_sanity_check",
        "not_a_real_world_accuracy_estimate": True,
        "stage_b": "not run (no training or recalibration on real data); evaluation.json real status unchanged",
        "name": args.name, "description": args.description,
        "artifacts": str(args.artifacts), "calibration_source": expert.calibration_source,
        "counts": {"windows": len(fs), "videos": int(len(set(fs.sim_ids))), "cameras": int(len(set(fs.groups))),
                   "mask_sources": dict(Counter(map(str, fs.mask_sources)))},
        "ranking_balanced_probs": ranking,
        "operating_points": {op: summarize(fs, preds[op]) for op in OPS},
        "windows": windows,
    }
    out = Path(args.out)
    write_json(out, doc)

    def f2(v):
        return "n/a" if v is None else f"{v:.2f}"

    L = [f"# Zero-shot check on real footage: {args.name}", "",
         "> Sanity check, not an accuracy estimate: the exported synthetic model and its synthetic thresholds, "
         "no fine-tuning or recalibration on real data (Stage B not run).", ""]
    if args.description:
        L += [args.description, ""]
    c = doc["counts"]
    L += [f"{c['windows']} windows from {c['videos']} videos ({c['cameras']} cameras); masks: "
          + ", ".join(f"{k} {v}" for k, v in c["mask_sources"].items()) + ".", ""]
    for op in OPS:
        s = doc["operating_points"][op]
        L += [f"**{op}**: abstained (either label) on {s['abstained_any']} / {c['windows']} windows "
              f"({', '.join(f'{k} {v}' for k, v in s['abstain_reasons'].items()) or 'none'})."]
        for name in LABEL_NAMES:
            r = s[name]
            L.append(f"* {name}: positives {r['positive_windows']} (decided {r['positives_decided']}: "
                     f"{r['detected']} detected, {r['missed']} missed); negatives {r['negative_windows']} "
                     f"(decided {r['negatives_decided']}: {r['false_alarms']} false alarms, "
                     f"{r['correct_rejections']} correct)")
        L.append("")
    L += ["Window-level AUROC of the balanced-point probabilities (tiny n, windows of one video are correlated): "
          + "; ".join(f"{n} {f2(r['auroc_windows'])} ({r['n_pos']} pos / {r['n_neg']} neg windows, "
                      f"{r['cameras_pos']} / {r['cameras_neg']} cameras)" for n, r in ranking.items()), "",
          "| window | labels smoke/fog | valid mask / flow frames | stabilisation | P(smoke) | P(fog) | "
          "balanced smoke/fog | low-false-alarm smoke/fog | abstain reasons (balanced) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for w in windows:
        d = {op: "/".join(DECISION_TEXT[w["decisions"][op][n]] for n in LABEL_NAMES) for op in OPS}
        L.append(f"| {w['window']} | {w['labels']['smoke']}/{w['labels']['fog']} | {w['valid_mask_frames']} / "
                 f"{w['valid_flow_frames']} | {w['stabilization']} | {w['probs']['smoke']:.2f} | "
                 f"{w['probs']['fog']:.2f} | {d['balanced']} | {d['low_false_alarm']} | "
                 f"{', '.join(w['abstain_reasons']['balanced']) or '-'} |")
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
