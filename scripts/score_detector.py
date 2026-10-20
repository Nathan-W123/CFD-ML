"""Score saved detectors side by side on labelled windows, with paired differences (nothing is fitted).

    D=data/real
    python scripts/score_detector.py --detector v1=artifacts_detector --detector v3=artifacts_detector_v3 \\
        --set figlib_test2:$D/figlib_test2/real.csv:$D/figlib_test2/clip.npz::$D/figlib_test2/clip_fine.npz:$D/figlib_test2/clip_zoom.npz:$D/figlib_test2/clip_ref.npz \\
        --set hpwren_bg_test:... --report reports/far/detector_v3_fresh.json

Sets as in scripts/train_detector.py. By default only the held-out test_iid
windows are scored (--all-splits: every window). Each detector is applied
with its own stored thresholds (deploy, strict and, if it has one, the fog
veto) and its own stack fallbacks, to the same windows, and reported with
train_detector.metrics_block (camera-bootstrap intervals). For every
detector after the first, the differences from the first (within-set smoke
AUROC, smoke AUROC per set, and at the deploy and strict points the false
alarms on smoke-free windows, overall and per set, and the smoke recall) get
a paired camera bootstrap: the same resampled cameras for both detectors.
--matched-recall r adds the false-alarm rates of each detector at the
threshold that catches r of these windows' smoke (descriptive: that
threshold is read off the scored windows themselves).
"""

import _bootstrap  # noqa: F401

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_detector as TD  # noqa: E402

from src.detector.model import Detector  # noqa: E402
from src.utils.io import write_json  # noqa: E402


def score(det, rows):
    """P(smoke), P(fog) and the stack used per window (windows grouped by which inputs they have)."""
    p_s, p_f = np.zeros(len(rows)), np.zeros(len(rows))
    used = {}
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["fine"] is not None, r["zoom"] is not None, r["ref"] is not None), []).append(i)
    for (has_fine, has_zoom, has_ref), idx in groups.items():
        rs = [rows[i] for i in idx]
        E = np.stack([r["emb"] for r in rs])
        fine = np.stack([r["fine"] for r in rs]) if has_fine else None
        zoom = np.stack([r["zoom"] for r in rs]) if has_zoom else None
        ref = np.stack([r["ref"] for r in rs]) if has_ref else None
        p = det.proba(E, None, fine, zoom, ref)
        p_s[idx], p_f[idx] = p["smoke"], p["fog"]
        avail = det.stack_inputs("smoke", E[:1], None, None if fine is None else fine[:1],
                                 None if zoom is None else zoom[:1], None if ref is None else ref[:1])
        key = "+".join(det.stack_used("smoke", avail))
        used[key] = used.get(key, 0) + len(idx)
    return p_s, p_f, used


def _alarm(p_s, p_f, thr):
    veto = thr.get("fog_veto")
    return np.where(p_f >= veto, -1.0, p_s) if veto is not None else p_s


def paired_stats(a, b, rows, thr_a, thr_b):
    """fn(idx) -> differences b - a of the compared statistics on rows idx."""
    y = np.array([r["smoke"] for r in rows])
    sets = np.array([r["set"] for r in rows])
    qa, qb = _alarm(a[0], a[1], thr_a), _alarm(b[0], b[1], thr_b)

    def fn(idx):
        idx = np.asarray(idx)
        k = idx[~np.isnan(y[idx])]
        out = {}
        wa, wb = TD.within_sets_auroc(a[0][k], y[k], sets[k]), TD.within_sets_auroc(b[0][k], y[k], sets[k])
        out["smoke_auroc_within_sets"] = None if wa is None or wb is None else wb - wa
        for s in sorted(set(sets.tolist())):
            m = k[sets[k] == s]
            if len(set(y[m].tolist())) == 2:
                out[f"smoke_auroc/{s}"] = TD.M.auroc(b[0][m], y[m]) - TD.M.auroc(a[0][m], y[m])
        for point in ("deploy", "strict"):
            ta, tb = thr_a.get(point), thr_b.get(point)
            if ta is None or tb is None:
                continue
            neg, pos = k[y[k] == 0], k[y[k] == 1]
            if len(neg):
                out[f"false_alarm_no_smoke/{point}"] = float(np.mean(qb[neg] >= tb) - np.mean(qa[neg] >= ta))
                for s in sorted(set(sets[neg].tolist())):
                    m = neg[sets[neg] == s]
                    out[f"false_alarm_no_smoke/{point}/{s}"] = float(np.mean(qb[m] >= tb) - np.mean(qa[m] >= ta))
            if len(pos):
                out[f"smoke_recall/{point}"] = float(np.mean(qb[pos] >= tb) - np.mean(qa[pos] >= ta))
        return out

    return fn


def matched_recall(p_s, p_f, thr, rows, r):
    """False alarms on smoke-free windows (overall, per set) at the threshold catching r of the smoke windows."""
    y = np.array([x["smoke"] for x in rows])
    sets = np.array([x["set"] for x in rows])
    q = _alarm(p_s, p_f, thr)
    pos = np.sort(q[y == 1])[::-1]
    if not len(pos):
        return None
    t = pos[max(int(np.ceil(r * len(pos))) - 1, 0)]
    neg = y == 0
    return {"threshold": float(t), "false_alarm_no_smoke": float(np.mean(q[neg] >= t)),
            "per_set": {s: float(np.mean(q[neg & (sets == s)] >= t)) for s in sorted(set(sets[neg].tolist()))}}


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--detector", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--set", action="append", required=True, dest="sets")
    ap.add_argument("--report", required=True)
    ap.add_argument("--all-splits", action="store_true", help="score every window, not only test_iid")
    ap.add_argument("--matched-recall", type=float, action="append", default=[])
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    rows = TD.load_sets(args.sets)
    if not args.all_splits:
        rows = [r for r in rows if r["split"] == "test_iid"]
    dets = [(spec.split("=", 1)[0], Detector.load(spec.split("=", 1)[1])) for spec in args.detector]
    cams = np.array([r["mount"] for r in rows])
    rep = {"status": "detector_scores", "sets": args.sets, "detectors": dict(s.split("=", 1) for s in args.detector),
           "results": {}, "paired": {}, "matched_recall": {}}
    probs = {}
    for name, det in dets:
        p_s, p_f, used = score(det, rows)
        thr = det.thresholds["smoke"]
        probs[name] = (p_s, p_f, thr)
        veto = thr.get("fog_veto")
        res = TD.metrics_block(p_s, p_f, rows, thr["deploy"], args.n_boot, thr.get("strict"),
                               p_f if veto is not None else None, veto)
        res["stacks_used"] = used
        res["thresholds"] = thr
        rep["results"][name] = res
        rep["matched_recall"][name] = {f"{r:.2f}": matched_recall(p_s, p_f, thr, rows, r) for r in args.matched_recall}
        print(f"{name}: within-set smoke AUROC {res['smoke_auroc_within_sets']}, false alarms "
              f"{res['false_alarm_no_smoke']}, recall {res['smoke_recall']}", file=sys.stderr)
    base = dets[0][0]
    for name, _ in dets[1:]:
        a, b = probs[base], probs[name]
        fn = paired_stats(a[:2], b[:2], rows, a[2], b[2])
        rep["paired"][f"{name}-{base}"] = {"value": fn(np.arange(len(rows))),
                                           "ci95": TD.M.grouped_bootstrap(fn, cams, n_boot=args.n_boot)}
    write_json(Path(args.report), rep)
    np.savez_compressed(Path(args.report).with_suffix(".probs.npz"), window_id=np.array([r["window_id"] for r in rows]),
                        set=np.array([r["set"] for r in rows]), smoke=np.array([r["smoke"] for r in rows]),
                        fog=np.array([r["fog"] for r in rows]),
                        **{f"p_smoke_{n}": probs[n][0] for n in probs}, **{f"p_fog_{n}": probs[n][1] for n in probs})
    print(json.dumps({k: v["value"] for k, v in rep["paired"].items()}, default=float, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
