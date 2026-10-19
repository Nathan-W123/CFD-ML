"""Train and evaluate the real-footage smoke / fog detector (far field), camera-disjoint.

    python scripts/train_detector.py \\
        --set figlib:data/real/figlib/real.csv:data/real/figlib/clip.npz:data/real/figlib/features.npz \\
        --set hpwren_fog:data/real/hpwren_fog/real.csv:data/real/hpwren_fog/clip.npz:data/real/hpwren_fog/features.npz \\
        --set hpwren_smoke:data/real/hpwren_smoke/real.csv:data/real/hpwren_smoke/clip.npz:data/real/hpwren_smoke/features.npz \\
        --out artifacts_detector --report reports/far/detector.json [--final]

Data: every --set is name:manifest:clip_npz[:features_npz]. A row's labels
come from its manifest (smoke / fog: 0, 1 or empty = unknown), its appearance
from the CLIP embeddings (scripts/embed_windows.py), its motion (optional
feature group "motion") from the extractor's FeatureSet (seq_id
<window_id>-w0). Splits come from the manifests (camera-mount based in
prepare_figlib.py and prepare_hpwren_fog.py, so one camera is in one split
across sets); "dev" is every split but test_iid.

Per head (smoke, fog), on the dev rows with a known label:

1. model selection: every candidate feature-group set x C (inverse L2
   strength) is scored by 5-fold cross-validation grouped by camera mount
   (out-of-fold AUROC); the best is refitted on all dev rows;
2. thresholds from the out-of-fold predictions of the chosen configuration
   (never from the refitted model's own training predictions): "balanced"
   (maximum Youden J) for both heads; for smoke also "deploy", the
   (1 - --max-false-smoke) quantile of P(smoke) over the dev fog windows
   (fog 1, smoke 0), i.e. at most 5 % of them raise a smoke alarm.

Reported: the cross-validated dev results, and with --final the held-out
test cameras (evaluated once): fog AUROC, smoke AUROC (all smoke-labelled
windows, and per set), and at the deploy point the false-smoke rate on the
test fog windows and the smoke recall on the test smoke windows, each with a
95 % bootstrap interval resampling cameras. The goal checks (fog AUROC >=
0.90, smoke AUROC >= 0.85, false smoke on fog <= 5 % with smoke recall >=
50 %) are written as booleans.
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_figlib import camera_mount  # noqa: E402

from src.data.dataset import FeatureSet  # noqa: E402
from src.detector.features import MODEL_ID, PROMPTS, text_embeddings, window_features  # noqa: E402
from src.detector.model import Detector, fit_head  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.training.baselines import pooled_features  # noqa: E402
from src.utils.io import write_json  # noqa: E402

GROUP_SETS = {"fog": (("full",), ("sims",), ("full", "sims"), ("full", "tiles", "sims"), ("full", "sims", "motion")),
              "smoke": (("tiles",), ("sims",), ("tiles", "sims"), ("full", "tiles", "sims"),
                        ("tiles", "sims", "motion"), ("full", "tiles", "sims", "motion"))}
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
GOALS = {"fog_auroc": 0.90, "smoke_auroc": 0.85, "max_false_smoke_on_fog": 0.05, "min_smoke_recall": 0.50}


def _label(v):
    v = (v or "").strip()
    return float(v) if v in ("0", "1", "0.0", "1.0") else np.nan


def load_sets(specs):
    """Rows of every set: dicts with window_id, set, camera, mount, split, smoke, fog, emb, motion."""
    rows = []
    for spec in specs:
        parts = spec.split(":")
        name, manifest, clip = parts[0], parts[1], parts[2]
        feats = parts[3] if len(parts) > 3 and parts[3] else None
        z = np.load(clip)
        emb = {str(w): e for w, e in zip(z["window_id"], z["emb"])}
        motion = {}
        if feats:
            fs = FeatureSet.load(feats)
            pf = pooled_features(fs.X)
            motion = {str(s): pf[i] for i, s in enumerate(fs.seq_ids)}
        with open(manifest, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                wid = r.get("window_id") or r["video_id"]
                if wid not in emb:
                    continue
                rows.append({"window_id": wid, "set": name, "camera": r["camera_id"],
                             "mount": camera_mount(r["camera_id"]), "split": r["split"],
                             "smoke": _label(r.get("smoke")), "fog": _label(r.get("fog")),
                             "emb": emb[wid].astype(np.float32), "motion": motion.get(f"{wid}-w0")})
    return rows


def _features(rows, T, groups):
    E = np.stack([r["emb"] for r in rows])
    mot = None
    if "motion" in groups:
        if any(r["motion"] is None for r in rows):
            return None
        mot = np.stack([r["motion"] for r in rows])
    return window_features(E, T, groups, mot)


def grouped_oof(X, y, groups, C, n_splits=5):
    from sklearn.model_selection import GroupKFold

    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        oof[te] = fit_head(X[tr], y[tr], (), C).proba(X[te])
    return oof


def select(rows, head, T):
    """(best config, all scores, out-of-fold predictions of the best) for one head on dev rows."""
    y = np.array([r[head] for r in rows])
    g = np.array([r["mount"] for r in rows])
    scores, best = [], None
    for groups, C in itertools.product(GROUP_SETS[head], C_GRID):
        X = _features(rows, T, groups)
        if X is None:
            continue
        oof = grouped_oof(X, y, g, C)
        auc = M.auroc(oof, y)
        scores.append({"groups": list(groups), "C": C, "cv_auroc": auc})
        if best is None or auc > best[0]:
            best = (auc, groups, C, oof)
    return {"groups": list(best[1]), "C": best[2], "cv_auroc": best[0]}, scores, best[3]


def youden(p, y):
    t, rec = M.threshold_max_youden(p, y)
    return float(t)


def metrics_block(p_smoke, p_fog, rows, thr, n_boot=1000):
    """AUROCs, deploy-point false smoke on fog and smoke recall, with camera-bootstrap intervals."""
    ys = np.array([r["smoke"] for r in rows])
    yf = np.array([r["fog"] for r in rows])
    sets = np.array([r["set"] for r in rows])
    cams = np.array([r["mount"] for r in rows])
    fog_win = (yf == 1) & (ys == 0)
    smk = ~np.isnan(ys)
    fgk = ~np.isnan(yf)

    def stats(idx):
        idx = np.asarray(idx)
        s, f, fw, ys_, yf_ = smk[idx], fgk[idx], fog_win[idx], ys[idx], yf[idx]
        ps, pf = p_smoke[idx], p_fog[idx]
        pos = s & (ys_ == 1)
        return {"smoke_auroc": M.auroc(ps[s], ys_[s]) if s.any() else None,
                "fog_auroc": M.auroc(pf[f], yf_[f]) if f.any() else None,
                "false_smoke_on_fog": float(np.mean(ps[fw] >= thr)) if fw.any() else None,
                "smoke_recall": float(np.mean(ps[pos] >= thr)) if pos.any() else None}

    out = stats(np.arange(len(rows)))
    out["ci95"] = M.grouped_bootstrap(stats, cams, n_boot=n_boot)
    out["n"] = {"windows": len(rows), "cameras": int(len(set(cams.tolist()))), "smoke_known": int(smk.sum()),
                "smoke_pos": int(np.sum(ys == 1)), "fog_known": int(fgk.sum()), "fog_pos": int(np.sum(yf == 1)),
                "fog_windows_smoke0": int(fog_win.sum())}
    out["per_set"] = {}
    for sname in sorted(set(sets.tolist())):
        m = sets == sname
        out["per_set"][sname] = {k: v for k, v in stats(np.flatnonzero(m)).items()}
    return out


def goal_checks(res):
    return {"fog_auroc": res["fog_auroc"] is not None and res["fog_auroc"] >= GOALS["fog_auroc"],
            "smoke_auroc": res["smoke_auroc"] is not None and res["smoke_auroc"] >= GOALS["smoke_auroc"],
            "deploy_point": (res["false_smoke_on_fog"] is not None and res["smoke_recall"] is not None
                             and res["false_smoke_on_fog"] <= GOALS["max_false_smoke_on_fog"]
                             and res["smoke_recall"] >= GOALS["min_smoke_recall"])}


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", action="append", required=True, dest="sets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--max-false-smoke", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--final", action="store_true", help="evaluate on the held-out test_iid cameras (once)")
    args = ap.parse_args(argv)

    rows = load_sets(args.sets)
    dev = [r for r in rows if r["split"] != "test_iid"]
    test = [r for r in rows if r["split"] == "test_iid"]
    T = {h: text_embeddings(PROMPTS[h]) for h in ("smoke", "fog")}
    heads, thresholds, selection, oof = {}, {}, {}, {}
    for head in ("smoke", "fog"):
        hr = [r for r in dev if not np.isnan(r[head])]
        best, scores, oof_p = select(hr, head, T[head])
        selection[head] = {"best": best, "scores": scores, "n_dev": len(hr)}
        X = _features(hr, T[head], best["groups"])
        heads[head] = fit_head(X, np.array([r[head] for r in hr]), best["groups"], best["C"])
        y = np.array([r[head] for r in hr])
        thresholds[head] = {"balanced": youden(oof_p, y)}
        if head == "smoke":
            fog_win = np.array([(r["fog"] == 1) and (r["smoke"] == 0) for r in hr])
            thresholds[head]["deploy"] = float(np.quantile(oof_p[fog_win], 1.0 - args.max_false_smoke))
        oof[head] = {r["window_id"]: p for r, p in zip(hr, oof_p)}
        print(f"{head}: best {best} on {len(hr)} dev windows", file=sys.stderr)
    det = Detector(heads=heads, text=T, thresholds=thresholds,
                   info={"model_id": MODEL_ID, "sets": args.sets, "max_false_smoke": args.max_false_smoke,
                         "selection": {h: selection[h]["best"] for h in selection}})
    det.save(args.out)

    # Cross-validated dev results: each window's out-of-fold probability for each head it has a label for
    dev_s = np.array([oof["smoke"].get(r["window_id"], np.nan) for r in dev])
    dev_f = np.array([oof["fog"].get(r["window_id"], np.nan) for r in dev])
    report = {"status": "detector", "goals": GOALS, "selection": selection, "thresholds": thresholds,
              "dev_cv": metrics_block(np.nan_to_num(dev_s, nan=0.0), np.nan_to_num(dev_f, nan=0.0), dev,
                                      thresholds["smoke"]["deploy"], args.n_boot)}
    report["dev_cv"]["goal_met"] = goal_checks(report["dev_cv"])
    if args.final:
        E = np.stack([r["emb"] for r in test])
        mot = None
        if any("motion" in heads[h].groups for h in heads):
            mot = np.stack([r["motion"] for r in test])
        p = det.proba(E, mot)
        report["test"] = metrics_block(p["smoke"], p["fog"], test, thresholds["smoke"]["deploy"], args.n_boot)
        report["test"]["goal_met"] = goal_checks(report["test"])
    write_json(Path(args.report), report)
    Path(args.report).with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(json.dumps({k: {kk: report[k][kk] for kk in ("smoke_auroc", "fog_auroc", "false_smoke_on_fog",
                                                         "smoke_recall", "goal_met")}
                      for k in ("dev_cv", "test") if k in report}, default=float))
    return 0


def _f(v, fmt="{:.3f}"):
    return "n/a" if v is None else fmt.format(v)


def _ci(ci, k, pct=False):
    c = (ci or {}).get(k) or {}
    if c.get("lo") is None:
        return ""
    return f" [{100 * c['lo']:.0f}, {100 * c['hi']:.0f}]" if pct else f" [{c['lo']:.2f}, {c['hi']:.2f}]"


def render(rep) -> str:
    L = ["# Real-footage detector (far field): smoke and fog", "",
         "> Linear heads on frozen CLIP appearance features (and, if selected, the motion features), trained on "
         "real labelled windows; model selection and thresholds on the dev cameras only; intervals resample "
         "cameras. Imagery courtesy of HPWREN (https://www.hpwren.ucsd.edu/).", "",
         "Goals: fog AUROC >= 0.90; smoke AUROC >= 0.85; at the deploy point at most 5 % false smoke on fog "
         "windows while detecting at least 50 % of the smoke windows.", "",
         "| head | selected feature groups | C | CV AUROC (dev) | dev windows |", "|---|---|---|---|---|"]
    for h, s in rep["selection"].items():
        b = s["best"]
        L.append(f"| {h} | {', '.join(b['groups'])} | {b['C']} | {_f(b['cv_auroc'])} | {s['n_dev']} |")
    L += ["", f"Thresholds: smoke deploy {rep['thresholds']['smoke']['deploy']:.3f}, smoke balanced "
          f"{rep['thresholds']['smoke']['balanced']:.3f}, fog balanced {rep['thresholds']['fog']['balanced']:.3f}.", ""]
    for key, title in (("dev_cv", "Dev cameras, out-of-fold (cross-validated)"), ("test", "Held-out test cameras")):
        if key not in rep:
            continue
        r = rep[key]
        n = r["n"]
        L += [f"## {title}", "",
              f"{n['windows']} windows, {n['cameras']} camera mounts; smoke labelled {n['smoke_known']} "
              f"({n['smoke_pos']} smoke), fog labelled {n['fog_known']} ({n['fog_pos']} fog), "
              f"{n['fog_windows_smoke0']} fog windows without smoke.", "",
              "| metric | value | goal | met |", "|---|---|---|---|",
              f"| fog AUROC | {_f(r['fog_auroc'])}{_ci(r['ci95'], 'fog_auroc')} | >= 0.90 | {r['goal_met']['fog_auroc']} |",
              f"| smoke AUROC | {_f(r['smoke_auroc'])}{_ci(r['ci95'], 'smoke_auroc')} | >= 0.85 | {r['goal_met']['smoke_auroc']} |",
              f"| false smoke on fog windows (deploy) | {_f(r['false_smoke_on_fog'], '{:.1%}')}"
              f"{_ci(r['ci95'], 'false_smoke_on_fog', True)} | <= 5 % | {r['goal_met']['deploy_point']} |",
              f"| smoke recall (deploy) | {_f(r['smoke_recall'], '{:.1%}')}{_ci(r['ci95'], 'smoke_recall', True)} "
              f"| >= 50 % | |", "",
              "| set | smoke AUROC | fog AUROC | false smoke on fog | smoke recall |", "|---|---|---|---|---|"]
        for s, v in r["per_set"].items():
            L.append(f"| {s} | {_f(v['smoke_auroc'])} | {_f(v['fog_auroc'])} | {_f(v['false_smoke_on_fog'], '{:.1%}')} "
                     f"| {_f(v['smoke_recall'], '{:.1%}')} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
