"""Train and evaluate the real-footage smoke / fog detector (far field), camera-disjoint.

    D=data/real; SETS=""
    for n in figlib hpwren_fog hpwren_fog2 hpwren_smoke hpwren_smoke2; do
        SETS="$SETS --set $n:$D/$n/real.csv:$D/$n/clip.npz:$D/$n/features.npz:$D/$n/clip_fine.npz"
        SETS="$SETS:$D/$n/clip_zoom.npz:$D/$n/clip_ref.npz"; done
    python scripts/train_detector.py $SETS --deploy-recall 0.8 --fog-veto 0.95 \\
        --out artifacts_detector --report reports/far/detector.json [--final]

Data: every --set is name:manifest:clip_npz[:features_npz[:fine_npz[:zoom_npz
[:ref_npz]]]]. A row's labels come from its manifest (smoke / fog: 0, 1 or
empty = unknown), its appearance from the CLIP embeddings
(scripts/embed_windows.py; fine_npz: the 4 x 6 tiles of frames 0 and 11,
--grid 4x6 --frames 0,11 --no-full; zoom_npz: full-resolution quarters of
the top-4 tiles, scripts/zoom_windows.py; ref_npz: the same camera 14 min
before frame 0, scripts/reference_frames.py), its motion (optional feature
group "motion") from the extractor's FeatureSet (seq_id <window_id>-w0).
Windows without zoom crops or a reference frame use a fallback stack.
Splits come from the manifests (camera-mount based in prepare_figlib.py and
prepare_hpwren_fog.py, so one camera is in one split across sets); "dev" is
every split but test_iid.

Per head (smoke, fog), on the dev rows with a known label:

1. model selection: every candidate feature-group set x C (inverse L2
   strength) is scored by 5-fold cross-validation grouped by camera mount
   (out-of-fold AUROC within sets, i.e. over smoke / no-smoke pairs of the
   same set, so easy negatives of another set cannot drive the choice;
   --select-on pooled for the pooled AUROC); the best is refitted on all
   dev rows;
   for smoke, when every row has fine embeddings, also a tile head (one
   model applied to every fine tile, the window scored by its best tile; its
   C by the same grouped cross-validation), a zoom head (the same on the
   zoom crops), and stacks (STACK_CONFIGS: logistic regressions on the
   heads' out-of-fold logits plus the change and reference scores of
   src/detector/features.py, the first whose inputs a window has), whose
   own out-of-fold predictions (a second grouped cross-validation) are the
   head's; the v1 stack (window + tile heads) is fitted and reported too;
2. thresholds from the out-of-fold predictions of the chosen configuration
   (never from the refitted model's own training predictions): "balanced"
   (maximum Youden J) for both heads; for smoke also "deploy": the lowest
   threshold at which at most --deploy-false-smoke (2.5 %) of the dev fog
   windows (fog 1, smoke 0) raise a smoke alarm, or, with --deploy-recall r
   (v2: 0.8, v1's deploy recall), the highest threshold that still catches
   r of the dev smoke windows if that is higher, so a better model spends
   its gain on fewer false alarms rather than on more recall (without it,
   v1's rule: the fog limit alone). The fog limit is half the goal's 5 %, a
   margin for new cameras (the dev estimate is out-of-fold, but a test set of
   about 50 fog windows moves the rate by 2 % per window); for smoke also
   "strict", at which at most --strict-false-alarm (5 %) of all dev no-smoke
   windows raise an alarm (the deploy point budgets fog windows only, and
   clear scenes minutes before a fire are harder); with --fog-veto u, a
   smoke alarm is withheld when the out-of-fold P(fog) is at least u, for
   the thresholds and in every reported rate.

Reported: the cross-validated dev results, and with --final the held-out
test cameras (scored with the one refitted model; nothing is fitted or
chosen on them): fog AUROC, smoke AUROC (all smoke-labelled
windows, and per set; "within sets" counts only smoke / no-smoke pairs from
the same set, so a set-level difference cannot help), and at the deploy point the false-smoke rate on the
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
from src.detector.boxes import fine_regions, instance_labels, load_boxes, zoom_regions  # noqa: E402
from src.detector.features import MODEL_ID, PROMPTS, text_embeddings, window_features  # noqa: E402
from src.detector.features import (change_scores, fog_tile_scores, ref_instances, reference_scores,  # noqa: E402
                                   tile_instances, zoom_instances)
from src.detector.model import Detector, fit_head, fit_tile_head, stack_logit, tile_logit  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.training.baselines import pooled_features  # noqa: E402
from src.utils.io import write_json  # noqa: E402

GROUP_SETS = {"fog": (("full",), ("sims",), ("full", "sims"), ("full", "tiles", "sims"), ("full", "sims", "motion")),
              "smoke": (("tiles",), ("sims",), ("tiles", "sims"), ("full", "tiles", "sims"),
                        ("tiles", "sims", "motion"), ("full", "tiles", "sims", "motion"),
                        ("full", "tiles", "sims", "fine"), ("full", "tiles", "sims", "fine", "motion"))}
C_GRID = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
TILE_C_GRID = (0.0001, 0.0003, 0.001, 0.003)
SELECT_ON = "within"            # model selection maximises the within-set AUROC (set by --select-on)
STACK_C = 1.0
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
        fine = {}
        if len(parts) > 4 and parts[4]:
            zf = np.load(parts[4])
            fine = {str(w): e for w, e in zip(zf["window_id"], zf["emb"])}
        zoom, zoom_tiles, ref = {}, {}, {}
        if len(parts) > 5 and parts[5]:
            zz = np.load(parts[5])
            zoom = {str(w): e for w, e in zip(zz["window_id"], zz["emb"])}
            if "tiles" in zz.files:
                zoom_tiles = {str(w): [int(t) for t in ts] for w, ts in zip(zz["window_id"], zz["tiles"])}
        if len(parts) > 6 and parts[6]:
            zr = np.load(parts[6])
            ref = {str(w): e for w, e in zip(zr["window_id"], zr["fine"])}
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
                             "emb": emb[wid].astype(np.float32), "motion": motion.get(f"{wid}-w0"),
                             "fine": fine[wid].astype(np.float32) if wid in fine else None,
                             "zoom": zoom[wid].astype(np.float32) if wid in zoom else None,
                             "zoom_tiles": zoom_tiles.get(wid),
                             "ref": ref[wid].astype(np.float32) if wid in ref else None})
    return rows


def _stack_rows(rows, key):
    if any(r[key] is None for r in rows):
        return None
    return np.stack([r[key] for r in rows])


def _features(rows, T, groups):
    E = np.stack([r["emb"] for r in rows])
    mot = fine = None
    if "motion" in groups:
        mot = _stack_rows(rows, "motion")
        if mot is None:
            return None
    if "fine" in groups:
        fine = _stack_rows(rows, "fine")
        if fine is None:
            return None
    return window_features(E, T, groups, mot, fine)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


def _folds(n, groups, n_splits=5):
    from sklearn.model_selection import GroupKFold

    return list(GroupKFold(n_splits=n_splits).split(np.zeros((n, 1)), np.zeros(n), groups))


def grouped_oof(X, y, groups, C, n_splits=5):
    """Out-of-fold logits of a linear head (folds grouped by camera mount)."""
    oof = np.zeros(len(y))
    for tr, te in _folds(len(y), groups, n_splits):
        oof[te] = fit_head(X[tr], y[tr], (), C).logit(X[te])
    return oof


def grouped_oof_tile(TF, y, groups, C, n_splits=5, inst=None, balance=False):
    """Out-of-fold window logits of a tile head (its best tile); inst: instance labels and weights (Y, W)."""
    oof = np.zeros(len(y))
    for tr, te in _folds(len(y), groups, n_splits):
        Y, W = (None, None) if inst is None else (inst[0][tr], inst[1][tr])
        oof[te] = tile_logit(fit_tile_head(TF[tr], y[tr], C, Y, W, balance), TF[te])
    return oof


def box_summary(inst, y):
    """How many windows and tiles the box labels use (smoke windows without a usable box are left out)."""
    Y, W = inst
    used = W.sum(1) > 0
    return {"smoke_windows": int(np.sum(y == 1)), "smoke_windows_boxed": int(np.sum(used & (y == 1))),
            "positive_tiles": int(np.sum((Y == 1) & (W > 0))), "negative_tiles": int(np.sum((Y == 0) & (W > 0))),
            "ignored_tiles_in_boxed_windows": int(np.sum((W == 0) & used[:, None]))}


def fit_stack(Z, y, sample_weight=None):
    """The stack: a logistic regression on the inputs Z (N, k), unstandardised: {"coef", "intercept"}."""
    h = fit_head(Z, y, ("stack",), STACK_C, sample_weight=sample_weight)
    coef = h.coef / h.scale
    return {"coef": [float(c) for c in coef], "intercept": float(h.intercept - np.sum(coef * h.mean))}


def cv_scores(z, y, sets):
    """Pooled, within-set and per-set out-of-fold AUROC, and the one model selection maximises (SELECT_ON)."""
    z, y, sets = np.asarray(z), np.asarray(y), np.asarray(sets)
    pooled, within = M.auroc(z, y), within_sets_auroc(z, y, sets)
    per_set = {str(s): M.auroc(z[sets == s], y[sets == s]) for s in sorted(set(sets.tolist()))
               if len(set(y[sets == s].tolist())) == 2}
    crit = within if (SELECT_ON == "within" and within is not None) else pooled
    return {"cv_auroc": pooled, "cv_auroc_within_sets": within, "cv_auroc_per_set": per_set}, crit


def select(rows, head, T):
    """(best config, all scores, out-of-fold logits of the best) for one head on dev rows."""
    y = np.array([r[head] for r in rows])
    g = np.array([r["mount"] for r in rows])
    sets = np.array([r["set"] for r in rows])
    scores, best = [], None
    for groups, C in itertools.product(GROUP_SETS[head], C_GRID):
        X = _features(rows, T, groups)
        if X is None:
            continue
        oof = grouped_oof(X, y, g, C)
        sc, crit = cv_scores(oof, y, sets)
        scores.append({"groups": list(groups), "C": C, **sc})
        if best is None or crit > best[0]:
            best = (crit, groups, C, oof, sc)
    return {"groups": list(best[1]), "C": best[2], **best[4]}, scores, best[3]


def deploy_threshold(p_fog, max_rate):
    """The lowest threshold at which at most max_rate of the fog windows have P(smoke) >= it: just above the
    (k+1)-th highest, k = floor(max_rate * n) (an interpolated quantile can let one more window through)."""
    p = np.sort(np.asarray(p_fog, np.float64))[::-1]
    k = int(np.floor(max_rate * len(p) + 1e-9))
    return float(np.nextafter(p[k], np.inf)) if k < len(p) else 0.0


def youden(p, y):
    t, rec = M.threshold_max_youden(p, y)
    return float(t)


def within_sets_auroc(p, y, sets):
    """AUROC over the smoke / no-smoke pairs drawn from the same set: the per-set AUROCs weighted by their
    number of pairs (sets with one class only contribute no pairs)."""
    num = den = 0.0
    for s in set(sets.tolist()):
        m = sets == s
        npos, nneg = int(np.sum(y[m] == 1)), int(np.sum(y[m] == 0))
        if npos and nneg:
            num += M.auroc(p[m], y[m]) * npos * nneg
            den += npos * nneg
    return num / den if den else None


def metrics_block(p_smoke, p_fog, rows, thr, n_boot=1000, thr_strict=None, p_veto=None, veto=None):
    """AUROCs; at the deploy point (thr) false smoke on fog, false alarms on every no-smoke window and smoke
    recall, and the same at the strict point (thr_strict) if given; camera-bootstrap intervals."""
    ys = np.array([r["smoke"] for r in rows])
    yf = np.array([r["fog"] for r in rows])
    sets = np.array([r["set"] for r in rows])
    cams = np.array([r["mount"] for r in rows])
    fog_win = (yf == 1) & (ys == 0)
    smk = ~np.isnan(ys)
    fgk = ~np.isnan(yf)
    vetoed = (np.asarray(p_veto) >= veto) if (p_veto is not None and veto is not None) else np.zeros(len(rows), bool)
    p_alarm = np.where(vetoed, -1.0, p_smoke)            # a vetoed window never raises a smoke alarm

    def stats(idx):
        idx = np.asarray(idx)
        s, f, fw, ys_, yf_ = smk[idx], fgk[idx], fog_win[idx], ys[idx], yf[idx]
        ps, pf, pa = p_smoke[idx], p_fog[idx], p_alarm[idx]
        pos = s & (ys_ == 1)
        neg = s & (ys_ == 0)
        out = {"smoke_auroc": M.auroc(ps[s], ys_[s]) if s.any() else None,
               "smoke_auroc_within_sets": within_sets_auroc(ps[s], ys_[s], sets[idx][s]),
               "fog_auroc": M.auroc(pf[f], yf_[f]) if f.any() else None,
               "fog_auroc_within_sets": within_sets_auroc(pf[f], yf_[f], sets[idx][f])}
        for suffix, t in (("", thr), ("_strict", thr_strict)):
            if t is None:
                continue
            out[f"false_smoke_on_fog{suffix}"] = float(np.mean(pa[fw] >= t)) if fw.any() else None
            out[f"false_alarm_no_smoke{suffix}"] = float(np.mean(pa[neg] >= t)) if neg.any() else None
            out[f"smoke_recall{suffix}"] = float(np.mean(pa[pos] >= t)) if pos.any() else None
        return out

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


_FOG_W, _FOG_R = ("fog_chg", "fogd_chg"), ("fog_ref", "fogd_ref")
STACK_CONFIGS = (("head", "tile", "zoom", "chg_max", "chg_rz", "ref_max", "ref_rz") + _FOG_W + _FOG_R,
                 ("head", "tile", "zoom", "chg_max", "chg_rz") + _FOG_W,
                 ("head", "tile", "chg_max", "chg_rz", "ref_max", "ref_rz") + _FOG_W + _FOG_R,
                 ("head", "tile", "chg_max", "chg_rz") + _FOG_W)


def fit_stacks(inputs, y, g, configs, weight=None):
    """Stacks for each input configuration (rows with every input present), the out-of-fold stacked logit of
    every row from the first configuration it has all inputs for (NaN = missing input), and per row the
    configuration used."""
    stacks, oof = [], np.full(len(y), np.nan)
    used = np.array([""] * len(y), dtype=object)
    for cfg in configs:
        if any(k not in inputs for k in cfg):
            continue
        Z = np.stack([inputs[k] for k in cfg], 1)
        ok = ~np.isnan(Z).any(1)
        if ok.sum() < 50 or len(set(y[ok].tolist())) < 2:
            continue
        w = np.ones(len(y)) if weight is None else np.asarray(weight, np.float64)
        stacks.append({"inputs": list(cfg), **fit_stack(Z[ok], y[ok], w[ok])})
        todo = ok & np.isnan(oof)
        used[todo] = "+".join(cfg)
        for tr, te in _folds(len(y), g):
            tr, te = tr[ok[tr]], te[todo[te]]
            if len(te):
                st = fit_stack(Z[tr], y[tr], w[tr])
                oof[te] = Z[te] @ np.asarray(st["coef"]) + st["intercept"]
    return stacks, oof, used


def oof_fog_veto(dev, Tf, fog_best, g_all):
    """Out-of-fold P(fog) for every dev row (folds by camera mount; fog head fitted on the fog-labelled rows of
    the other folds), the input of the fog veto."""
    X = _features(dev, Tf, fog_best["groups"])
    yf = np.array([r["fog"] for r in dev])
    out = np.zeros(len(dev))
    for tr, te in _folds(len(dev), g_all):
        tr = tr[~np.isnan(yf[tr])]
        out[te] = fit_head(X[tr], yf[tr], (), fog_best["C"]).proba(X[te])
    return out


def smoke_thresholds(p, y, fog_win, veto_mask, args):
    """deploy, strict and balanced thresholds on out-of-fold probabilities after the fog veto."""
    q = np.where(veto_mask, -1.0, p)                   # a vetoed window never alarms
    thr = {"balanced": youden(p, y)}
    deploy = deploy_threshold(q[fog_win], args.deploy_false_smoke)
    if args.deploy_recall is not None:
        pos = np.sort(q[y == 1])[::-1]
        k = int(np.ceil(args.deploy_recall * len(pos))) - 1
        deploy = max(deploy, float(pos[max(k, 0)]))
    thr["deploy"] = deploy
    thr["strict"] = deploy_threshold(q[y == 0], args.strict_false_alarm)
    if args.fog_veto is not None:
        thr["fog_veto"] = float(args.fog_veto)
    return thr


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", action="append", required=True, dest="sets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--deploy-false-smoke", type=float, default=0.025,
                    help="share of the dev fog windows allowed to raise a smoke alarm at the deploy threshold")
    ap.add_argument("--deploy-recall", type=float, default=None,
                    help="dev smoke recall the deploy threshold keeps (never above the fog limit); none: fog only")
    ap.add_argument("--strict-false-alarm", type=float, default=0.05,
                    help="share of all dev no-smoke windows allowed to raise a smoke alarm at the strict threshold")
    ap.add_argument("--fog-veto", type=float, default=None,
                    help="withhold a smoke alarm when P(fog) is at least this (none: no veto)")
    ap.add_argument("--stack-fog-weight", type=float, default=1.0,
                    help="sample weight of the fog windows (fog 1, smoke 0) when the v2 stacks are fitted")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--select-on", choices=("within", "pooled"), default="within",
                    help="out-of-fold AUROC that model selection maximises: within sets (default) or pooled")
    ap.add_argument("--boxes", action="append", default=[],
                    help="smoke box CSV (scripts/box_labels.py): tile and zoom heads fitted on instance labels")
    ap.add_argument("--box-min-confidence", choices=("low", "medium", "high"), default="medium")
    ap.add_argument("--box-pos-frac", type=float, default=0.25,
                    help="a tile is positive when it and the box overlap by this share of the smaller of the two")
    ap.add_argument("--box-balance", action="store_true",
                    help="positive tiles together weigh as much as negative ones in the box-supervised heads")
    ap.add_argument("--ref-tile-head", action="store_true",
                    help="also a tile head on each fine tile's change since the reference frame (stack input "
                         "'tileref' in the stacks that use the reference)")
    ap.add_argument("--zoom-selector",
                    help="detector whose tile head selected the stored zoom tiles (kept as the zoom selector)")
    ap.add_argument("--final", action="store_true", help="evaluate on the held-out test_iid cameras (once)")
    ap.add_argument("--save-oof", help="also write the dev out-of-fold probabilities and stack inputs (.npz)")
    args = ap.parse_args(argv)
    global SELECT_ON
    SELECT_ON = args.select_on

    rows = load_sets(args.sets)
    boxes = {}
    for path in args.boxes:
        boxes.update(load_boxes(path))
    dev = [r for r in rows if r["split"] != "test_iid"]
    test = [r for r in rows if r["split"] == "test_iid"]
    T = {h: text_embeddings(PROMPTS[h]) for h in ("smoke", "fog")}
    heads, thresholds, selection, oof = {}, {}, {}, {}
    tile_heads, zoom_heads, stacks, ref_tile_heads = {}, {}, {}, {}
    oof_v1, smoke_inputs = None, {}
    for head in ("fog", "smoke"):
        hr = [r for r in dev if not np.isnan(r[head])]
        best, scores, oof_z = select(hr, head, T[head])
        selection[head] = {"best": best, "scores": scores, "n_dev": len(hr)}
        X = _features(hr, T[head], best["groups"])
        y = np.array([r[head] for r in hr])
        heads[head] = fit_head(X, y, best["groups"], best["C"])
        fine = _stack_rows(hr, "fine")
        if head == "smoke" and fine is not None:
            g = np.array([r["mount"] for r in hr])
            sets = np.array([r["set"] for r in hr])

            def side_head(TI, kind, m, inst=None):
                """C of a tile-type head by grouped CV on rows m (inst: instance labels of those rows);
                out-of-fold logits (NaN outside m)."""
                best_side = None
                side_scores = []
                for C in TILE_C_GRID:
                    zt = grouped_oof_tile(TI, y[m], g[m], C, inst=inst, balance=args.box_balance)
                    sc, crit = cv_scores(zt, y[m], sets[m])
                    side_scores.append({"C": C, **sc})
                    if best_side is None or crit > best_side[0]:
                        best_side = (crit, C, zt, sc)
                full = np.full(len(y), np.nan)
                full[m] = best_side[2]
                selection[head][kind] = {"C": best_side[1], **best_side[3], "n": int(m.sum()), "scores": side_scores}
                print(f"{head}: {kind} head C {best_side[1]} on {int(m.sum())} windows, CV {best_side[3]}",
                      file=sys.stderr)
                return best_side[1], full

            TF = tile_instances(fine, T[head])
            wids = [r["window_id"] for r in hr]
            box_kw = {"min_confidence": args.box_min_confidence, "pos_frac": args.box_pos_frac}
            inst = None
            if boxes:
                regions = fine_regions()
                inst = instance_labels(wids, y, boxes, lambda i: regions, **box_kw)
                selection[head]["box_labels"] = box_summary(inst, y)
            C_tile, z_tile = side_head(TF, "tile", np.ones(len(y), bool), inst)
            tile_heads[head] = fit_tile_head(TF, y, C_tile, *(inst or (None, None)), balance=args.box_balance)
            inputs = {"head": oof_z, "tile": z_tile}
            c = change_scores(fine, T[head])
            inputs["chg_max"], inputs["chg_rz"] = c[:, 0], c[:, 1]
            has_zoom = np.array([r["zoom"] is not None for r in hr])
            if has_zoom.any():
                TZ = zoom_instances(np.stack([r["zoom"] for r in hr if r["zoom"] is not None]), T[head])
                inst_z = None
                if boxes:
                    zr = [r for r in hr if r["zoom"] is not None]
                    if any(r["zoom_tiles"] is None for r in zr):
                        raise SystemExit("--boxes needs the zoom tiles of every zoom row (the 'tiles' array of "
                                         "scripts/zoom_windows.py's output)")
                    inst_z = instance_labels([r["window_id"] for r in zr], y[has_zoom], boxes,
                                             lambda i: zoom_regions(zr[i]["zoom_tiles"]), **box_kw)
                    selection[head]["box_labels_zoom"] = box_summary(inst_z, y[has_zoom])
                C_zoom, inputs["zoom"] = side_head(TZ, "zoom", has_zoom, inst_z)
                zoom_heads[head] = fit_tile_head(TZ, y[has_zoom], C_zoom, *(inst_z or (None, None)),
                                                 balance=args.box_balance)
            has = np.array([r["ref"] is not None for r in hr])
            fs = fog_tile_scores(fine, None, T[head], T["fog"])
            if has.any():
                refs = np.stack([r["ref"] for r in hr if r["ref"] is not None])
                rs = np.full((len(hr), 2), np.nan)
                rs[has] = reference_scores(fine[has], refs, T[head])
                inputs["ref_max"], inputs["ref_rz"] = rs[:, 0], rs[:, 1]
                fs[has] = fog_tile_scores(fine[has], refs, T[head], T["fog"])
                if args.ref_tile_head:
                    TR = ref_instances(fine[has], refs, T[head])
                    inst_r = None if inst is None else (inst[0][has], inst[1][has])
                    C_ref, inputs["tileref"] = side_head(TR, "tileref", has, inst_r)
                    ref_tile_heads[head] = fit_tile_head(TR, y[has], C_ref, *(inst_r or (None, None)),
                                                         balance=args.box_balance)
            inputs["fog_chg"], inputs["fogd_chg"] = fs[:, 0], fs[:, 1]
            if has.any():
                inputs["fog_ref"], inputs["fogd_ref"] = fs[:, 2], fs[:, 3]
            # v1: window head + tile head only (kept for the paired comparison)
            st_v1, oof_v1, _ = fit_stacks(inputs, y, g, (("head", "tile"),))
            fog_neg = np.array([(r["fog"] == 1) and (r["smoke"] == 0) for r in hr])
            configs = tuple(c + ("tileref",) if (args.ref_tile_head and "ref_max" in c) else c
                            for c in STACK_CONFIGS)
            stacks[head], oof_s, used = fit_stacks(inputs, y, g, configs,
                                                   np.where(fog_neg, args.stack_fog_weight, 1.0))
            smoke_inputs = {"window_id": np.array([r["window_id"] for r in hr]), **{k: np.asarray(v, np.float64)
                                                                                    for k, v in inputs.items()}}
            selection[head]["stacked_v1"] = {**cv_scores(oof_v1, y, sets)[0], "stack": st_v1[0]}
            selection[head]["stacked"] = {**cv_scores(oof_s, y, sets)[0], "stacks": stacks[head],
                                          "rows_per_stack": {k: int(np.sum(used == k)) for k in sorted(set(used))}}
            print(f"{head}: stacked v1 {selection[head]['stacked_v1']['cv_auroc_within_sets']:.3f} within sets, "
                  f"v2 {selection[head]['stacked']['cv_auroc_within_sets']:.3f}", file=sys.stderr)
            oof_z = oof_s
        oof_p = _sigmoid(oof_z)
        if head == "fog":
            thresholds[head] = {"balanced": youden(oof_p, y)}
        else:
            g_all = np.array([r["mount"] for r in dev])
            pv_all = oof_fog_veto(dev, T["fog"], selection["fog"]["best"], g_all)
            pv = dict(zip([r["window_id"] for r in dev], pv_all))
            veto = (np.array([pv[r["window_id"]] for r in hr]) >= args.fog_veto) if args.fog_veto is not None \
                else np.zeros(len(hr), bool)
            fog_win = np.array([(r["fog"] == 1) and (r["smoke"] == 0) for r in hr])
            thresholds[head] = smoke_thresholds(oof_p, y, fog_win, veto, args)
            if oof_v1 is not None:
                thr_v1 = smoke_thresholds(_sigmoid(oof_v1), y, fog_win, veto, args)
                oof["smoke_v1"] = dict(zip([r["window_id"] for r in hr], _sigmoid(oof_v1)))
        oof[head] = {r["window_id"]: p for r, p in zip(hr, oof_p)}
        print(f"{head}: best {best} on {len(hr)} dev windows", file=sys.stderr)
    zoom_selector = {}
    if args.zoom_selector and zoom_heads:
        zoom_selector = {h: Detector.load(args.zoom_selector).tile_heads[h] for h in zoom_heads}
    det = Detector(heads=heads, text=T, thresholds=thresholds,
                   info={"model_id": MODEL_ID, "sets": args.sets, "deploy_false_smoke": args.deploy_false_smoke,
                         "deploy_recall": args.deploy_recall, "strict_false_alarm": args.strict_false_alarm,
                         "fog_veto": args.fog_veto, "stack_fog_weight": args.stack_fog_weight,
                         "boxes": args.boxes, "box_min_confidence": args.box_min_confidence,
                         "box_pos_frac": args.box_pos_frac, "box_balance": args.box_balance,
                         "zoom_selector": args.zoom_selector, "ref_tile_head": args.ref_tile_head,
                         "selection": {h: selection[h]["best"] for h in selection},
                         "tile_heads": {h: selection[h]["tile"]["C"] for h in tile_heads},
                         "zoom_heads": {h: selection[h]["zoom"]["C"] for h in zoom_heads}},
                   tile_heads=tile_heads, stacks=stacks, zoom_heads=zoom_heads, zoom_selector=zoom_selector,
                   ref_tile_heads=ref_tile_heads)
    det.save(args.out)

    # Cross-validated dev results: each window's out-of-fold probability for each head it has a label for
    dev_s = np.array([oof["smoke"].get(r["window_id"], np.nan) for r in dev])
    dev_f = np.array([oof["fog"].get(r["window_id"], np.nan) for r in dev])
    dev_v = pv_all if args.fog_veto is not None else None
    report = {"status": "detector", "goals": GOALS, "selection": selection, "thresholds": thresholds,
              "rules": {"deploy_false_smoke": args.deploy_false_smoke, "deploy_recall": args.deploy_recall,
                        "strict_false_alarm": args.strict_false_alarm, "fog_veto": args.fog_veto,
                        "stack_fog_weight": args.stack_fog_weight, "select_on": args.select_on,
                        "boxes": args.boxes, "box_min_confidence": args.box_min_confidence,
                        "box_pos_frac": args.box_pos_frac, "box_balance": args.box_balance,
                        "ref_tile_head": args.ref_tile_head},
              "dev_cv": metrics_block(np.nan_to_num(dev_s, nan=0.0), np.nan_to_num(dev_f, nan=0.0), dev,
                                      thresholds["smoke"]["deploy"], args.n_boot, thresholds["smoke"]["strict"],
                                      dev_v, args.fog_veto)}
    report["dev_cv"]["goal_met"] = goal_checks(report["dev_cv"])
    if args.save_oof:
        np.savez_compressed(args.save_oof, window_id=np.array([r["window_id"] for r in dev]),
                            set=np.array([r["set"] for r in dev]), smoke=np.array([r["smoke"] for r in dev]),
                            fog=np.array([r["fog"] for r in dev]), p_smoke=dev_s, p_fog=dev_f,
                            p_fog_veto=dev_v if dev_v is not None else np.full(len(dev), np.nan),
                            p_smoke_v1=np.array([oof.get("smoke_v1", {}).get(r["window_id"], np.nan) for r in dev]),
                            **{f"input_{k}": v for k, v in smoke_inputs.items()})
    if "smoke_v1" in oof:
        dev_s1 = np.array([oof["smoke_v1"].get(r["window_id"], np.nan) for r in dev])
        report["thresholds_v1_stack"] = thr_v1
        report["dev_cv_v1_stack"] = metrics_block(np.nan_to_num(dev_s1, nan=0.0), np.nan_to_num(dev_f, nan=0.0),
                                                  dev, thr_v1["deploy"], args.n_boot, thr_v1["strict"], dev_v,
                                                  args.fog_veto)
    if args.final:
        p_s, p_f, used = np.zeros(len(test)), np.zeros(len(test)), []
        mot_needed = any("motion" in heads[h].groups for h in heads)
        groups = {}
        for i, r in enumerate(test):
            key = (r["fine"] is not None, r.get("zoom") is not None, r.get("ref") is not None)
            groups.setdefault(key, []).append(i)
        for (has_fine, has_zoom, has_ref), idx in groups.items():
            rs = [test[i] for i in idx]
            E = np.stack([r["emb"] for r in rs])
            mot = np.stack([r["motion"] for r in rs]) if mot_needed else None
            fine = np.stack([r["fine"] for r in rs]) if has_fine else None
            zoom = np.stack([r["zoom"] for r in rs]) if has_zoom else None
            ref = np.stack([r["ref"] for r in rs]) if has_ref else None
            p = det.proba(E, mot, fine, zoom, ref)
            p_s[idx], p_f[idx] = p["smoke"], p["fog"]
            avail = det.stack_inputs("smoke", E[:1], None if mot is None else mot[:1],
                                     None if fine is None else fine[:1], None if zoom is None else zoom[:1],
                                     None if ref is None else ref[:1])
            used.append({"stack": det.stack_used("smoke", avail), "windows": len(idx)})
        report["test_stacks_used"] = used
        report["test"] = metrics_block(p_s, p_f, test, thresholds["smoke"]["deploy"], args.n_boot,
                                       thresholds["smoke"]["strict"], p_f if args.fog_veto is not None else None,
                                       args.fog_veto)
        report["test"]["goal_met"] = goal_checks(report["test"])
        if "smoke_v1" in oof:
            det_v1 = Detector(heads=heads, text=T, thresholds={"smoke": thr_v1, "fog": thresholds["fog"]},
                              tile_heads=tile_heads, stacks={"smoke": st_v1})
            fine_t = np.stack([r["fine"] for r in test])
            mot_t = np.stack([r["motion"] for r in test]) if mot_needed else None
            p1 = det_v1.proba(np.stack([r["emb"] for r in test]), mot_t, fine_t)
            report["test_v1_stack"] = metrics_block(p1["smoke"], p1["fog"], test, thr_v1["deploy"], args.n_boot,
                                                    thr_v1["strict"], p1["fog"] if args.fog_veto is not None
                                                    else None, args.fog_veto)
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
         "> Linear heads on frozen CLIP appearance features (and, if selected, the motion features; for smoke "
         "also a tile head on 4 x 6 fine tiles and, if given, a zoom head on full-resolution crops and the "
         "prompt-contrast change within the window and against a reference frame 14 min earlier, all stacked "
         "with the window head), trained on "
         "real labelled windows; model selection and thresholds on the dev cameras only; intervals resample "
         "cameras. Imagery courtesy of HPWREN (https://www.hpwren.ucsd.edu/).", "",
         "Goals: fog AUROC >= 0.90; smoke AUROC >= 0.85; at the deploy point at most 5 % false smoke on fog "
         "windows while detecting at least 50 % of the smoke windows.", "",
         "| head | selected feature groups | C | CV AUROC pooled | CV AUROC within sets | dev windows |",
         "|---|---|---|---|---|---|"]
    for h, s in rep["selection"].items():
        b = s["best"]
        L.append(f"| {h} | {', '.join(b['groups'])} | {b['C']} | {_f(b['cv_auroc'])} "
                 f"| {_f(b.get('cv_auroc_within_sets'))} | {s['n_dev']} |")
        boxed = " (box labels)" if rep.get("rules", {}).get("boxes") else ""
        for kind, label in (("tile", f"fine tiles, best tile{boxed}"),
                            ("zoom", f"full-resolution zoom crops, best crop{boxed}"),
                            ("tileref", f"fine tiles against the reference frame, best tile{boxed}")):
            if kind in s:
                t = s[kind]
                L.append(f"| {h}: {kind} head | {label} | {t['C']} | {_f(t['cv_auroc'])} "
                         f"| {_f(t.get('cv_auroc_within_sets'))} | {t.get('n', s['n_dev'])} |")
        if "stacked_v1" in s:
            st = s["stacked_v1"]
            L.append(f"| {h}: v1 stack | head + tile head | | {_f(st['cv_auroc'])} "
                     f"| {_f(st.get('cv_auroc_within_sets'))} | {s['n_dev']} |")
        if "stacked" in s:
            st = s["stacked"]
            used = st.get("rows_per_stack") or {}
            L.append(f"| {h}: stacked (deployed) | {'; '.join(f'{k} ({v})' for k, v in used.items()) or 'head + tile head'} "
                     f"| | {_f(st['cv_auroc'])} | {_f(st.get('cv_auroc_within_sets'))} | {s['n_dev']} |")
    ts = rep["thresholds"]["smoke"]
    L += ["", f"Thresholds: smoke deploy {ts['deploy']:.3f}, smoke strict {ts.get('strict', float('nan')):.3f}, "
          f"smoke balanced {ts['balanced']:.3f}, fog balanced {rep['thresholds']['fog']['balanced']:.3f}"
          + (f"; smoke alarms withheld when P(fog) >= {ts['fog_veto']}" if ts.get("fog_veto") is not None else "")
          + ".", ""]
    if rep.get("rules"):
        L += ["Rules: " + ", ".join(f"{k} {v}" for k, v in rep["rules"].items()) + ".", ""]
    if "dev_cv_v1_stack" in rep:
        L += ["## Deployed stack against the v1 stack (window + tile heads), same operating rules", "",
              "| | dev, v1 stack | dev, deployed | test, v1 stack | test, deployed |", "|---|---|---|---|---|"]
        cols = [rep.get("dev_cv_v1_stack"), rep.get("dev_cv"), rep.get("test_v1_stack"), rep.get("test")]
        set_names = sorted({sn for c in cols if c for sn in c.get("per_set", {})})
        smoke_sets = [sn for sn in set_names if any(c and c["per_set"].get(sn, {}).get("smoke_auroc") is not None
                                                    for c in cols)]
        neg_sets = [sn for sn in set_names if any(c and c["per_set"].get(sn, {}).get("false_alarm_no_smoke")
                                                  is not None for c in cols)]

        def row(label, get, fmt="{:.3f}"):
            vals = []
            for c in cols:
                try:
                    vals.append(_f(get(c), fmt) if c is not None else "")
                except (KeyError, TypeError):
                    vals.append("n/a")
            L.append(f"| {label} | " + " | ".join(vals) + " |")

        row("smoke AUROC within sets", lambda c: c["smoke_auroc_within_sets"])
        for sname in smoke_sets:
            row(f"smoke AUROC, {sname}", lambda c, sname=sname: c["per_set"][sname]["smoke_auroc"])
        row("fog AUROC", lambda c: c["fog_auroc"])
        row("deploy: smoke recall", lambda c: c["smoke_recall"], "{:.1%}")
        row("deploy: alarms on fog windows", lambda c: c["false_smoke_on_fog"], "{:.1%}")
        row("deploy: alarms on all smoke-free windows", lambda c: c["false_alarm_no_smoke"], "{:.1%}")
        for sname in neg_sets:
            row(f"deploy: alarms on smoke-free windows, {sname}",
                lambda c, sname=sname: c["per_set"][sname]["false_alarm_no_smoke"], "{:.1%}")
        row("strict: smoke recall", lambda c: c["smoke_recall_strict"], "{:.1%}")
        row("strict: alarms on all smoke-free windows", lambda c: c["false_alarm_no_smoke_strict"], "{:.1%}")
        L.append("")
    if rep.get("test_stacks_used"):
        L += ["Test windows by stack: " + "; ".join(f"{'+'.join(u['stack'])}: {u['windows']}"
                                                   for u in rep["test_stacks_used"]) + ".", ""]
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
              f"| fog AUROC within sets | {_f(r.get('fog_auroc_within_sets'))}{_ci(r['ci95'], 'fog_auroc_within_sets')} | | |",
              f"| smoke AUROC | {_f(r['smoke_auroc'])}{_ci(r['ci95'], 'smoke_auroc')} | >= 0.85 | {r['goal_met']['smoke_auroc']} |",
              f"| smoke AUROC within sets | {_f(r.get('smoke_auroc_within_sets'))}"
              f"{_ci(r['ci95'], 'smoke_auroc_within_sets')} | | |",
              f"| false smoke on fog windows (deploy) | {_f(r['false_smoke_on_fog'], '{:.1%}')}"
              f"{_ci(r['ci95'], 'false_smoke_on_fog', True)} | <= 5 % | {r['goal_met']['deploy_point']} |",
              f"| smoke recall (deploy) | {_f(r['smoke_recall'], '{:.1%}')}{_ci(r['ci95'], 'smoke_recall', True)} "
              f"| >= 50 % | |",
              f"| false alarms on all no-smoke windows (deploy) | {_f(r.get('false_alarm_no_smoke'), '{:.1%}')}"
              f"{_ci(r['ci95'], 'false_alarm_no_smoke', True)} | | |"]
        if r.get("smoke_recall_strict") is not None:
            L += [f"| strict point: false alarms on all no-smoke windows | "
                  f"{_f(r.get('false_alarm_no_smoke_strict'), '{:.1%}')}{_ci(r['ci95'], 'false_alarm_no_smoke_strict', True)} | | |",
                  f"| strict point: false smoke on fog windows | {_f(r.get('false_smoke_on_fog_strict'), '{:.1%}')}"
                  f"{_ci(r['ci95'], 'false_smoke_on_fog_strict', True)} | | |",
                  f"| strict point: smoke recall | {_f(r.get('smoke_recall_strict'), '{:.1%}')}"
                  f"{_ci(r['ci95'], 'smoke_recall_strict', True)} | | |"]
        L += ["",
              "| set | smoke AUROC | fog AUROC | false smoke on fog | false alarms, no-smoke | smoke recall "
              "| strict: false alarms, no-smoke | strict: smoke recall |", "|---|---|---|---|---|---|---|---|"]
        for s, v in r["per_set"].items():
            L.append(f"| {s} | {_f(v['smoke_auroc'])} | {_f(v['fog_auroc'])} | {_f(v['false_smoke_on_fog'], '{:.1%}')} "
                     f"| {_f(v.get('false_alarm_no_smoke'), '{:.1%}')} | {_f(v['smoke_recall'], '{:.1%}')} "
                     f"| {_f(v.get('false_alarm_no_smoke_strict'), '{:.1%}')} | {_f(v.get('smoke_recall_strict'), '{:.1%}')} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
