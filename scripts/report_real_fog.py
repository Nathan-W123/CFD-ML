"""Zero-shot test of the far-field model on real fog: the HPWREN fog set (and FIgLib smoke), as a report.

    python scripts/report_real_fog.py --features data/real/hpwren_fog/features.npz \\
        --hpwren-dir data/real/hpwren_fog --labels configs/real/hpwren_fog_labels.csv \\
        --artifacts artifacts_far --figlib data/real/figlib/features.npz \\
        --out reports/far/real_fog_hpwren.json

The exported model and its synthetic thresholds, unchanged (Stage B, training
on real footage, is reported by scripts/evaluate_physics.py --real). Per
window the labels are the consolidated visual labels (prepare_hpwren_fog.py
consolidate); fog 1 / 0 windows with smoke 0 are the fog_only / neither
categories. Reported, on every labelled window and on the test_iid cameras:

* the labels: counts, fog type and extent, labeller agreement, and how often
  the METAR evidence that proposed a window matches what is visible;
* fog: AUROC of P(fog) (fog windows against clear ones), with a 95 %
  bootstrap interval resampling cameras, and its mean +- sd over the training
  seeds; the fog decisions at both operating points;
* smoke on fog: the smoke decisions on the fog windows (false smoke on
  fog_only: alarms over all fog windows, abstentions counted as no alarm, and
  over the decided ones) and on the clear windows, and the AUROC of P(smoke)
  for fog against clear (0.5 = fog raises the smoke score no more than a
  clear morning);
* what was measurable: valid-input rates and abstention reasons per class,
  and the fog results per fog type;
* with --figlib: P(smoke) of FIgLib smoke windows against the HPWREN fog
  windows (the smoke-vs-fog ranking on real footage). The two sets differ in
  years, cameras' state and time of day (FIgLib fires at any hour, these fog
  windows are mornings), so this comparison is confounded and only indicative.
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.data.dataset import FeatureSet
from src.inference.predict import PhysicsExpert
from src.training import metrics as M
from src.utils.io import write_json

OPS = ("balanced", "low_false_alarm")


def _f(v, fmt="{:.3f}"):
    return "n/a" if v is None else fmt.format(v)


def _pct(num, den):
    return None if not den else num / den


def _ci(c):
    return "" if not c or c.get("lo") is None else f" [{c['lo']:.2f}, {c['hi']:.2f}]"


def _seed_dirs(artifacts: Path) -> list:
    return sorted(p for p in (artifacts / "seeds").glob("seed_*") if (p / "model_config.yaml").is_file())


def window_records(fs: FeatureSet, hpwren_dir: Path, labels_csv: Path) -> list:
    """Per row of the HPWREN FeatureSet: window id, camera, fog / smoke labels, fog type / extent, METAR kind."""
    with open(labels_csv, newline="", encoding="utf-8") as fh:
        labels = {r["window_id"]: r for r in csv.DictReader(fh)}
    recs = []
    for i, m in enumerate(fs.meta):
        wid = m.get("window_id") or str(fs.sim_ids[i])
        meta_path = hpwren_dir / wid / "window.json"
        win = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        lab = labels.get(wid, {})
        recs.append({"window_id": wid, "camera": str(fs.groups[i]), "split": str(fs.splits[i]),
                     "fog": fs.y[i, 1], "smoke": fs.y[i, 0], "fog_type": lab.get("fog_type") or "none",
                     "fog_extent": lab.get("fog_extent") or "none", "kind_candidate": win.get("kind_candidate", "")})
    return recs


def label_summary(labels_csv: Path, hpwren_dir: Path) -> dict:
    with open(labels_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    kinds = {}
    for r in rows:
        p = hpwren_dir / r["window_id"] / "window.json"
        kinds[r["window_id"]] = json.loads(p.read_text()).get("kind_candidate", "") if p.is_file() else ""
    metar = Counter((kinds[r["window_id"]], r["fog"] or "unknown") for r in rows)
    stats_path = labels_csv.with_suffix(".stats.json")
    return {"windows_labelled": len(rows),
            "fog": dict(Counter(r["fog"] or "unknown" for r in rows)),
            "smoke": dict(Counter(r["smoke"] or "unknown" for r in rows)),
            "quality": dict(Counter(r["quality"] for r in rows)),
            "fog_type": dict(Counter(r["fog_type"] for r in rows if r["fog"] == "1")),
            "fog_extent": dict(Counter(r["fog_extent"] for r in rows if r["fog"] == "1")),
            "metar_vs_visual": {f"{k}->{f}": n for (k, f), n in sorted(metar.items())},
            "agreement": json.loads(stats_path.read_text()) if stats_path.is_file() else None}


def evaluate(expert: PhysicsExpert, fs: FeatureSet, recs: list, idx: np.ndarray, n_boot: int) -> dict:
    """Zero-shot results of one model on the rows idx of the HPWREN set."""
    fog = np.array([recs[i]["fog"] for i in idx], dtype=np.float64)
    groups = np.array([recs[i]["camera"] for i in idx])
    out = {"n": int(len(idx)), "n_fog": int(np.sum(fog == 1)), "n_clear": int(np.sum(fog == 0)),
           "cameras": int(len(set(groups.tolist())))}
    res = {op: expert.predict(fs.X[idx], op, meta=[fs.meta[i] for i in idx]) for op in OPS}
    p = res["balanced"]["probs"]
    out["auroc_fog"] = M.auroc(p[:, 1], fog)
    out["auroc_smoke_fog_vs_clear"] = M.auroc(p[:, 0], fog)
    out["auroc_ci"] = M.grouped_bootstrap(
        lambda j: {"fog": M.auroc(p[j, 1], fog[j]), "smoke_fog_vs_clear": M.auroc(p[j, 0], fog[j])},
        groups, n_boot=n_boot)
    out["p_smoke_median"] = {"fog": float(np.median(p[fog == 1, 0])) if np.any(fog == 1) else None,
                             "clear": float(np.median(p[fog == 0, 0])) if np.any(fog == 0) else None}
    out["p_fog_median"] = {"fog": float(np.median(p[fog == 1, 1])) if np.any(fog == 1) else None,
                           "clear": float(np.median(p[fog == 0, 1])) if np.any(fog == 0) else None}
    valid = res["balanced"]["valid_input"]
    out["valid_input"] = {"fog": float(np.mean(valid[fog == 1])) if np.any(fog == 1) else None,
                          "clear": float(np.mean(valid[fog == 0])) if np.any(fog == 0) else None}
    out["ops"] = {}
    for op in OPS:
        d = res[op]["decisions"]
        reasons = defaultdict(Counter)
        for i, rs in enumerate(res[op]["abstain_reasons"]):
            for r in rs:
                reasons["fog" if fog[i] == 1 else "clear"][r] += 1
        cls = {}
        for name, m in (("fog", fog == 1), ("clear", fog == 0)):
            sm, fg = d[m, 0], d[m, 1]
            cls[name] = {"n": int(m.sum()),
                         "smoke": {"yes": int(np.sum(sm == 1)), "no": int(np.sum(sm == 0)), "abstain": int(np.sum(sm == -1))},
                         "fog": {"yes": int(np.sum(fg == 1)), "no": int(np.sum(fg == 0)), "abstain": int(np.sum(fg == -1))},
                         "abstain_reasons": dict(reasons[name])}
        f = cls["fog"]
        out["ops"][op] = {**cls,
                          "false_smoke_on_fog": _pct(f["smoke"]["yes"], f["n"]),
                          "false_smoke_on_fog_decided": _pct(f["smoke"]["yes"], f["smoke"]["yes"] + f["smoke"]["no"]),
                          "false_smoke_on_clear": _pct(cls["clear"]["smoke"]["yes"], cls["clear"]["n"]),
                          "fog_recall_decided": _pct(f["fog"]["yes"], f["fog"]["yes"] + f["fog"]["no"]),
                          "fog_false_alarm_decided": _pct(cls["clear"]["fog"]["yes"],
                                                          cls["clear"]["fog"]["yes"] + cls["clear"]["fog"]["no"])}
    by_type = {}
    neg = fog == 0
    for t in sorted({recs[i]["fog_type"] for i, f in zip(idx, fog) if f == 1}):
        m = np.array([recs[i]["fog_type"] == t for i in idx]) & (fog == 1)
        sel = m | neg
        by_type[t] = {"n": int(m.sum()), "valid_input": float(np.mean(valid[m])),
                      "p_fog_median": float(np.median(p[m, 1])), "auroc_vs_clear": M.auroc(p[sel, 1], fog[sel]),
                      "fog_yes_balanced": int(np.sum(res["balanced"]["decisions"][m, 1] == 1)),
                      "smoke_yes_balanced": int(np.sum(res["balanced"]["decisions"][m, 0] == 1))}
    out["by_fog_type"] = by_type
    out["_probs"] = p
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", required=True, help="HPWREN fog FeatureSet (extract_features.py on its real.csv)")
    ap.add_argument("--hpwren-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--artifacts", default="artifacts_far")
    ap.add_argument("--figlib", default=None, help="FIgLib FeatureSet (smoke labels) for the smoke-vs-fog ranking")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    fs = FeatureSet.load(args.features)
    keep = np.flatnonzero(~np.isnan(fs.y[:, 1]))
    fs = fs.subset(keep)
    recs = window_records(fs, Path(args.hpwren_dir), Path(args.labels))
    art = Path(args.artifacts)
    expert = PhysicsExpert.load(art)
    seeds = [PhysicsExpert.load(d) for d in _seed_dirs(art)]
    doc = {"status": "zero_shot_real_fog", "artifacts": str(art), "features": args.features,
           "not_a_deployment_estimate": True, "labels": label_summary(Path(args.labels), Path(args.hpwren_dir)),
           "subsets": {}}
    subsets = {"all": np.arange(len(fs)), "test_iid": np.flatnonzero(fs.splits == "test_iid")}
    probs = {}
    for name, idx in subsets.items():
        if len(idx) == 0:
            continue
        r = evaluate(expert, fs, recs, idx, args.n_boot)
        probs[name] = r.pop("_probs")
        vals = []
        for s in seeds:
            pr = s.predict(fs.X[idx], "balanced", meta=[fs.meta[i] for i in idx])["probs"]
            vals.append(M.auroc(pr[:, 1], fs.y[idx, 1]))
        vals = [v for v in vals if v is not None]
        r["auroc_fog_seeds"] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                "values": vals} if vals else None
        doc["subsets"][name] = r
    if args.figlib:
        fig = FeatureSet.load(args.figlib)
        pos = np.flatnonzero(fig.y[:, 0] == 1)
        pf = expert.predict(fig.X[pos], "balanced", meta=[fig.meta[i] for i in pos])["probs"][:, 0]
        cross = {}
        for name, idx in subsets.items():
            if name not in probs:
                continue
            fogm = fs.y[idx, 1] == 1
            p_h = probs[name][:, 0]
            sel_fig = pf if name == "all" else pf[fig.splits[pos] == name]
            cross[name] = {
                "n_figlib_smoke": int(len(sel_fig)), "n_hpwren_fog": int(fogm.sum()), "n_hpwren_clear": int((~fogm).sum()),
                "auroc_smoke_figlib_smoke_vs_hpwren_fog": M.auroc(np.r_[sel_fig, p_h[fogm]],
                                                                   np.r_[np.ones(len(sel_fig)), np.zeros(fogm.sum())]),
                "auroc_smoke_figlib_smoke_vs_hpwren_clear": M.auroc(np.r_[sel_fig, p_h[~fogm]],
                                                                     np.r_[np.ones(len(sel_fig)), np.zeros((~fogm).sum())]),
                "p_smoke_median_figlib_smoke": float(np.median(sel_fig)) if len(sel_fig) else None}
        doc["smoke_vs_fog_cross_set"] = {"confounded": True, **cross}
    write_json(Path(args.out), doc)
    Path(args.out).with_suffix(".md").write_text(render(doc), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


def render(doc: dict) -> str:
    lab = doc["labels"]
    ag = (lab.get("agreement") or {}).get("rounds_1_2") or {}
    L = ["# Real fog test: HPWREN fog set (far-field model, zero-shot)", "",
         "> The exported far-field model and its synthetic thresholds, unchanged. Labels are visual (blind contact "
         "sheets, independent votes); METAR fog reports only proposed the windows. Imagery courtesy of HPWREN "
         "(https://www.hpwren.ucsd.edu/).", "",
         "## Labels", "",
         f"* {lab['windows_labelled']} windows labelled; fog {lab['fog']}; smoke {lab['smoke']}; quality {lab['quality']}.",
         f"* Fog types (fog = 1): {lab['fog_type']}; extent: {lab['fog_extent']}.",
         f"* Votes 1 and 2 agree on fog for {_f(ag.get('fog_agreement'), '{:.0%}')} of the sheets both decided "
         f"({ag.get('both_definite', 'n/a')}); Cohen's kappa {_f(ag.get('fog_cohen_kappa'), '{:.2f}')}.",
         f"* METAR candidate kind -> visual fog label: {lab['metar_vs_visual']}.", ""]
    for name, r in doc["subsets"].items():
        sd = r.get("auroc_fog_seeds") or {}
        ci = r["auroc_ci"]
        L += [f"## {name} ({r['n']} windows: {r['n_fog']} fog, {r['n_clear']} clear; {r['cameras']} cameras)", "",
              f"* **Fog AUROC** (P(fog), fog against clear): {_f(r['auroc_fog'])}{_ci(ci.get('fog'))}; over the "
              f"seeds {_f(sd.get('mean'))} ± {_f(sd.get('sd'))}.",
              f"* P(smoke), fog against clear: AUROC {_f(r['auroc_smoke_fog_vs_clear'])}"
              f"{_ci(ci.get('smoke_fog_vs_clear'))} (0.5: fog raises the smoke score no more than a clear morning).",
              f"* Median P(fog): fog {_f(r['p_fog_median']['fog'])}, clear {_f(r['p_fog_median']['clear'])}; "
              f"median P(smoke): fog {_f(r['p_smoke_median']['fog'])}, clear {_f(r['p_smoke_median']['clear'])}.",
              f"* Valid input (enough masks and flow): fog {_f(r['valid_input']['fog'], '{:.0%}')}, clear "
              f"{_f(r['valid_input']['clear'], '{:.0%}')}.", "",
              "| operating point | false smoke on fog (all / decided) | false smoke on clear | fog found (decided) | "
              "fog called on clear (decided) | fog windows: smoke yes / no / abstain | clear windows: fog yes / no / abstain |",
              "|---|---|---|---|---|---|---|"]
        for op, o in r["ops"].items():
            fs_, fc = o["fog"]["smoke"], o["clear"]["fog"]
            L.append(f"| {op} | {_f(o['false_smoke_on_fog'], '{:.0%}')} / {_f(o['false_smoke_on_fog_decided'], '{:.0%}')} | "
                     f"{_f(o['false_smoke_on_clear'], '{:.0%}')} | {_f(o['fog_recall_decided'], '{:.0%}')} | "
                     f"{_f(o['fog_false_alarm_decided'], '{:.0%}')} | {fs_['yes']} / {fs_['no']} / {fs_['abstain']} | "
                     f"{fc['yes']} / {fc['no']} / {fc['abstain']} |")
        L += ["", f"Abstention reasons (balanced): fog {r['ops']['balanced']['fog']['abstain_reasons']}; clear "
              f"{r['ops']['balanced']['clear']['abstain_reasons']}.", "",
              "| fog type | n | valid input | median P(fog) | AUROC against clear | fog yes (balanced) | smoke yes (balanced) |",
              "|---|---|---|---|---|---|---|"]
        for t, b in r["by_fog_type"].items():
            L.append(f"| {t} | {b['n']} | {b['valid_input']:.0%} | {b['p_fog_median']:.3f} | {_f(b['auroc_vs_clear'])} | "
                     f"{b['fog_yes_balanced']} | {b['smoke_yes_balanced']} |")
        L.append("")
    cross = doc.get("smoke_vs_fog_cross_set")
    if cross:
        L += ["## Smoke against fog across datasets (confounded, indicative only)", "",
              "FIgLib smoke windows (fires 2016-2024, any hour) against HPWREN windows (2026 mornings): different "
              "years, cameras' state and time of day.", ""]
        for name, c in cross.items():
            if name == "confounded":
                continue
            L.append(f"* {name}: P(smoke) AUROC FIgLib smoke vs HPWREN fog {_f(c['auroc_smoke_figlib_smoke_vs_hpwren_fog'])} "
                     f"({c['n_figlib_smoke']} vs {c['n_hpwren_fog']}); vs HPWREN clear "
                     f"{_f(c['auroc_smoke_figlib_smoke_vs_hpwren_clear'])} ({c['n_hpwren_clear']}).")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
