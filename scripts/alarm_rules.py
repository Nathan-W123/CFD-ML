"""Tune an alarm rule over per-minute window scores on dev cameras, then evaluate it on test cameras.

    python scripts/alarm_rules.py tune --scores fires_dev.npz --scores ordinary_dev.npz \\
        --out reports/far/alarm_rules_dev.json
    python scripts/alarm_rules.py evaluate --chosen reports/far/alarm_rules_dev.json \\
        --scores fires_test.npz --scores ordinary_test.npz --out reports/far/alarm_rules_test.json

Rules and metrics: src/detector/temporal.py; the protocol: configs/real/detector_temporal_rules.json.
Scores come from scripts/score_dense.py (fog-vetoed windows count as P(smoke) = 0).

tune: R0 (one window >= T) at the v3 deploy and strict thresholds is the baseline. Candidates: R1 (k of the
last m windows >= T1, m 2-5, k 2..m, optionally or one window >= T2 > T1) and R2 (mean logit of the last m
windows >= logit(T), m 2-6), thresholds from a grid of dev window-score quantiles (0.50 to 0.995). Among the
candidates whose dev fire detection within 15 and within 30 minutes are both at least the baseline's, the one
with the fewest dev false-alarm events per camera-hour (fire sequences' pre-plume footage and ordinary
footage pooled) is chosen; ties go to the simpler rule (single < smooth < persist < persist with bypass, then
smaller m), then to more fires within 15 minutes.

evaluate: the baselines and the chosen rule on the given scores, each with 95 % intervals resampling camera
mounts, and the paired difference chosen - baseline (deploy) on the same resampled mounts.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from src.detector.temporal import Rule, evaluate
from src.utils.io import write_json

DEPLOY, STRICT = 0.234, 0.354            # artifacts_detector_v3 thresholds (deploy, strict)
QUANTILES = np.r_[np.arange(0.50, 0.90, 0.02), np.arange(0.90, 0.9951, 0.005)]
KEYS = ("fires_within_5", "fires_within_10", "fires_within_15", "fires_within_20", "fires_within_30",
        "fires_detected", "median_delay_detected", "fire_pre_events_per_hour", "ordinary_events_per_hour",
        "events_per_hour", "fire_pre_window_share", "ordinary_window_share")


def load_sequences(paths):
    """Sequences on a minute grid: [{"p", "off", "kind", "mount", "seq"}] (fog-vetoed windows -> P(smoke) 0)."""
    seqs, all_p = [], []
    for path in paths:
        z = np.load(path)
        kind = str(z["kind"])
        p = z["p_smoke"].astype(np.float64)
        veto = z["fog_veto"].astype(np.float64)
        p = np.where(~np.isnan(veto) & (z["p_fog"] >= veto), 0.0, p)
        all_p.append(p)
        seq = z["seq"].astype(str)
        for s in np.unique(seq):
            i = np.flatnonzero(seq == s)
            t = z["t_end"][i]
            o = np.argsort(t)
            i, t = i[o], t[o]
            minute = np.round((t - t[0]) / 60.0).astype(int)
            grid_p = np.full(minute[-1] + 1, np.nan)
            grid_off = np.full(minute[-1] + 1, np.nan)
            grid_p[minute] = p[i]
            if kind == "fire":
                grid_off[minute] = z["off_end_s"][i] / 60.0
            seqs.append({"p": grid_p, "off": grid_off, "kind": kind, "mount": str(z["mount"][i[0]]), "seq": s})
    return seqs, np.concatenate(all_p)


def candidates(grid):
    yield Rule("single", DEPLOY)
    for m in range(2, 7):
        for T in grid:
            yield Rule("smooth", float(T), m=m)
    for m in range(2, 6):
        for k in range(2, m + 1):
            for T1 in grid:
                yield Rule("persist", float(T1), k=k, m=m)
                for T2 in grid:
                    if T2 > T1:
                        yield Rule("persist", float(T1), k=k, m=m, T2=float(T2))


def complexity(rule):
    return ({"single": 0, "smooth": 1, "persist": 2}[rule.kind] + (rule.T2 is not None), rule.m, rule.k)


def rule_dict(rule):
    return {"kind": rule.kind, "T": rule.T, "k": rule.k, "m": rule.m, "T2": rule.T2, "name": rule.name()}


def cmd_tune(args) -> int:
    seqs, all_p = load_sequences(args.scores)
    grid = np.unique(np.round(np.quantile(all_p, QUANTILES), 4))
    base = {name: evaluate(Rule("single", T), seqs) for name, T in (("R0_deploy", DEPLOY), ("R0_strict", STRICT))}
    b = base["R0_deploy"]
    feasible, n = [], 0
    for rule in candidates(grid):
        n += 1
        r = evaluate(rule, seqs)
        if r["fires_within_15"] >= b["fires_within_15"] and r["fires_within_30"] >= b["fires_within_30"]:
            feasible.append((r["events_per_hour"], complexity(rule), -r["fires_within_15"], rule, r))
    feasible.sort(key=lambda x: x[:3])
    chosen_rule, chosen = feasible[0][3], feasible[0][4]
    best_by_kind = {}
    for f in feasible:
        best_by_kind.setdefault(f[3].kind + ("+bypass" if f[3].T2 is not None else ""), {**f[4], **rule_dict(f[3])})
    rep = {"status": "alarm_rule_tuning", "scores": args.scores, "grid": grid.tolist(), "candidates": n,
           "feasible": len(feasible), "baselines": base, "chosen": {**chosen, **rule_dict(chosen_rule)},
           "best_feasible_by_kind": best_by_kind,
           "sequences": {"fire": sum(s["kind"] == "fire" for s in seqs),
                         "ordinary": sum(s["kind"] == "ordinary" for s in seqs)}}
    write_json(Path(args.out), rep)
    print(json.dumps({"chosen": rep["chosen"]["name"], **{k: rep["chosen"][k] for k in KEYS},
                      "baseline": {k: b[k] for k in KEYS}}, default=float, indent=1))
    return 0


def _boot(fn, seqs, n_boot, seed=0):
    mounts = sorted({s["mount"] for s in seqs})
    by = {m: [s for s in seqs if s["mount"] == m] for m in mounts}
    rng = np.random.default_rng(seed)
    samples = {}
    for _ in range(n_boot):
        pick = rng.integers(0, len(mounts), len(mounts))
        sub = [s for j in pick for s in by[mounts[j]]]
        for k, v in fn(sub).items():
            if v is not None and np.isfinite(v):
                samples.setdefault(k, []).append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in samples.items()
            if len(v) * 2 >= n_boot}


def cmd_evaluate(args) -> int:
    seqs, _ = load_sequences(args.scores)
    ch = json.loads(Path(args.chosen).read_text())["chosen"]
    rules = {"R0_deploy": Rule("single", DEPLOY), "R0_strict": Rule("single", STRICT),
             "chosen": Rule(ch["kind"], ch["T"], k=ch["k"], m=ch["m"], T2=ch["T2"])}
    pick = lambda r: {k: r[k] for k in KEYS}  # noqa: E731
    rep = {"status": "alarm_rule_evaluation", "scores": args.scores, "rules": {k: r.name() for k, r in rules.items()},
           "results": {}, "ci95": {}, "paired_chosen_minus_deploy": {}}
    for name, rule in rules.items():
        rep["results"][name] = evaluate(rule, seqs)
        rep["ci95"][name] = _boot(lambda sub, rule=rule: pick(evaluate(rule, sub)), seqs, args.n_boot)

    def diff(sub):
        a, c = evaluate(rules["R0_deploy"], sub), evaluate(rules["chosen"], sub)
        return {k: (None if a[k] is None or c[k] is None else c[k] - a[k]) for k in KEYS}

    rep["paired_chosen_minus_deploy"] = {"value": diff(seqs), "ci95": _boot(diff, seqs, args.n_boot)}
    r0, c = rep["results"]["R0_deploy"], rep["results"]["chosen"]
    n_fires = max(c["fires"], 1)
    rep["targets"] = {
        "fire_pre_events": c["fire_pre_events_per_hour"] is not None and r0["fire_pre_events_per_hour"] is not None
        and c["fire_pre_events_per_hour"] <= 0.6 * r0["fire_pre_events_per_hour"],
        "ordinary_events": c["ordinary_events_per_hour"] is not None and r0["ordinary_events_per_hour"] is not None
        and c["ordinary_events_per_hour"] <= 0.6 * r0["ordinary_events_per_hour"],
        "fires_within_30": c["fires_within_30"] >= r0["fires_within_30"] - 1.0 / n_fires - 1e-9,
        "fires_within_15": c["fires_within_15"] >= r0["fires_within_15"] - 3.0 / n_fires - 1e-9}
    rep["targets"]["all_met"] = all(rep["targets"].values())
    write_json(Path(args.out), rep)
    Path(args.out).with_suffix(".md").write_text(render(rep), encoding="utf-8")
    print(json.dumps({"targets": rep["targets"], **{n: pick(r) for n, r in rep["results"].items()}},
                     default=float, indent=1))
    return 0


def render(rep) -> str:
    names = list(rep["results"])
    L = ["# Alarm rules over per-minute window scores", "",
         "Rules: " + "; ".join(f"{n}: {rep['rules'][n]}" for n in names) + ". Intervals resample camera mounts (95 %).",
         "", "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]

    def fmt(k, v):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return "n/a"
        if "share" in k or "fires_" in k:
            return f"{100 * v:.1f}%"
        return f"{v:.2f}"

    for k in KEYS:
        cells = []
        for n in names:
            v = rep["results"][n].get(k)
            ci = rep["ci95"][n].get(k)
            cells.append(fmt(k, v) + (f" [{fmt(k, ci[0])}, {fmt(k, ci[1])}]" if ci else ""))
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    r = rep["results"][names[0]]
    L += ["", f"{r['fires']} fire sequences; smoke-free footage: {r['fire_pre_hours']:.1f} camera-hours before the "
          f"first visible plume, {r['ordinary_hours']:.1f} camera-hours on ordinary days.", "",
          "Paired, chosen - R0_deploy (same resampled mounts):", "", "| | difference | 95 % |", "|---|---|---|"]
    pv, pc = rep["paired_chosen_minus_deploy"]["value"], rep["paired_chosen_minus_deploy"]["ci95"]
    for k in KEYS:
        v = pv.get(k)
        ci = pc.get(k)
        L.append(f"| {k} | {fmt(k, v) if v is not None else 'n/a'} | "
                 + (f"[{fmt(k, ci[0])}, {fmt(k, ci[1])}]" if ci else "") + " |")
    L += ["", "Targets: " + ", ".join(f"{k} {'met' if v else 'not met'}" for k, v in rep["targets"].items()) + "."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--scores", action="append", required=True)
    t.add_argument("--out", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--chosen", required=True)
    e.add_argument("--scores", action="append", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    return cmd_tune(args) if args.cmd == "tune" else cmd_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
