"""Render evaluation.json (and optional diagnostics) as a Markdown results report.

    python scripts/summarize_results.py --evaluation reports/evaluation.json \\
        [--oracle reports/oracle_diagnostic.json] [--benchmark reports/benchmark_cpu.json] \\
        [--qa reports/qa_train.json] --out reports/results.md

Every number is copied from the inputs; nothing is recomputed. The report
keeps the synthetic-only disclaimer and the real-world status at the top.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

LABELS = ("smoke", "fog")
SPLITS = ("test_iid", "test_ood")
OPS = ("balanced", "low_false_alarm")


def f3(v):
    return "n/a" if v is None else f"{v:.3f}"


def pct(v):
    return "n/a" if v is None else f"{100 * v:.1f} %"


def ci(doc, path):
    c = (doc or {}).get(path) or {}
    if c.get("lo") is None:
        return ""
    return f" [{c['lo']:.2f}, {c['hi']:.2f}]"


def ms(mean_sd, path, fmt=f3):
    r = (mean_sd or {}).get(path)
    if not r:
        return "n/a"
    return f"{fmt(r['mean'])} ± {fmt(r['sd'])}" if fmt is f3 else f"{fmt(r['mean'])} ± {100 * r['sd']:.1f}"


def headline_rows(name, test):
    rows = []
    for split in SPLITS:
        for op in OPS:
            r = (test or {}).get(split, {}).get(op)
            if not r:
                continue
            rows.append(f"| {name} | {split} | {op} | {f3(r['smoke']['auroc'])} | {f3(r['fog']['auroc'])} | "
                        f"{f3(r['smoke']['f1'])} | {f3(r['fog']['f1'])} | {pct(r['false_smoke_on_fog_only'])} | "
                        f"{f3(r['smoke']['ece'])} / {f3(r['fog']['ece'])} | {pct(r['abstain_rate_any'])} |")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--oracle", default=None)
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--qa", default=None, help="qa_dataset.py output (optional)")
    ap.add_argument("--out", default="reports/results.md")
    args = ap.parse_args(argv)
    ev = json.loads(Path(args.evaluation).read_text())
    syn, abl = ev["synthetic"], ev["ablations"]
    L = ["# Results (synthetic evaluation)", "", f"> {ev['disclaimer']}", "",
         f"**Real-world status: {ev['real'].get('status')}.** {ev['real'].get('reason', '')}", ""]

    counts = (ev.get("data") or {}).get("counts") or {}
    if counts:
        L += ["## Data", "", "| split | sequences | simulations | per category (sequences) |", "|---|---|---|---|"]
        for split, c in counts.items():
            if not isinstance(c, dict):
                continue
            cats = ", ".join(f"{k} {v.get('sequences')}" for k, v in (c.get("per_category") or {}).items())
            L.append(f"| {split} | {c.get('sequences', 'n/a')} | {c.get('simulations', c.get('groups', 'n/a'))} | {cats} |")
        gen = (ev.get("data") or {}).get("generation") or {}
        if gen:
            keep = {k: gen[k] for k in ("simulations_accepted", "simulation_attempts_rejected", "render_retries",
                                        "sequences", "slots_without_valid_simulation") if k in gen}
            if keep:
                L += ["", "Generation: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in keep.items()) + "."]
        L.append("")

    if args.qa and Path(args.qa).exists():
        qa = json.loads(Path(args.qa).read_text())
        L += [f"## Data QA (qa_dataset.py, {qa.get('split')} split, {qa.get('n')} sequences)", "",
              "| category | n | mask valid steps | flow valid steps |", "|---|---|---|---|"]
        for c, r in (qa.get("per_category") or {}).items():
            L.append(f"| {c} | {r['n']} | {pct(r['mask_valid_rate'])} | {pct(r['flow_valid_rate'])} |")
        fc = qa.get("flow_consistency") or {}
        if fc:
            L += ["", f"Optical flow vs CFD velocity ({fc.get('n')} sequences): " + "; ".join(
                f"{d} correlation {f3((fc.get(d) or {}).get('correlation'))}, sign agreement where the CFD moves "
                f"{pct((fc.get(d) or {}).get('sign_agreement_where_cfd_moves'))}" for d in ("horizontal", "vertical")) + "."]
        flags = qa.get("flags") or []
        L += ["", f"Validity flags / parameters with folded univariate AUROC above "
                  f"{(qa.get('parameter_audit') or {}).get('threshold')}:" + (" none." if not flags else "")]
        L += [f"* {f}" for f in flags]
        L.append("")

    L += ["## Headline: experiment C (CFD-pretrained GRU) against baselines", "",
          "AUROC is threshold-free; F1, false smoke on fog_only and abstention use the operating point "
          "(thresholds fitted on val_calib only). Temperature scaling and abstention apply to C only (its "
          "temperatures and abstention margins are also fitted on val_calib), so only C's ECE is after "
          "temperature scaling; A and the pooled logistic baseline report raw sigmoid probabilities and never "
          "abstain. Exported model = the seed with the best val_select loss.", "",
          "| model | split | operating point | smoke AUROC | fog AUROC | smoke F1 | fog F1 | "
          "false smoke on fog_only | ECE smoke / fog | abstain (any) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    L += headline_rows("C: GRU (exported seed)", (abl.get("C") or {}).get("test"))
    L += headline_rows("A: motion-cue heuristic", (abl.get("A") or {}).get("test"))
    L += headline_rows("pooled logistic (14 x mean/std)", ((abl.get("extra_baselines") or {})
                                                            .get("pooled_logistic") or {}).get("test"))
    L.append("")
    temps, cal = syn.get("temperatures") or {}, syn.get("calibration") or {}
    notes = ["C temperatures: " + ", ".join(f"{k} T = {f3(v)}" for k, v in temps.items()) + "."] if temps else []
    for key, what in (("temperature_fallback", "fell back to T = 1"), ("temperature_clipped", "clipped to a bound")):
        hit = {k: v for k, v in (cal.get(key) or {}).items() if v is not None}  # key absent in older runs
        if hit:
            notes.append(f"Temperature {what}: " + ", ".join(f"{k} ({v})" for k, v in hit.items()) + ".")
    if notes:
        L += [" ".join(notes), ""]

    msd = (abl.get("C") or {}).get("mean_sd_over_seeds") or {}
    if msd:
        L += [f"Mean ± sd over the {len(syn.get('seeds', {}).get('seeds', []))} training seeds (balanced point):", "",
              "| split | smoke AUROC | fog AUROC | false smoke on fog_only | smoke ECE | fog ECE |",
              "|---|---|---|---|---|---|"]
        for split in SPLITS:
            p = f"{split}.balanced"
            L.append(f"| {split} | {ms(msd, p + '.smoke.auroc')} | {ms(msd, p + '.fog.auroc')} | "
                     f"{ms(msd, p + '.false_smoke_on_fog_only')} | {ms(msd, p + '.smoke.ece')} | "
                     f"{ms(msd, p + '.fog.ece')} |")
        L.append("")

    L += ["95 % grouped-bootstrap intervals (resampling simulations), exported model:", "",
          "| split | operating point | smoke AUROC | fog AUROC | false smoke on fog_only |", "|---|---|---|---|---|"]
    for split in SPLITS:
        for op in OPS:
            o = (syn.get("test", {}).get(split, {}).get(op) or {}).get("overall")
            if not o:
                continue
            c = o.get("ci95") or {}
            L.append(f"| {split} | {op} | {f3(o['per_label']['smoke']['auroc'])}{ci(c, 'smoke.auroc')} | "
                     f"{f3(o['per_label']['fog']['auroc'])}{ci(c, 'fog.auroc')} | "
                     f"{pct(o['false_smoke_on_fog_only']['rate'])}{ci(c, 'false_smoke_on_fog_only.rate')} |")
    L.append("")

    for split in SPLITS:
        sl = (syn.get("test", {}).get(split, {}).get("balanced") or {}).get("slices") or {}
        for name in ("category", "holdout_tag", "regime", "mask_source", "wind_bin"):
            part = sl.get(name)
            if not part or (name == "holdout_tag" and split == "test_iid"):
                continue
            L += [f"### C on {split} by {name} (balanced point)", "",
                  f"| {name} | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |",
                  "|---|---|---|---|---|---|---|---|"]
            for k, r in part.items():
                L.append(f"| {k} | {r['n']} | {f3(r['smoke']['recall'])} | {f3(r['fog']['recall'])} | "
                         f"{f3(r['smoke']['auroc'])} | {f3(r['fog']['auroc'])} | "
                         f"{pct(r['false_smoke_on_fog_only'])} | {pct(r['abstain_rate_any'])} |")
            L.append("")

    rob = ev.get("robustness") or {}
    if rob.get("suites"):
        L += ["## Robustness (re-renders of the test_iid simulations, balanced point)", "",
              "| suite | n | smoke AUROC | fog AUROC | false smoke on fog_only | abstain | Δ smoke AUROC | Δ fog AUROC |",
              "|---|---|---|---|---|---|---|---|"]
        for suite, r in rob["suites"].items():
            d = r.get("delta_vs_nominal") or {}
            L.append(f"| {suite} | {r['n']} | {f3(r['smoke']['auroc'])} | {f3(r['fog']['auroc'])} | "
                     f"{pct(r['false_smoke_on_fog_only'])} | {pct(r['abstain_rate_any'])} | "
                     f"{f3(d.get('smoke.auroc'))} | {f3(d.get('fog.auroc'))} |")
        L.append("")

    loro = ev.get("loro") or {}
    if loro.get("regimes"):
        L += [f"## Leave-one-regime-out (seed {loro.get('seed')}, evaluated on {loro.get('eval_split')})", "",
              "For a held-out regime positive for a label: recall and AUROC of its sequences against the split's "
              "negatives. Negative for a label: false-positive rate and AUROC of the split's positives against it.", "",
              "| held-out regime | n | smoke | fog |", "|---|---|---|---|"]

        def lab(r):
            if not r:
                return "n/a"
            if r["role"] == "positive":
                return f"recall {f3(r['recall'])}, AUROC {f3(r['auroc_vs_split_negatives'])}"
            if r["role"] == "negative":
                return f"FPR {f3(r['false_positive_rate'])}, AUROC {f3(r['auroc_split_positives_vs_held_out'])}"
            return f"accuracy {f3(r.get('accuracy'))}"

        for regime, r in loro["regimes"].items():
            if r.get("status") == "skipped":
                L.append(f"| {regime} | {r.get('n_eval_held_out', 0)} | skipped | skipped |")
            else:
                L.append(f"| {regime} | {r['n_eval_held_out']} | {lab(r['labels'].get('smoke'))} | "
                         f"{lab(r['labels'].get('fog'))} |")
        L.append("")

    audit = ev.get("shortcut_audit") or {}
    if audit:
        L += ["## Shortcut audit", "", f"Overall flag: **{audit.get('flag')}**."]
        for r in audit.get("flag_reasons") or []:
            L.append(f"* {r}")
        fo = audit.get("flags_only") or {}
        if fo.get("pooled"):
            L.append(f"* Flags-only probe (validity flags alone), pooled held-out AUROC: "
                     + ", ".join(f"{k} {f3(v)}" for k, v in (fo["pooled"].get("auroc") or {}).items()))
        params = audit.get("parameters") or {}
        for lab_, res in (params.get("per_label") or {}).items():
            top = sorted(res.items(), key=lambda kv: -(kv[1] or 0.0))[:4]
            if top:
                L.append(f"* Highest context / render parameter AUROCs for {lab_} (folded, threshold "
                         f"{params.get('threshold')}): " + ", ".join(f"{k} {f3(v)}" for k, v in top))
        L.append("")

    proxy = ev.get("synthetic_proxy")
    if proxy:
        L += ["## Synthetic transfer proxy (NOT real-world evidence)", "", proxy.get("description", ""), "",
              "| model | smoke AUROC | fog AUROC | false smoke on fog_only (balanced) |", "|---|---|---|---|"]
        for name in ("C_zero_shot", "B_scratch", "D_finetuned"):
            r = ((proxy.get(name) or {}).get("test") or {}).get("test_iid", {}).get("balanced")
            if r:
                L.append(f"| {name} | {f3(r['smoke']['auroc'])}{ci(r.get('ci95'), 'smoke.auroc')} | "
                         f"{f3(r['fog']['auroc'])}{ci(r.get('ci95'), 'fog.auroc')} | "
                         f"{pct(r['false_smoke_on_fog_only'])} |")
        L.append("")

    if args.oracle and Path(args.oracle).exists():
        od = json.loads(Path(args.oracle).read_text())
        L += ["## CFD-oracle diagnostic (oracle features are never model inputs)", "", od["description"], "",
              "Pooled logistic regression on per-feature time mean and std, test AUROC [95 % CI]:", "",
              "| features | label | oracle test_iid | camera test_iid | oracle test_ood | camera test_ood |",
              "|---|---|---|---|---|---|"]

        def cell(r):
            if not r or r.get("auroc") is None:
                return "n/a"
            lo, hi = r["ci95"]
            return f"{r['auroc']:.3f}" + ("" if lo is None else f" [{lo:.2f}, {hi:.2f}]")

        for g, res in od["pooled_logistic"].items():
            for lab_ in LABELS:
                o, c = res["oracle"][lab_], res["camera"][lab_]
                L.append(f"| {g} | {lab_} | {cell(o.get('test_iid'))} | {cell(c.get('test_iid'))} | "
                         f"{cell(o.get('test_ood'))} | {cell(c.get('test_ood'))} |")
        L.append("")

    if args.benchmark and Path(args.benchmark).exists():
        table = Path(args.benchmark).with_suffix(".md")   # benchmark_cpu.py writes both
        if table.exists():
            L += ["## CPU benchmark (one core)", "", table.read_text().strip(), ""]
        else:
            bm = json.loads(Path(args.benchmark).read_text())
            L += ["## CPU benchmark (one core)", "", "```", json.dumps(bm, indent=2)[:3000], "```", ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
