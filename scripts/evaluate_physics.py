"""Evaluate the trained physics expert and write evaluation.json, figures and audits.

    python scripts/evaluate_physics.py --features data/synthetic/features.npz --artifacts artifacts \\
        --reports reports [--robustness data/synthetic/features_robustness.npz] \\
        [--summary data/synthetic/summary.json] [--loro] [--proxy data/synthetic_proxy/features.npz] \\
        [--real data/real/features.npz]

    # Stage B on its own (keeps the synthetic sections of an existing evaluation.json):
    python scripts/evaluate_physics.py --artifacts artifacts --reports reports --real data/real/features.npz

What it reports (all synthetic unless --real is given):
* C, the CFD-pretrained GRU: test_iid and test_ood (overall, per held-out
  recipe, category, regime, wind bin, mask source), both operating points,
  95 % grouped-bootstrap intervals, mean +- sd over the three seeds;
* A, the motion-feature heuristic baseline, plus a pooled logistic baseline;
* the shortcut audit (flags-only probe, univariate feature AUCs, context /
  render parameter AUCs on the accepted data);
* robustness suites (re-renders of test_iid beyond the training ranges);
* --loro: leave-one-regime-out generalisation (one model per regime);
* --proxy: a synthetic transfer proxy (pretrained vs scratch on a small,
  separately generated, shifted synthetic domain), stored with
  not_real_world_evidence = true;
* real-world results only with --real (labelled footage, camera-disjoint
  splits, groups = camera ids): B (scratch, training.lr) and D (fine-tuned
  C, finetune.lr) are trained on the real train split and B, C (zero-shot)
  and D evaluated on the real test split; otherwise "real": {"status":
  "pending"} and B / D are null. Targets may be NaN (unknown, e.g. smoke-only
  supervision with unannotated fog): that label is left out of training and
  calibration, its real metrics are null and its per-label status is
  "no_labels" (the real status is "partial", never "complete" for it).

evaluation.json is written into the artifact directory and copied to --reports.
"""

import _bootstrap  # noqa: F401

import argparse
import shutil
import sys
from pathlib import Path

from src.data.dataset import FeatureSet, assert_no_group_leakage
from src.training.evaluate import evaluate_real, run_evaluation, run_synthetic_proxy, update_real_section
from src.training.train import run_loro
from src.utils.config import load_training_config
from src.utils.io import read_json, write_json


def run_stage_b(real_path, cfg, artifacts, out_dir, n_boot=None):
    """Train B and D on the real FeatureSet's camera-disjoint splits and evaluate B, C, D on its test split."""
    from src.training.train import run_finetune

    fs_real = FeatureSet.load(real_path)
    if fs_real.source != "real":
        raise SystemExit(f"{real_path} is not a real-footage FeatureSet (source = {fs_real.source!r})")
    assert_no_group_leakage(fs_real)
    out = Path(out_dir)
    # B: random init on the real train split (training.lr); D: fine-tune C (finetune.lr).
    # Same splits, epochs, patience, seeds and normaliser policy for both.
    run_finetune(fs_real, cfg, artifacts, out / "B", from_scratch=True)
    run_finetune(fs_real, cfg, artifacts, out / "D", from_scratch=False)
    return evaluate_real(fs_real, cfg, {"B": out / "B", "C": artifacts, "D": out / "D"}, n_boot=n_boot)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", default=None,
                    help="synthetic FeatureSet (required unless only --real is evaluated)")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--robustness", default=None, help="FeatureSet of robustness re-renders")
    ap.add_argument("--summary", default=None, help="dataset summary.json (acceptance rates)")
    ap.add_argument("--loro", action="store_true", help="run leave-one-regime-out experiments")
    ap.add_argument("--proxy", default=None, help="separately generated shifted synthetic FeatureSet")
    ap.add_argument("--proxy-out", default="runs/synthetic_proxy")
    ap.add_argument("--real", default=None, help="real-footage FeatureSet (camera-disjoint splits)")
    ap.add_argument("--real-out", default="runs/real")
    ap.add_argument("--n-boot", type=int, default=None, help="bootstrap resamples (default: evaluation config)")
    args = ap.parse_args(argv)
    if not args.features and not args.real:
        ap.error("give --features (synthetic evaluation) and / or --real (Stage B)")

    cfg = load_training_config(args.config, args.set)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    real = run_stage_b(args.real, cfg, args.artifacts, args.real_out, args.n_boot) if args.real else None
    eval_path = Path(args.artifacts) / "evaluation.json"
    if args.features:
        fs = FeatureSet.load(args.features)
        assert_no_group_leakage(fs)
        robust = FeatureSet.load(args.robustness) if args.robustness else None
        summary = read_json(args.summary) if args.summary else None
        loro = run_loro(fs, cfg) if args.loro else None
        proxy = None
        if args.proxy:
            proxy = run_synthetic_proxy(
                FeatureSet.load(args.proxy), cfg, args.artifacts, args.proxy_out, n_boot=args.n_boot,
                description=("Separately generated synthetic simulations (different base seed) rendered with "
                             "detector / box masks and shifted rendering; re-split by simulation. Not real footage."))
        doc = run_evaluation(fs, cfg, args.artifacts, reports, robustness_fs=robust, loro=loro,
                             data_summary=summary, real=real, synthetic_proxy=proxy, n_boot=args.n_boot)
    else:
        # Stage B only: keep the synthetic sections of the existing evaluation.json.
        existing = read_json(eval_path) if eval_path.exists() else None
        if existing is None:
            print(f"warning: no {eval_path}; the synthetic sections are recorded as not run", file=sys.stderr)
        doc = update_real_section(existing, real)
        write_json(eval_path, doc)
        print(f"wrote {eval_path}")
    if eval_path.exists() and eval_path.resolve() != (reports / "evaluation.json").resolve():
        shutil.copyfile(eval_path, reports / "evaluation.json")
    r = doc.get("real", {})
    print(f"real-world status: {r.get('status')}")
    for name, st in (r.get("label_status") or {}).items():
        print(f"  {name}: {st.get('status')}" + (f" ({st['reason']})" if st.get("reason") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
