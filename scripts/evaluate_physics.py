"""Evaluate the trained physics expert and write evaluation.json, figures and audits.

    python scripts/evaluate_physics.py --features data/synthetic/features.npz --artifacts artifacts \\
        --reports reports [--robustness data/synthetic/features_robustness.npz] \\
        [--summary data/synthetic/summary.json] [--loro] [--proxy data/synthetic_proxy/features.npz] \\
        [--real data/real/features.npz]

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
  splits); otherwise "real": {"status": "pending"} and B / D are null.

evaluation.json is written into the artifact directory and copied to --reports.
"""

import _bootstrap  # noqa: F401

import argparse
import shutil
import sys
from pathlib import Path

from src.data.dataset import FeatureSet, assert_no_group_leakage
from src.training.evaluate import evaluate_real, run_evaluation, run_synthetic_proxy
from src.training.train import run_loro
from src.utils.config import load_training_config
from src.utils.io import read_json


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", required=True)
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
    args = ap.parse_args(argv)

    cfg = load_training_config(args.config, args.set)
    fs = FeatureSet.load(args.features)
    assert_no_group_leakage(fs)
    robust = FeatureSet.load(args.robustness) if args.robustness else None
    summary = read_json(args.summary) if args.summary else None
    loro = run_loro(fs, cfg) if args.loro else None
    proxy = None
    if args.proxy:
        proxy = run_synthetic_proxy(
            FeatureSet.load(args.proxy), cfg, args.artifacts, args.proxy_out,
            description=("Separately generated synthetic simulations (different base seed) rendered with "
                         "detector / box masks and shifted rendering; re-split by simulation. Not real footage."))
    real = None
    if args.real:
        from src.training.train import run_finetune

        fs_real = FeatureSet.load(args.real)
        assert_no_group_leakage(fs_real)
        out = Path(args.real_out)
        # B: random init on the real train split (training.lr); D: fine-tune C (finetune.lr).
        # Same splits, epochs, patience, seeds and normaliser policy for both.
        run_finetune(fs_real, cfg, args.artifacts, out / "B", from_scratch=True)
        run_finetune(fs_real, cfg, args.artifacts, out / "D", from_scratch=False)
        real = evaluate_real(fs_real, cfg, {"B": out / "B", "C": args.artifacts, "D": out / "D"})
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    doc = run_evaluation(fs, cfg, args.artifacts, reports, robustness_fs=robust, loro=loro,
                         data_summary=summary, real=real, synthetic_proxy=proxy)
    src = Path(args.artifacts) / "evaluation.json"
    if src.exists() and src.resolve() != (reports / "evaluation.json").resolve():
        shutil.copyfile(src, reports / "evaluation.json")
    print(f"real-world status: {doc.get('real', {}).get('status')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
