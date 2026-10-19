"""Train the physics GRU: Stage A (CFD pretraining) or Stage B (real-footage fine-tuning).

Stage A (synthetic):
    python scripts/train_physics.py --features data/synthetic/features.npz --out artifacts

    Fits the normaliser on the train split, trains one model per seed in
    configs/training.yaml (one CPU thread, deterministic), early-stops on
    val_select, picks the best seed on val_select, fits temperatures,
    thresholds and the abstain band on val_calib, and writes the artifacts
    (physics_encoder.pt, physics_classifier.pt, feature_normalizer.json,
    model_config.yaml) plus per-seed logs and training curves.

Stage B (real labelled footage, camera-disjoint splits):
    python scripts/train_physics.py --stage finetune --features data/real/features.npz \\
        --init artifacts --out runs/finetune_real            # experiment D (lr 1e-4)
    python scripts/train_physics.py --stage scratch --features data/real/features.npz \\
        --out runs/scratch_real                              # experiment B (same data, random init)

Targets may be NaN = unknown (e.g. real footage whose fog is not annotated):
they are left out of the loss, early stopping and every calibration fit; a
label with too few known val_calib targets keeps the pretrained calibration
(finetune) or the neutral defaults (scratch), recorded in model_config.yaml
under calibration.label_support.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.constants import LABEL_NAMES
from src.data.dataset import FeatureSet, assert_no_group_leakage
from src.training.train import run_finetune, run_pretraining
from src.utils.config import load_training_config
from src.utils.io import write_yaml


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", required=True, help="FeatureSet .npz")
    ap.add_argument("--out", required=True, help="output directory (artifacts)")
    ap.add_argument("--config", default=None, help="training YAML (default configs/training.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--stage", choices=["pretrain", "finetune", "scratch"], default="pretrain")
    ap.add_argument("--init", default=None, help="pretrained artifact directory (finetune)")
    args = ap.parse_args(argv)
    cfg = load_training_config(args.config, args.set)
    fs = FeatureSet.load(args.features)
    assert_no_group_leakage(fs)
    for j, name in enumerate(LABEL_NAMES):
        known = fs.known[:, j]
        if not known.all():
            per_split = ", ".join(f"{s} {int(known[fs.splits == s].sum())}/{int(np.sum(fs.splits == s))}"
                                  for s in sorted(set(fs.splits)))
            print(f"note: {name} targets known for {int(known.sum())} / {len(fs)} sequences ({per_split}); "
                  "unknown targets are left out of the loss and calibration", file=sys.stderr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_yaml(out / "training_config_used.yaml", cfg)
    if args.stage == "pretrain":
        summary = run_pretraining(fs, cfg, out)
    else:
        if fs.source != "real":
            print("warning: Stage B is meant for real footage; this FeatureSet is "
                  f"{fs.source!r} (any result is NOT real-world evidence)", file=sys.stderr)
        if args.stage == "finetune" and not args.init:
            ap.error("--stage finetune needs --init (pretrained artifacts)")
        summary = run_finetune(fs, cfg, args.init, out, from_scratch=(args.stage == "scratch"))
    print(json.dumps(summary, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
