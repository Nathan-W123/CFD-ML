"""Concatenate FeatureSets (e.g. real footage from two sources) into one, checking for split leakage.

    python scripts/merge_features.py --out data/real/far_combined/features.npz \\
        data/real/figlib/features.npz data/real/hpwren_fog/features.npz

All inputs must have the same source (all real or all synthetic) and distinct
sequence ids. The result must pass assert_no_group_leakage: a group (for real
footage, a camera) may not appear in two splits, so the inputs' splits must be
assigned consistently (prepare_figlib.py and prepare_hpwren_fog.py both split
by camera mount with the same fractions).
"""

import _bootstrap  # noqa: F401

import argparse
import json
import sys
from collections import Counter

from src.data.dataset import FeatureSet, assert_no_group_leakage


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args(argv)
    sets = [FeatureSet.load(p) for p in args.inputs]
    fs = FeatureSet.concat(sets)
    assert_no_group_leakage(fs)
    fs.save(args.out)
    print(json.dumps({"rows": len(fs), "inputs": {p: len(s) for p, s in zip(args.inputs, sets)},
                      "splits": dict(Counter(fs.splits.tolist())),
                      "known_labels": {"smoke": int(fs.known[:, 0].sum()), "fog": int(fs.known[:, 1].sum())}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
