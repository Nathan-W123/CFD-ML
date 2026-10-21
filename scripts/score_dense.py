"""Score a window ending at every frame of embedded sequences (scripts/embed_frames.py) with a frozen detector.

    # fresh test fires and ordinary-day stretches, final model
    python scripts/score_dense.py --dense-dir data/real/dense/figlib_test2 --kind fire \\
        --manifest data/real/figlib_test2/real.csv --detector artifacts_detector_v3 --out reports/far/dense/fires_test.npz
    python scripts/score_dense.py --dense-dir data/real/dense/stretches_test --kind ordinary \\
        --stretches data/real/stretches_test/stretches.csv --detector artifacts_detector_v3 --out .../ordinary_test.npz
    # dev sequences, each scored by the fold model that did not see its camera mount
    python scripts/score_dense.py --dense-dir data/real/dense/figlib_dev --kind fire --manifest data/real/figlib/real.csv \\
        --fold-models runs/folds --folds configs/real/temporal_dev_folds.json --out .../fires_dev.npz

Windows are assembled by src/detector/dense.py (frames 0, 6, 11 and the reference 14 min before frame 0, each
within 30 s of its nominal time); windows missing any of them are not scored. No zoom crops are used (the
detector's fallback stack without zoom). Fire sequences (FIgLib) keep each window's end time relative to the
first visible plume (from the frame names); ordinary stretches keep only window ends inside the stretch's
scored range. Written: an .npz of one row per scored window: seq, camera, mount, kind, t_end (unix),
off_end_s (fire sequences; NaN otherwise), p_smoke, p_fog, and the fog veto of the model that scored it.
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from src.detector.dense import assemble, window_index
from src.detector.model import Detector
from src.utils.io import write_json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dense-dir", required=True)
    ap.add_argument("--kind", choices=("fire", "ordinary"), required=True)
    ap.add_argument("--manifest", help="FIgLib manifest: sequence folder -> camera, mount")
    ap.add_argument("--stretches", help="stretches.csv of scripts/hpwren_stretches.py")
    ap.add_argument("--detector", help="one detector for every sequence")
    ap.add_argument("--fold-models", help="folder with fold<k>/ detectors")
    ap.add_argument("--folds", help="JSON {camera mount: fold k}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    meta = {}
    if args.manifest:
        with open(args.manifest, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                meta[Path(r["video_path"]).name] = {"camera": r["camera_id"], "mount": r["camera_mount"]}
    if args.stretches:
        with open(args.stretches, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                meta[r["stretch_id"]] = {"camera": r["camera"], "mount": r["camera_mount"],
                                         "lo": int(r["t_first_end"]) - 30, "hi": int(r["t_last_end"]) + 30}
    if args.detector:
        det = Detector.load(args.detector)
        pick = lambda mount: det  # noqa: E731
    else:
        folds = json.loads(Path(args.folds).read_text())
        models = {}

        def pick(mount):
            k = folds[mount]
            if k not in models:
                models[k] = Detector.load(Path(args.fold_models) / f"fold{k}")
            return models[k]
    cols = {k: [] for k in ("seq", "camera", "mount", "t_end", "off_end_s", "p_smoke", "p_fog", "fog_veto")}
    skipped = {}
    files = sorted(Path(args.dense_dir).glob("*.npz"))
    for n, f in enumerate(files, 1):
        name = f.stem
        if name not in meta:
            skipped[name] = "no metadata"
            continue
        m = meta[name]
        z = np.load(f)
        unix = z["unix"]
        idx = window_index(unix)
        ok = (idx[:, 0] >= 0) & (idx[:, 1] >= 0) & (idx[:, 3] >= 0)
        if args.kind == "ordinary":
            ok &= (unix >= m["lo"]) & (unix <= m["hi"])
        if not ok.any():
            skipped[name] = "no complete window"
            continue
        det = pick(m["mount"])
        x = assemble(z["coarse"], z["fine"], idx[ok])
        p = det.proba(x["E"], None, x["fine"], None, x["ref"])
        k = int(ok.sum())
        cols["seq"] += [name] * k
        cols["camera"] += [m["camera"]] * k
        cols["mount"] += [m["mount"]] * k
        cols["t_end"] += unix[ok].tolist()
        cols["off_end_s"] += (z["offset_s"][ok].astype(float).tolist() if args.kind == "fire" else [np.nan] * k)
        cols["p_smoke"] += p["smoke"].tolist()
        cols["p_fog"] += p["fog"].tolist()
        veto = det.thresholds["smoke"].get("fog_veto")
        cols["fog_veto"] += [np.nan if veto is None else float(veto)] * k
        if n % 20 == 0:
            print(f"  {n}/{len(files)} sequences", file=sys.stderr, flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, kind=np.array(args.kind), **{k: np.array(v) for k, v in cols.items()})
    write_json(out.with_suffix(".json"), {"sequences": len(files) - len(skipped), "windows": len(cols["seq"]),
                                          "skipped": skipped, "kind": args.kind,
                                          "detector": args.detector or args.fold_models})
    print(json.dumps({"sequences": len(files) - len(skipped), "windows": len(cols["seq"]), "skipped": len(skipped)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
