"""Write a real-footage manifest for downloaded MIVIA Smoke Detection Dataset videos.

The dataset (149 videos of about 15 min, one fixed camera, July 2012,
https://mivia.unisa.it/datasets/video-analysis-datasets/smoke-detection-dataset/)
is downloadable only by registered users of the MIVIA site, so it is not
fetched here. After downloading (all or part of it) into a folder:

    python scripts/prepare_mivia.py --videos /path/to/mivia_mp4s --out data/real/mivia
    python scripts/extract_features.py --real-manifest data/real/mivia/real.csv --artifacts artifacts \\
        --out data/real/mivia/features.npz
    python scripts/evaluate_real_zero_shot.py --features data/real/mivia/features.npz --artifacts artifacts \\
        --name mivia --out reports/real_zero_shot_mivia.json

Labels come from configs/real/mivia_smoke_labels.csv, transcribed from the
dataset's own per-video list (MiviaSmokeDataset.pdf on the dataset page):
"Smoke" and "Smoke with red sun reflections" -> smoke = 1; "Mountains",
"Clouds", "Sun", "Red reflections" -> smoke = 0; fog = 0 for every video (the
dataset has no fog class; clouds are a look-alike, not fog). The labels are
per video: a window of a smoke video taken before the smoke appears is still
labelled smoke, so the per-window smoke recall is a lower bound. All videos
come from the same camera ("3_" prefix), so there is one camera group.
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--videos", required=True, help="folder containing the downloaded MIVIA .mp4 files (searched recursively)")
    ap.add_argument("--labels", default="configs/real/mivia_smoke_labels.csv")
    ap.add_argument("--out", default="data/real/mivia")
    ap.add_argument("--windows-per-video", type=int, default=4, help="56 s windows taken from each video")
    ap.add_argument("--stride-s", type=float, default=200.0, help="spacing of those windows (s)")
    args = ap.parse_args(argv)

    labels = {r["video"]: r for r in csv.DictReader(open(args.labels, newline="", encoding="utf-8"))}
    found = {p.name: p for p in Path(args.videos).rglob("*.mp4")}
    rows, unknown = [], sorted(set(found) - set(labels))
    for name, lab in labels.items():
        if name not in found:
            continue
        rows.append({"video_path": str(found[name]), "camera_id": "mivia_cam3", "video_id": name[:-4],
                     "smoke": lab["smoke"], "fog": lab["fog"], "split": "test_iid",
                     "stride_s": args.stride_s, "max_windows": args.windows_per_video, "event": lab["event"]})
    if not rows:
        raise SystemExit(f"no labelled MIVIA videos found under {args.videos}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "real.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    events = {}
    for r in rows:
        events[r["event"]] = events.get(r["event"], 0) + 1
    print(f"{len(rows)} of {len(labels)} labelled videos found -> {out / 'real.csv'}; by event: {events}")
    if unknown:
        print(f"{len(unknown)} .mp4 files without a label were ignored, e.g. {unknown[:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
