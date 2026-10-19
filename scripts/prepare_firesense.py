"""Download the FIRESENSE smoke videos and write a real-footage manifest for them.

    python scripts/prepare_firesense.py --out data/real/firesense
    python scripts/extract_features.py --real-manifest data/real/firesense/real.csv --artifacts artifacts \\
        --out data/real/firesense/features.npz
    python scripts/evaluate_real_zero_shot.py --features data/real/firesense/features.npz --artifacts artifacts \\
        --name firesense --out reports/real_zero_shot_firesense.json

FIRESENSE (CERTH / Bilkent University; Zenodo record 836749, CC BY 4.0,
https://zenodo.org/records/836749): 13 smoke ("pos") and 9 non-smoke ("neg")
clips. Labels: smoke from the folder; fog = 0 for every clip (none shows fog,
checked by eye). Every clip is its own camera except testpos06 / testpos07
(the same courtyard CCTV view). testpos03 is left out: OpenCV's decoder
crashes (segfault) on it, and at 21.5 s it is too short anyway. Clips
shorter than one 56 s window are listed as skipped by the extractor.
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/836749/files/smoke_videos.1407.zip?download=1"
MD5 = "5ee6bc867c4a9332b4e75f308c26fc23"
EXCLUDE = {"testpos03": "OpenCV decoder segfault; 21.5 s, too short for a window anyway"}
SAME_CAMERA = {"testpos06": "firesense_cctv_courtyard", "testpos07": "firesense_cctv_courtyard"}
# Second 56 s window at the end of clips long enough for one more (window starts 1 s + k * stride).
STRIDE_S = {"testneg04": 56.0, "testpos05": 53.0, "testpos04": 22.0, "testpos07": 14.0}


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/real/firesense")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = out.parent / f"{out.name}_raw" / "smoke_videos.1407.zip"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, archive)
    if md5sum(archive) != MD5:
        raise SystemExit(f"{archive}: MD5 mismatch (expected {MD5})")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        unsafe = [n for n in names if n.startswith("/") or ".." in Path(n).parts]
        if unsafe:
            raise SystemExit(f"refusing to extract unsafe paths: {unsafe}")
        z.extractall(out)
    rows = []
    for p in sorted(out.glob("*/*.avi")):
        vid = p.name.split(".")[0]
        if vid in EXCLUDE:
            print(f"excluded {vid}: {EXCLUDE[vid]}")
            continue
        rows.append({"video_path": str(p), "camera_id": SAME_CAMERA.get(vid, f"firesense_{vid}"),
                     "video_id": vid, "smoke": 1 if p.parent.name == "pos" else 0, "fog": 0,
                     "split": "test_iid", "stride_s": STRIDE_S.get(vid, ""),
                     "max_windows": 2 if vid in STRIDE_S else 1})
    with open(out / "real.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} videos -> {out / 'real.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
