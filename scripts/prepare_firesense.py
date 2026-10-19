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
(the same courtyard CCTV view). Left out, with the reason printed (EXCLUDE):
testpos03 (OpenCV's decoder crashes on it; too short anyway), testpos04
(edited footage: a hard cut to a different shot inside both windows) and
testpos05 (its container clock runs ~27 % slow against the burned-in DVR
clock, so the 5 s / 1 s window timing would be wrong). Clips shorter than
one 56 s window are listed as skipped by the extractor.

The archive is downloaded to a temporary file and renamed only when
complete; an archive whose MD5 does not match (partial or corrupt download)
is deleted with a message, so the next run downloads it again.
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/836749/files/smoke_videos.1407.zip?download=1"
MD5 = "5ee6bc867c4a9332b4e75f308c26fc23"
EXCLUDE = {
    "testpos03": "OpenCV decoder segfault; 21.5 s, too short for a window anyway",
    "testpos04": ("edited footage: a hard cut to a different shot at ~41.5 s falls inside both windows, "
                  "and the end is a burnt patch without smoke"),
    "testpos05": ("container clock ~27 % slow against the burned-in DVR clock (149 s of DVR time over "
                  "109.37 s of container time), so the 5 s / 1 s window timing would be wrong"),
}
SAME_CAMERA = {"testpos06": "firesense_cctv_courtyard", "testpos07": "firesense_cctv_courtyard"}
# Second 56 s window at the end of clips long enough for one more (window starts 1 s + k * stride).
STRIDE_S = {"testneg04": 56.0, "testpos07": 14.0}


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Download to <dest>.part, then rename: an interrupted download never looks complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(part, "wb") as fh:
            shutil.copyfileobj(resp, fh, length=1 << 20)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, dest)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/real/firesense")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = out.parent / f"{out.name}_raw" / "smoke_videos.1407.zip"
    if not archive.exists():
        download(URL, archive)
    digest = md5sum(archive)
    if digest != MD5:
        archive.unlink()
        raise SystemExit(f"{archive}: MD5 {digest} does not match the published {MD5} (partial or corrupt "
                         "download); the archive was deleted, run the script again to re-download it")
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
