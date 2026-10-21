"""Ordinary-day stretches of consecutive minutes around labelled smoke-free HPWREN windows (for alarm rates over time).

    python scripts/hpwren_stretches.py select --manifest data/real/hpwren_bg_test/real.csv --per-camera 3 \\
        --seed temporal-test --out data/real/stretches_test
    python scripts/hpwren_stretches.py download --out data/real/stretches_test --workers 4
    python scripts/embed_frames.py --out-dir data/real/dense/stretches_test --delete-frames \\
        data/real/stretches_test/<stretch> ...

select: from a manifest of labelled windows (scripts/prepare_hpwren_fog.py manifest), up to --per-camera
smoke-free windows per camera (seeded draw, one per camera-day; --max-cameras keeps a seeded draw of
cameras). Each becomes a stretch whose scored window
ends run from --before minutes before to --after minutes after the labelled window's last frame, so its frames
span that range plus the 25 minutes a window and its reference frame reach back. Written:
<out>/stretches.csv (stretch_id, camera, camera_mount, split, window_id, t_label_end, t_first_end, t_last_end).

download: for every stretch and every minute of its span, the archived frame nearest that minute (within
--max-offset-s), downscaled to --max-width px like the stored windows, saved as <out>/<stretch_id>/<unix>.jpg.
Existing files are kept. Imagery courtesy of HPWREN (https://www.hpwren.ucsd.edu/).
"""

import _bootstrap  # noqa: F401

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_hpwren_fog import IMAGE_URL, TZ, _downscale, _get, image_list  # noqa: E402

REACH_MIN = 25                      # a window ending at t uses frames from t - 11 and its reference at t - 25
FIELDS = ["stretch_id", "camera", "camera_mount", "split", "window_id", "t_label_end", "t_first_end", "t_last_end"]


def cmd_select(args) -> int:
    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("smoke") in ("0", "0.0")]
    by_cam = {}
    for r in sorted(rows, key=lambda r: r["window_id"]):
        by_cam.setdefault(r["camera_id"], []).append(r)
    rng = random.Random(args.seed)
    out_rows = []
    for cam in sorted(by_cam):
        days, picks = set(), []
        cands = by_cam[cam][:]
        rng.shuffle(cands)
        for r in cands:
            meta = json.loads((Path(r["video_path"]) / "window.json").read_text())
            if meta["day"] in days:
                continue
            days.add(meta["day"])
            picks.append((r, meta))
            if len(picks) >= args.per_camera:
                break
        for r, meta in picks:
            t_end = int(meta["unix_times"][-1])
            out_rows.append({"stretch_id": f"stretch-{r['window_id'].split('-', 1)[1]}", "camera": cam,
                             "camera_mount": r["camera_mount"], "split": r["split"], "window_id": r["window_id"],
                             "t_label_end": t_end, "t_first_end": t_end - 60 * args.before,
                             "t_last_end": t_end + 60 * args.after})
    if args.max_cameras:
        cams = sorted({r["camera"] for r in out_rows})
        keep = set(rng.sample(cams, min(args.max_cameras, len(cams))))
        out_rows = [r for r in out_rows if r["camera"] in keep]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "stretches.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    print(json.dumps({"stretches": len(out_rows), "cameras": len({r['camera'] for r in out_rows})}))
    return 0


def _nearest(names, target, max_offset):
    times = np.array([int(Path(n).stem) for n in names if Path(n).stem.isdigit()], np.int64)
    if not len(times):
        return None
    i = int(np.argmin(np.abs(times - target)))
    return int(times[i]) if abs(times[i] - target) <= max_offset else None


def fetch_stretch(row, out: Path, max_offset: float, max_width: int) -> dict:
    cam = row["camera"]
    folder = out / row["stretch_id"]
    folder.mkdir(parents=True, exist_ok=True)
    have = {int(p.stem) for p in folder.glob("*.jpg")}
    got = missing = 0
    t0 = int(row["t_first_end"]) - 60 * REACH_MIN
    for target in range(t0, int(row["t_last_end"]) + 1, 60):
        local = dt.datetime.fromtimestamp(target, TZ)
        day, q = f"{local:%Y%m%d}", 1 + local.hour // 3
        unix = _nearest(image_list(out, cam, day, q), target, max_offset)
        if unix is None:
            missing += 1
            continue
        if unix in have:
            continue
        img = cv2.imdecode(np.frombuffer(_get(IMAGE_URL.format(cam=cam, day=day, q=q, name=f"{unix}.jpg")), np.uint8),
                           cv2.IMREAD_COLOR)
        if img is None:
            missing += 1
            continue
        tmp = folder / f"{unix}.tmp.jpg"
        cv2.imwrite(str(tmp), _downscale(img, max_width), [cv2.IMWRITE_JPEG_QUALITY, 90])
        tmp.rename(folder / f"{unix}.jpg")
        have.add(unix)
        got += 1
    return {"stretch_id": row["stretch_id"], "fetched": got, "missing_minutes": missing}


def cmd_download(args) -> int:
    out = Path(args.out)
    with open(out / "stretches.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.only:
        keep = set(Path(args.only).read_text().split())
        rows = [r for r in rows if r["stretch_id"] in keep]
    results, failed = [], []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_stretch, r, out, args.max_offset_s, args.max_width): r for r in rows}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001 - listed, not fatal
                failed.append({"stretch_id": futs[fut]["stretch_id"], "reason": str(exc)})
            print(f"  {n}/{len(rows)} stretches", file=sys.stderr, flush=True)
    (out / "download.json").write_text(json.dumps({"results": results, "failed": failed}, indent=1))
    print(json.dumps({"stretches": len(results), "failed": len(failed),
                      "missing_minutes": sum(r["missing_minutes"] for r in results)}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--per-camera", type=int, default=3)
    s.add_argument("--seed", default="stretches")
    s.add_argument("--max-cameras", type=int, default=0, help="keep a seeded draw of this many cameras (0: all)")
    s.add_argument("--before", type=int, default=20, help="scored window ends start this many minutes earlier")
    s.add_argument("--after", type=int, default=19)
    d = sub.add_parser("download")
    d.add_argument("--out", required=True)
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--max-offset-s", type=float, default=30.0)
    d.add_argument("--max-width", type=int, default=1024)
    d.add_argument("--only", help="file with the stretch ids to download")
    args = ap.parse_args(argv)
    return cmd_select(args) if args.cmd == "select" else cmd_download(args)


if __name__ == "__main__":
    sys.exit(main())
