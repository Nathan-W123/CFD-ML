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
Existing files are kept.

audit-sheets: for windows listed in a CSV (stretch_id, camera, t_end: the window's last frame), the window's 12
frames again (the stretch frames are deleted once embedded) and a blind contact sheet of minutes 0, 4, 8 and 11
(prepare_hpwren_fog.write_sheet) under an opaque id: <out>/_audit_sheets/<sheet_id>.jpg and
<out>/audit_sheets.csv (sheet_id, stretch_id, camera, t_end).
Imagery courtesy of HPWREN (https://www.hpwren.ucsd.edu/).
"""

import _bootstrap  # noqa: F401

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import hashlib
import json
import shutil
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_hpwren_fog import IMAGE_URL, TZ, _downscale, _get, image_list, write_sheet  # noqa: E402

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


def _fetch_frame(out: Path, cam: str, target: int, max_offset: float, max_width: int, folder: Path):
    local = dt.datetime.fromtimestamp(target, TZ)
    day, q = f"{local:%Y%m%d}", 1 + local.hour // 3
    unix = _nearest(image_list(out, cam, day, q), target, max_offset)
    if unix is None:
        raise ValueError(f"no archived frame near {local:%Y-%m-%d %H:%M}")
    path = folder / f"{unix}.jpg"
    if not path.is_file():
        img = cv2.imdecode(np.frombuffer(_get(IMAGE_URL.format(cam=cam, day=day, q=q, name=f"{unix}.jpg")), np.uint8),
                           cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cannot decode a frame")
        cv2.imwrite(str(path), _downscale(img, max_width), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path.name


def cmd_audit_sheets(args) -> int:
    out = Path(args.out)
    with open(args.windows, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    index, failed = [], []
    for r in rows:
        t_end = int(float(r["t_end"]))
        sid = hashlib.sha1(f"stretch-audit:{r['stretch_id']}:{t_end}".encode()).hexdigest()[:12]
        tmp = out / "_audit_frames" / sid
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            files = [_fetch_frame(out, r["camera"], t_end - 60 * (11 - k), args.max_offset_s, args.max_width, tmp)
                     for k in range(12)]
            write_sheet(tmp, files, out / "_audit_sheets" / f"{sid}.jpg")
            index.append({"sheet_id": sid, "stretch_id": r["stretch_id"], "camera": r["camera"], "t_end": t_end})
        except Exception as exc:  # noqa: BLE001 - listed, not fatal
            failed.append({"stretch_id": r["stretch_id"], "t_end": t_end, "reason": str(exc)})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    with open(out / "audit_sheets.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sheet_id", "stretch_id", "camera", "t_end"])
        w.writeheader()
        w.writerows(index)
    print(json.dumps({"sheets": len(index), "failed": failed}))
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
    a = sub.add_parser("audit-sheets")
    a.add_argument("--out", required=True)
    a.add_argument("--windows", required=True, help="CSV with stretch_id, camera, t_end")
    a.add_argument("--max-offset-s", type=float, default=30.0)
    a.add_argument("--max-width", type=int, default=1024)
    args = ap.parse_args(argv)
    return {"select": cmd_select, "download": cmd_download, "audit-sheets": cmd_audit_sheets}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
