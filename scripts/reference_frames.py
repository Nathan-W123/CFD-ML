"""Same-camera reference frames a fixed time before each window, embedded with CLIP like the window itself.

    # HPWREN: fetch first (the archive keeps about 96 days), oldest windows first, then embed
    python scripts/reference_frames.py fetch --manifest data/real/hpwren_smoke/real.csv \\
        --manifest data/real/hpwren_fog/real.csv --artifacts artifacts_far
    python scripts/reference_frames.py embed --manifest data/real/hpwren_smoke/real.csv --source hpwren \\
        --artifacts artifacts_far --out data/real/hpwren_smoke/clip_ref.npz
    python scripts/reference_frames.py embed --manifest data/real/figlib/real.csv --source figlib \\
        --artifacts artifacts_far --out data/real/figlib/clip_ref.npz

Smoke is new; a scene that merely looks smoky (haze, a bright slope, a cloud
line on a ridge) looked the same a little earlier. The detector compares the
window's last frame with one frame of the same camera --lag-min (14) minutes
before the window's frame 0 (frames selected as scripts/embed_windows.py
selects them). The lag is the same for every window and every label, so it
cannot encode the label; a deployed camera has that frame in its own recent
history. With FIgLib's windows (first frames at about -24, -12, +1 and +13 min
from the first visible plume) a 14-min lag puts every reference before the
first visible plume (-38, -26, -13 and -1 min). Later fire-day windows
(+20 to +45 min after the reported start) get references that may already
show smoke, as a deployed camera's would.

* fetch (HPWREN 2026 windows): the archived original nearest the reference
  time (within --max-offset-s), downscaled to 1024 px like the stored frames,
  saved as <set>/_ref/<window_id>.jpg; oldest windows first.
* embed: coarse crops (whole frame + 2 x 3 tiles) and fine crops (4 x 6 tiles)
  of the reference frame (FIgLib: the sequence's own frame nearest the
  reference time; HPWREN: the fetched file). Written: an .npz with window_id,
  emb (N, 7, 512), fine (N, 24, 512) float16 and lag_s (frame 0 minus the
  reference frame). Rows without a reference go to <out>.skipped.json.

Imagery courtesy of HPWREN (https://www.hpwren.ucsd.edu/).
"""

import _bootstrap  # noqa: F401

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_windows import ClipEmbedder, crops  # noqa: E402
from prepare_hpwren_fog import IMAGE_URL, TZ, _downscale, _get, image_list  # noqa: E402
from region_boxes import model_timing, window_frames  # noqa: E402

LAG_MIN = 14.0


def _unix(path) -> int:
    return int(Path(path).stem.split("_")[0])


def _nearest(times, target, max_offset):
    times = np.asarray(list(times), dtype=np.int64)
    if not len(times):
        return None
    i = int(np.argmin(np.abs(times - target)))
    return int(times[i]) if abs(times[i] - target) <= max_offset else None


def reference_time(row, timing, lag_s) -> int:
    files, idx = window_frames(row, timing)
    return _unix(files[idx[0]]) - lag_s


def ref_path(row) -> Path:
    return Path(row["video_path"]).parent / "_ref" / f"{row.get('window_id') or row['video_id']}.jpg"


def fetch_one(row, timing, lag_s, max_offset, max_width) -> str:
    out = ref_path(row)
    if out.is_file():
        return "kept"
    target = reference_time(row, timing, lag_s)
    meta = json.loads((Path(row["video_path"]) / "window.json").read_text())
    cam, cache_root = meta["camera"], Path(row["video_path"]).parent
    local = dt.datetime.fromtimestamp(target, TZ)
    day, q = f"{local:%Y%m%d}", 1 + local.hour // 3
    names = image_list(cache_root, cam, day, q)
    unix = _nearest([int(Path(n).stem) for n in names if Path(n).stem.isdigit()], target, max_offset)
    if unix is None:
        raise ValueError(f"no archived image within {max_offset:.0f} s of {local:%Y-%m-%d %H:%M}")
    img = cv2.imdecode(np.frombuffer(_get(IMAGE_URL.format(cam=cam, day=day, q=q, name=f"{unix}.jpg")), np.uint8),
                       cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode the reference frame")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{unix}.tmp.jpg")
    cv2.imwrite(str(tmp), _downscale(img, max_width), [cv2.IMWRITE_JPEG_QUALITY, 90])
    tmp.rename(out)
    return "fetched"


def cmd_fetch(args, timing) -> int:
    rows = []
    for m in args.manifest:
        with open(m, newline="", encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    lag_s = int(round(args.lag_min * 60))
    rows.sort(key=lambda r: reference_time(r, timing, lag_s))          # oldest first: the archive is rolling
    counts, failed = {"kept": 0, "fetched": 0}, []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_one, r, timing, lag_s, args.max_offset_s, args.max_width): r for r in rows}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            r = futs[fut]
            try:
                counts[fut.result()] += 1
            except Exception as exc:  # noqa: BLE001 - listed, not fatal
                failed.append({"window_id": r.get("window_id") or r["video_id"], "reason": str(exc)})
            if n % 200 == 0:
                print(f"  {n}/{len(rows)} references", file=sys.stderr, flush=True)
    print(json.dumps({**counts, "failed": len(failed), "failures": failed[:20]}))
    return 0


def cmd_embed(args, timing) -> int:
    with open(args.manifest[0], newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    lag_s = int(round(args.lag_min * 60))
    out = Path(args.out)
    done = {}
    if out.is_file() and not args.force:
        z = np.load(out)
        done = {str(w): (e, f, l) for w, e, f, l in zip(z["window_id"], z["emb"], z["fine"], z["lag_s"])}
    emb = ClipEmbedder(threads=args.threads)
    ids, embs, fines, lags, skipped = [], [], [], [], []
    for n, r in enumerate(rows, 1):
        wid = r.get("window_id") or r["video_id"]
        if wid in done:
            e, f, l = done[wid]
            ids.append(wid), embs.append(e), fines.append(f), lags.append(int(l))
            continue
        try:
            files, idx = window_frames(r, timing)
            t0 = _unix(files[idx[0]])
            if args.source == "figlib":
                seq = sorted(p for p in Path(r["video_path"]).iterdir() if p.suffix.lower() == ".jpg")
                by_time = {_unix(p): p for p in seq}
                t = _nearest(by_time, t0 - lag_s, args.max_offset_s)
                if t is None:
                    raise ValueError("no sequence frame near the reference time")
                path, t_ref = by_time[t], t
            else:
                path = ref_path(r)
                if not path.is_file():
                    raise ValueError("reference frame not fetched")
                t_ref = t0 - lag_s
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"cannot read {path}")
            z = emb(np.stack(crops(img) + crops(img, 4, 6, False)))
        except Exception as exc:  # noqa: BLE001 - listed, not fatal
            skipped.append({"window_id": wid, "reason": str(exc)})
            continue
        ids.append(wid), embs.append(z[:7].astype(np.float16)), fines.append(z[7:].astype(np.float16))
        lags.append(int(t0 - t_ref))
        if n % 100 == 0:
            print(f"  {n}/{len(rows)} windows", file=sys.stderr, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, window_id=np.array(ids), emb=np.stack(embs), fine=np.stack(fines),
                        lag_s=np.array(lags, np.int64))
    out.with_suffix(".skipped.json").write_text(json.dumps(skipped, indent=1))
    print(json.dumps({"windows": len(ids), "skipped": len(skipped), "out": str(out)}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("fetch", "embed"))
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--source", choices=("figlib", "hpwren"), default="hpwren")
    ap.add_argument("--artifacts", required=True, help="exported model whose extractor timing selects the frames")
    ap.add_argument("--out")
    ap.add_argument("--lag-min", type=float, default=LAG_MIN)
    ap.add_argument("--max-offset-s", type=float, default=90.0)
    ap.add_argument("--max-width", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    timing = model_timing(Path(args.artifacts))
    if args.command == "fetch":
        return cmd_fetch(args, timing)
    if not args.out:
        ap.error("embed needs --out")
    return cmd_embed(args, timing)


if __name__ == "__main__":
    sys.exit(main())
