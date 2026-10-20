"""Full-resolution zoom crops of a window's most suspicious tiles, embedded with CLIP (streamed, nothing kept).

    python scripts/zoom_windows.py --manifest data/real/figlib/real.csv --source figlib \\
        --selection runs/zoom/selection.json --artifacts artifacts_far --out data/real/figlib/clip_zoom.npz
    python scripts/zoom_windows.py --manifest data/real/hpwren_smoke/real.csv --source hpwren \\
        --selection runs/zoom/selection.json --artifacts artifacts_far --out data/real/hpwren_smoke/clip_zoom.npz

The stored frames are at most 1024 px wide; the originals are 2-3x larger
(FIgLib 3072 x 2048 or 2048 x 1536, HPWREN 3072 x 2048), so a plume a few
pixels wide in a stored frame is a few dozen in the original. For every
manifest row this fetches the originals of frames 0 and 11 of the window
(the frames scripts/embed_windows.py uses for the fine 4 x 6 tiles; FIgLib
from the sequence's tarball, read once per sequence and deleted; HPWREN
from the camera archive), and for each of the window's selected 4 x 6 tiles
(--selection: JSON {window_id: [tile index 0-23, ...]}, tiles numbered row
by row) cuts the tile from the original, splits it 2 x 2, and embeds each
quarter with the same frozen CLIP encoder (224 px input, so about one input
pixel per original pixel). The camera banner (top 3 %) is blanked first.

Written: an .npz with window_id, tiles (N, K) and emb (N, 2, 4 K, 512)
float16 (frames 0 and 11; per frame the K tiles' quarters, tile-major,
quarters row by row). Rows that cannot be fetched are listed in
<out>.skipped.json. Existing rows of --out are reused (--force recomputes).
Imagery courtesy of HPWREN (https://www.hpwren.ucsd.edu/).
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import datetime as dt
import json
import sys
import tarfile
import tempfile
import threading
import queue
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_windows import BANNER_FRAC, CLIP_MEAN, CLIP_STD, SIZE, ClipEmbedder  # noqa: E402
from prepare_hpwren_fog import IMAGE_URL, TZ, _get  # noqa: E402
from region_boxes import model_timing, window_frames  # noqa: E402

FINE_ROWS, FINE_COLS = 4, 6
FRAMES = (0, 11)
FIGLIB_TAR = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/{name}.tgz"


def zoom_crops(img_bgr: np.ndarray, tiles, mean=CLIP_MEAN, std=CLIP_STD) -> list:
    """For each 4 x 6 tile index, the tile's 2 x 2 quarters of the (full-resolution) frame, (3, 224, 224) each."""
    img = img_bgr.copy()
    h, w = img.shape[:2]
    img[: int(round(BANNER_FRAC * h))] = 0
    out = []
    for t in tiles:
        r, c = divmod(int(t), FINE_COLS)
        y0, y1 = r * h // FINE_ROWS, (r + 1) * h // FINE_ROWS
        x0, x1 = c * w // FINE_COLS, (c + 1) * w // FINE_COLS
        ym, xm = (y0 + y1) // 2, (x0 + x1) // 2
        for ya, yb in ((y0, ym), (ym, y1)):
            for xa, xb in ((x0, xm), (xm, x1)):
                x = cv2.resize(img[ya:yb, xa:xb], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                out.append(((x - mean) / std).transpose(2, 0, 1))
    return out


def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode an original frame")
    return img


def figlib_originals(name: str, wanted: set, tmp_dir: Path) -> dict:
    """{frame file name: decoded original} for the wanted frames of one FIgLib sequence (tarball deleted)."""
    tgz = tmp_dir / f"{name}.tgz"
    tgz.write_bytes(_get(FIGLIB_TAR.format(name=name), timeout=300.0))
    out = {}
    try:
        with tarfile.open(tgz, "r:*") as tf:
            for member in tf:
                base = PurePosixPath(member.name).name
                if member.isfile() and base in wanted:
                    out[base] = _decode(tf.extractfile(member).read())
    finally:
        tgz.unlink(missing_ok=True)
    return out


def hpwren_url(cam: str, unix: int) -> str:
    """Archive URL of one original frame; the day folder and 3-hour quarter come from the frame's own local time."""
    local = dt.datetime.fromtimestamp(unix, TZ)
    return IMAGE_URL.format(cam=cam, day=f"{local:%Y%m%d}", q=1 + local.hour // 3, name=f"{unix}.jpg")


def hpwren_original(window_dir: Path, file_name: str) -> np.ndarray:
    meta = json.loads((window_dir / "window.json").read_text())
    return _decode(_get(hpwren_url(meta["camera"], int(Path(file_name).stem)), timeout=120.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--source", choices=("figlib", "hpwren"), required=True)
    ap.add_argument("--selection", required=True, help="JSON {window_id: [tile indices]}")
    ap.add_argument("--artifacts", required=True, help="exported model whose extractor timing selects the frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    timing = model_timing(Path(args.artifacts))
    selection = json.loads(Path(args.selection).read_text())
    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("window_id") or r["video_id"]) in selection]
    out = Path(args.out)
    done = {}
    if out.is_file() and not args.force:
        z = np.load(out)
        done = {str(w): (e, t) for w, e, t in zip(z["window_id"], z["emb"], z["tiles"])}
    todo = [r for r in rows if (r.get("window_id") or r["video_id"]) not in done]
    emb = ClipEmbedder(threads=args.threads)

    # fetch jobs, grouped so a FIgLib tarball is read once for all its windows
    jobs = {}
    for r in todo:
        key = r["video_id"] if args.source == "figlib" else r.get("window_id") or r["video_id"]
        jobs.setdefault(key, []).append(r)
    q = queue.Queue(maxsize=4)
    skipped = []

    def fetch():
        with tempfile.TemporaryDirectory() as td:
            for key, rs in jobs.items():
                try:
                    picks = []
                    for r in rs:
                        files, idx = window_frames(r, timing)
                        picks.append((r, [Path(files[idx[k]]).name for k in FRAMES]))
                    if args.source == "figlib":
                        imgs = figlib_originals(key, {f for _, fs in picks for f in fs}, Path(td))
                    else:
                        imgs = {f: hpwren_original(Path(r["video_path"]), f) for r, fs in picks for f in fs}
                    q.put(("ok", picks, imgs))
                except Exception as exc:  # noqa: BLE001 - listed, not fatal
                    q.put(("err", [r for r in rs], str(exc)))
        q.put(None)

    threading.Thread(target=fetch, daemon=True).start()
    ids, embs, tiles_out = [], [], []
    n = 0
    while True:
        item = q.get()
        if item is None:
            break
        if item[0] == "err":
            for r in item[1]:
                skipped.append({"window_id": r.get("window_id") or r["video_id"], "reason": item[2]})
            continue
        _, picks, imgs = item
        for r, fs in picks:
            wid = r.get("window_id") or r["video_id"]
            tiles = [int(t) for t in selection[wid]]
            try:
                crops = [c for f in fs for c in zoom_crops(imgs[f], tiles)]
            except KeyError as exc:
                skipped.append({"window_id": wid, "reason": f"original frame missing: {exc}"})
                continue
            arr = np.stack(crops)
            z = np.concatenate([emb(arr[i:i + args.batch]) for i in range(0, len(arr), args.batch)])
            embs.append(z.reshape(len(FRAMES), 4 * len(tiles), -1).astype(np.float16))
            ids.append(wid)
            tiles_out.append(tiles)
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(todo)} windows", file=sys.stderr, flush=True)
    for wid, (e, t) in done.items():
        ids.append(wid), embs.append(e), tiles_out.append(list(t))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, window_id=np.array(ids), emb=np.stack(embs), tiles=np.array(tiles_out, np.int64),
                        frames=np.array(FRAMES))
    out.with_suffix(".skipped.json").write_text(json.dumps(skipped, indent=1))
    print(json.dumps({"windows": len(ids), "skipped": len(skipped), "out": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
