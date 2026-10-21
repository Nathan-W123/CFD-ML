"""Per-frame CLIP embeddings of whole image sequences, for scoring a window that ends every minute.

    python scripts/embed_frames.py --out-dir data/real/dense/figlib_test2 data/real/figlib_test2/<sequence> ...
    python scripts/embed_frames.py --out-dir data/real/dense/figlib_test2 --list sequences.txt

Every frame of each sequence folder (*.jpg named <unix>.jpg or <unix>_<offset>.jpg, FIgLib's offset being
seconds from the first visible plume) is embedded once with the crops the detector uses
(scripts/embed_windows.crops: banner blanked; the whole frame and a 2 x 3 grid, "coarse", 7 crops; a 4 x 6
grid, "fine", 24 crops). A window ending at any frame is then assembled from these without embedding a frame
twice (src/detector/dense.py). Written per sequence: <out-dir>/<folder name>.npz with unix (F,), offset_s (F,)
(0 where the file name has no offset), coarse (F, 7, 512) and fine (F, 24, 512) float16. Existing outputs are
kept (--force recomputes); --delete-frames removes a folder's frames once its embeddings are written (for
streamed downloads). Frames are read and cropped in a background thread while CLIP runs.
"""

import _bootstrap  # noqa: F401

import argparse
import queue
import shutil
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_windows import ClipEmbedder, crops  # noqa: E402

N_COARSE, N_FINE = 7, 24


def frame_times(path: Path):
    """(unix, offset_s) from a frame file name <unix>[_<offset>].jpg."""
    parts = path.stem.split("_")
    return int(parts[0]), (int(parts[1]) if len(parts) > 1 else 0)


def sequence_frames(folder: Path) -> list:
    files = [p for p in folder.iterdir() if p.suffix.lower() == ".jpg" and p.stem.split("_")[0].isdigit()]
    return sorted(files, key=lambda p: frame_times(p)[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folders", nargs="*")
    ap.add_argument("--list", help="text file with one sequence folder per line")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--batch", type=int, default=62)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--delete-frames", action="store_true", help="delete each folder's frames after embedding")
    args = ap.parse_args(argv)
    folders = [Path(f) for f in args.folders]
    if args.list:
        folders += [Path(line.strip()) for line in Path(args.list).read_text().splitlines() if line.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [f for f in folders if args.force or not (out_dir / f"{f.name}.npz").is_file()]
    print(f"{len(folders)} sequences, {len(todo)} to embed", file=sys.stderr, flush=True)
    emb = ClipEmbedder(threads=args.threads)
    q = queue.Queue(maxsize=8)                 # frames, not sequences: one frame's crops are about 19 MB

    def reader():
        for folder in todo:
            q.put(("start", folder))
            for p in sequence_frames(folder):
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is not None:
                    q.put(("frame", frame_times(p), np.stack(crops(img) + crops(img, 4, 6, False))))
            q.put(("end", folder))
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    n, per_batch = 0, max(1, args.batch // (N_COARSE + N_FINE))
    times, zs, pending = [], [], []

    def flush():
        if pending:
            z = emb(np.concatenate(pending))
            zs.append(z.reshape(len(pending), N_COARSE + N_FINE, -1).astype(np.float16))
            pending.clear()

    while True:
        item = q.get()
        if item is None:
            break
        if item[0] == "start":
            times, zs = [], []
        elif item[0] == "frame":
            times.append(item[1])
            pending.append(item[2])
            if len(pending) >= per_batch:
                flush()
        else:
            flush()
            folder = item[1]
            if not times:
                print(f"  {folder}: no readable frames", file=sys.stderr)
                continue
            z = np.concatenate(zs)
            out = out_dir / f"{folder.name}.npz"
            tmp = out.with_name(out.stem + ".tmp.npz")
            np.savez_compressed(tmp, unix=np.array([t[0] for t in times], np.int64),
                                offset_s=np.array([t[1] for t in times], np.int64),
                                coarse=z[:, :N_COARSE], fine=z[:, N_COARSE:])
            tmp.rename(out)
            if args.delete_frames:
                shutil.rmtree(folder, ignore_errors=True)
            n += 1
            print(f"  {n}/{len(todo)} {folder.name}: {len(times)} frames", file=sys.stderr, flush=True)
    print(f"done: {n} sequences -> {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
