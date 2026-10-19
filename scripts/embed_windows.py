"""Frozen CLIP image embeddings of real windows: the frames the extractor selects, whole frame and 6 tiles.

    python scripts/embed_windows.py --manifest data/real/figlib/real.csv --artifacts artifacts_far \\
        --out data/real/figlib/clip.npz
    python scripts/embed_windows.py --manifest data/real/hpwren_fog/real.csv --artifacts artifacts_far \\
        --out data/real/hpwren_fog/clip.npz

The appearance input of the detector (src/detector): for every manifest row
(one window, max_windows 1) the 12 frames are selected exactly as
scripts/extract_features.py selects them (the exported model's timing and the
row's start_s, via region_boxes.window_frames), and frames FRAMES of the window
are embedded with a frozen CLIP ViT-B/32 image encoder (openai/clip-vit-base-
patch32; projected, L2-normalised 512-d embeddings):

* crop 0: the whole frame, squashed to 224 x 224 (nothing cropped away);
* crops 1-6: a 2 x 3 grid of tiles (rows top to bottom, columns left to right),
  each resized to 224 x 224, so a plume a few dozen pixels wide in a 1024-px
  frame is still a few dozen pixels in its tile.

The camera's burned-in banner (top rows) is blanked first. Written: an .npz
with window_id, seq_id (the extractor's id, <window_id>-w0), the embeddings
(N, len(FRAMES), 7, 512) float16 and the model id. Rows whose frames cannot be
selected are listed in <out>.skipped.json. Rows already in an existing --out
are reused (--force recomputes).
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from region_boxes import model_timing, window_frames  # noqa: E402

MODEL_ID = "openai/clip-vit-base-patch32"
FRAMES = (0, 6, 11)
N_CROPS = 7
SIZE = 224
BANNER_FRAC = 0.03                 # top rows blanked: the camera banner (camera name, date, time)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def crops(img_bgr: np.ndarray) -> list:
    """The 7 crops of a frame (whole frame, then the 2 x 3 tiles), RGB float32 normalised, (3, 224, 224) each."""
    img = img_bgr.copy()
    h, w = img.shape[:2]
    img[: int(round(BANNER_FRAC * h))] = 0
    out = [img]
    for r in range(2):
        for c in range(3):
            out.append(img[r * h // 2:(r + 1) * h // 2, c * w // 3:(c + 1) * w // 3])
    res = []
    for x in out:
        x = cv2.resize(x, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        res.append(((x - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1))
    return res


class ClipEmbedder:
    def __init__(self, model_id: str = MODEL_ID, threads: int = 4):
        import torch
        from transformers import CLIPModel

        torch.set_num_threads(threads)
        self.torch = torch
        self.model = CLIPModel.from_pretrained(model_id).eval()

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            z = self.model.get_image_features(pixel_values=self.torch.from_numpy(batch))
            z = z / z.norm(dim=-1, keepdim=True)
        return z.numpy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--artifacts", required=True, help="exported model whose extractor timing selects the frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    timing = model_timing(Path(args.artifacts))
    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = Path(args.out)
    done = {}
    if out.is_file() and not args.force:
        z = np.load(out, allow_pickle=False)
        done = {str(w): e for w, e in zip(z["window_id"], z["emb"])}
    emb = ClipEmbedder(threads=args.threads)
    ids, seqs, embs, skipped = [], [], [], []
    pending, pending_ids = [], []

    def flush():
        if not pending:
            return
        arr = np.stack([c for p in pending for c in p])
        z = np.concatenate([emb(arr[i:i + args.batch]) for i in range(0, len(arr), args.batch)])
        k = 0
        for wid, p in zip(pending_ids, pending):
            n = len(p)
            embs.append(z[k:k + n].reshape(len(FRAMES), N_CROPS, -1).astype(np.float16))
            ids.append(wid)
            seqs.append(f"{wid}-w0")
            k += n
        pending.clear()
        pending_ids.clear()

    for i, row in enumerate(rows):
        wid = row.get("window_id") or row["video_id"]
        if wid in done:
            ids.append(wid), seqs.append(f"{wid}-w0"), embs.append(done[wid])
            continue
        try:
            files, idx = window_frames(row, timing)
            cs = []
            for k in FRAMES:
                img = cv2.imread(str(files[idx[k]]), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"cannot read {files[idx[k]]}")
                cs += crops(img)
        except Exception as exc:  # noqa: BLE001 - listed, not fatal
            skipped.append({"window_id": wid, "reason": str(exc)})
            continue
        pending.append(cs)
        pending_ids.append(wid)
        if len(pending) >= 8:
            flush()
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)} windows", file=sys.stderr)
    flush()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, window_id=np.array(ids), seq_id=np.array(seqs), emb=np.stack(embs),
                        model=np.array(MODEL_ID), frames=np.array(FRAMES))
    out.with_suffix(".skipped.json").write_text(json.dumps(skipped, indent=1))
    print(json.dumps({"windows": len(ids), "skipped": len(skipped), "out": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
