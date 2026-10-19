"""Boxes drawn around a candidate region in real windows, turned into per-frame masks for the extractor.

    # 1. box sheets: the frames extract_features.py will select (exported model timing), 0 / 4 / 8 / 11 of
    #    each window in a 2 x 2 grid with a coordinate grid, under opaque ids
    python scripts/region_boxes.py sheets --manifest data/real/hpwren_fog/real.csv --artifacts artifacts_far \\
        --out data/real/boxes/hpwren
    # 2. draw the boxes: a CSV with sheet_id, tile (0-3), x0, y0, x1, y1 (fractions of the tile, 0-1; all
    #    empty = no region in that tile)
    # 3. masks and a manifest whose rows point extract_features.py at them (mask_dir)
    python scripts/region_boxes.py masks --manifest data/real/hpwren_fog/real.csv --artifacts artifacts_far \\
        --sheets-dir data/real/boxes/hpwren --boxes configs/real/hpwren_fog_boxes.csv \\
        --out-manifest data/real/hpwren_fog/real_boxes.csv

Why: the built-in fallback detector (src/data/masks.detect_candidate_masks)
finds change against the window's median frame, so a static veil, such as a
fog sea lying below a mountain camera for the whole window, is never a
candidate. The model was trained with masks around the aerosol (degraded true
masks, jittered per-frame boxes, or that detector on evolving synthetic
aerosol), so testing its physics on real fog needs a mask on the fog. A box
around the region in each drawn frame stands in for a vision detector's box.

sheets
  For each manifest row (one window: max_windows 1, as written by
  prepare_hpwren_fog.py and prepare_figlib.py), the frame times come from the
  folder's timestamps (extract_features.folder_timestamps) and the window from
  features.select_window with the model's exported timing (interval,
  tolerance, consecutive flow) and the row's start_s, exactly as the
  extractor will select it. Frames 0, 4, 8 and 11 of the window are drawn in a
  2 x 2 sheet (tiles TILE pixels, banner rows blanked) with a 10 x 10
  coordinate grid labelled 0.0-1.0 on each tile. Written: <out>/<sheet_id>.jpg
  and <out>/box_sheets.csv (sheet_id, window_id, video_path, frame indices
  of the 12 selected frames, frame size).

masks
  For each sheet with boxes, the box of every selected frame is interpolated
  linearly in time between the drawn tiles (frames 0, 4, 8, 11; before the
  first drawn tile and after the last one the nearest drawn box is held; a
  tile drawn empty means no region in that frame and is not interpolated
  across), scaled to the frame size, and written as
  <out-mask-root>/<window_id>/frame_<index:06d>.png (255 inside the box) for
  the frame indices the extractor will load. The output manifest repeats the
  input rows that have boxes, with mask_dir set; rows without boxes are
  dropped (listed in the summary).
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.data.feature_spec import N_STEPS
from src.data.features import select_window
from src.utils.io import read_yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import folder_timestamps  # noqa: E402

TILES = (0, 4, 8, 11)
TILE = (768, 512)
BANNER_PX = 20
GRID = 10
SALT = "region-boxes-v1"


def model_timing(artifacts: Path) -> dict:
    """interval, tolerance and flow lag the exported model's extractor uses (consecutive flow: lag None)."""
    fc = ((read_yaml(artifacts / "model_config.yaml").get("preprocessing") or {}).get("feature_config") or {})
    consecutive = fc.get("flow_mode") == "consecutive"
    return {"interval_s": float(fc["expected_interval_s"]), "max_offset_s": float(fc["max_offset_s"]),
            "flow_lag_s": None if consecutive else float(fc["flow_lag_s"])}


def window_frames(row: dict, timing: dict) -> tuple:
    """(sorted image paths, indices of the 12 selected frames) of a manifest row, as the extractor selects them."""
    path = Path(row["video_path"])
    if not path.is_dir():
        raise ValueError(f"{path} is not an image folder (box sheets need image folders)")
    ts = folder_timestamps(path)
    from src.data.features import _IMAGE_EXTENSIONS, _folder_timestamps

    files = sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
    listed, times = _folder_timestamps(path, files)
    order = np.argsort(times)
    listed = [listed[i] for i in order]
    start = float(row["start_s"]) if row.get("start_s") not in (None, "") else None
    win = select_window(ts, n=N_STEPS, interval_s=timing["interval_s"], start=start,
                        max_offset_s=timing["max_offset_s"], flow_lag_s=timing["flow_lag_s"])
    return listed, [int(i) for i in win["obs_idx"]]


def sheet_id_for(window_id: str) -> str:
    return hashlib.sha1(f"{SALT}:{window_id}".encode("utf-8")).hexdigest()[:12]


def _grid(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    over = img.copy()
    for k in range(1, GRID):
        x, y = int(round(k * w / GRID)), int(round(k * h / GRID))
        cv2.line(over, (x, 0), (x, h - 1), (0, 255, 255), 1)
        cv2.line(over, (0, y), (w - 1, y), (0, 255, 255), 1)
        cv2.putText(over, f"{k / GRID:.1f}", (x + 2, BANNER_PX + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(over, f"{k / GRID:.1f}", (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return cv2.addWeighted(over, 0.6, img, 0.4, 0)


def write_box_sheet(paths: list, out_path: Path) -> list:
    """2 x 2 sheet of the given 4 frames with coordinate grids; returns the frames' (H, W)."""
    tiles, sizes = [], []
    for k, p in zip(TILES, paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cannot read {p}")
        sizes.append(img.shape[:2])
        img = cv2.resize(img, TILE, interpolation=cv2.INTER_AREA)
        img[:BANNER_PX] = 0
        img = _grid(img)
        cv2.rectangle(img, (0, 0), (150, 34), (0, 0, 0), -1)
        cv2.putText(img, f"tile {TILES.index(k)}: t={k}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return sizes


def _read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list, fields: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def cmd_sheets(args) -> int:
    timing = model_timing(Path(args.artifacts))
    out = Path(args.out)
    rows, skipped = [], []
    for row in _read_csv(Path(args.manifest)):
        wid = row.get("window_id") or row["video_id"]
        try:
            files, idx = window_frames(row, timing)
        except Exception as exc:  # noqa: BLE001 - listed, not fatal
            skipped.append({"window_id": wid, "reason": str(exc)})
            continue
        sid = sheet_id_for(wid)
        sheet = out / f"{sid}.jpg"
        h, w = cv2.imread(str(files[idx[0]]), cv2.IMREAD_GRAYSCALE).shape
        if args.force or not sheet.is_file():
            write_box_sheet([files[idx[k]] for k in TILES], sheet)
        rows.append({"sheet_id": sid, "window_id": wid, "video_path": row["video_path"],
                     "frame_indices": " ".join(map(str, idx)), "height": h, "width": w})
    _write_csv(out / "box_sheets.csv", rows, ["sheet_id", "window_id", "video_path", "frame_indices", "height", "width"])
    print(json.dumps({"sheets": len(rows), "skipped": skipped[:5], "n_skipped": len(skipped)}))
    return 0


def parse_box(r: dict):
    """(x0, y0, x1, y1) fractions of a box row, or None for an empty (no region) row."""
    vals = [r.get(k, "") for k in ("x0", "y0", "x1", "y1")]
    if all(v in ("", None) for v in vals):
        return None
    x0, y0, x1, y1 = (float(v) for v in vals)
    x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
    y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return x0, y0, x1, y1


def frame_boxes(tile_boxes: dict, n: int = N_STEPS) -> list:
    """Per selected frame, the box interpolated between drawn tiles (see the module docstring), or None."""
    drawn = sorted((TILES[t], b) for t, b in tile_boxes.items())
    out = []
    for k in range(n):
        before = [(f, b) for f, b in drawn if f <= k]
        after = [(f, b) for f, b in drawn if f >= k]
        if not before:
            out.append(after[0][1])
        elif not after:
            out.append(before[-1][1])
        else:
            (f0, b0), (f1, b1) = before[-1], after[0]
            if b0 is None or b1 is None:
                out.append(b0 if f0 == k else b1 if f1 == k else None)
            elif f1 == f0:
                out.append(b0)
            else:
                a = (k - f0) / (f1 - f0)
                out.append(tuple((1 - a) * u + a * v for u, v in zip(b0, b1)))
    return out


def cmd_masks(args) -> int:
    sheets = {r["sheet_id"]: r for r in _read_csv(Path(args.sheets_dir) / "box_sheets.csv")}
    boxes = defaultdict(dict)
    for r in _read_csv(Path(args.boxes)):
        boxes[r["sheet_id"]][int(r["tile"])] = parse_box(r)
    by_window = {s["window_id"]: (sid, s) for sid, s in sheets.items()}
    root = Path(args.mask_root) if args.mask_root else Path(args.sheets_dir) / "masks"
    out_rows, dropped = [], defaultdict(int)
    rows = _read_csv(Path(args.manifest))
    fields = list(rows[0].keys()) + (["mask_dir"] if "mask_dir" not in rows[0] else [])
    for row in rows:
        wid = row.get("window_id") or row["video_id"]
        if wid not in by_window:
            dropped["no_sheet"] += 1
            continue
        sid, s = by_window[wid]
        tb = boxes.get(sid)
        if not tb or all(b is None for b in tb.values()):
            dropped["no_box"] += 1
            continue
        h, w = int(s["height"]), int(s["width"])
        idx = [int(i) for i in s["frame_indices"].split()]
        mdir = root / wid
        mdir.mkdir(parents=True, exist_ok=True)
        for i, b in zip(idx, frame_boxes(tb, len(idx))):
            m = np.zeros((h, w), np.uint8)
            if b is not None:
                x0, y0, x1, y1 = b
                m[int(round(y0 * (h - 1))):int(round(y1 * (h - 1))) + 1,
                  int(round(x0 * (w - 1))):int(round(x1 * (w - 1))) + 1] = 255
            cv2.imwrite(str(mdir / f"frame_{i:06d}.png"), m)
        out_rows.append({**row, "mask_dir": str(mdir)})
    _write_csv(Path(args.out_manifest), out_rows, fields)
    print(json.dumps({"rows": len(out_rows), "dropped": dict(dropped)}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheets")
    s.add_argument("--manifest", required=True)
    s.add_argument("--artifacts", required=True, help="exported model whose extractor timing selects the frames")
    s.add_argument("--out", required=True)
    s.add_argument("--force", action="store_true")
    m = sub.add_parser("masks")
    m.add_argument("--manifest", required=True)
    m.add_argument("--sheets-dir", required=True)
    m.add_argument("--boxes", required=True)
    m.add_argument("--out-manifest", required=True)
    m.add_argument("--mask-root", default=None, help="default <sheets-dir>/masks")
    m.add_argument("--artifacts", default=None, help="accepted for symmetry; frame indices come from box_sheets.csv")
    args = ap.parse_args(argv)
    return {"sheets": cmd_sheets, "masks": cmd_masks}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
