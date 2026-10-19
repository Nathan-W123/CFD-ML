"""scripts/region_boxes.py: box sheets select the extractor's frames; boxes become per-frame masks it loads."""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import region_boxes as R  # noqa: E402


def _folder(root: Path, n: int = 16, h: int = 60, w: int = 90, jitter=None) -> Path:
    folder = root / "win"
    folder.mkdir(parents=True)
    rows = []
    for k in range(n):
        t = 60.0 * k + (jitter[k] if jitter else 0.0)
        name = f"{1_790_000_000 + k * 60}.jpg"
        cv2.imwrite(str(folder / name), np.full((h, w, 3), 10 * k, np.uint8))
        rows.append((name, t))
    with open(folder / "timestamps.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["filename", "timestamp_s"])
        wr.writerows(rows)
    return folder


def _artifacts(root: Path) -> Path:
    art = root / "art"
    art.mkdir()
    (art / "model_config.yaml").write_text(
        "preprocessing:\n  feature_config:\n    expected_interval_s: 60.0\n    max_offset_s: 10.0\n"
        "    flow_mode: consecutive\n    flow_lag_s: null\n")
    return art


def test_frame_boxes_interpolate_hold_and_gap():
    a, b = (0.0, 0.5, 0.4, 1.0), (0.2, 0.5, 0.6, 1.0)
    fb = R.frame_boxes({0: a, 1: b, 2: b, 3: b})
    assert fb[0] == a and fb[4] == b and fb[11] == b
    assert fb[2] == pytest.approx((0.1, 0.5, 0.5, 1.0))
    held = R.frame_boxes({1: a})                     # one drawn tile (frame 4): held before and after
    assert all(x == a for x in held)
    gap = R.frame_boxes({0: a, 1: None, 2: b, 3: b})  # an empty tile is not interpolated across
    assert gap[0] == a and gap[2] is None and gap[4] is None and gap[6] is None and gap[8] == b
    assert R.parse_box({"x0": "", "y0": "", "x1": "", "y1": ""}) is None
    assert R.parse_box({"x0": "0.8", "y0": "1.2", "x1": "0.2", "y1": "0.3"}) == (0.2, 0.3, 0.8, 1.0)


def test_sheets_and_masks_match_the_extractor_window(tmp_path):
    folder = _folder(tmp_path, jitter=[0, 2, -1, 3, 0, -2, 1, 0, 2, -3, 1, 0, 2, 0, 1, 0])
    art = _artifacts(tmp_path)
    manifest = tmp_path / "real.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["video_path", "camera_id", "video_id", "smoke", "fog", "split",
                                           "start_s", "max_windows", "window_id"])
        w.writeheader()
        w.writerow({"video_path": str(folder), "camera_id": "cam-a", "video_id": "v", "smoke": "0", "fog": "1",
                    "split": "test_iid", "start_s": "120", "max_windows": 1, "window_id": "w1"})
    sheets = tmp_path / "sheets"
    assert R.main(["sheets", "--manifest", str(manifest), "--artifacts", str(art), "--out", str(sheets)]) == 0
    with open(sheets / "box_sheets.csv", newline="") as fh:
        (row,) = list(csv.DictReader(fh))
    assert row["frame_indices"].split() == [str(i) for i in range(2, 14)]      # start 120 s -> frames 2..13
    sheet = cv2.imread(str(sheets / f"{row['sheet_id']}.jpg"))
    assert sheet.shape[:2] == (2 * R.TILE[1], 2 * R.TILE[0])
    boxes = tmp_path / "boxes.csv"
    with open(boxes, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sheet_id", "tile", "x0", "y0", "x1", "y1"])
        for t in range(4):
            w.writerow([row["sheet_id"], t, 0.0, 0.5, 0.5, 1.0])
    out_manifest = tmp_path / "real_boxes.csv"
    assert R.main(["masks", "--manifest", str(manifest), "--sheets-dir", str(sheets), "--boxes", str(boxes),
                   "--out-manifest", str(out_manifest)]) == 0
    with open(out_manifest, newline="") as fh:
        (mrow,) = list(csv.DictReader(fh))
    mdir = Path(mrow["mask_dir"])
    files = sorted(mdir.glob("frame_*.png"))
    assert [f.name for f in files] == [f"frame_{i:06d}.png" for i in range(2, 14)]
    m = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    assert m.shape == (60, 90) and m[59, 0] == 255 and m[0, 89] == 0 and 0.2 < (m > 0).mean() < 0.3
    # the extractor accepts the masks (provided) for exactly this window
    ex = __import__("extract_features")
    from src.data.features import select_window

    ts = ex.folder_timestamps(folder)
    win = select_window(ts, n=12, interval_s=60.0, start=120.0, max_offset_s=10.0, flow_lag_s=None)
    masks = ex.load_masks(mdir, win["obs_idx"], (60, 90))
    assert masks.shape == (12, 60, 90) and masks.any(axis=(1, 2)).all()
