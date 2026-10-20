"""Smoke boxes -> instance labels for the tile and zoom heads.

A box file (configs/real/smoke_dev_boxes.csv, written by scripts/box_labels.py)
holds one row per smoke window: the target ("smoke" or "none"), the labeller's
confidence, and for each of the window's minutes 0, 4, 8 and 11 whether the
plume is visible and its box (x0, y0, x1, y1 as fractions of the frame).

The fine tiles (4 x 6) and the zoom crops (2 x 2 quarters of selected fine
tiles) are scored on frame 11, so their labels come from the minute-11 box:

* 1: the region and the box overlap by at least pos_frac of the smaller of the
  two (a tile inside a large plume, or a small plume inside a tile);
* 0: the region does not touch the box grown by margin on every side;
* -1 (ignored): anything in between, a box edge grazing the region.

Windows without a usable box (target "none", confidence below the minimum,
plume not visible at minute 11, or no region labelled 1) give no instance
labels; smoke-free windows give 0 for every region.
"""

from __future__ import annotations

import csv
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

FINE_ROWS, FINE_COLS = 4, 6
MINUTES = (0, 4, 8, 11)
CONFIDENCE = {"low": 0, "medium": 1, "high": 2}

Rect = Tuple[float, float, float, float]


def load_boxes(path) -> Dict[str, Dict]:
    """{window_id: {"target", "confidence", "boxes": {minute: (x0, y0, x1, y1) or None}}}."""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            boxes = {}
            for m in MINUTES:
                vis = r.get(f"m{m}_visible", "").strip().lower() in ("1", "true")
                boxes[m] = (tuple(float(r[f"m{m}_{k}"]) for k in ("x0", "y0", "x1", "y1")) if vis else None)
            out[r["window_id"]] = {"target": r["target"], "confidence": r["confidence"], "boxes": boxes}
    return out


def fine_regions(rows: int = FINE_ROWS, cols: int = FINE_COLS) -> list:
    """The fine tiles as frame fractions, row by row (the order of scripts/embed_windows.py's crops)."""
    return [(c / cols, r / rows, (c + 1) / cols, (r + 1) / rows) for r in range(rows) for c in range(cols)]


def zoom_regions(tiles: Sequence[int]) -> list:
    """The zoom crops of the given fine tiles: each tile's 2 x 2 quarters, tile-major, quarters row by row (the
    order of scripts/zoom_windows.py)."""
    fine = fine_regions()
    out = []
    for t in tiles:
        x0, y0, x1, y1 = fine[int(t)]
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        for ya, yb in ((y0, ym), (ym, y1)):
            for xa, xb in ((x0, xm), (xm, x1)):
                out.append((xa, ya, xb, yb))
    return out


def _area(r: Rect) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def _inter(a: Rect, b: Rect) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def region_labels(regions: Sequence[Rect], box: Rect, pos_frac: float = 0.25, margin: float = 0.02) -> np.ndarray:
    """1 / 0 / -1 (ignored) per region for one box (see the module docstring)."""
    x0, y0, x1, y1 = box
    box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    grown = (box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin)
    out = np.full(len(regions), -1, np.int64)
    for i, r in enumerate(regions):
        inter = _inter(r, box)
        if inter > 0 and inter >= pos_frac * min(_area(r), max(_area(box), 1e-12)):
            out[i] = 1
        elif _inter(r, grown) == 0:
            out[i] = 0
    return out


def usable_box(entry: Optional[Dict], min_confidence: str = "medium") -> Optional[Rect]:
    """The minute-11 box of a box-file entry if it is usable, else None."""
    if not entry or entry["target"] != "smoke":
        return None
    if CONFIDENCE.get(entry["confidence"], -1) < CONFIDENCE[min_confidence]:
        return None
    box = entry["boxes"].get(11)
    if box is None or _area((min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3]))) <= 0:
        return None
    return box


def instance_labels(window_ids: Sequence[str], y: np.ndarray, boxes: Dict[str, Dict], regions_of,
                    min_confidence: str = "medium", pos_frac: float = 0.25,
                    margin: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """Instance labels Y and weights W, both (N, K), for N windows with window labels y (0 / 1).

    regions_of(i) gives window i's K regions (fine_regions() for the tile head, zoom_regions(tiles_i) for the
    zoom head). Smoke-free windows: every region 0, weight 1. Smoke windows with a usable box: the region
    labels, weight 1 (0 where ignored); smoke windows without one, or whose box labels no region 1: weight 0.
    """
    y = np.asarray(y)
    K = len(regions_of(0))
    Y = np.zeros((len(y), K), np.float64)
    W = np.zeros((len(y), K), np.float64)
    for i, wid in enumerate(window_ids):
        if y[i] == 0:
            W[i] = 1.0
            continue
        box = usable_box(boxes.get(str(wid)), min_confidence)
        if box is None:
            continue
        lab = region_labels(regions_of(i), box, pos_frac, margin)
        if not (lab == 1).any():
            continue
        Y[i] = lab == 1
        W[i] = lab >= 0
    return Y, W
