"""Smoke boxes drawn on box sheets -> the box CSV src/detector/boxes.py reads.

    # 1. sheets of the dev smoke windows (minutes 0, 4, 8, 11 in a 2 x 2 grid with a coordinate grid; the frames
    #    the embeddings were computed from), from a manifest of those windows
    python scripts/region_boxes.py sheets --manifest smoke_dev.csv --artifacts artifacts_far \\
        --out data/real/boxes/smoke_dev
    # 2. boxes drawn on the sheets: a JSON list (or {"boxes": [...]}) of {sheet_id, target ("smoke" or "none"),
    #    confidence, what, tiles: [{tile 0-3, visible, x0, y0, x1, y1}]}, coordinates fractions of the frame
    python scripts/box_labels.py --sheets data/real/boxes/smoke_dev/box_sheets.csv --boxes boxes.json \\
        --out configs/real/smoke_dev_boxes.csv

Sheet tile k shows minute (0, 4, 8, 11)[k] of the window. Written: one row
per window (window_id, sheet_id, target, confidence, what, labeler, and per
minute m: m<m>_visible, m<m>_x0, m<m>_y0, m<m>_x1, m<m>_y1, corners ordered
and clipped to [0, 1]; an empty box or target "none" is not visible), and
<out>.stats.json.
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import json
import sys
from pathlib import Path

MINUTES = (0, 4, 8, 11)
FIELDS = ["window_id", "sheet_id", "target", "confidence", "what", "labeler"] + \
    [f"m{m}_{k}" for m in MINUTES for k in ("visible", "x0", "y0", "x1", "y1")]


def _clean(t: dict):
    """A tile's box as (visible, x0, y0, x1, y1): corners ordered and clipped to [0, 1]; empty -> not visible."""
    if not t.get("visible"):
        return 0, 0.0, 0.0, 0.0, 0.0
    x0, x1 = sorted((min(max(float(t["x0"]), 0.0), 1.0), min(max(float(t["x1"]), 0.0), 1.0)))
    y0, y1 = sorted((min(max(float(t["y0"]), 0.0), 1.0), min(max(float(t["y1"]), 0.0), 1.0)))
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    return 1, x0, y0, x1, y1


def cmd_collect(args) -> int:
    with open(args.sheets, newline="", encoding="utf-8") as fh:
        by_sheet = {r["sheet_id"]: r["window_id"] for r in csv.DictReader(fh)}
    entries = json.loads(Path(args.boxes).read_text())
    if isinstance(entries, dict):
        entries = entries.get("boxes", entries.get("result", {}).get("boxes", []))
    rows, unknown = {}, 0
    for e in entries:
        wid = by_sheet.get(e.get("sheet_id"))
        if wid is None:
            unknown += 1
            continue
        tiles = {int(t["tile"]): t for t in e.get("tiles", [])}
        row = {"window_id": wid, "sheet_id": e["sheet_id"], "target": e["target"], "confidence": e["confidence"],
               "what": e.get("what", ""), "labeler": e.get("labeler", "")}
        for k, m in enumerate(MINUTES):
            vis, x0, y0, x1, y1 = _clean(tiles.get(k, {}))
            if e["target"] != "smoke":
                vis, x0, y0, x1, y1 = 0, 0.0, 0.0, 0.0, 0.0
            row.update({f"m{m}_visible": vis, f"m{m}_x0": f"{x0:.3f}", f"m{m}_y0": f"{y0:.3f}",
                        f"m{m}_x1": f"{x1:.3f}", f"m{m}_y1": f"{y1:.3f}"})
        rows[wid] = row
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(sorted(rows.values(), key=lambda r: r["window_id"]))
    vals = list(rows.values())
    stats = {"windows": len(vals), "sheets_in_index": len(by_sheet), "unknown_sheets": unknown,
             "target": {t: sum(r["target"] == t for r in vals) for t in ("smoke", "none")},
             "confidence": {c: sum(r["confidence"] == c and r["target"] == "smoke" for r in vals)
                            for c in ("high", "medium", "low")},
             "visible_at_minute_11": sum(r["m11_visible"] == 1 for r in vals)}
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sheets", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out", required=True)
    return cmd_collect(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
