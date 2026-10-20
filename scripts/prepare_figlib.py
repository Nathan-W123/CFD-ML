"""Download HPWREN FIgLib wildfire ignition sequences and write a real-footage manifest for them.

    python scripts/prepare_figlib.py --out data/real/figlib --n 40
    python scripts/extract_features.py --real-manifest data/real/figlib/real.csv \\
        --artifacts <far-field model> --out data/real/figlib/features.npz
    # before a far-field model exists, take the timing from the profile instead:
    #   ... --profile configs/simulation_far.yaml

FIgLib, the HPWREN Fire Ignition images Library (https://www.hpwren.ucsd.edu/FIgLib/):
fixed wildfire-watch cameras in Southern California recording ONE FRAME PER
MINUTE, each sequence about 81 frames named <unix_ts>_<offset_s>.jpg, offset
in seconds relative to the first frame with a visible plume (about 40 min
before to 40 min after). Imagery courtesy of HPWREN, the High Performance
Wireless Research and Education Network, UC San Diego; credit
https://www.hpwren.ucsd.edu/ whenever it is used (written to <out>/CREDIT.txt).

The chosen N are split-aware: N is divided over train / val_select /
val_calib / test_iid in proportion to --fractions (every split with a
non-zero fraction gets at least one sequence when N allows; a split with
too few sequences hands its share to the others), and within a split the
sequences are spread over its cameras (round robin over the cameras in a
fixed hashed order, oldest sequence of each camera first). With --zero-shot
it is that round robin over every camera. Per sequence:

1. the tarball https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/<name>.tgz
   is downloaded to <out>/_downloads/<name>.tgz.part (resumed with an HTTP
   range request when a partial file exists) and renamed only when its size
   matches the server's (Content-Length, or the total of Content-Range when
   resuming); a short transfer keeps the .part, is listed as skipped and
   resumes on the next run;
2. it is read member by member, refusing absolute paths, ".." and anything
   that is not a regular file; only <unix_ts>_<offset_s>.jpg images are used
   (nothing from the archive is written to disk as is); a truncated or
   corrupt archive is deleted and the sequence listed as skipped (a rerun
   downloads it again);
3. every frame is downscaled to at most --max-width pixels (default 1024) and
   re-encoded as JPEG quality 90 into <out>/<name>/; the tarball is deleted;
4. <out>/<name>/timestamps.csv (filename,timestamp_s relative to the first
   frame: the sidecar features.load_frames reads) and sequence.json (plume
   offset, camera, unix times, credit) are written last: a sequence with both
   is done and skipped on a rerun.

The manifest <out>/real.csv (scripts/extract_features.py --real-manifest) has,
per sequence, --pre-windows windows (default 1; the latest first) of 12 frames
--interval-s (60) apart lying entirely before offset 0 (smoke = 0) and
--post-windows windows (default 1) starting at or after offset +60 s
(smoke = 1); windows straddling offset 0 are never written, and every window
is checked with features.select_window at the far-field timing (no lag
frames, tolerance --max-offset-s, default 10 s) before it is listed. fog is
left empty (unknown: FIgLib does not annotate fog). camera_id is the camera
part of the name: the trailing <site>[-<direction>]-<camera model> (e.g.
rm-n-mobo-c; 20220302-Jimfire-0921-stgo-s-mobo-c -> stgo-s-mobo-c), so
different fires on one camera share a group; camera_mount (its
site-direction, e.g. stgo-s) is written to every row too. The split is
camera-disjoint and deterministic: crc32 of the camera's mount (so the
colour and mono cameras mobo-c / mobo-m of one mount stay together) against
--fractions (default train 0.5, val_select 0.15, val_calib 0.15, test_iid
0.2), or test_iid for everything with --zero-shot. Every window has its own
window_id, so window ids never collide.
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import http.client
import json
import os
import re
import sys
import tarfile
import urllib.error
import urllib.request
import zlib
from collections import defaultdict
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from src.data.feature_spec import N_STEPS
from src.data.features import IncompatibleFrameTiming, select_window
from src.utils.io import write_json

BASE_URL = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data"
INDEX_URL = f"{BASE_URL}/index.html"
TAR_URL = BASE_URL + "/Tar/{name}.tgz"
CREDIT = ("HPWREN FIgLib (Fire Ignition images Library): imagery courtesy of HPWREN, the High Performance "
          "Wireless Research and Education Network, University of California San Diego, "
          "https://www.hpwren.ucsd.edu/ (library: https://www.hpwren.ucsd.edu/FIgLib/).")
SPLIT_NAMES = ("train", "val_select", "val_calib", "test_iid")
# Camera part of a sequence name: <site>[-<direction>]-<camera model>, e.g. rm-n-mobo-c, smer-tcs3-mobo-m.
CAMERA_RE = re.compile(r"(?<![A-Za-z0-9])((?:[a-z0-9]+-){1,2}(?:mobo-c|mobo-m|mobo|iqeye|axis))$")
CAMERA_MODEL_RE = re.compile(r"-(?:mobo-c|mobo-m|mobo|iqeye|axis)$")
FRAME_RE = re.compile(r"^(\d+)_([+-]?\d+)\.jpe?g$", re.IGNORECASE)
INDEX_LINK_RE = re.compile(r"""href=["']?([^"'\s>/]+)/index\.html""", re.IGNORECASE)
JPEG_QUALITY = 90
CHUNK = 1 << 20


# =============================================================================
# Names, cameras, splits
# =============================================================================


def camera_id(name: str) -> str:
    """The camera part of a sequence name (site-direction-model), e.g. 20160604_FIRE_rm-n-mobo-c -> rm-n-mobo-c."""
    m = CAMERA_RE.search(name)
    if m is None:
        raise ValueError(f"cannot find the camera in sequence name {name!r}")
    return m.group(1)


def camera_mount(camera: str) -> str:
    """The mount (site-direction) of a camera: rm-n-mobo-c and rm-n-mobo-m share rm-n."""
    return CAMERA_MODEL_RE.sub("", camera)


def _unit_hash(text: str) -> float:
    return zlib.crc32(text.encode("utf-8")) / 2.0 ** 32


def split_for(camera: str, fractions, zero_shot: bool = False) -> str:
    """Deterministic camera-disjoint split from the crc32 of the camera's mount (stable as the set grows)."""
    if zero_shot:
        return "test_iid"
    u = _unit_hash(camera_mount(camera))
    cum = np.cumsum(fractions) / np.sum(fractions)
    return SPLIT_NAMES[min(int(np.searchsorted(cum, u, side="right")), len(SPLIT_NAMES) - 1)]


def list_sequences(index_url: str = INDEX_URL) -> list:
    """Sequence folder names listed on the FIgLib index page (sorted)."""
    with urllib.request.urlopen(index_url, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    names = sorted({m.group(1) for m in INDEX_LINK_RE.finditer(html)} - {"Tar"})
    if not names:
        raise SystemExit(f"no sequences found on {index_url}")
    return names


def _round_robin(by_camera: dict) -> list:
    """Every name of {camera: [names, oldest first]}: round robin over the cameras in fixed crc32 order."""
    order = sorted(by_camera, key=lambda c: (_unit_hash(c), c))
    out, depth = [], 0
    while any(len(v) > depth for v in by_camera.values()):
        out.extend(by_camera[cam][depth] for cam in order if len(by_camera[cam]) > depth)
        depth += 1
    return out


def split_quotas(n: int, fractions) -> list:
    """Sequences per split (SPLIT_NAMES order) for n in proportion to the fractions: every split with a
    non-zero fraction gets at least one when n allows, the rest by largest remainder (ties: earlier split)."""
    f = np.asarray(fractions, dtype=np.float64)
    ideal = n * f / f.sum()
    q = np.floor(ideal).astype(int)
    pos = np.flatnonzero(f > 0)
    if n >= len(pos):
        q[pos] = np.maximum(q[pos], 1)
    while q.sum() > n:                     # trim where the quota most exceeds its share, keeping one each
        cand = [i for i in range(len(q)) if q[i] > 1] or [i for i in range(len(q)) if q[i] > 0]
        q[max(cand, key=lambda i: (q[i] - ideal[i], -i))] -= 1
    while q.sum() < n:
        q[max(pos, key=lambda i: (ideal[i] - q[i], -i))] += 1
    return [int(v) for v in q]


def choose(names, n: int, fractions=None, zero_shot: bool = False) -> list:
    """N names, split-aware: N divided over the splits in proportion to `fractions` (split_quotas; a
    split short of sequences hands its share to the splits furthest below their share), and within a split
    spread over its cameras (round robin, fixed crc32 order, oldest sequence first). Without fractions, or
    with zero_shot (everything is test_iid), the round robin over every camera."""
    by_camera = defaultdict(list)
    for name in sorted(names):
        try:
            by_camera[camera_id(name)].append(name)
        except ValueError:
            print(f"skipping {name}: no camera in the name", file=sys.stderr)
    if fractions is None or zero_shot:
        return _round_robin(by_camera)[:n]
    per_split = {s: _round_robin({c: v for c, v in by_camera.items() if split_for(c, fractions) == s})
                 for s in SPLIT_NAMES}
    quota = dict(zip(SPLIT_NAMES, split_quotas(n, fractions)))
    take = {s: min(quota[s], len(per_split[s])) for s in SPLIT_NAMES}
    ideal = {s: n * f / float(np.sum(fractions)) for s, f in zip(SPLIT_NAMES, fractions)}
    while sum(take.values()) < n:
        spare = [s for s in SPLIT_NAMES if take[s] < len(per_split[s]) and ideal[s] > 0]
        if not spare:
            break
        take[max(spare, key=lambda s: (ideal[s] - take[s], -SPLIT_NAMES.index(s)))] += 1
    return [name for s in SPLIT_NAMES for name in per_split[s][:take[s]]]


# =============================================================================
# Download and frame preparation
# =============================================================================


def _expected_size(resp, have: int):
    """Full size of the file a response completes: Content-Length for 200, the total after '/' in
    Content-Range for 206 (else what we have + Content-Length); None when the server does not say."""
    length = resp.headers.get("Content-Length")
    if resp.status == 206:
        m = re.match(r"\s*bytes\s+\d+-\d+/(\d+)\s*$", resp.headers.get("Content-Range") or "")
        if m:
            return int(m.group(1))
        return have + int(length) if length is not None else None
    return int(length) if length is not None else None


def download(url: str, dest: Path) -> Path:
    """Download to <dest>.part (resuming a partial file with a range request), then rename to dest.

    The .part is renamed only when its size is the server's (`_expected_size`); a short transfer
    raises OSError and keeps the .part, so the next run resumes it.
    """
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code != 416 or not have:        # 416: the partial file is already complete
            raise
        os.replace(part, dest)
        return dest
    with resp:
        mode = "ab" if have and resp.status == 206 else "wb"     # 200: the server ignored the range
        total = resp.headers.get("Content-Length")
        expected = _expected_size(resp, have)
        print(f"  downloading {url} ({'resuming at ' + str(have) + ' B, ' if mode == 'ab' else ''}"
              f"{int(total) / 1e6:.1f} MB to go)" if total else f"  downloading {url}", flush=True)
        with open(part, mode) as fh:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
    size = part.stat().st_size
    if expected is not None and size != expected:
        raise OSError(f"incomplete download of {url}: {size} of {expected} bytes in {part}; rerun to resume")
    os.replace(part, dest)
    return dest


def _safe_member(member: tarfile.TarInfo) -> bool:
    """A regular file whose path is relative and never climbs out ('..'); links, devices etc. are refused."""
    path = PurePosixPath(member.name)
    if member.name.startswith(("/", "\\")) or path.is_absolute() or ".." in path.parts or ":" in member.name:
        raise ValueError(f"refusing unsafe path {member.name!r} in the archive")
    return member.isfile()


def downscale_jpeg(data: bytes, max_width: int) -> tuple:
    """Decoded image bytes -> (JPEG q90 bytes at most max_width wide, original (h, w))."""
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR) if data else None
    if img is None:
        raise ValueError("undecodable image")
    h, w = img.shape[:2]
    if max_width and w > max_width:
        img = cv2.resize(img, (max_width, max(1, round(h * max_width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes(), (h, w)


def prepare_sequence(name: str, out: Path, max_width: int) -> dict:
    """Download, check, downscale and index one sequence; returns its sequence.json record.

    A truncated or corrupt archive (tar / gzip / zlib errors, EOF) is deleted and raises ValueError,
    so the caller lists the sequence as skipped and a rerun downloads it again.
    """
    seq_dir = out / name
    record_path = seq_dir / "sequence.json"
    if record_path.exists() and (seq_dir / "timestamps.csv").exists():
        return json.loads(record_path.read_text())
    tgz = download(TAR_URL.format(name=name), out / "_downloads" / f"{name}.tgz")
    frames, jpegs = [], []
    sizes = set()
    try:
        with tarfile.open(tgz, "r:*") as tf:
            for member in tf:
                if not _safe_member(member):
                    continue
                base = PurePosixPath(member.name).name
                m = FRAME_RE.match(base)
                if m is None:
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                try:
                    jpeg, size = downscale_jpeg(fh.read(), max_width)
                except ValueError:                  # an empty or corrupt frame: drop it, keep the sequence
                    print(f"  dropped undecodable frame {base}", file=sys.stderr)
                    continue
                jpegs.append((base, jpeg))
                sizes.add(size)
                frames.append({"filename": base, "unix_s": int(m.group(1)), "offset_s": int(m.group(2))})
    except (tarfile.TarError, EOFError, zlib.error, OSError) as exc:
        tgz.unlink(missing_ok=True)
        raise ValueError(f"{name}: truncated or corrupt archive ({type(exc).__name__}: {exc}); deleted, "
                         "a rerun downloads it again") from exc
    seq_dir.mkdir(parents=True, exist_ok=True)       # written only once the whole archive has been read
    for base, jpeg in jpegs:
        (seq_dir / base).write_bytes(jpeg)
    if len(frames) < N_STEPS:
        raise ValueError(f"{name}: only {len(frames)} frames in the archive")
    frames.sort(key=lambda f: f["unix_s"])
    t0 = frames[0]["unix_s"]
    plume = [f["unix_s"] - f["offset_s"] for f in frames]       # unix time of the first visible plume
    record = {
        "name": name, "camera_id": camera_id(name), "mount": camera_mount(camera_id(name)),
        "n_frames": len(frames), "first_frame_unix_s": t0,
        "plume_unix_s": int(np.median(plume)), "plume_time_s": float(np.median(plume) - t0),
        "plume_clock_spread_s": float(np.max(plume) - np.min(plume)),
        "original_sizes": sorted([list(s) for s in sizes]), "max_width": max_width, "jpeg_quality": JPEG_QUALITY,
        "frames": [{**f, "timestamp_s": float(f["unix_s"] - t0)} for f in frames],
        "source": TAR_URL.format(name=name), "credit": CREDIT,
    }
    with open(seq_dir / "timestamps.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "timestamp_s", "offset_s"])
        for f in record["frames"]:
            w.writerow([f["filename"], f"{f['timestamp_s']:.3f}", f["offset_s"]])
    write_json(record_path, record)
    tgz.unlink()
    return record


# =============================================================================
# Windows and manifest
# =============================================================================


def _window(ts: np.ndarray, start: float, interval: float, max_offset: float):
    try:
        return select_window(ts, n=N_STEPS, interval_s=interval, start=start, max_offset_s=max_offset,
                             flow_lag_s=None)
    except IncompatibleFrameTiming:
        return None


def plan_windows(record: dict, interval: float, max_offset: float, n_pre: int, n_post: int,
                 post_start_s: float = 60.0) -> list:
    """[(kind, k, start_s, first_offset_s, last_offset_s)]: pre windows wholly before offset 0 (latest first,
    not overlapping), post windows starting at offset >= post_start_s (earliest first, not overlapping)."""
    ts = np.array([f["timestamp_s"] for f in record["frames"]])
    off = np.array([f["offset_s"] for f in record["frames"]], dtype=np.float64)
    out = []
    limit = np.inf                                   # pre windows: last frame before this time
    for i in range(len(ts) - 1, -1, -1):
        if sum(1 for o in out if o[0] == "pre") >= n_pre:
            break
        win = _window(ts, float(ts[i]), interval, max_offset)
        if win is None:
            continue
        sel = win["obs_idx"]
        if np.all(off[sel] < 0) and ts[sel[-1]] < limit:
            out.append(("pre", sum(1 for o in out if o[0] == "pre"), float(ts[i]), float(off[sel[0]]),
                        float(off[sel[-1]])))
            limit = ts[sel[0]]
    floor = -np.inf                                  # post windows: first frame after this time
    for i in range(len(ts)):
        if sum(1 for o in out if o[0] == "post") >= n_post:
            break
        if off[i] < post_start_s or ts[i] <= floor:
            continue
        win = _window(ts, float(ts[i]), interval, max_offset)
        if win is None:
            continue
        sel = win["obs_idx"]
        out.append(("post", sum(1 for o in out if o[0] == "post"), float(ts[i]), float(off[sel[0]]),
                    float(off[sel[-1]])))
        floor = ts[sel[-1]]
    return out


def manifest_rows(record: dict, seq_dir: Path, args) -> list:
    rows = []
    split = split_for(record["camera_id"], args.fractions, args.zero_shot)
    for kind, k, start, off0, off1 in plan_windows(record, args.interval_s, args.max_offset_s,
                                                   args.pre_windows, args.post_windows):
        rows.append({"video_path": str(seq_dir), "camera_id": record["camera_id"], "video_id": record["name"],
                     "smoke": 1 if kind == "post" else 0, "fog": "", "split": split,
                     "start_s": f"{start:.3f}", "max_windows": 1, "window_id": f"{record['name']}-{kind}{k}",
                     "plume_offset_first_s": int(off0), "plume_offset_last_s": int(off1),
                     "camera_mount": record.get("mount") or camera_mount(record["camera_id"])})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/real/figlib")
    ap.add_argument("--n", type=int, default=20,
                    help="number of sequences (divided over the splits by --fractions, spread over cameras)")
    ap.add_argument("--names", default=None, help="comma list of sequence names instead of --n")
    ap.add_argument("--max-width", type=int, default=1024, help="downscale frames to at most this width (px)")
    ap.add_argument("--interval-s", type=float, default=60.0, help="observation spacing of a window")
    ap.add_argument("--max-offset-s", type=float, default=10.0, help="frame-time tolerance of a window")
    ap.add_argument("--pre-windows", type=int, default=1, help="smoke-free windows per sequence")
    ap.add_argument("--post-windows", type=int, default=1, help="smoke windows per sequence")
    ap.add_argument("--fractions", type=lambda s: [float(v) for v in s.split(",")], default=[0.5, 0.15, 0.15, 0.2],
                    help="train,val_select,val_calib,test_iid fractions (by camera mount)")
    ap.add_argument("--zero-shot", action="store_true", help="every sequence in test_iid (no Stage B training)")
    ap.add_argument("--index-url", default=INDEX_URL)
    args = ap.parse_args(argv)
    if len(args.fractions) != len(SPLIT_NAMES) or min(args.fractions) < 0 or sum(args.fractions) <= 0:
        ap.error("--fractions needs four non-negative values (train,val_select,val_calib,test_iid)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "CREDIT.txt").write_text(CREDIT + "\n", encoding="utf-8")
    print(CREDIT)
    names = (args.names.split(",") if args.names else
             choose(list_sequences(args.index_url), args.n, args.fractions, args.zero_shot))
    rows, skipped = [], []
    for i, name in enumerate(names):
        print(f"[{i + 1}/{len(names)}] {name}", flush=True)
        try:
            record = prepare_sequence(name, out, args.max_width)
        except (OSError, ValueError, EOFError, zlib.error, tarfile.TarError, urllib.error.URLError,
                http.client.HTTPException) as exc:
            print(f"  skipped: {exc}", file=sys.stderr)
            skipped.append({"name": name, "reason": str(exc)})
            continue
        new = manifest_rows(record, out / name, args)
        if not any(r["smoke"] == 0 for r in new) or not any(r["smoke"] == 1 for r in new):
            skipped.append({"name": name, "reason": "missing a pre- or post-ignition window at the window timing",
                            "windows": len(new)})
        rows.extend(new)
    if not rows:
        raise SystemExit("no windows: nothing was written")
    with open(out / "real.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_json(out / "prepare_skipped.json", skipped)
    splits = {s: sum(r["split"] == s for r in rows) for s in SPLIT_NAMES}
    print(f"{len(rows)} windows ({sum(r['smoke'] for r in rows)} smoke) from {len({r['video_id'] for r in rows})} "
          f"sequences, {len({r['camera_id'] for r in rows})} cameras -> {out / 'real.csv'}; windows per split "
          f"{splits}; {len(skipped)} problems listed in {out / 'prepare_skipped.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
