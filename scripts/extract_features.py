"""Extract 12 x 14 motion features from stored synthetic sequences or from real footage.

Synthetic (re-extraction from data/<name>/seqs, e.g. to measure the mask-source shift):
    python scripts/extract_features.py --data data/synthetic --out data/synthetic/features_detector.npz \\
        --mask-source detector [--splits test_iid,test_ood]

    Each row is re-extracted with the extractor settings it was generated with
    (its stored feature_config and roi_scale: observation interval, flow lag
    or consecutive flow, timing tolerance); --interval-s / --flow-lag-s /
    --max-offset-s override them.

Real footage (Stage B; the same extractor as for synthetic data):
    python scripts/extract_features.py --real-manifest real.csv --artifacts artifacts --out data/real/features.npz
    python scripts/extract_features.py --real-manifest data/real/figlib/real.csv \\
        --profile configs/simulation_far.yaml --out data/real/figlib/features.npz     # no model yet

    The extractor settings come from, in order: --artifacts (the exported
    model's preprocessing.feature_config, exactly as
    PhysicsExpert.features_from_frames uses it), else --profile (a simulation
    profile's time block: frame_interval_s, flow_lag_s, max_offset_s), else
    the FeatureConfig defaults. An explicit --interval-s / --flow-lag-s /
    --max-offset-s that contradicts the exported model is an error unless
    --force-timing is given (features at another timing would be judged
    against the model's own timing anyway, and the windows would be refused).

    Two flow modes, the configuration's: "lag" (near-field: 12 observations
    `interval` apart, each with a frame `flow_lag_s` earlier; footage whose
    frame spacing cannot give such pairs, e.g. one-minute imagery, is skipped)
    and "consecutive" (flow_lag_s 0 / none, far-field one-frame-per-minute
    footage such as HPWREN / FIgLib: 12 observations `interval` apart within
    max_offset_s, no lag frames, flow between consecutive observations).

real.csv columns: video_path (a video file or an image folder with a
timestamps.csv sidecar or *_t<seconds>s.<ext> names), camera_id, video_id,
smoke, fog (0 / 1, or empty = unknown: stored as NaN and left out of training
and evaluation for that label), optional: split (train / val_select /
val_calib / test_iid), start_s, stride_s (window spacing, default 12
observation intervals), max_windows, window_id (unique id of the row's
windows; needed when one video appears in several rows, else the row's start_s
goes into the id), mask_dir (one PNG per frame named frame_<index>.png with a
6-digit zero-padded index, e.g. frame_000123.png, where index is the frame's
position on the clip's time line: the decoded frame number of a video, the
time-sorted image position of a folder; same size as the frames, non-zero =
candidate region; without it the image-only fallback detector is used and
recorded as mask_source).
Groups are camera ids, so one camera (and every video and window from it)
never spans two splits. Videos whose frame timing cannot provide the
configured windows are skipped and listed in <out>.skipped.json, as are
single windows that fail (undecodable frames, missing or mis-sized masks),
with their window index.
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import json
import sys
import zlib
from collections import Counter
from dataclasses import asdict
from dataclasses import fields as dc_fields
from pathlib import Path

import cv2
import numpy as np

from src.constants import LABEL_NAMES
from src.data.dataset import FeatureSet, label_category, load_sequence, parse_label
from src.data.feature_spec import N_STEPS
from src.data.features import (
    FeatureConfig, IncompatibleFrameTiming, _container_timestamps_usable, extract_features_with_meta,
    load_frames, select_window,
)
from src.data.masks import detect_candidate_masks
from src.utils.io import read_yaml, write_json

REAL_SPLIT_FRACTIONS = (("train", 0.6), ("val_select", 0.1), ("val_calib", 0.1), ("test_iid", 0.2))
FRAME_TIME_TOL_S = 1e-3   # decoded frame times vs the selected window (same time base: ~exact)
TIMING_KEYS = {"interval_s": "expected_interval_s", "flow_lag_s": "flow_lag_s", "max_offset_s": "max_offset_s"}


def _lag_arg(text: str) -> float:
    """--flow-lag-s value: seconds, or 0 / none / null for consecutive-observation flow."""
    return 0.0 if text.strip().lower() in ("none", "null", "consecutive") else float(text)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="synthetic dataset directory (with features.npz and seqs/)")
    src.add_argument("--real-manifest", help="CSV of labelled real videos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-source", choices=["stored", "detector"], default="stored",
                    help="synthetic: use the stored candidate masks or re-detect them from the frames")
    ap.add_argument("--splits", default=None, help="synthetic: comma list of splits to process")
    ap.add_argument("--interval-s", type=float, default=None, help="observation spacing (s)")
    ap.add_argument("--flow-lag-s", type=_lag_arg, default=None,
                    help="flow lag (s); 0 or 'none' = consecutive-observation flow (no lag frames)")
    ap.add_argument("--max-offset-s", type=float, default=None, help="frame-selection tolerance (s)")
    ap.add_argument("--artifacts", default=None,
                    help="real footage: take the extractor settings from this exported model "
                         "(preprocessing.feature_config), exactly as PhysicsExpert.features_from_frames does")
    ap.add_argument("--profile", default=None,
                    help="real footage without --artifacts: take the timing from this simulation profile "
                         "(e.g. configs/simulation_far.yaml: time.frame_interval_s / flow_lag_s / max_offset_s)")
    ap.add_argument("--force-timing", action="store_true",
                    help="allow --interval-s / --flow-lag-s / --max-offset-s to differ from the exported model")
    return ap.parse_args(argv)


def _same_timing(key: str, a, b) -> bool:
    if key == "flow_lag_s":            # None and 0 both mean consecutive-observation flow
        a, b = (0.0 if v is None else float(v) for v in (a, b))
    return abs(float(a) - float(b)) <= 1e-9


def real_feature_config(args):
    """(FeatureConfig, record) for real footage: exported model > --profile > defaults, then explicit flags.

    With --artifacts an explicit timing flag that contradicts the model's
    exported value raises SystemExit unless --force-timing (review finding F3).
    """
    known = {f.name for f in dc_fields(FeatureConfig)}
    stored, source = {}, "feature_config_defaults"
    if args.artifacts:
        cfg = read_yaml(Path(args.artifacts) / "model_config.yaml")
        stored = {k: v for k, v in ((cfg.get("preprocessing") or {}).get("feature_config") or {}).items()
                  if k in known}
        source = f"artifacts:{args.artifacts}"
    elif args.profile:
        from src.data.pipeline import feature_timing

        stored = dict(feature_timing(read_yaml(args.profile)))
        source = f"profile:{args.profile}"
    overridden, conflicts = {}, []
    for arg, key in TIMING_KEYS.items():
        v = getattr(args, arg)
        if v is None:
            continue
        if args.artifacts and key in stored and not _same_timing(key, stored[key], v):
            conflicts.append(f"{key}: model {stored[key]!r}, requested {v!r}")
        overridden[key] = v
        stored[key] = v
    if conflicts and not args.force_timing:
        raise SystemExit(f"the requested timing contradicts the exported model in {args.artifacts} ("
                         + "; ".join(conflicts) + "); features at another timing would be refused by the model. "
                         "Pass --force-timing to extract them anyway (recorded in each row's meta).")
    record = {"source": source, "overridden": overridden, "forced": bool(conflicts)}
    return FeatureConfig(**stored), record


def synthetic(args):
    base = Path(args.data)
    fs = FeatureSet.load(base / "features.npz")
    keep = np.ones(len(fs.X), dtype=bool)
    if args.splits:
        keep &= np.isin(fs.splits, args.splits.split(","))
    fs = fs.subset(keep)
    X = np.zeros_like(fs.X)
    sources = []
    known = {f.name for f in dc_fields(FeatureConfig)}
    overrides = {key: getattr(args, arg) for arg, key in TIMING_KEYS.items() if getattr(args, arg) is not None}
    for i, meta in enumerate(fs.meta):
        seq = load_sequence(base / meta["seq_path"])
        masks = seq["masks"]
        if args.mask_source == "detector":
            masks = detect_candidate_masks(seq["frames"], seq["timestamps"])
        stored = {k: v for k, v in (meta.get("feature_config") or {}).items() if k in known}
        fcfg = FeatureConfig(**{**stored, "roi_scale": float(meta.get("roi_scale", 3.0)), **overrides})
        X[i], _ = extract_features_with_meta(seq["frames"], seq["timestamps"], masks,
                                             seq["lag_frames"], seq["lag_timestamps"], fcfg)
        sources.append("detector" if args.mask_source == "detector" else fs.mask_sources[i])
    out = FeatureSet(X=X, y=fs.y, seq_ids=fs.seq_ids, sim_ids=fs.sim_ids, groups=fs.groups,
                     splits=fs.splits, categories=fs.categories, regimes=fs.regimes, tags=fs.tags,
                     mask_sources=np.array(sources), meta=fs.meta, source="synthetic")
    out.save(args.out)
    print(f"{len(X)} sequences -> {args.out}")


def camera_split(camera_id: str) -> str:
    """Deterministic camera-level split (crc32, not the salted hash())."""
    u = (zlib.crc32(camera_id.encode("utf-8")) % 10_000) / 10_000.0
    acc = 0.0
    for name, frac in REAL_SPLIT_FRACTIONS:
        acc += frac
        if u < acc:
            return name
    return REAL_SPLIT_FRACTIONS[-1][0]


def video_timestamps(path: Path) -> np.ndarray:
    """Per-frame times (s) on load_frames' time base: container times when its probe accepts
    them (they must then keep increasing, else IncompatibleFrameTiming), else frame index / fps."""
    use_container = _container_timestamps_usable(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    ts = []
    try:
        if not use_container and fps <= 0:
            raise IncompatibleFrameTiming(f"{path}: no container timestamps and no frame rate")
        while cap.grab():
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if use_container else len(ts) / fps
            if ts and not t > ts[-1]:
                raise IncompatibleFrameTiming(f"{path}: frame times stop increasing at frame {len(ts)}")
            ts.append(t)
    finally:
        cap.release()
    if not ts:
        raise ValueError(f"no frames decoded from {path}")
    return np.asarray(ts, dtype=np.float64)


def folder_timestamps(path: Path) -> np.ndarray:
    """Sorted image times (s) of a folder on load_frames' time base (sidecar or file names), without decoding."""
    try:
        from src.data.features import _IMAGE_EXTENSIONS, _folder_timestamps
    except ImportError:                     # the loader's helpers moved: decode once instead
        return load_frames(path)[1]
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"no images in {path}")
    return np.sort(_folder_timestamps(path, files)[1])


def load_masks(mask_dir: Path, frame_idx, shape) -> np.ndarray:
    """Boolean candidate masks from mask_dir/frame_<index:06d>.png for the given frame indices."""
    masks = []
    for i in frame_idx:
        f = mask_dir / f"frame_{int(i):06d}.png"
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise ValueError(f"missing or unreadable mask {f}")
        if m.shape != tuple(shape):
            raise ValueError(f"mask {f} has shape {m.shape}, frames have {tuple(shape)}")
        masks.append(m > 0)
    return np.stack(masks)


def _window_base(row: dict, vid: str, row_index: int, rows_per_video: int) -> str:
    """Id prefix of a manifest row's windows: window_id, else the video id (+ start_s when the video
    appears in several rows, so window ids never collide)."""
    if row.get("window_id"):
        return str(row["window_id"])
    if rows_per_video > 1:
        return f"{vid}-s{row['start_s']}" if row.get("start_s") else f"{vid}-r{row_index}"
    return vid


def real(args):
    with open(args.real_manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    feats, y, seq_ids, sim_ids, groups, splits, metas, sources = [], [], [], [], [], [], [], []
    skipped = []
    fcfg, timing_record = real_feature_config(args)
    consecutive = fcfg.flow_mode == "consecutive"
    lag = None if consecutive else float(fcfg.flow_lag_s)
    interval = float(fcfg.expected_interval_s)
    print(f"extractor: interval {interval} s, {'consecutive flow (no lag frames)' if consecutive else f'lag {lag} s'}, "
          f"max offset {fcfg.max_offset_s} s ({timing_record['source']})", file=sys.stderr)
    rows_per_path = Counter(str(Path(r["video_path"])) for r in rows)
    times_cache = {}
    seen = set()
    for r_i, row in enumerate(rows):
        path = Path(row["video_path"])
        key = str(path)
        try:
            labels = [parse_label(row.get("smoke")), parse_label(row.get("fog"))]
        except ValueError as exc:
            skipped.append({"video": key, "row": r_i, "reason": f"bad label: {exc}"})
            continue
        try:
            if key not in times_cache:
                times_cache[key] = video_timestamps(path) if path.is_file() else folder_timestamps(path)
            ts = times_cache[key]
        except Exception as exc:  # unreadable video / folder without timestamps
            skipped.append({"video": key, "row": r_i, "reason": str(exc)})
            continue
        # Selection and decoding share one time base: a video's times are handed to
        # load_frames as `timestamps`; a folder's are re-derived by the same loader.
        decode_ts = ts if path.is_file() else None
        start = float(row["start_s"]) if row.get("start_s") not in (None, "") else float(ts[0]) + (lag or 0.0)
        stride = float(row.get("stride_s") or N_STEPS * interval)
        max_w = int(row.get("max_windows") or 1000)
        vid = row.get("video_id") or path.stem
        base = _window_base(row, vid, r_i, rows_per_path[key])
        k = 0
        while k < max_w:
            try:
                win = select_window(ts, n=N_STEPS, interval_s=interval, start=start + k * stride,
                                    max_offset_s=float(fcfg.max_offset_s), flow_lag_s=lag)
            except IncompatibleFrameTiming as exc:
                if k == 0:
                    skipped.append({"video": key, "row": r_i, "reason": str(exc)})
                break
            try:
                want = win["obs_times"] if consecutive else np.concatenate([win["obs_times"], win["lag_times"]])
                frames, got = load_frames(path, times=want, timestamps=decode_ts)
                err = float(np.max(np.abs(got - want)))
                if err > FRAME_TIME_TOL_S:
                    raise IncompatibleFrameTiming(f"decoded frame times differ from the selected window "
                                                  f"by up to {err:.3g} s")
                obs, obs_t = frames[:N_STEPS], got[:N_STEPS]
                lag_frames, lag_t = (None, None) if consecutive else (frames[N_STEPS:], got[N_STEPS:])
                if row.get("mask_dir"):
                    masks = load_masks(Path(row["mask_dir"]), win["obs_idx"], obs.shape[1:])
                    msrc = "provided"
                else:
                    masks = detect_candidate_masks(obs, obs_t)
                    msrc = "fallback_detector"
                x, meta = extract_features_with_meta(obs, obs_t, masks, lag_frames, lag_t, fcfg)
            except (IncompatibleFrameTiming, ValueError, OSError, cv2.error) as exc:
                skipped.append({"video": key, "row": r_i, "window": k, "reason": str(exc)})
                k += 1
                continue
            seq_id = f"{base}-w{k}"
            if seq_id in seen:
                raise SystemExit(f"duplicate window id {seq_id!r} (row {r_i}): give the manifest rows of "
                                 f"video {vid!r} distinct window_id or start_s values")
            seen.add(seq_id)
            cam = row["camera_id"]
            feats.append(x)
            y.append(labels)
            seq_ids.append(seq_id)
            sim_ids.append(vid)
            groups.append(cam)
            splits.append(row.get("split") or camera_split(cam))
            sources.append(msrc)
            metas.append({"video_path": key, "camera_id": cam, "window": k, "window_id": base,
                          "manifest_row": r_i, "native_dt": float(win["native_dt"]),
                          "obs_times": obs_t.tolist(), "flow_mode": meta.get("flow_mode"),
                          "labels_known": {n: bool(np.isfinite(v)) for n, v in zip(LABEL_NAMES, labels)},
                          "feature_config": asdict(fcfg), "timing_source": timing_record,
                          "feature_meta": {kk: meta[kk] for kk in ("roi", "flow_mode", "timing") if kk in meta},
                          "stabilization": {kk: (meta.get("stabilization") or {}).get(kk)
                                            for kk in ("applied", "reason", "textured_frac", "n_corners")}})
            k += 1
    if not feats:
        print("no windows extracted", file=sys.stderr)
        write_json(Path(args.out).with_suffix(".skipped.json"), skipped)
        return 1
    yy = np.array(y, dtype=np.float32)
    cats = [label_category(s, f) for s, f in yy]
    fs = FeatureSet(X=np.stack(feats).astype(np.float32), y=yy,
                    seq_ids=np.array(seq_ids), sim_ids=np.array(sim_ids), groups=np.array(groups),
                    splits=np.array(splits), categories=np.array(cats), regimes=np.array(["real"] * len(y)),
                    tags=np.array([""] * len(y)), mask_sources=np.array(sources), meta=metas, source="real")
    fs.save(args.out)
    write_json(Path(args.out).with_suffix(".skipped.json"), skipped)
    print(json.dumps({"windows": len(feats), "videos_skipped": sum("window" not in s for s in skipped),
                      "windows_skipped": sum("window" in s for s in skipped),
                      "cameras": len(set(groups)), "splits": {s: splits.count(s) for s in sorted(set(splits))},
                      "known_labels": {n: int(fs.known[:, j].sum()) for j, n in enumerate(LABEL_NAMES)},
                      "flow_mode": fcfg.flow_mode}))
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.data:
        synthetic(args)
        return 0
    return real(args)


if __name__ == "__main__":
    sys.exit(main())
