"""Extract 12 x 14 motion features from stored synthetic sequences or from real footage.

Synthetic (re-extraction from data/<name>/seqs, e.g. to measure the mask-source shift):
    python scripts/extract_features.py --data data/synthetic --out data/synthetic/features_detector.npz \\
        --mask-source detector [--splits test_iid,test_ood]

Real footage (Stage B; the same extractor as for synthetic data):
    python scripts/extract_features.py --real-manifest real.csv --out data/real/features.npz

real.csv columns: video_path, camera_id, video_id, smoke (0/1), fog (0/1),
optional: split (train/val_select/val_calib/test_iid), start_s, stride_s
(window spacing, default 60), max_windows, mask_dir (one PNG per frame named
frame_<index>.png with a 6-digit zero-padded index, e.g. frame_000123.png,
where index is the frame's position on the clip's time line: the decoded frame
number of a video, the time-sorted image position of a folder; same size as the
frames, non-zero = candidate region; without it the image-only fallback
detector is used and recorded as mask_source).
Groups are camera ids, so one camera (and every video and window from it)
never spans two splits. Videos whose frame timing cannot provide 1 s
optical-flow pairs (e.g. one-minute HPWREN imagery) are skipped and listed in
<out>.skipped.json, as are single windows that fail (undecodable frames,
missing or mis-sized masks), with their window index.
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import json
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

from src.data.dataset import FeatureSet, load_sequence
from src.data.features import (
    FeatureConfig, IncompatibleFrameTiming, _container_timestamps_usable, extract_features_with_meta,
    load_frames, select_window,
)
from src.data.masks import detect_candidate_masks
from src.utils.io import write_json

REAL_SPLIT_FRACTIONS = (("train", 0.6), ("val_select", 0.1), ("val_calib", 0.1), ("test_iid", 0.2))
FRAME_TIME_TOL_S = 1e-3   # decoded frame times vs the selected window (same time base: ~exact)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="synthetic dataset directory (with features.npz and seqs/)")
    src.add_argument("--real-manifest", help="CSV of labelled real videos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-source", choices=["stored", "detector"], default="stored",
                    help="synthetic: use the stored candidate masks or re-detect them from the frames")
    ap.add_argument("--splits", default=None, help="synthetic: comma list of splits to process")
    ap.add_argument("--interval-s", type=float, default=5.0)
    ap.add_argument("--flow-lag-s", type=float, default=1.0)
    return ap.parse_args(argv)


def synthetic(args):
    base = Path(args.data)
    fs = FeatureSet.load(base / "features.npz")
    keep = np.ones(len(fs.X), dtype=bool)
    if args.splits:
        keep &= np.isin(fs.splits, args.splits.split(","))
    fs = fs.subset(keep)
    X = np.zeros_like(fs.X)
    sources = []
    for i, meta in enumerate(fs.meta):
        seq = load_sequence(base / meta["seq_path"])
        masks = seq["masks"]
        if args.mask_source == "detector":
            masks = detect_candidate_masks(seq["frames"], seq["timestamps"])
        fcfg = FeatureConfig(roi_scale=float(meta.get("roi_scale", 3.0)), flow_lag_s=args.flow_lag_s,
                             expected_interval_s=args.interval_s)
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


def real(args):
    rows = list(csv.DictReader(open(args.real_manifest, newline="", encoding="utf-8")))
    feats, y, seq_ids, sim_ids, groups, splits, metas, sources = [], [], [], [], [], [], [], []
    skipped = []
    fcfg = FeatureConfig(flow_lag_s=args.flow_lag_s, expected_interval_s=args.interval_s)
    for row in rows:
        path = Path(row["video_path"])
        try:
            ts = video_timestamps(path) if path.is_file() else load_frames(path)[1]
        except Exception as exc:  # unreadable video / folder without timestamps
            skipped.append({"video": str(path), "reason": str(exc)})
            continue
        # Selection and decoding share one time base: a video's times are handed to
        # load_frames as `timestamps`; a folder's are re-derived by the same loader.
        decode_ts = ts if path.is_file() else None
        start = float(row.get("start_s") or (ts[0] + args.flow_lag_s))
        stride = float(row.get("stride_s") or 60.0)
        max_w = int(row.get("max_windows") or 1000)
        k = 0
        while k < max_w:
            try:
                win = select_window(ts, n=12, interval_s=args.interval_s, start=start + k * stride,
                                    flow_lag_s=args.flow_lag_s)
            except IncompatibleFrameTiming as exc:
                if k == 0:
                    skipped.append({"video": str(path), "reason": str(exc)})
                break
            try:
                want = np.concatenate([win["obs_times"], win["lag_times"]])
                frames, got = load_frames(path, times=want, timestamps=decode_ts)
                err = float(np.max(np.abs(got - want)))
                if err > FRAME_TIME_TOL_S:
                    raise IncompatibleFrameTiming(f"decoded frame times differ from the selected window "
                                                  f"by up to {err:.3g} s")
                obs, lag = frames[:12], frames[12:]
                obs_t, lag_t = got[:12], got[12:]
                if row.get("mask_dir"):
                    masks = load_masks(Path(row["mask_dir"]), win["obs_idx"], obs.shape[1:])
                    msrc = "provided"
                else:
                    masks = detect_candidate_masks(obs, obs_t)
                    msrc = "fallback_detector"
                x, meta = extract_features_with_meta(obs, obs_t, masks, lag, lag_t, fcfg)
            except (IncompatibleFrameTiming, ValueError, OSError, cv2.error) as exc:
                skipped.append({"video": str(path), "window": k, "reason": str(exc)})
                k += 1
                continue
            vid = row.get("video_id") or path.stem
            cam = row["camera_id"]
            feats.append(x)
            y.append([float(row["smoke"]), float(row["fog"])])
            seq_ids.append(f"{vid}-w{k}")
            sim_ids.append(vid)
            groups.append(cam)
            splits.append(row.get("split") or camera_split(cam))
            sources.append(msrc)
            metas.append({"video_path": str(path), "camera_id": cam, "window": k,
                          "native_dt": float(win["native_dt"]), "obs_times": obs_t.tolist(),
                          "feature_meta": {kk: meta[kk] for kk in ("roi", "flow_mode") if kk in meta}})
            k += 1
    if not feats:
        print("no windows extracted", file=sys.stderr)
        write_json(Path(args.out).with_suffix(".skipped.json"), skipped)
        return 1
    cats = ["smoke_only" if s and not f else "fog_only" if f and not s else "both" if s and f else "neither"
            for s, f in y]
    fs = FeatureSet(X=np.stack(feats).astype(np.float32), y=np.array(y, dtype=np.float32),
                    seq_ids=np.array(seq_ids), sim_ids=np.array(sim_ids), groups=np.array(groups),
                    splits=np.array(splits), categories=np.array(cats), regimes=np.array(["real"] * len(y)),
                    tags=np.array([""] * len(y)), mask_sources=np.array(sources), meta=metas, source="real")
    fs.save(args.out)
    write_json(Path(args.out).with_suffix(".skipped.json"), skipped)
    print(json.dumps({"windows": len(feats), "videos_skipped": sum("window" not in s for s in skipped),
                      "windows_skipped": sum("window" in s for s in skipped),
                      "cameras": len(set(groups)), "splits": {s: splits.count(s) for s in set(splits)}}))
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.data:
        synthetic(args)
        return 0
    return real(args)


if __name__ == "__main__":
    sys.exit(main())
