"""Real-vs-synthetic image statistics: is the renderer tuned to real footage?

    python scripts/image_stats.py --real-manifest data/real/firesense/real.csv --synthetic data/synthetic \\
        --out reports/image_stats_firesense.json [--n-synthetic 200] [--max-frames 12]

The motion features only transfer to real cameras if the rendered frames look
like camera frames in the ways the extractor is sensitive to: noise, contrast,
compression, flicker and camera shake. This script measures robust grayscale
statistics per clip on real footage and on a generated synthetic dataset and
reports, per statistic, the percentiles of both, the fraction of real clips
outside the synthetic [P1, P99] range and a plain-language note on which
`render:` ranges of the simulation config to widen.

Statistics (per frame, then the median over the clip's frames; the last two per clip):

* noise_sigma (gray levels): Gaussian-equivalent noise std. High-pass residual
  with Immerkaer's 3x3 kernel (the outer product of two second differences:
  planes, ramps and axis-aligned edges cancel; white noise of std s gives std
  6 s), then 1.4826 median|residual| / 6 over unsaturated pixels. The
  residual of a uint8 frame is integer-valued, so the median is interpolated
  within its unit bin; a plain median (or the integer image-minus-3x3-median
  residual) would quantise the estimate in steps of a quarter of a gray level
  or more, too coarse for the 1-3 gray levels of real cameras. It measures the
  finest-scale noise left in the frame, so on compressed frames it is what the
  codec kept (JPEG removes most of it), on real and synthetic frames alike.
* contrast: (P95 - P5) / 255 of the gray values.
* brightness: mean / 255.
* sharpness: variance of the 3x3 Laplacian / (P95 - P5)^2 (fine detail and
  noise relative to the contrast; NaN on nearly flat frames).
* blockiness: mean |gradient| across the 8x8 block boundaries / mean |gradient|
  inside the blocks (about 1 uncompressed, above 1 with JPEG / MPEG blocking).
* flicker: sample std of the frame means / their mean (exposure flicker plus
  any scene change, so an upper bound of the camera's own flicker).
* motion_px: median |phase-correlation shift| between consecutive sampled
  frames, px per frame interval (camera-shake proxy; pans and large moving
  structures also count), over the pairs that register (peak response >=
  MIN_PHASE_RESPONSE). A pair that does not (scene change, frames with little
  in common) gives the position of a noise peak, tens of px at random, so it
  is left out; motion_registered_frac is the fraction of pairs kept. A noise
  peak can still clear the cutoff now and then, so a clip gets a motion value
  only when more than half of its pairs register (MIN_REGISTERED_FRAC);
  otherwise NaN: such clips drop out of the motion comparison and are counted
  separately.

Real input: a manifest in the scripts/extract_features.py --real-manifest
format (only video_path is required; camera_id, video_id, smoke, fog and
start_s are used when present). video_path is a video file or an image folder,
read with src.data.features.load_frames on the extractor's time base. At most
--max-frames (K) frames are sampled per clip, --interval-s apart (default: the
synthetic observation spacing, time.frame_interval_s), so flicker and motion
are measured over the same interval as the synthetic frames; a clip too short
for --min-frames such frames is sampled evenly instead, and a clip sampled
evenly or left with fewer than --min-frames distinct frames has its flicker
and motion left out of the comparison. When the real spacing differs from the
synthetic one by more than INTERVAL_TOL, the script warns and compares no
flicker or motion at all (verdict no_data). Synthetic input: --n-synthetic (M)
sequences drawn (seeded) from <data>/seqs/<split>/*.npz of --splits (default
every split but test_ood, whose conditions are shifted on purpose), their
first K observation frames.

Resolution: real frames are measured at their native size and again after
resizing to the synthetic frame size (render.native_size from
<data>/config_used.yaml; the same pixel count with the aspect ratio kept, so a
pan is not rescaled differently along the two axes). The resized values are
the comparable ones (noise, sharpness and shift in px depend on the pixel
scale), except blockiness, which exists only on the encoder's native 8x8 grid
(the renderer, like a camera, compresses at its native size): it is compared
at native resolution.

Outputs: <out> (JSON: settings, per-clip values, percentiles, coverage,
notes, and on the synthetic set the rank correlation of each statistic with
the render parameter meant to drive it) and <out>.md. A diagnostic of the
rendering, never a model input; with a handful of real clips the coverage
fractions are coarse (one clip of 20 is 5 %).
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import math
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from src.data.features import IncompatibleFrameTiming, load_frames
from src.utils.io import read_yaml, write_json

STATS = ("noise_sigma", "contrast", "brightness", "sharpness", "blockiness", "flicker", "motion_px")
FRAME_STATS = STATS[:5]        # per frame, median over the clip
TEMPORAL_STATS = STATS[5:]     # per clip, need frames at the comparison interval
PERCENTILES = (5, 25, 50, 75, 95)
COVER_PCT = (1.0, 99.0)        # the synthetic range a real clip should fall in
OUTSIDE_FLAG = 0.10            # more real clips than this outside the range: widen
BLOCK = 8                      # JPEG / MPEG transform block (px)
MAD_TO_SIGMA = 1.4826          # median |x| -> std of a zero-mean normal
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
_IMMERKAER_GAIN = 6.0          # sqrt of the sum of squared kernel weights
MIN_NOISE_PX = 64              # unsaturated pixels needed for a noise estimate
MIN_SPREAD_GRAY = 2.0          # P95 - P5 below this: sharpness undefined
MIN_MEAN_GRAY = 1.0            # mean frame brightness below this: flicker undefined
INTERVAL_TOL = 0.2             # achieved / requested interval within 1 +- this: temporal stats comparable
MIN_PHASE_RESPONSE = 0.15      # phase-correlation peak below this: the pair did not register (shift is noise)
MIN_REGISTERED_FRAC = 0.5      # a clip's motion counts only when more than this fraction of its pairs register
DEFAULT_SPLITS = "train,val_select,val_calib,test_iid"   # synthetic splits sampled (test_ood is shifted on purpose)
MD_MAX_CLIPS = 60              # per-clip rows in the Markdown report (all of them in the JSON)
SHAKE_SHIFT_PER_STD = math.sqrt(2.0) * math.sqrt(2.0 * math.log(2.0))
# ^ median |2D shift| between two frames with independent shake / per-axis shake std (about 1.67)

STAT_INFO = {
    "noise_sigma": {"unit": "gray levels", "basis": "resized"},
    "contrast": {"unit": "(P95 - P5) / 255", "basis": "resized"},
    "brightness": {"unit": "mean / 255", "basis": "resized"},
    "sharpness": {"unit": "var(Laplacian) / (P95 - P5)^2", "basis": "resized"},
    "blockiness": {"unit": "boundary / interior gradient", "basis": "native"},
    "flicker": {"unit": "std / mean of frame means", "basis": "resized"},
    "motion_px": {"unit": "px per frame interval", "basis": "resized"},
}
# Render parameters kept per synthetic sequence (for the tracking check) and config ranges quoted in notes.
RENDER_KEYS = ("noise_sigma", "contrast", "global_haze", "brightness_offset", "jpeg", "jpeg_quality",
               "exposure_flicker", "camera_shake_px", "camera_shake_tau_s")
CONFIG_KEYS = ("noise_sigma", "contrast", "global_haze", "brightness_offset", "jpeg_quality", "jpeg_prob",
               "exposure_flicker", "camera_shake_px", "camera_shake_tau_s", "background.sky_brightness",
               "background.ground_brightness", "texture.gain", "strong_artifact.exposure_flicker",
               "strong_artifact.camera_shake_px")
CONFIG_UNITS = {"camera_shake_tau_s": " s"}
# (statistic, render parameter it should follow); jpeg_strength = 100 - quality for compressed clips, else 0.
TRACKING = (("noise_sigma", "noise_sigma"), ("contrast", "contrast"), ("contrast", "global_haze"),
            ("brightness", "brightness_offset"), ("sharpness", "noise_sigma"), ("blockiness", "jpeg_strength"),
            ("flicker", "exposure_flicker"), ("motion_px", "camera_shake_px"))

ADVICE = {
    "noise_sigma": {
        "high": "Real frames are noisier (real P95 {r95:.2f} gray levels vs synthetic P99 {s99:.2f}). Raise the "
                "upper end of render.noise_sigma (now {noise_sigma}; the real P95 is {r95_fs:.3f} of full scale, "
                "and JPEG removes part of the rendered noise, so go somewhat above it), or compress fewer clips "
                "(render.jpeg_prob {jpeg_prob}).",
        "low": "Real frames are cleaner (real P5 {r5:.2f} gray levels vs synthetic P1 {s1:.2f}): in-camera "
               "denoising and video codecs remove fine noise. Lower the lower end of render.noise_sigma (now "
               "{noise_sigma}; the real P5 is {r5_fs:.4f} of full scale) and/or compress more often or harder "
               "(render.jpeg_prob {jpeg_prob}, render.jpeg_quality {jpeg_quality}).",
    },
    "contrast": {
        "high": "Real frames span a wider gray range (real P95 {r95:.2f} vs synthetic P99 {s99:.2f}). Raise the "
                "upper end of render.contrast (now {contrast}) and/or lower render.global_haze (now {global_haze}), "
                "which compresses the range; the scene's own range comes from render.background.sky_brightness / "
                "ground_brightness (now {background_sky_brightness} / {background_ground_brightness}).",
        "low": "Real frames are flatter (real P5 {r5:.2f} vs synthetic P1 {s1:.2f}; haze, fog, overcast light). "
               "Lower the lower end of render.contrast (now {contrast}) and/or raise the upper end of "
               "render.global_haze (now {global_haze}) or narrow the gap between render.background.sky_brightness "
               "and ground_brightness (now {background_sky_brightness} / {background_ground_brightness}).",
    },
    "brightness": {
        "high": "Real frames are brighter (real P95 {r95:.2f} vs synthetic P99 {s99:.2f}). Widen "
                "render.brightness_offset upward (now {brightness_offset}) or raise "
                "render.background.sky_brightness / ground_brightness (now {background_sky_brightness} / "
                "{background_ground_brightness}).",
        "low": "Real frames are darker (real P5 {r5:.2f} vs synthetic P1 {s1:.2f}). Widen "
               "render.brightness_offset downward (now {brightness_offset}) or lower "
               "render.background.sky_brightness / ground_brightness (now {background_sky_brightness} / "
               "{background_ground_brightness}).",
    },
    "sharpness": {
        "high": "Real frames carry more fine detail relative to their contrast (real P95 {r95:.3g} vs synthetic "
                "P99 {s99:.3g}). No render parameter sets blur directly: sharpness rises with render.noise_sigma "
                "(now {noise_sigma}) and texture detail (render.texture.gain {texture_gain}) and falls with JPEG "
                "compression, so raise the upper end of render.jpeg_quality (now {jpeg_quality}) or lower "
                "render.jpeg_prob ({jpeg_prob}).",
        "low": "Real frames are softer (real P5 {r5:.3g} vs synthetic P1 {s1:.3g}; defocus, motion blur, haze, "
               "codec smoothing). Compress harder or more often (render.jpeg_quality {jpeg_quality}, "
               "render.jpeg_prob {jpeg_prob}); a defocus blur would need a new render parameter.",
    },
    "blockiness": {
        "high": "Real frames are blockier (real P95 {r95:.3f} vs synthetic P99 {s99:.3f}, native resolution). "
                "Lower the lower end of render.jpeg_quality (now {jpeg_quality}) and/or raise render.jpeg_prob "
                "(now {jpeg_prob}).",
        "low": "Real frames are less blocky (real P5 {r5:.3f} vs synthetic P1 {s1:.3f}, native resolution; "
               "deblocking codecs such as H.264, or high-quality encoding). Raise the upper end of "
               "render.jpeg_quality (now {jpeg_quality}) and/or lower render.jpeg_prob (now {jpeg_prob}).",
    },
    "flicker": {
        "high": "Real clips flicker more (real P95 {r95:.3f} vs synthetic P99 {s99:.3f}). Raise the upper end of "
                "render.exposure_flicker (now {exposure_flicker}; strong artefacts "
                "{strong_artifact_exposure_flicker}): the renderer's per-frame gain gives a flicker statistic "
                "close to its exposure_flicker value. Scene changes (moving smoke, passing objects, auto-exposure) "
                "also count, so the real values are an upper bound.",
        "low": "Real clips are steadier than the synthetic ones (real P5 {r5:.3f} vs synthetic P1 {s1:.3f}). "
               "{flicker_floor}",
    },
    "motion_px": {
        "high": "Real cameras move more between frames (real P95 {r95:.2f} px vs synthetic P99 {s99:.2f} px at the "
                "synthetic frame size, per {interval:g} s). Raise the upper end of render.camera_shake_px (now "
                "{camera_shake_px}; strong artefacts {strong_artifact_camera_shake_px}): with frames farther apart "
                "than render.camera_shake_tau_s ({camera_shake_tau_s}) a per-axis shake std s gives a median "
                "shift of about {k_shake:.2f} s, so the real P95 corresponds to s = {r95_shake:.2f} px. Pans and "
                "large moving structures also count here.",
        "low": "Real cameras are steadier (real P5 {r5:.2f} px vs synthetic P1 {s1:.2f} px per {interval:g} s). "
               "{shake_floor}",
    },
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real-manifest", required=True, help="CSV in the extract_features.py --real-manifest format")
    ap.add_argument("--synthetic", required=True, help="synthetic dataset directory (seqs/, config_used.yaml)")
    ap.add_argument("--out", required=True, help="JSON report; the Markdown goes next to it (.md)")
    ap.add_argument("--n-synthetic", type=int, default=200, help="synthetic sequences sampled (M)")
    ap.add_argument("--max-frames", type=int, default=12, help="frames per clip at most (K)")
    ap.add_argument("--interval-s", type=float, default=None,
                    help="real frame spacing (default: the synthetic time.frame_interval_s)")
    ap.add_argument("--min-frames", type=int, default=3,
                    help="a real clip with fewer frames at --interval-s is sampled evenly instead; flicker and "
                         "motion need at least this many distinct frames")
    ap.add_argument("--splits", default=DEFAULT_SPLITS,
                    help="comma list of synthetic splits to sample from (default: every split but test_ood; "
                         "'all' for every split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="title of the report (default: the manifest's folder)")
    return ap.parse_args(argv)


# =============================================================================
# Image statistics
# =============================================================================


def _check_u8(img) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype != np.uint8 or a.ndim != 2 or min(a.shape) < 3:
        raise ValueError(f"expected a grayscale uint8 frame (H, W) with H, W >= 3, got {a.dtype} {a.shape}")
    return a


def _grouped_median(q: np.ndarray) -> float:
    """Median of non-negative integers read as a continuous variable: |x| in [k - 0.5, k + 0.5) maps to k
    ([0, 0.5) to 0), and the median is interpolated linearly inside its bin."""
    counts = np.bincount(q)
    cdf = np.cumsum(counts)
    half = 0.5 * float(cdf[-1])
    k = int(np.searchsorted(cdf, half))
    below = float(cdf[k - 1]) if k > 0 else 0.0
    lo, width = (k - 0.5, 1.0) if k > 0 else (0.0, 0.5)
    return lo + (half - below) / float(counts[k]) * width


def noise_sigma(img) -> float:
    """Gaussian-equivalent noise std (gray levels) of a uint8 frame; NaN when nearly all of it is clipped.

    See the module docstring: Immerkaer's kernel, then a grouped MAD about zero
    over pixels whose 3x3 neighbourhood is not at 0 or 255 (clipped noise is
    not noise the camera lacks).
    """
    a = _check_u8(img)
    r = cv2.filter2D(a.astype(np.float32), cv2.CV_32F, _IMMERKAER, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    clipped = ((a == 0) | (a == 255)).astype(np.uint8)
    ok = cv2.dilate(clipped, np.ones((3, 3), np.uint8))[1:-1, 1:-1] == 0
    if int(ok.sum()) < MIN_NOISE_PX:
        return math.nan
    q = np.abs(np.rint(r[ok])).astype(np.int64)    # exact integers on uint8 input
    return MAD_TO_SIGMA * _grouped_median(q) / _IMMERKAER_GAIN


def _u8_percentiles(a: np.ndarray, qs) -> list:
    """Lower percentiles of a uint8 image: the smallest value whose CDF reaches q %."""
    cdf = np.cumsum(np.bincount(a.ravel(), minlength=256)) / float(a.size)
    return [int(np.searchsorted(cdf, q / 100.0 - 1e-12)) for q in qs]


def blockiness(img, block: int = BLOCK) -> float:
    """Mean |gradient| across block boundaries / mean |gradient| inside blocks (both axes; grid at pixel 0)."""
    a = _check_u8(img).astype(np.int16)
    on_sum = off_sum = 0.0
    for d in (np.abs(np.diff(a, axis=1)), np.abs(np.diff(a, axis=0)).T):
        if d.shape[1] < 2 * block:
            continue
        # column j of d lies between pixels j and j + 1: a block boundary when j % block == block - 1
        on = np.zeros(d.shape[1], dtype=bool)
        on[block - 1::block] = True
        on_sum += float(d[:, on].mean())
        off_sum += float(d[:, ~on].mean())
    return on_sum / off_sum if off_sum > 1e-9 else math.nan


def frame_stats(img) -> dict:
    """The per-frame statistics of one uint8 grayscale frame, plus its mean gray level ("mean")."""
    a = _check_u8(img)
    p5, p95 = _u8_percentiles(a, (5, 95))
    spread = float(p95 - p5)
    lap = cv2.Laplacian(a, cv2.CV_32F)[1:-1, 1:-1]       # 3x3 aperture: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    mean = float(a.mean())
    return {
        "noise_sigma": noise_sigma(a),
        "contrast": spread / 255.0,
        "brightness": mean / 255.0,
        "sharpness": float(lap.var()) / spread ** 2 if spread >= MIN_SPREAD_GRAY else math.nan,
        "blockiness": blockiness(a),
        "mean": mean,
    }


@lru_cache(maxsize=8)
def _hanning(h: int, w: int) -> np.ndarray:
    return cv2.createHanningWindow((w, h), cv2.CV_64F)


def global_shift(a, b) -> tuple:
    """(dx, dy, response): phase-correlation shift of frame b relative to frame a (px) and its peak response."""
    a, b = _check_u8(a), _check_u8(b)
    (dx, dy), resp = cv2.phaseCorrelate(a.astype(np.float64), b.astype(np.float64), _hanning(*a.shape))
    return float(dx), float(dy), float(resp)


def _finite_median(values) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else math.nan


def clip_stats(frames, timestamps=None) -> dict:
    """Statistics of one clip (T, H, W) uint8: per-frame statistics as medians over the frames, flicker over
    the frame means, motion over the consecutive pairs that register; NaN where undefined (e.g. one frame)."""
    frames = np.asarray(frames)
    if frames.ndim != 3 or len(frames) == 0:
        raise ValueError(f"expected frames (T, H, W), got shape {frames.shape}")
    per = [frame_stats(f) for f in frames]
    out = {k: _finite_median([p[k] for p in per]) for k in FRAME_STATS}
    means = np.array([p["mean"] for p in per])
    ok = len(means) >= 2 and means.mean() >= MIN_MEAN_GRAY
    out["flicker"] = float(means.std(ddof=1) / means.mean()) if ok else math.nan
    shifts = [global_shift(frames[i - 1], frames[i]) for i in range(1, len(frames))]
    # Only registered pairs: an unregistered pair's "shift" is wherever the strongest noise peak lies.
    registered = [math.hypot(dx, dy) for dx, dy, r in shifts if r >= MIN_PHASE_RESPONSE]
    out["motion_registered_frac"] = len(registered) / len(shifts) if shifts else math.nan
    out["motion_px"] = (_finite_median(registered) if out["motion_registered_frac"] > MIN_REGISTERED_FRAC
                        else math.nan)
    out["phase_response"] = _finite_median([r for _, _, r in shifts])
    ts = None if timestamps is None else np.asarray(timestamps, dtype=np.float64).ravel()
    out["interval_s"] = float(np.median(np.diff(ts))) if ts is not None and ts.size >= 2 else math.nan
    out["n_frames"] = int(len(frames))
    return out


def resize_to_pixels(frames, size) -> tuple:
    """(frames resized to the pixel count of `size` (H, W) with their aspect ratio kept, scale factor)."""
    frames = np.asarray(frames)
    _, H, W = frames.shape
    s = math.sqrt(float(size[0]) * float(size[1]) / float(H * W))
    w, h = max(2 * BLOCK, int(round(W * s))), max(2 * BLOCK, int(round(H * s)))
    if (h, w) == (H, W):
        return frames.copy(), 1.0
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    return np.stack([cv2.resize(f, (w, h), interpolation=interp) for f in frames]), s


def sample_times(t0: float, t1: float, interval_s: float, k: int, min_frames: int) -> tuple:
    """(target times, mode): at most k targets interval_s apart from t0 ("interval") when [t0, t1] holds at
    least min_frames of them, else k targets spread evenly over it ("even")."""
    fit = int(math.floor((t1 - t0) / interval_s + 1e-9)) + 1 if t1 >= t0 else 0
    if fit >= min_frames:
        return t0 + interval_s * np.arange(min(fit, k), dtype=np.float64), "interval"
    return np.linspace(t0, t1, max(k, 1)), "even"


# =============================================================================
# Inputs
# =============================================================================


def _resolve(path_text: str, manifest: Path) -> Path:
    """The manifest path as written (relative to the working directory), else relative to the manifest."""
    p = Path(path_text)
    if p.exists() or p.is_absolute():
        return p
    alt = manifest.parent / p
    return alt if alt.exists() else p


def real_clip_frames(path: Path, interval_s: float, k: int, min_frames: int, start_s=None) -> tuple:
    """(frames (n, H, W) uint8, times (n,), mode) of one real clip, read with load_frames.

    Videos: the extractor's per-frame times (container times when usable, else
    index / fps), handed to load_frames as `timestamps` so selection and
    decoding share one time base. Image folders: load_frames' own timestamps
    (sidecar CSV or filename pattern); the first and last are probed by
    requesting the frames nearest to -inf and +inf. Repeated frames (targets
    closer than the native spacing) are dropped.
    """
    if path.is_file():
        from extract_features import video_timestamps   # lazy: the script imports the dataset stack

        ts = video_timestamps(path)
        t0, t1, decode_ts = float(ts[0]), float(ts[-1]), ts
    elif path.is_dir():
        _, probe = load_frames(path, times=[-1e12, 1e12])
        t0, t1, decode_ts = float(probe[0]), float(probe[-1]), None
    else:
        raise FileNotFoundError(f"no video file or image folder at {path}")
    if start_s not in (None, ""):
        t0 = min(max(t0, float(start_s)), t1)
    targets, mode = sample_times(t0, t1, interval_s, k, min_frames)
    frames, got = load_frames(path, times=targets, timestamps=decode_ts)
    _, first = np.unique(got, return_index=True)
    keep = np.sort(first)
    return frames[keep], got[keep], mode


def real_records(manifest: Path, size, interval_s: float, k: int, min_frames: int) -> tuple:
    """Per-clip statistics (native and resized to `size`) of every manifest row, and the skipped rows."""
    with open(manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    records, skipped = [], []
    for row in rows:
        path = _resolve(row["video_path"], manifest)
        try:
            frames, ts, mode = real_clip_frames(path, interval_s, k, min_frames, row.get("start_s"))
            native = clip_stats(frames, ts)
            small, scale = resize_to_pixels(frames, size)
            resized = clip_stats(small, ts)
        except (IncompatibleFrameTiming, ValueError, OSError, cv2.error) as exc:
            skipped.append({"video": str(path), "reason": str(exc)})
            continue
        achieved = native["interval_s"]
        temporal = (mode == "interval" and len(frames) >= max(2, min_frames) and math.isfinite(achieved)
                    and abs(achieved / interval_s - 1.0) <= INTERVAL_TOL)
        records.append({
            "video_path": str(path), "camera_id": row.get("camera_id") or "",
            "video_id": row.get("video_id") or path.stem,
            "labels": {n: row.get(n) for n in ("smoke", "fog") if row.get(n) not in (None, "")},
            "native_size": [int(v) for v in frames.shape[1:]], "resized_size": [int(v) for v in small.shape[1:]],
            "resize_scale": scale, "sampling": mode, "temporal_comparable": bool(temporal),
            "native": native, "resized": resized,
        })
    return records, skipped


def synthetic_records(data_dir: Path, n: int, k: int, seed: int, splits=None) -> tuple:
    """Per-sequence statistics of n sequences drawn (seeded) from data_dir/seqs/*/*.npz, and how many exist."""
    from src.data.dataset import load_sequence

    files = sorted((data_dir / "seqs").glob("*/*.npz"))
    if splits:
        files = [f for f in files if f.parent.name in splits]
    if not files:
        raise SystemExit(f"no sequences under {data_dir / 'seqs'}" + (f" in splits {splits}" if splits else ""))
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(files), size=min(n, len(files)), replace=False))
    records = []
    for i in pick:
        seq = load_sequence(files[i])
        frames = np.asarray(seq["frames"])[:k]
        ts = np.asarray(seq["timestamps"], dtype=np.float64)[:k]
        rp = seq.get("render_params") or {}
        params = {key: rp.get(key) for key in RENDER_KEYS}
        params["jpeg_strength"] = (100.0 - float(rp["jpeg_quality"])) if rp.get("jpeg") else 0.0
        records.append({
            "seq_id": seq.get("seq_id") or files[i].stem, "split": files[i].parent.name,
            "category": seq.get("category"), "path": str(files[i].relative_to(data_dir)),
            "native_size": [int(v) for v in frames.shape[1:]], "stats": clip_stats(frames, ts),
            "render_params": params,
        })
    return records, len(files)


# =============================================================================
# Comparison
# =============================================================================


def _values(records, key: str, stat: str, temporal_only: bool = False) -> np.ndarray:
    v = [r[key][stat] for r in records if not (temporal_only and not r.get("temporal_comparable", True))]
    v = np.asarray(v, dtype=np.float64)
    return v[np.isfinite(v)]


def _n_unregistered(records, key: str) -> int:
    """Records (temporal-comparable ones) whose pairs mostly fail to register: no motion value."""
    def unregistered(frac) -> bool:
        return frac is not None and math.isfinite(frac) and frac <= MIN_REGISTERED_FRAC

    return sum(1 for r in records
               if r.get("temporal_comparable", True) and unregistered(r[key].get("motion_registered_frac")))


def spacing_matches(real_s: float, synthetic_s: float, tol: float = INTERVAL_TOL) -> bool:
    """False when both frame spacings are known and differ by more than tol (relative to the synthetic one):
    flicker and motion depend on the spacing, so they are then not compared."""
    if not (math.isfinite(real_s) and math.isfinite(synthetic_s) and synthetic_s > 0):
        return True
    return abs(real_s / synthetic_s - 1.0) <= tol


def _summary(v: np.ndarray) -> dict:
    out = {"n": int(v.size)}
    out.update({f"p{p}": (float(np.percentile(v, p)) if v.size else None) for p in PERCENTILES})
    return out


def _coverage(v: np.ndarray, lo, hi) -> dict:
    if not v.size or lo is None:
        return {"below": None, "above": None, "outside": None}
    below, above = float(np.mean(v < lo)), float(np.mean(v > hi))
    return {"below": below, "above": above, "outside": below + above}


def verdict(cov: dict, flag: float = OUTSIDE_FLAG) -> str:
    """"no_data", "covered", or "widen_low" / "widen_high" / "widen_both" (sides with more than flag / 2)."""
    if cov.get("outside") is None:
        return "no_data"
    if cov["outside"] <= flag:
        return "covered"
    low, high = cov["below"] > flag / 2.0, cov["above"] > flag / 2.0
    return "widen_both" if low and high else "widen_low" if low else "widen_high"


def _cfg_value(render_cfg: dict, dotted: str) -> str:
    v = render_cfg
    for part in dotted.split("."):
        v = v.get(part) if isinstance(v, dict) else None
    if v is None:
        return "unset"
    if isinstance(v, (list, tuple)):
        text = "[" + ", ".join(f"{float(x):g}" for x in v) + "]"
    else:
        text = f"{float(v):g}" if isinstance(v, (int, float)) else str(v)
    return text + CONFIG_UNITS.get(dotted, "")


def _starts_at_zero(render_cfg: dict, key: str) -> bool:
    v = render_cfg.get(key)
    return isinstance(v, (list, tuple)) and len(v) > 0 and float(v[0]) <= 0.0


def advice(stat: str, entry: dict, render_cfg: dict, interval_s: float, flag: float = OUTSIDE_FLAG) -> str:
    """Plain-language note for one statistic: covered, or which render ranges to widen and by about how much."""
    basis = entry["comparable_basis"]
    real, syn = entry[f"real_{basis}"], entry["synthetic"]
    v = entry["verdict"]
    if v == "no_data":
        why = ("no real clip has frames at the comparison spacing" if stat in TEMPORAL_STATS
               else "no real or synthetic values")
        if (entry.get("unregistered") or {}).get("real") and not real["n"]:
            why = "no real clip with frames at the comparison spacing has a frame pair that registers"
        return f"No comparison: {why}."
    if v == "covered":
        return (f"Covered: {real['outside']:.0%} of {real['n']} real clips fall outside the synthetic P1-P99 range "
                f"[{syn['p1']:.3g}, {syn['p99']:.3g}]; no change needed.")
    fmt = {key.replace(".", "_"): _cfg_value(render_cfg, key) for key in CONFIG_KEYS}
    # Low-side notes for ranges that may already start at 0 (then there is no lower end to lower).
    fmt["flicker_floor"] = (
        "render.exposure_flicker already starts at 0, so the synthetic floor comes from the scenes themselves "
        "(aerosol growing or drifting through the view changes the frame mean); nothing to widen in the camera "
        "model." if _starts_at_zero(render_cfg, "exposure_flicker") else
        f"Lower the lower end of render.exposure_flicker (now {fmt['exposure_flicker']}).")
    fmt["shake_floor"] = (
        f"render.camera_shake_px already starts at 0 (now {fmt['camera_shake_px']}) but is drawn uniformly, so "
        "few synthetic clips are as steady as a rigid mount: make near-zero shake more common or lengthen "
        f"render.camera_shake_tau_s (now {fmt['camera_shake_tau_s']}) so that frames stay correlated; part of the "
        "synthetic floor can also be aerosol motion that the phase correlation follows."
        if _starts_at_zero(render_cfg, "camera_shake_px") else
        f"Lower the lower end of render.camera_shake_px (now {fmt['camera_shake_px']}) or lengthen "
        f"render.camera_shake_tau_s (now {fmt['camera_shake_tau_s']}) so that frames stay correlated.")
    fmt.update(r5=real["p5"], r95=real["p95"], s1=syn["p1"], s99=syn["p99"], interval=interval_s,
               r5_fs=real["p5"] / 255.0, r95_fs=real["p95"] / 255.0, k_shake=SHAKE_SHIFT_PER_STD,
               r95_shake=real["p95"] / SHAKE_SHIFT_PER_STD)
    head = (f"{real['outside']:.0%} of {real['n']} real clips fall outside the synthetic P1-P99 range "
            f"[{syn['p1']:.3g}, {syn['p99']:.3g}] ({real['below']:.0%} below, {real['above']:.0%} above).")
    sides = {"widen_low": ("low",), "widen_high": ("high",), "widen_both": ("low", "high")}[v]
    return " ".join([head] + [ADVICE[stat][s].format(**fmt) for s in sides])


def _ranks(v: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged."""
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), dtype=np.float64)
    r[order] = np.arange(len(v), dtype=np.float64)
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=r) / counts)[inv]


def spearman(x, y):
    """Spearman rank correlation over the finite pairs; None with fewer than 3 pairs or a constant input."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    rx, ry = _ranks(x[ok]), _ranks(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def tracking(synthetic: list) -> dict:
    """On the synthetic set: rank correlation of each statistic with the render parameter that should drive it."""
    out = {}
    for stat, param in TRACKING:
        x = [r["stats"][stat] for r in synthetic]
        y = [np.nan if r["render_params"].get(param) is None else float(r["render_params"][param])
             for r in synthetic]
        out[f"{stat}~{param}"] = spearman(x, y)
    return out


def compare(real: list, synthetic: list, render_cfg: dict, interval_s: float, flag: float = OUTSIDE_FLAG,
            synthetic_interval_s=None) -> dict:
    """Per statistic: synthetic and real (native, resized) percentiles, coverage of the real clips by the
    synthetic [P1, P99] range, verdict and note (on the comparable basis). With a synthetic_interval_s that
    spacing_matches rejects, flicker and motion get no coverage and verdict "no_data" (values kept)."""
    spacing_ok = synthetic_interval_s is None or spacing_matches(interval_s, synthetic_interval_s)
    out = {}
    for stat in STATS:
        temporal = stat in TEMPORAL_STATS
        syn = _values(synthetic, "stats", stat)
        lo, hi = (float(np.percentile(syn, p)) for p in COVER_PCT) if syn.size else (None, None)
        cover = (None, None) if temporal and not spacing_ok else (lo, hi)
        entry = {"unit": STAT_INFO[stat]["unit"], "comparable_basis": STAT_INFO[stat]["basis"],
                 "synthetic": dict(_summary(syn), p1=lo, p99=hi)}
        for basis in ("native", "resized"):
            v = _values(real, basis, stat, temporal_only=temporal)
            entry[f"real_{basis}"] = dict(_summary(v), **_coverage(v, *cover))
        if stat == "motion_px":        # clips without a registered pair have no value, so no coverage above
            entry["unregistered"] = {"real": _n_unregistered(real, entry["comparable_basis"]),
                                     "synthetic": _n_unregistered(synthetic, "stats")}
        entry["verdict"] = verdict(entry[f"real_{entry['comparable_basis']}"], flag)
        if temporal and not spacing_ok:
            entry["note"] = (f"No comparison: the real frames are sampled {interval_s:g} s apart but the synthetic "
                             f"ones {synthetic_interval_s:g} s, and {stat} depends on the spacing; rerun with "
                             f"--interval-s {synthetic_interval_s:g}.")
        else:
            entry["note"] = advice(stat, entry, render_cfg, interval_s, flag)
        out[stat] = entry
    return out


# =============================================================================
# Report
# =============================================================================


def _clean(value):
    """NaN / inf -> None, recursively (JSON holds finite numbers only)."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def _fmt(v, spec: str = ".3g") -> str:
    return "n/a" if v is None else format(v, spec)


def _pcts(s: dict) -> str:
    return " / ".join(_fmt(s[f"p{p}"]) for p in PERCENTILES)


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:.0%}"


def markdown(doc: dict) -> str:
    st, real, syn = doc["settings"], doc["real"], doc["synthetic"]
    n_temporal = sum(r["temporal_comparable"] for r in real["clips"])
    unreg = doc["stats"]["motion_px"].get("unregistered") or {}
    sizes = sorted({tuple(r["native_size"]) for r in real["clips"]})
    L = [f"# Image statistics, real vs synthetic: {doc['name']}", "",
         "> Diagnostic of the rendering, not a model input. Real clips are few, so a coverage fraction moves "
         f"in steps of 1 / n; a statistic is flagged when more than {doc['outside_flag']:.0%} of the real clips "
         "fall outside the synthetic P1-P99 range.", ""]
    if not st["temporal_spacing_ok"]:
        L += [f"> **Warning**: the real frames are sampled {st['interval_s']:g} s apart but the synthetic ones "
              f"{_fmt(syn['interval_s'], 'g')} s apart: flicker and motion depend on the spacing and are not "
              "compared.", ""]
    L += [f"* Real: `{real['manifest']}`, {len(real['clips'])} clips ({len(real['skipped'])} skipped), native "
          f"sizes (H x W) {', '.join(f'{h}x{w}' for h, w in sizes) or 'none'}; up to {st['max_frames']} frames "
          f"{st['interval_s']:g} s apart, {n_temporal} clips with at least {st['min_frames']} distinct frames at "
          f"that spacing for flicker and motion, {unreg.get('real', 0)} of them left out of motion because no frame "
          f"pair registers (phase-correlation response below {st['min_phase_response']:g}).",
          f"* Synthetic: `{syn['data']}`, {len(syn['sequences'])} of {syn['available']} sequences in splits "
          f"{', '.join(st['splits']) if st['splits'] else 'all'} (seed {st['seed']}), native size "
          f"{st['synthetic_size'][0]} x {st['synthetic_size'][1]}, frames {_fmt(syn['interval_s'], 'g')} s apart, "
          f"{unreg.get('synthetic', 0)} without a registered frame pair.",
          "* Comparable basis: real frames resized to the synthetic pixel count (aspect kept), except blockiness, "
          "which exists only on the encoder's native 8x8 grid and is compared at native resolution.", "",
          "## Comparison (comparable basis)", "",
          "| statistic | unit | basis | real P5 / P25 / P50 / P75 / P95 | synthetic P5 / P25 / P50 / P75 / P95 | "
          "synthetic P1 - P99 | real below / above | verdict |",
          "|---|---|---|---|---|---|---|---|"]
    for stat, e in doc["stats"].items():
        r, s = e[f"real_{e['comparable_basis']}"], e["synthetic"]
        L.append(f"| {stat} | {e['unit']} | {e['comparable_basis']} | {_pcts(r)} (n {r['n']}) | {_pcts(s)} "
                 f"(n {s['n']}) | {_fmt(s['p1'])} - {_fmt(s['p99'])} | {_pct(r['below'])} / {_pct(r['above'])} | "
                 f"{e['verdict']} |")
    L += ["", "## Real footage at native resolution (reference)", "",
          "| statistic | real P5 / P25 / P50 / P75 / P95 | outside synthetic P1 - P99 (below / above) |",
          "|---|---|---|"]
    for stat, e in doc["stats"].items():
        r = e["real_native"]
        L.append(f"| {stat} | {_pcts(r)} (n {r['n']}) | {_pct(r['outside'])} ({_pct(r['below'])} / "
                 f"{_pct(r['above'])}) |")
    L += ["", "## What to change in `render:`", ""]
    L += [f"* **{stat}**: {e['note']}" for stat, e in doc["stats"].items()]
    L += ["", "## Do the statistics follow their render parameters? (synthetic, Spearman rank correlation)", "",
          "| statistic ~ parameter | rho |", "|---|---|"]
    L += [f"| {k} | {_fmt(v, '.2f')} |" for k, v in doc["synthetic_tracking"].items()]
    clips = real["clips"][:MD_MAX_CLIPS]
    if clips:
        L += ["", "## Per real clip (comparable basis; flicker and motion only at the comparison spacing)", "",
              "| clip | native H x W | frames | spacing (s) | " + " | ".join(STATS) + " |",
              "|---|---|---|---|" + "---|" * len(STATS)]
        for r in clips:
            vals = []
            for stat in STATS:
                v = r[STAT_INFO[stat]["basis"]][stat]
                vals.append(_fmt(v) if (stat not in TEMPORAL_STATS or r["temporal_comparable"]) else "-")
            L.append(f"| {r['video_id']} | {r['native_size'][0]} x {r['native_size'][1]} | "
                     f"{r['resized']['n_frames']} | {_fmt(r['resized']['interval_s'], '.3g')} "
                     f"({r['sampling']}) | " + " | ".join(vals) + " |")
        if len(real["clips"]) > MD_MAX_CLIPS:
            L.append(f"\n({len(real['clips']) - MD_MAX_CLIPS} more clips in the JSON report.)")
    L += ["", "## Notes", "",
          "* Flicker and motion include scene changes (moving smoke, passing objects, pans, auto-exposure): they "
          "bound the camera's own flicker and shake from above, on both sides.",
          "* Noise includes the finest scene texture; sharpness includes noise. Real values below the synthetic "
          "range usually mean in-camera denoising or codec smoothing.",
          "* The synthetic frames carry the rendered nuisances of every class (strong artefacts included), so "
          "their upper percentiles reflect deliberately harsh clips."]
    if real["skipped"]:
        L += ["", "## Skipped real clips", ""] + [f"* `{s['video']}`: {s['reason']}" for s in real["skipped"]]
    return "\n".join(L) + "\n"


def main(argv=None):
    args = parse_args(argv)
    manifest, data = Path(args.real_manifest), Path(args.synthetic)
    cfg_path = data / "config_used.yaml"
    cfg = read_yaml(cfg_path) if cfg_path.is_file() else {}
    render_cfg = cfg.get("render") or {}
    splits = args.splits.split(",") if args.splits not in (None, "", "all") else None
    synthetic, available = synthetic_records(data, args.n_synthetic, args.max_frames, args.seed, splits)
    size = render_cfg.get("native_size") or synthetic[0]["native_size"]
    syn_interval = _finite_median([r["stats"]["interval_s"] for r in synthetic])
    interval = args.interval_s or (cfg.get("time") or {}).get("frame_interval_s") or syn_interval
    if not (interval and math.isfinite(float(interval)) and float(interval) > 0):
        raise SystemExit("no frame interval: pass --interval-s")
    interval = float(interval)
    spacing_ok = spacing_matches(interval, syn_interval)
    if not spacing_ok:
        print(f"warning: real frames sampled {interval:g} s apart, synthetic frames {syn_interval:g} s apart "
              f"(more than {INTERVAL_TOL:.0%} off): flicker and motion are not compared", file=sys.stderr)
    real, skipped = real_records(manifest, size, interval, args.max_frames, args.min_frames)
    if not real:
        print(f"no real clip could be read from {manifest}", file=sys.stderr)
    doc = _clean({
        "status": "image_statistics_diagnostic",
        "name": args.name or manifest.parent.name,
        "outside_flag": OUTSIDE_FLAG,
        "settings": {"interval_s": interval, "max_frames": args.max_frames, "min_frames": args.min_frames,
                     "n_synthetic": args.n_synthetic, "seed": args.seed, "splits": splits,
                     "synthetic_size": [int(v) for v in size], "cover_percentiles": list(COVER_PCT),
                     "temporal_spacing_ok": spacing_ok, "interval_tol": INTERVAL_TOL,
                     "min_phase_response": MIN_PHASE_RESPONSE, "min_registered_frac": MIN_REGISTERED_FRAC,
                     "resize": "real frames to the synthetic pixel count, aspect kept (INTER_AREA when shrinking)"},
        "stats": compare(real, synthetic, render_cfg, interval, synthetic_interval_s=syn_interval),
        "synthetic_tracking": tracking(synthetic),
        "real": {"manifest": str(manifest), "clips": real, "skipped": skipped},
        "synthetic": {"data": str(data), "available": available, "interval_s": syn_interval,
                      "render_config": {k: _cfg_value(render_cfg, k) for k in CONFIG_KEYS},
                      "sequences": synthetic},
    })
    out = Path(args.out)
    write_json(out, doc)
    text = markdown(doc)
    out.with_suffix(".md").write_text(text, encoding="utf-8")
    print(text)
    return 0 if real else 1


if __name__ == "__main__":
    sys.exit(main())
