"""Candidate-region masks: degraded detectability masks, YOLO-like boxes, image-only detector.

A real deployment gets its candidate regions from a detector or segmentation
model, never from clean simulation fields. The synthetic pipeline therefore
never hands the network a clean mask either:

* `degrade_masks` turns the class-agnostic VISIBLE-CONTRAST map of a rendered
  sequence (`alpha * |A - B|`, plus the brightness change of a moving shadow,
  see render.py) into detector-like masks: a random threshold, erosion or
  dilation, small boundary noise, dropped and fragmented components,
  false-positive blobs that may persist for a few frames, whole-frame misses
  and occluded pixels removed. The same statistics apply to every class.
* `boxes_from_masks` turns masks into per-frame jittered, filled bounding
  boxes of their components, like a box detector (YOLO-style) would give.
* `detect_candidate_masks` is a grayscale, class-agnostic fallback detector
  that needs nothing but the frames: a temporal-median background over the
  window (after a similarity alignment against camera shake), cues
  for brightness change and for the loss of local contrast that a veil of
  smoke, fog or haze causes, then morphology and a minimum area. It is used
  for a share of the synthetic variants (so the network sees its error
  statistics) and for real footage that comes without masks.

Where this fits: `render.render_sequence` calls all three; real-footage
inference may call `detect_candidate_masks` directly.
Calls into: numpy and OpenCV only.
Start reading at: degrade_masks, then detect_candidate_masks.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

import cv2
import numpy as np

# Defaults of the degradation (mirroring configs/simulation.yaml render.masks);
# a value may be a scalar or a [lo, hi] range drawn per call.
_DEGRADE_DEFAULTS: Dict[str, Any] = {
    "threshold_contrast": [0.02, 0.08],   # on the visible-contrast map
    "threshold_jitter": 0.1,              # per-frame log-normal jitter of the threshold
    "morph_px": [-2, 2],                  # erosion (< 0) / dilation (> 0), pixels
    "boundary_noise": [0.0, 0.15],
    "drop_component_prob": 0.1,
    "fragment_prob": None,                # None: same as drop_component_prob
    "false_positive_prob": 0.15,          # per frame: a new false-positive blob appears
    "false_positive_prob_boost": 0.5,     # used instead when params["fp_boost"] is true
    "fp_boost": False,
    "frame_dropout_prob": 0.05,
    "min_area_frac": 0.001,
}

_DETECTOR_DEFAULTS: Dict[str, Any] = {
    "blur_sigma": 1.0,          # pre-smoothing against noise and JPEG blocks, pixels
    "align": True,              # similarity alignment to the temporal median (corners + LK + RANSAC)
    "max_shift_px": 12.0,       # larger estimated motions are ignored (unreliable)
    "max_rotation_deg": 3.0,
    "min_track_points": 12,
    "change_k": 4.0,            # brightness change threshold in robust noise sigmas ...
    "change_min": 0.025,        # ... but never below this (fraction of full scale)
    "change_sigma": 1.5,        # smoothing of |I - background| before thresholding
    "contrast_sigma": 3.0,      # window of the local-contrast (local std) estimate
    "veil_drop": 0.5,           # relative loss of local contrast that counts as a veil
    "veil_min_texture": 0.02,   # background local std needed before contrast loss is trusted
    "open_px": 1,
    "close_px": 3,
    "min_area_frac": 0.001,
    "sensitivity": 1.0,         # > 1 lowers every threshold (more detections, more false positives)
}


# ---------------------------------------------------------------------- helpers
def _draw(value: Any, rng: np.random.Generator, integer: bool = False) -> Any:
    """A scalar as is, or a uniform draw from a [lo, hi] range (inclusive integers)."""
    if isinstance(value, (list, tuple)):
        lo, hi = float(value[0]), float(value[1])
        u = float(rng.random())
        if integer:
            return int(min(math.floor(lo + (hi - lo + 1.0) * u), hi))
        return lo + (hi - lo) * u
    return int(round(float(value))) if integer else float(value)


def _resolve_degrade_params(params: Optional[Mapping[str, Any]], rng: np.random.Generator) -> Dict[str, Any]:
    p = dict(_DEGRADE_DEFAULTS)
    p.update({k: v for k, v in (params or {}).items() if v is not None or k == "fragment_prob"})
    # The draw order is fixed, so the stream does not depend on which values are ranges.
    out = {
        "threshold_contrast": _draw(p["threshold_contrast"], rng),
        "morph_px": _draw(p["morph_px"], rng, integer=True),
        "boundary_noise": _draw(p["boundary_noise"], rng),
        "threshold_jitter": float(p["threshold_jitter"]),
        "drop_component_prob": float(p["drop_component_prob"]),
        "frame_dropout_prob": float(p["frame_dropout_prob"]),
        "min_area_frac": float(p["min_area_frac"]),
    }
    frag = p.get("fragment_prob")
    out["fragment_prob"] = out["drop_component_prob"] if frag is None else float(frag)
    fp = float(p["false_positive_prob"])
    if bool(p.get("fp_boost", False)):
        fp = max(fp, float(p["false_positive_prob_boost"]))
    out["false_positive_prob"] = fp
    return out


def _ellipse_kernel(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _morph(mask: np.ndarray, px: int) -> np.ndarray:
    """Erode (px < 0) or dilate (px > 0) a bool mask by |px| pixels."""
    if px == 0 or not mask.any():
        return mask
    m = mask.astype(np.uint8)
    k = _ellipse_kernel(abs(int(px)))
    m = cv2.dilate(m, k) if px > 0 else cv2.erode(m, k)
    return m.astype(bool)


def _smooth_noise(rng: np.random.Generator, h: int, w: int, cell: int) -> np.ndarray:
    """Zero-mean, unit-std noise with correlation length ~cell pixels."""
    nh, nw = h // cell + 3, w // cell + 3
    low = rng.standard_normal((nh, nw)).astype(np.float32)
    up = cv2.resize(low, (nw * cell, nh * cell), interpolation=cv2.INTER_CUBIC)
    oy, ox = cell // 2, cell // 2
    out = up[oy:oy + h, ox:ox + w]
    return (out - out.mean()) / (out.std() + 1e-6)


def _boundary_noise(mask: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Jitter the boundary of a mask by ~1-3 px (strength 0.15) with smooth random noise."""
    h, w = mask.shape
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.5)
    noise = 0.7 * _smooth_noise(rng, h, w, 8) + 0.3 * _smooth_noise(rng, h, w, 3)
    return (soft + 2.0 * strength * noise) > 0.5


def _components(mask: np.ndarray):
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)


def _remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    if not mask.any():
        return mask
    n, lab, stats, _ = _components(mask)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[lab]


def _drop_and_fragment(mask: np.ndarray, rng: np.random.Generator, drop_p: float,
                       frag_p: float, min_area: int) -> np.ndarray:
    """Drop whole components (detector misses) and cut some apart (fragmentation)."""
    if not mask.any():
        return mask
    n, lab, stats, cents = _components(mask)
    keep = np.ones(n, dtype=bool)
    keep[0] = False
    for j in range(1, n):
        if rng.random() < drop_p:
            keep[j] = False
    out = keep[lab]
    cut = out.astype(np.uint8)
    for j in range(1, n):
        if not keep[j] or stats[j, cv2.CC_STAT_AREA] < 4 * min_area:
            continue
        if rng.random() < frag_p:
            # A random crack through the centroid, 2-3 px wide.
            ang = rng.uniform(0.0, math.pi)
            half = 0.5 * math.hypot(stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]) + 2
            cx, cy = cents[j]
            p0 = (int(round(cx - half * math.cos(ang))), int(round(cy - half * math.sin(ang))))
            p1 = (int(round(cx + half * math.cos(ang))), int(round(cy + half * math.sin(ang))))
            cv2.line(cut, p0, p1, 0, thickness=int(rng.integers(2, 4)))
    return cut.astype(bool)


def _false_positive_tracks(masks: np.ndarray, rng: np.random.Generator, prob: float,
                           min_area: int) -> np.ndarray:
    """Add elliptical false-positive blobs IN PLACE; each lives 1-4 frames and drifts slightly.

    Real detectors fire on bright rocks, glare or clouds for several frames in
    a row, so some false positives survive a persistence filter.
    """
    T, h, w = masks.shape
    if prob <= 0.0:
        return masks
    max_area = max(2.0 * min_area, 0.02 * h * w)
    tracks: List[List[float]] = []
    for t in range(T):
        if rng.random() < prob:
            area = math.exp(rng.uniform(math.log(1.5 * min_area), math.log(max_area)))
            r = math.sqrt(area / math.pi)
            aspect = rng.uniform(0.5, 2.0)
            tracks.append([
                rng.uniform(0, w - 1), rng.uniform(0, h - 1),          # centre
                r * math.sqrt(aspect), r / math.sqrt(aspect),           # semi-axes
                rng.uniform(0.0, 180.0),                                # angle
                float(rng.normal(0.0, 1.0)), float(rng.normal(0.0, 1.0)),  # drift px / frame
                float(t + int(rng.integers(1, 5))),                     # end (exclusive)
            ])
        layer = masks[t].astype(np.uint8)
        for tr in tracks:
            if t < tr[7]:
                cv2.ellipse(layer, (int(round(tr[0])), int(round(tr[1]))),
                            (max(1, int(round(tr[2]))), max(1, int(round(tr[3])))),
                            tr[4], 0, 360, 1, thickness=-1)
                tr[0] += tr[5]
                tr[1] += tr[6]
        masks[t] = layer.astype(bool)
        tracks = [tr for tr in tracks if t + 1 < tr[7]]
    return masks


# ---------------------------------------------------------------------- public API
def degrade_masks(detectability: np.ndarray, rng: np.random.Generator, params: Optional[Mapping[str, Any]],
                  occluder: Optional[np.ndarray] = None) -> np.ndarray:
    """(T, H, W) visible-contrast map -> (T, H, W) bool detector-like candidate masks.

    `params` uses the keys of `render.masks` (scalars or [lo, hi] ranges):
    threshold_contrast, morph_px, boundary_noise, drop_component_prob,
    false_positive_prob (false_positive_prob_boost when fp_boost is true),
    frame_dropout_prob, min_area_frac; optional threshold_jitter and
    fragment_prob. `occluder` (T, H, W) or (H, W) in [0, 1]: pixels with
    occluder >= 0.5 are removed from the masks (a detector cannot see behind
    a foreground object). Nothing here depends on the scenario class.
    """
    D = np.asarray(detectability, dtype=np.float32)
    if D.ndim == 2:
        D = D[None]
    T, h, w = D.shape
    p = _resolve_degrade_params(params, rng)
    min_area = max(4, int(round(p["min_area_frac"] * h * w)))
    out = np.zeros((T, h, w), dtype=bool)
    for t in range(T):
        th = p["threshold_contrast"] * math.exp(p["threshold_jitter"] * float(rng.standard_normal()))
        m = D[t] > th
        m = _morph(m, p["morph_px"])
        if p["boundary_noise"] > 0.0 and m.any():
            m = _boundary_noise(m, rng, p["boundary_noise"])
        m = _remove_small(m, min_area)
        m = _drop_and_fragment(m, rng, p["drop_component_prob"], p["fragment_prob"], min_area)
        out[t] = m
    _false_positive_tracks(out, rng, p["false_positive_prob"], min_area)
    for t in range(T):
        if rng.random() < p["frame_dropout_prob"]:
            out[t] = False
    if occluder is not None:
        occ = np.asarray(occluder)
        out &= ~(np.broadcast_to(occ, out.shape) >= 0.5)
    return out


def boxes_from_masks(masks: np.ndarray, rng: np.random.Generator, jitter: float = 0.1) -> np.ndarray:
    """Per frame, every 8-connected component becomes a filled, jittered bounding box.

    Each box edge moves by N(0, jitter * box size) independently, like the
    localisation error of a box detector; boxes are clipped to the image.
    """
    m = np.asarray(masks).astype(bool)
    squeeze = m.ndim == 2
    if squeeze:
        m = m[None]
    T, h, w = m.shape
    out = np.zeros_like(m)
    for t in range(T):
        if not m[t].any():
            continue
        n, _, stats, _ = _components(m[t])
        for j in range(1, n):
            x, y, bw, bh = (int(v) for v in stats[j, :4])
            e = rng.normal(0.0, float(jitter), 4)
            x0 = int(np.clip(round(x + e[0] * bw), 0, w - 1))
            x1 = int(np.clip(round(x + bw - 1 + e[1] * bw), 0, w - 1))
            y0 = int(np.clip(round(y + e[2] * bh), 0, h - 1))
            y1 = int(np.clip(round(y + bh - 1 + e[3] * bh), 0, h - 1))
            if x1 >= x0 and y1 >= y0:
                out[t, y0:y1 + 1, x0:x1 + 1] = True
    return out[0] if squeeze else out


def _to_gray_float(frames: np.ndarray) -> np.ndarray:
    """(T, H, W) or (T, H, W, 3) uint8 / float in [0, 1] -> (T, H, W) float32 in [0, 1]."""
    f = np.asarray(frames)
    if f.ndim == 4:
        if f.shape[-1] == 1:
            f = f[..., 0]
        else:
            f = np.stack([cv2.cvtColor(np.ascontiguousarray(x[..., :3]), cv2.COLOR_BGR2GRAY) for x in f])
    if f.ndim != 3:
        raise ValueError(f"expected (T, H, W) or (T, H, W, C) frames, got shape {f.shape}")
    if f.dtype == np.uint8:
        return f.astype(np.float32) / 255.0
    f = f.astype(np.float32)
    return f / 255.0 if f.max(initial=0.0) > 1.5 else f


def _local_std(img: np.ndarray, sigma: float) -> np.ndarray:
    mu = cv2.GaussianBlur(img, (0, 0), sigma)
    mu2 = cv2.GaussianBlur(img * img, (0, 0), sigma)
    return np.sqrt(np.maximum(mu2 - mu * mu, 0.0))


def _to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _median(stack: np.ndarray) -> np.ndarray:
    """Exact per-pixel median of a short (T, H, W) stack by an odd-even transposition network
    (elementwise min / max; ~10x faster than np.median along the time axis)."""
    a = [np.array(s, dtype=np.float32) for s in stack]
    n = len(a)
    tmp = np.empty_like(a[0])
    for r in range(n):
        for i in range(r % 2, n - 1, 2):
            np.minimum(a[i], a[i + 1], out=tmp)
            np.maximum(a[i], a[i + 1], out=a[i + 1])
            a[i], tmp = tmp, a[i]
    return a[n // 2] if n % 2 else 0.5 * (a[n // 2 - 1] + a[n // 2])


def _estimate_alignment(ref8: np.ndarray, pts: Optional[np.ndarray], frame: np.ndarray,
                        p: Mapping[str, Any]) -> Optional[np.ndarray]:
    """(2, 3) similarity mapping `frame` pixel coordinates onto the reference, or None.

    Corners of the reference are tracked into the frame with pyramidal LK and
    a RANSAC similarity is fitted, so a moving object (a minority of the
    corners) does not bias the estimate the way it biases phase correlation.
    """
    n_min = int(p["min_track_points"])
    if pts is None or len(pts) < n_min:
        return None
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(ref8, _to_u8(frame), pts, None, winSize=(9, 9), maxLevel=2)
    ok = status.ravel() == 1
    if ok.sum() < n_min:
        return None
    M, inliers = cv2.estimateAffinePartial2D(nxt[ok], pts[ok], method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if M is None or inliers is None or int(inliers.sum()) < n_min:
        return None
    scale = math.hypot(M[0, 0], M[1, 0])
    angle = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    if (abs(scale - 1.0) > 0.08 or abs(angle) > float(p["max_rotation_deg"])
            or math.hypot(M[0, 2], M[1, 2]) > float(p["max_shift_px"]) + 0.1 * max(ref8.shape)):
        return None
    return M


def detect_candidate_masks(frames: np.ndarray, timestamps: Optional[np.ndarray] = None,
                           params: Optional[Mapping[str, Any]] = None) -> np.ndarray:
    """Image-only, grayscale, class-agnostic candidate masks (T, H, W) bool.

    1. Frames -> grayscale float, lightly smoothed; each frame aligned to the
       temporal median with a similarity transform (corners tracked by LK,
       RANSAC fit; frames where it fails or exceeds the bounds stay as they are).
    2. Background = per-pixel temporal median of the aligned window.
    3. Cues per frame: |I - background| in robust noise units (brightness or
       greying change, moving texture), and the relative LOSS of local
       contrast against the background (a veil over textured terrain).
    4. Threshold, open / close, drop regions below `min_area_frac`, shift the
       mask back into the frame's own (unaligned) coordinates.
    A static veil present in most frames is part of the median and is not
    detected; this is a fallback, not a segmentation model. `timestamps` is
    accepted for interface symmetry and is not needed by this detector.
    """
    p = dict(_DETECTOR_DEFAULTS)
    p.update(params or {})
    g = _to_gray_float(frames)
    T, h, w = g.shape
    out = np.zeros((T, h, w), dtype=bool)
    if T < 3:
        return out
    sm = np.stack([cv2.GaussianBlur(f, (0, 0), float(p["blur_sigma"])) for f in g])
    warps: List[Optional[np.ndarray]] = [None] * T
    aligned = sm
    if p["align"]:
        # Two rounds: the median of the raw frames is blurred by the shake, the second
        # reference (median of the aligned frames) is sharp.
        for _ in range(2):
            ref8 = _to_u8(_median(aligned))
            pts = cv2.goodFeaturesToTrack(ref8, maxCorners=80, qualityLevel=0.01, minDistance=7, blockSize=7)
            warps = [_estimate_alignment(ref8, pts, sm[t], p) for t in range(T)]
            aligned = np.stack([sm[t] if warps[t] is None else
                                cv2.warpAffine(sm[t], warps[t], (w, h), flags=cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_REPLICATE) for t in range(T)])
    bg = _median(aligned)
    sens = max(float(p["sensitivity"]), 1e-6)
    std_bg = _local_std(bg, float(p["contrast_sigma"]))
    textured = std_bg > float(p["veil_min_texture"])
    min_area = max(4, int(round(float(p["min_area_frac"]) * h * w)))
    k_open = _ellipse_kernel(int(p["open_px"])) if int(p["open_px"]) > 0 else None
    k_close = _ellipse_kernel(int(p["close_px"])) if int(p["close_px"]) > 0 else None
    for t in range(T):
        d = np.abs(aligned[t] - bg)
        sigma = max(1.4826 * float(np.median(d)), 0.003)
        change = cv2.GaussianBlur(d, (0, 0), float(p["change_sigma"]))
        change_thr = max(float(p["change_k"]) * sigma, float(p["change_min"])) / sens
        m = change > change_thr
        std_t = _local_std(aligned[t], float(p["contrast_sigma"]))
        loss = (std_bg - std_t) / (std_bg + 1e-3)
        m |= textured & (loss > float(p["veil_drop"]) / sens)
        mu8 = m.astype(np.uint8)
        if k_open is not None:
            mu8 = cv2.morphologyEx(mu8, cv2.MORPH_OPEN, k_open)
        if k_close is not None:
            mu8 = cv2.morphologyEx(mu8, cv2.MORPH_CLOSE, k_close)
        m = _remove_small(mu8.astype(bool), min_area)
        if warps[t] is not None and m.any():
            # Back into the frame's own (shaken) coordinates.
            m = cv2.warpAffine(m.astype(np.uint8), warps[t], (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        out[t] = m
    return out
