"""Motion features (T x 14) from frames and candidate masks, identical for synthetic and real footage.

The physics expert never sees CFD fields. It sees what a camera pipeline can
measure: optical flow inside a candidate region and the region's geometry.
This module turns a short clip (observation frames every `expected_interval_s`,
each with an optional partner frame `flow_lag_s` earlier, plus one candidate
mask per observation) into the 14-column array defined in `feature_spec.py`.

Two flow modes, fixed by the `FeatureConfig` a model was trained with:

* "lag" (near-field profile: observations 5 s apart, lag frames 1 s
  earlier): flow over each short lag pair. Without lag frames the extractor
  falls back to consecutive flow from a zero start and flags it
  (`timing.reasons` gets "consecutive_flow"; the inference API refuses it
  unless allowed).
* "consecutive" (`flow_lag_s` None or 0; far-field profile, one frame per
  minute as camera networks such as HPWREN / AlertWildfire record): flow
  between consecutive observations. This is a first-class mode: lag frames,
  if given, are ignored (recorded) so training and inference measure the same
  thing, and because displacements over a minute can exceed what DIS tracks
  from a zero start (~15 working px), each pair is also tried from the
  displacement of the mask centroid and the candidate whose flow best
  reconstructs the frame (photometric residual) is kept.

Pipeline (docs/design.md section 7):

1. `_to_gray_u8` converts every image input (uint8, float in [0, 1], BGR) to
   one uint8 grayscale representation. It is the only conversion, so a float
   copy of a uint8 clip gives bit-identical features.
2. `_estimate_stabilization`: a bounded similarity transform of every
   observation and lag frame onto observation frame 0, estimated from
   corners tracked (band-passed Lucas-Kanade + RANSAC) ONLY outside the
   dilated union of the candidate masks. A frame whose estimate fails the
   quality check or the bound is left unwarped and its flow is marked
   invalid. The whole step is skipped (and recorded) when less than 3 % of
   the image is textured background or fewer than 12 background corners
   exist, when most estimates fail, or when the "background" drifts steadily
   over the window (unmasked fog, not camera shake), so a frame-filling fog
   bank is never "stabilised away". A skipped step leaves camera motion in
   the flow, so with `require_stabilization` (default) the flow gate is
   closed for the whole clip ("stabilization_unavailable"); set it to False
   only for footage known to come from a rigid, static mount.
3. `_persistent_filter` drops mask components that touch nothing in the
   previous or next observation (single-frame detector false positives).
4. `_roi_box`: one square analysis frame per clip around the union of the
   filtered masks (`roi_scale` x the larger bounding-box side), resized so its
   longer side is `work_size`. Every length is then measured in
   L = sqrt(H * W) of the resized ROI and every rate per second, so the
   features do not depend on how large the scene appears in the image.
5. `_flow_stats`: DIS optical flow from the lag frame to the observation frame
   (or between consecutive observations: from zero, and in the consecutive
   mode also from the centroid displacement), a forward-backward check, and
   winsorised statistics, affine divergence and mean |curl| on the eroded
   mask.
6. Geometry of the resized masks: area, log-area growth and centroid
   velocity between observations, `_roughness` and `_anchor_stability`.

Image rows grow downward; every vertical feature is converted to +UP here.

Where this fits: `dataset.py` calls `extract_features_with_meta` on rendered
sequences, `src.inference.predict.PhysicsExpert.features_from_frames` on real
footage (after `select_window` / `load_frames`). Closed gates are exactly 0.0
and the output is always finite.
Calls into: OpenCV (DIS / Farneback flow, LK tracking, morphology, video
I/O), numpy, `feature_spec`.
Start reading at: extract_features_with_meta.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.data.feature_spec import (
    FEATURE_GATES,
    FEATURE_INDEX,
    FLOW_VALID_INDEX,
    MASK_VALID_INDEX,
    N_FEATURES,
    gate_masks,
)
from src.utils.io import PathLike

# ---- stabilisation ---------------------------------------------------------
_STAB_DILATE_FRAC = 0.04        # mask union dilation, fraction of min(H, W) (>= 5 px)
_STAB_MIN_TEXTURED_FRAC = 0.03  # textured background required, fraction of the image (the corner,
                                # inlier and residual checks below guard the estimate itself)
_STAB_TEXTURE_BLUR = 2.5        # Gaussian sigma before the texture gradient (suppresses noise)
_STAB_TEXTURE_GRAD = 0.75       # gray levels / px (9x9 mean gradient); pixel noise up to sigma ~12 stays below
_STAB_EST_MAX_SIDE = 384        # estimate on a copy whose shorter side is at most this
_STAB_MAX_CORNERS = 50
_STAB_MIN_CORNERS = 12
_STAB_BORDER_FRAC = 0.06        # corners keep this far (fraction of min(H, W)) from the border
_STAB_BANDPASS_BOX = 11         # box size of the broad blur removed before tracking (applied twice)
_STAB_LK_WIN = 11
_STAB_LK_LEVELS = 3
_STAB_RANSAC_PX = 1.0
_STAB_MIN_INLIERS = 10
_STAB_MIN_INLIER_RATIO = 0.4
_STAB_MAX_RESIDUAL_PX = 0.7     # RMS inlier reprojection error (estimation resolution)
_STAB_MAX_SHIFT_FRAC = 0.12     # bound: image-centre displacement / min(H, W)
_STAB_MAX_ROTATION_DEG = 3.0
_STAB_MAX_SCALE_DEV = 0.06
_STAB_MAX_FAILED_FRAC = 0.5     # more estimates failing the quality check: background unusable
_STAB_MAX_DRIFT_FRAC = 0.04     # steady drift of the "background" over the window above this
_STAB_DRIFT_T = 5.0             # ... and this significant means it is scene content, not shake

# ---- masks, flow, geometry -------------------------------------------------
_PERSIST_DILATE_FRAC = 0.03     # neighbour dilation for the persistence test (>= 3 px)
_MIN_FLOW_REGION_PX = 20        # eroded region below this closes the flow gate
_WINSOR_PCT = (5.0, 95.0)
_FLOW_MARGIN_PX = 16            # flow is computed on the region's box + this margin (working px)
_FLOW_MARGIN_CONSECUTIVE_PX = 32
_FLOW_MIN_SIDE_PX = 32
_GUESS_MIN_PX = 3.0             # consecutive mode: try the centroid displacement as a start if at least this
_GUESS_MIN_GAIN = 0.05          # ... and keep it only if its photometric residual is this much lower
_STATS_SUBSAMPLE_ABOVE_PX = 4096  # larger flow regions: statistics on a 2x2-strided subset
_ANCHOR_DZ_SCALE = 0.02         # anchor decay length, fraction of L
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
_SIDECAR_NAMES = ("timestamps.csv",)
_FILENAME_TIME = re.compile(r"_t(\d+(?:\.\d+)?)s$")   # e.g. cam3_t12.5s.jpg


class IncompatibleFrameTiming(ValueError):
    """The clip cannot provide the observation spacing or the short flow lag the model expects."""


@dataclass
class FeatureConfig:
    work_size: int = 128            # longer side of the analysis frame (ROI) after resizing
    roi: str = "union_bbox"         # "union_bbox" | "full"
    roi_scale: float = 2.0          # ROI side = roi_scale * max(region extent w, h) (synthetic: drawn from [1.5, 2.5])
    roi_min_px: int = 32            # minimum ROI side in native pixels
    stabilize: bool = True
    require_stabilization: bool = True  # no camera-motion estimate -> flow gate closed (see module doc)
    min_mask_frac: float = 0.002    # of the analysis frame
    flow_method: str = "dis"        # "dis" (default) | "farneback"
    flow_erode_px: int = 2          # flow statistics on the mask eroded by this (working px)
    fb_max_px: float = 1.0          # forward-backward consistency (median, working px)
    anchor_band_px: int = 6         # bottom band (working px) whose columns define the anchor
    min_roughness_px: int = 100
    expected_interval_s: float = 5.0          # observation spacing the model expects (far-field profile: 60)
    flow_lag_s: Optional[float] = 1.0         # lag-frame offset; None or 0: consecutive-observation flow
    max_offset_s: float = 0.5                 # timing tolerance: per frame; observation intervals get 2x

    @property
    def flow_mode(self) -> str:
        """The flow mode this configuration measures: "lag" or "consecutive" (flow_lag_s None / 0)."""
        return "lag" if self.flow_lag_s is not None and float(self.flow_lag_s) > 0.0 else "consecutive"


# =============================================================================
# Public API
# =============================================================================


def extract_features(frames, timestamps, masks, lag_frames=None, lag_timestamps=None,
                     cfg: Optional[FeatureConfig] = None) -> np.ndarray:
    """(T, 14) float32 features; see `extract_features_with_meta`."""
    return extract_features_with_meta(frames, timestamps, masks, lag_frames, lag_timestamps, cfg)[0]


def extract_features_with_meta(frames, timestamps, masks, lag_frames=None, lag_timestamps=None,
                               cfg: Optional[FeatureConfig] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Features and the preprocessing record of one clip.

    frames      (T, H, W) or (T, H, W, 3) BGR; uint8, or float in [0, 1]
    timestamps  (T,) seconds of the observations (ideally ~`expected_interval_s` apart)
    masks       (T, H, W) candidate masks (bool / 0-1 / 0-255), or None (no candidate)
    lag_frames  (T, H, W[, 3]) frames ~`flow_lag_s` before each observation, or None;
                ignored (meta "lag_frames_ignored") when cfg.flow_mode is "consecutive"
    lag_timestamps (T,) seconds of the lag frames (default: timestamps - cfg.flow_lag_s)

    Returns X (T, 14) float32 (finite; closed gates exactly 0.0) and a JSON-able
    meta dict: ROI box, stabilisation per frame, flow mode ("lag" |
    "consecutive"), achieved intervals and timing compatibility, per-frame flow
    status, persistent-filter removals and valid-frame counts.
    """
    cfg = cfg or FeatureConfig()
    _check_config(cfg)
    # A consecutive-flow configuration measures the same thing whatever the caller has.
    lag_ignored = lag_frames is not None and cfg.flow_mode == "consecutive"
    if lag_ignored:
        lag_frames = lag_timestamps = None
    if lag_frames is not None:
        obs, lag = _to_gray_u8(frames, lag_frames)
    else:
        (obs,) = _to_gray_u8(frames)
        lag = None
    n, h, w = obs.shape
    if n == 0:
        raise ValueError("no frames")
    if lag is not None and lag.shape != obs.shape:
        raise ValueError(f"lag_frames shape {lag.shape} does not match frames {obs.shape}")
    ts = _as_times(timestamps, n, "timestamps")
    if lag is not None:
        lag_ts = ts - cfg.flow_lag_s if lag_timestamps is None else _as_times(lag_timestamps, n, "lag_timestamps")
    else:
        lag_ts = None
    raw_masks = _to_bool_masks(masks, (n, h, w))

    # Stabilisation (transforms only; frames are warped straight into the ROI below).
    stab = _estimate_stabilization(obs, raw_masks, lag, enabled=cfg.stabilize, obs_times=ts, lag_times=lag_ts)
    stab_masks = np.stack([_warp_mask(raw_masks[t], stab["obs_M"][t]) for t in range(n)])

    # Persistent-component filter, then the analysis frame.
    filt_masks, removed, persist_px = _persistent_filter(stab_masks)

    box, roi_reason = _roi_box(filt_masks.sum(axis=0), cfg)
    x0, y0, x1, y1 = box
    work_h, work_w = _work_shape(y1 - y0, x1 - x0, cfg.work_size)
    scale = work_w / float(x1 - x0)
    L = math.sqrt(work_h * work_w)
    obs_w = np.stack([_to_work(obs[t], stab["obs_M"][t], box, (work_h, work_w)) for t in range(n)])
    lag_w = None
    if lag is not None:
        lag_w = np.stack([_to_work(lag[t], stab["lag_M"][t], box, (work_h, work_w)) for t in range(n)])
    masks_w = np.stack([_mask_to_work(filt_masks[t, y0:y1, x0:x1], (work_h, work_w)) for t in range(n)])

    X = np.zeros((n, N_FEATURES), dtype=np.float64)

    # Mask geometry (step 6; the flow gate below needs mask_valid).
    area_px = masks_w.reshape(n, -1).sum(axis=1).astype(np.float64)
    area_frac = area_px / float(work_h * work_w)
    mask_valid = (area_px > 0) & (area_frac >= cfg.min_mask_frac)
    obs_dt = np.full(n, np.nan)
    obs_dt[1:] = np.diff(ts)
    pair_valid = np.zeros(n, dtype=bool)
    pair_valid[1:] = mask_valid[1:] & mask_valid[:-1]
    centroids = np.zeros((n, 2))
    bottoms: List[Optional[np.ndarray]] = [None] * n
    for t in range(n):
        if not mask_valid[t]:
            continue
        centroids[t] = _centroid(masks_w[t])
        bottoms[t] = _bottom_rows(masks_w[t])
        X[t, FEATURE_INDEX["region_area"]] = area_frac[t]
        X[t, FEATURE_INDEX["boundary_roughness"]] = _roughness(masks_w[t], cfg.min_roughness_px)
    for t in range(1, n):
        dt = obs_dt[t]
        if not pair_valid[t] or not (dt > 0):
            continue
        X[t, FEATURE_INDEX["area_growth"]] = (math.log(area_px[t]) - math.log(area_px[t - 1])) / dt
        d = centroids[t] - centroids[t - 1]
        X[t, FEATURE_INDEX["centroid_dx"]] = d[0] / L / dt
        X[t, FEATURE_INDEX["centroid_dz"]] = -d[1] / L / dt          # rows grow downward
        X[t, FEATURE_INDEX["anchor_stability"]] = _anchor_stability(
            bottoms[t], bottoms[t - 1], work_h, L, cfg.anchor_band_px)

    # Optical flow (step 5).
    flow_mode = "lag" if lag is not None else "consecutive"
    flow_valid = np.zeros(n, dtype=bool)
    flow_dt = np.full(n, np.nan)
    fb_err: List[Optional[float]] = [None] * n
    region_px = [0] * n
    status: List[str] = [""] * n
    flow_start: List[Optional[str]] = [None] * n
    engine = _FlowEngine(cfg.flow_method)
    # stabilize=False asserts a static camera; otherwise an unremoved camera motion invalidates the flow.
    stab_required = bool(cfg.stabilize and cfg.require_stabilization)
    for t in range(n):
        if flow_mode == "lag":
            prev, prev_mask_ok, prev_stab_ok = lag_w[t], True, stab["lag_ok"][t]
            dt = ts[t] - lag_ts[t]
            crop_masks = (masks_w[t],)
        else:
            if t == 0:
                status[t] = "no_pair"
                continue
            prev, prev_mask_ok, prev_stab_ok = obs_w[t - 1], mask_valid[t - 1], stab["obs_ok"][t - 1]
            dt = ts[t] - ts[t - 1]
            crop_masks = (masks_w[t], masks_w[t - 1])
        flow_dt[t] = dt
        if not (mask_valid[t] and prev_mask_ok):
            status[t] = "mask_invalid"
            continue
        if not (stab["obs_ok"][t] and prev_stab_ok):
            status[t] = "stabilization_failed"
            continue
        if stab_required and not stab["info"]["applied"]:
            status[t] = "stabilization_unavailable"
            continue
        if not (np.isfinite(dt) and dt > 0):
            status[t] = "bad_timing"
            continue
        if flow_mode == "lag":
            margin, guesses = _FLOW_MARGIN_PX, ()
        else:
            margin, guesses = _FLOW_MARGIN_CONSECUTIVE_PX, ()
            if cfg.flow_mode == "consecutive":
                # First-class consecutive mode only (a lag configuration's fallback keeps the zero
                # start): over a long spacing the region may move further than DIS tracks from rest,
                # so also start from its centroid displacement (working px, image axes).
                d = centroids[t] - centroids[t - 1]
                if math.hypot(*d) >= _GUESS_MIN_PX:
                    guesses = ((float(d[0]), float(d[1])),)
        res = _flow_stats(prev, obs_w[t], masks_w[t], crop_masks, margin, cfg, engine, guesses)
        region_px[t] = res["region_px"]
        fb_err[t] = res["fb_px"]
        flow_start[t] = res["start"]
        if res["status"] != "ok":
            status[t] = res["status"]
            continue
        u, v, su, sv, div, curl = res["stats"]
        X[t, FEATURE_INDEX["flow_u_mean"]] = u / L / dt
        X[t, FEATURE_INDEX["flow_v_mean"]] = -v / L / dt               # rows grow downward
        X[t, FEATURE_INDEX["flow_u_std"]] = su / L / dt
        X[t, FEATURE_INDEX["flow_v_std"]] = sv / L / dt
        X[t, FEATURE_INDEX["flow_divergence"]] = div / dt              # trace: unchanged by the flip
        X[t, FEATURE_INDEX["flow_vorticity_abs"]] = curl / dt
        flow_valid[t] = True
        status[t] = "ok"

    X[:, MASK_VALID_INDEX] = mask_valid.astype(np.float64)
    X[:, FLOW_VALID_INDEX] = flow_valid.astype(np.float64)

    # Last resort: a non-finite flow statistic closes the flow gate; any other
    # non-finite entry becomes 0. Then every closed gate is forced to exactly 0.
    flow_cols = [i for i, gate in enumerate(FEATURE_GATES) if gate == "flow"]
    bad_flow = ~np.isfinite(X[:, flow_cols]).all(axis=1) & flow_valid
    X[bad_flow, FLOW_VALID_INDEX] = 0.0
    n_nonfinite = int((~np.isfinite(X)).sum())
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.where(gate_masks(X), X, 0.0).astype(np.float32) + np.float32(0.0)   # + 0 turns -0.0 into 0.0
    for t in np.flatnonzero(bad_flow):
        status[t] = "nonfinite"

    meta = {
        "n_frames": int(n),
        "frame_shape": [int(h), int(w)],
        "flow_mode": flow_mode,
        "lag_frames_ignored": bool(lag_ignored),
        "flow_method": cfg.flow_method,
        "roi": {
            "mode": cfg.roi,
            "box": [int(x0), int(y0), int(x1), int(y1)],      # x0, y0, x1, y1 (exclusive), native px
            "reason": roi_reason,
            "work_shape": [int(work_h), int(work_w)],
            "L_px": float(L),
            "work_px_per_native_px": float(scale),
        },
        "stabilization": stab["info"],
        "persistent_filter": {"removed_components": removed, "dilate_px": int(persist_px)},
        "timing": _timing_meta(ts, lag_ts, cfg, flow_mode),
        "flow": {
            "status": status,
            "dt_s": [_opt_float(v) for v in flow_dt],
            "fb_error_px": fb_err,
            "region_px": [int(v) for v in region_px],
            "start": flow_start,               # "zero" | "centroid" (consecutive mode), None without flow
        },
        "counts": {
            "mask_valid": int(X[:, MASK_VALID_INDEX].sum()),
            "flow_valid": int(X[:, FLOW_VALID_INDEX].sum()),
            "mask_pair_valid": int(pair_valid.sum()),
        },
        "nonfinite_repaired": n_nonfinite,
        "config": asdict(cfg),
    }
    return X, meta


def stabilize_sequence(frames, masks, lag_frames=None
                       ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Warp every observation and lag frame (and the masks) onto observation frame 0.

    Returns (frames uint8 (T, H, W), masks bool (T, H, W), lag frames uint8 or
    None, info). `info["obs_ok"]` / `info["lag_ok"]` say per frame whether the
    frame may be used for flow (False when its estimate failed or exceeded the
    bound; such a frame is returned unwarped). When the background is
    insufficient, unreliable or moving the whole step is skipped: frames come
    back unchanged, every frame is usable and `info["applied"]` is False with
    the reason ("insufficient_textured_background", "too_few_background_corners",
    "unreliable_background", "background_moving"). Observations are assumed
    5 s apart with lag frames 1 s earlier for the drift check (its threshold is
    in pixels over the window and its t-statistic is scale-free, so evenly
    spaced observations at any interval give the same answer);
    `extract_features_with_meta` passes the real timestamps.
    """
    if lag_frames is not None:
        obs, lag = _to_gray_u8(frames, lag_frames)
    else:
        (obs,) = _to_gray_u8(frames)
        lag = None
    m = _to_bool_masks(masks, obs.shape)
    stab = _estimate_stabilization(obs, m, lag, enabled=True)
    out = np.stack([_warp_image(obs[t], stab["obs_M"][t]) for t in range(len(obs))]) if len(obs) else obs
    out_m = np.stack([_warp_mask(m[t], stab["obs_M"][t]) for t in range(len(m))]) if len(m) else m
    out_lag = None
    if lag is not None:
        out_lag = np.stack([_warp_image(lag[t], stab["lag_M"][t]) for t in range(len(lag))])
    info = dict(stab["info"])
    info["obs_ok"] = [bool(v) for v in stab["obs_ok"]]
    info["lag_ok"] = None if lag is None else [bool(v) for v in stab["lag_ok"]]
    return out, out_m, out_lag, info


def select_window(timestamps, n: int = 12, interval_s: float = 5.0, start: Optional[float] = None,
                  max_offset_s: float = 0.5, flow_lag_s: Optional[float] = 1.0) -> Dict[str, Any]:
    """Pick `n` observations `interval_s` apart, each with a lag frame `flow_lag_s` earlier.

    `timestamps` are the native frame times (s, strictly increasing). The first
    target is `start` (default: the earliest time that still has a lag frame).
    Returns {"obs_idx" (n,), "lag_idx" (n,), "offsets" (n,) obs time - target,
    "lag_offsets", "targets", "obs_times", "lag_times", "native_dt", "flow_mode"}.

    Raises IncompatibleFrameTiming when the native frame spacing exceeds
    flow_lag_s / 2 (one-minute HPWREN / FIgLib imagery cannot give lag pairs
    and is refused, never stretched into "motion"), the clip is too short, a
    selected frame is more than `max_offset_s` from its target, or indices
    repeat.

    flow_lag_s None or 0 selects for consecutive-observation flow (a model
    trained that way, e.g. the far-field one-frame-per-minute profile): no lag
    frames ("lag_idx", "lag_offsets", "lag_times" are None, "flow_mode" is
    "consecutive"), the first target defaults to the first frame, and any
    native frame rate that puts a frame within `max_offset_s` of every target
    is accepted (one-minute imagery included).
    """
    consecutive = flow_lag_s is None or (np.isfinite(flow_lag_s) and float(flow_lag_s) == 0.0)
    if n < 1 or not (interval_s > 0) or not (consecutive or flow_lag_s > 0) or not (max_offset_s >= 0):
        raise ValueError("select_window needs n >= 1, interval_s > 0, flow_lag_s > 0 (or None / 0 for "
                         "consecutive flow), max_offset_s >= 0")
    ts = np.asarray(timestamps, dtype=np.float64).ravel()
    if ts.size < 2:
        raise IncompatibleFrameTiming(f"need at least two frames, got {ts.size}")
    if not np.all(np.isfinite(ts)):
        raise IncompatibleFrameTiming("timestamps contain non-finite values")
    d = np.diff(ts)
    if np.any(d <= 0):
        raise IncompatibleFrameTiming("timestamps must be strictly increasing (repeated or unordered frames)")
    native_dt = float(np.median(d))
    lag = 0.0 if consecutive else float(flow_lag_s)
    if not consecutive and native_dt > lag / 2.0:
        raise IncompatibleFrameTiming(
            f"native frame spacing {native_dt:.3g} s exceeds flow_lag_s / 2 = {lag / 2:.3g} s: "
            "this footage (e.g. one-minute HPWREN/FIgLib imagery) cannot provide short-lag optical flow")
    if start is None:
        start = float(ts[0]) + lag
    targets = float(start) + interval_s * np.arange(n, dtype=np.float64)
    if targets[-1] > ts[-1] + max_offset_s or targets[0] - lag < ts[0] - max_offset_s:
        raise IncompatibleFrameTiming(
            f"clip [{ts[0]:.3f}, {ts[-1]:.3f}] s is too short for {n} observations {interval_s} s apart "
            f"starting at {targets[0]:.3f} s" + ("" if consecutive else f" with a {lag} s flow lag"))
    obs_idx = _nearest_index(ts, targets)
    offsets = ts[obs_idx] - targets
    if np.any(np.abs(offsets) > max_offset_s + 1e-9):
        k = int(np.argmax(np.abs(offsets)))
        raise IncompatibleFrameTiming(f"no frame within {max_offset_s} s of target {targets[k]:.3f} s")
    if np.any(np.diff(obs_idx) <= 0):
        raise IncompatibleFrameTiming("observation indices repeat: interval_s is shorter than the frame spacing")
    if consecutive:
        return {
            "obs_idx": obs_idx,
            "lag_idx": None,
            "offsets": offsets,
            "lag_offsets": None,
            "targets": targets,
            "obs_times": ts[obs_idx],
            "lag_times": None,
            "native_dt": native_dt,
            "flow_mode": "consecutive",
        }
    lag_targets = ts[obs_idx] - lag
    lag_idx = _nearest_index(ts, lag_targets)
    lag_offsets = ts[lag_idx] - lag_targets
    if np.any(np.abs(lag_offsets) > max_offset_s + 1e-9):
        k = int(np.argmax(np.abs(lag_offsets)))
        raise IncompatibleFrameTiming(f"no lag frame within {max_offset_s} s of {lag_targets[k]:.3f} s")
    if np.any(lag_idx >= obs_idx):
        raise IncompatibleFrameTiming("a lag frame coincides with its observation (repeated index)")
    return {
        "obs_idx": obs_idx,
        "lag_idx": lag_idx,
        "offsets": offsets,
        "lag_offsets": lag_offsets,
        "targets": targets,
        "obs_times": ts[obs_idx],
        "lag_times": ts[lag_idx],
        "native_dt": native_dt,
        "flow_mode": "lag",
    }


def load_frames(path: PathLike, times=None, timestamps=None) -> Tuple[np.ndarray, np.ndarray]:
    """Grayscale uint8 frames (N, H, W) and their timestamps (N,) s from a video or an image folder.

    Video: timestamps from the container (CAP_PROP_POS_MSEC, else frame index /
    fps; `timestamps` overrides them). Image folder: timestamps from
    `timestamps` (one per image, sorted by filename), else a sidecar
    `timestamps.csv` (columns filename,timestamp_s), else filenames ending in
    `_t<seconds>s` (e.g. `cam3_t12.5s.jpg`); otherwise IncompatibleFrameTiming
    is raised (a folder of images is not assumed to have any frame rate).

    `times` (seconds): decode only the frame nearest each requested time (in
    the requested order); None decodes everything.
    """
    p = Path(path)
    if p.is_dir():
        return _load_image_folder(p, times, timestamps)
    if p.is_file():
        return _load_video(p, times, timestamps)
    raise FileNotFoundError(f"no video file or image folder at {p}")


# =============================================================================
# Input conversion
# =============================================================================


def _to_gray_u8(*arrays) -> List[np.ndarray]:
    """THE conversion of every image input to uint8 grayscale stacks (T, H, W).

    Each argument is a stack (T, H, W) or (T, H, W, C). uint8 passes through;
    bool -> {0, 255}; float is read as [0, 1] (as [0, 255] when its maximum
    exceeds 1.5; decided once over all float arguments, so observation and
    lag frames always share the scale) and rounded, so a float copy of a uint8
    clip converts back bit-identically; uint16 is scaled by 255/65535; other
    integers are clipped to [0, 255]. C = 3 (BGR) or 4 (BGRA) is converted with
    OpenCV's luma weights; C = 1 is squeezed.
    """
    arrs = [np.asarray(x) for x in arrays]
    float_max = [float(np.nanmax(a)) for a in arrs if a.dtype.kind == "f" and np.isfinite(a).any()]
    float_scale = 255.0 if not float_max or max(float_max) <= 1.5 else 1.0
    out = []
    for a in arrs:
        if a.ndim not in (3, 4):
            raise ValueError(f"expected frames (T, H, W) or (T, H, W, C), got shape {a.shape}")
        if a.dtype == np.uint8:
            u8 = a
        elif a.dtype == np.bool_:
            u8 = a.astype(np.uint8) * 255
        elif a.dtype.kind == "f":
            u8 = np.rint(np.clip(np.nan_to_num(a * float_scale, nan=0.0), 0.0, 255.0)).astype(np.uint8)
        elif a.dtype == np.uint16:
            u8 = np.rint(a.astype(np.float64) * (255.0 / 65535.0)).astype(np.uint8)
        elif a.dtype.kind in "iu":
            u8 = np.clip(a, 0, 255).astype(np.uint8)
        else:
            raise TypeError(f"unsupported image dtype {a.dtype}")
        if u8.ndim == 4:
            ch = u8.shape[-1]
            if ch == 1:
                u8 = u8[..., 0]
            elif ch in (3, 4) and len(u8):
                code = cv2.COLOR_BGR2GRAY if ch == 3 else cv2.COLOR_BGRA2GRAY
                u8 = np.stack([cv2.cvtColor(np.ascontiguousarray(f), code) for f in u8])
            elif ch in (3, 4):
                u8 = u8[..., 0]
            else:
                raise ValueError(f"unsupported channel count {ch}")
        out.append(np.ascontiguousarray(u8))
    return out


def _to_bool_masks(masks, shape: Tuple[int, int, int]) -> np.ndarray:
    """(T, H, W) bool; None -> all empty. Floats are thresholded at 0.5, integers at > 0."""
    if masks is None:
        return np.zeros(shape, dtype=bool)
    m = np.asarray(masks)
    if m.shape != tuple(shape):
        raise ValueError(f"masks shape {m.shape} does not match frames {tuple(shape)}")
    if m.dtype == np.bool_:
        return m
    if m.dtype.kind == "f":
        return np.nan_to_num(m, nan=0.0) > 0.5
    return m > 0


def _as_times(values, n: int, name: str) -> np.ndarray:
    ts = np.asarray(values, dtype=np.float64).ravel()
    if ts.shape != (n,):
        raise ValueError(f"{name} must have {n} entries, got {ts.shape}")
    return ts


def _check_config(cfg: FeatureConfig) -> None:
    if cfg.roi not in ("union_bbox", "full"):
        raise ValueError(f"FeatureConfig.roi must be 'union_bbox' or 'full', got {cfg.roi!r}")
    if cfg.flow_method not in ("dis", "farneback"):
        raise ValueError(f"FeatureConfig.flow_method must be 'dis' or 'farneback', got {cfg.flow_method!r}")
    if cfg.work_size < 16:
        raise ValueError("FeatureConfig.work_size must be >= 16")
    if not (float(cfg.expected_interval_s) > 0) or not (float(cfg.max_offset_s) >= 0):
        raise ValueError("FeatureConfig needs expected_interval_s > 0 and max_offset_s >= 0")
    if cfg.flow_lag_s is not None and not (float(cfg.flow_lag_s) >= 0):
        raise ValueError(f"FeatureConfig.flow_lag_s must be > 0, or None / 0 for consecutive flow, "
                         f"got {cfg.flow_lag_s!r}")


# =============================================================================
# Stabilisation
# =============================================================================


def _texture_map(img: np.ndarray) -> np.ndarray:
    """Pixels with trackable structure: 9x9 mean gradient of the blurred image above a noise floor."""
    g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), _STAB_TEXTURE_BLUR)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) * 0.125
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) * 0.125
    mag = cv2.boxFilter(cv2.magnitude(gx, gy), -1, (9, 9))
    return mag > _STAB_TEXTURE_GRAD


def _identity() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _estimate_stabilization(obs: np.ndarray, masks: np.ndarray, lag: Optional[np.ndarray],
                            enabled: bool = True, obs_times: Optional[np.ndarray] = None,
                            lag_times: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Similarity transforms (2x3, frame -> reference) of every frame onto observation 0.

    Returns {"obs_M", "lag_M": lists of 2x3 arrays, "obs_ok", "lag_ok": bool
    arrays (frame usable for flow), "info": JSON-able record}. The times (s)
    only feed the background-drift check; without them observation k is
    taken at 5 k s and its lag frame 1 s earlier.
    """
    n, h, w = obs.shape
    n_lag = 0 if lag is None else len(lag)
    obs_M = [_identity() for _ in range(n)]
    lag_M = [_identity() for _ in range(n_lag)]
    obs_ok = np.ones(n, dtype=bool)
    lag_ok = np.ones(n_lag, dtype=bool)
    info: Dict[str, Any] = {"enabled": bool(enabled), "applied": False, "reason": None,
                            "background_frac": None, "textured_frac": None, "n_corners": 0,
                            "background_drift_px": None, "frames": [], "lag_frames": []}

    def done(reason: Optional[str]) -> Dict[str, Any]:
        info["reason"] = reason
        if not info["frames"]:
            info["frames"] = [{"used": False, "reason": reason} for _ in range(n)]
            info["lag_frames"] = [{"used": False, "reason": reason} for _ in range(n_lag)]
        return {"obs_M": obs_M, "lag_M": lag_M, "obs_ok": obs_ok, "lag_ok": lag_ok, "info": info}

    if not enabled:
        return done("disabled")
    if n == 0 or (n < 2 and n_lag == 0):
        return done("single_frame")

    # Background = outside the dilated union of every candidate mask in the window.
    r = max(5, int(round(_STAB_DILATE_FRAC * min(h, w))))
    union = masks.any(axis=0).astype(np.uint8)
    dil = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    background = ~dil
    textured = background & _texture_map(obs[0])
    info["background_frac"] = float(background.mean())
    info["textured_frac"] = float(textured.mean())
    if textured.mean() < _STAB_MIN_TEXTURED_FRAC:
        return done("insufficient_textured_background")

    # Estimate on a (possibly downscaled) copy; transforms are scaled back.
    s = min(1.0, _STAB_EST_MAX_SIDE / float(min(h, w)))
    ref = _prep_estimation(obs[0], s)
    tex_small = textured if s == 1.0 else cv2.resize(
        textured.astype(np.uint8), (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    tex_u8 = tex_small.astype(np.uint8) * 255
    # Corners keep away from the image border: content there enters / leaves with the shake.
    margin = max(4, int(round(_STAB_BORDER_FRAC * min(ref.shape))))
    corner_mask = np.zeros_like(tex_u8)
    corner_mask[margin:-margin, margin:-margin] = tex_u8[margin:-margin, margin:-margin]
    corners = cv2.goodFeaturesToTrack(ref, _STAB_MAX_CORNERS, 0.01, 7, mask=corner_mask, blockSize=7)
    info["n_corners"] = 0 if corners is None else int(len(corners))
    if corners is None or len(corners) < _STAB_MIN_CORNERS:
        return done("too_few_background_corners")

    def estimate(img: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        tgt = _prep_estimation(img, s)
        M_small, rec = _similarity_to_reference(ref, tgt, corners, margin)
        if M_small is None:
            return _identity(), False, rec
        M = M_small.copy()
        M[:, 2] /= s                                   # translation back to native px
        rec.update(_transform_record(M, h, w))
        within = (rec["shift_px"] <= _STAB_MAX_SHIFT_FRAC * min(h, w)
                  and abs(rec["angle_deg"]) <= _STAB_MAX_ROTATION_DEG
                  and abs(rec["scale"] - 1.0) <= _STAB_MAX_SCALE_DEV)
        if not within:
            rec["used"], rec["reason"] = False, "bound_exceeded"
            return _identity(), False, rec
        rec["used"], rec["reason"] = True, None
        return M, True, rec

    info["frames"].append({"used": True, "reason": "reference", "dx": 0.0, "dy": 0.0, "angle_deg": 0.0,
                           "scale": 1.0, "shift_x": 0.0, "shift_y": 0.0, "shift_px": 0.0,
                           "inliers": None, "residual_px": None})
    for t in range(1, n):
        obs_M[t], obs_ok[t], rec = estimate(obs[t])
        info["frames"].append(rec)
    for t in range(n_lag):
        lag_M[t], lag_ok[t], rec = estimate(lag[t])
        info["lag_frames"].append(rec)

    # Most estimates failing the quality check means the background does not
    # support tracking (e.g. heavy noise counted as texture): skip, as for an
    # untextured background, rather than invalidating every frame's flow.
    recs = info["frames"][1:] + info["lag_frames"]
    n_bad = sum(r["reason"] in ("tracking_failed", "estimate_failed", "low_quality") for r in recs)
    if recs and n_bad > _STAB_MAX_FAILED_FRAC * len(recs):
        _reset(obs_M, obs_ok, lag_M, lag_ok, info, "unreliable_background")
        return done("unreliable_background")

    # A camera shakes about a fixed pose; a "background" that drifts steadily is
    # unmasked scene content (e.g. a fog bank the detector only partly covers).
    # Aligning to it would remove the very motion we want to measure, so the
    # whole step is skipped instead.
    t_obs = np.arange(n) * 5.0 if obs_times is None else np.asarray(obs_times, dtype=np.float64)
    t_lag = (np.arange(n_lag) * 5.0 - 1.0) if lag_times is None else np.asarray(lag_times, dtype=np.float64)
    recs = info["frames"] + info["lag_frames"]
    oks = np.concatenate([obs_ok, lag_ok])
    times = np.concatenate([t_obs, t_lag])
    shifts = np.array([[r.get("shift_x", 0.0), r.get("shift_y", 0.0)] for r in recs])
    drift, t_stat = _drift(times[oks], shifts[oks])
    info["background_drift_px"] = drift
    if drift > _STAB_MAX_DRIFT_FRAC * min(h, w) and t_stat > _STAB_DRIFT_T:
        _reset(obs_M, obs_ok, lag_M, lag_ok, info, "background_moving")
        return done("background_moving")
    info["applied"] = True
    return done(None)


def _reset(obs_M, obs_ok, lag_M, lag_ok, info, reason: str) -> None:
    """Undo every estimate (identity, all frames usable) and record why, keeping the per-frame records."""
    for rec in info["frames"] + info["lag_frames"]:
        rec["used"], rec["reason"] = False, rec["reason"] if rec["reason"] == "reference" else reason
    for t in range(len(obs_M)):
        obs_M[t], obs_ok[t] = _identity(), True
    for t in range(len(lag_M)):
        lag_M[t], lag_ok[t] = _identity(), True


def _drift(times: np.ndarray, shifts: np.ndarray) -> Tuple[float, float]:
    """Linear drift of the estimated camera offsets over the window (px) and its t-statistic."""
    if len(times) < 4 or np.ptp(times) <= 0:
        return 0.0, 0.0
    tc = times - times.mean()
    sxx = float(tc @ tc)
    slope = tc @ (shifts - shifts.mean(axis=0)) / sxx                  # (2,) px / s
    resid = shifts - shifts.mean(axis=0) - np.outer(tc, slope)
    dof = max(len(times) - 2, 1)
    se = np.sqrt((resid ** 2).sum(axis=0) / dof / sxx) + 1e-9
    return float(np.hypot(*slope) * np.ptp(times)), float(np.max(np.abs(slope) / se))


def _prep_estimation(img: np.ndarray, s: float) -> np.ndarray:
    """Downscale (if needed) and band-pass the image for corner tracking.

    A light blur suppresses pixel noise; subtracting a broad blur removes
    exposure flicker and slow illumination changes, which bias Lucas-Kanade at
    strong edges (a global gain/offset match measured on the shaken frame was
    worse: ~0.3 px errors against ~0.05 px with the band-pass).
    """
    if s < 1.0:
        img = cv2.resize(img, (max(1, int(round(img.shape[1] * s))), max(1, int(round(img.shape[0] * s)))),
                         interpolation=cv2.INTER_AREA)
    fine = cv2.GaussianBlur(img, (5, 5), 1.0)
    k = (_STAB_BANDPASS_BOX, _STAB_BANDPASS_BOX)
    broad = cv2.blur(cv2.blur(img, k), k)            # ~Gaussian, sigma ~4.5 px
    return cv2.addWeighted(fine, 2.0, broad, -2.0, 128.0)


def _similarity_to_reference(ref: np.ndarray, tgt: np.ndarray, corners: np.ndarray, margin: int
                             ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """LK-track reference corners into `tgt`, RANSAC similarity tgt -> ref, quality check.

    Tracks that end within `margin` / 2 of the border are dropped (half-visible windows).
    """
    rec: Dict[str, Any] = {"used": False, "reason": None, "inliers": 0, "residual_px": None}
    pts, st, _ = cv2.calcOpticalFlowPyrLK(
        ref, tgt, corners, None, winSize=(_STAB_LK_WIN, _STAB_LK_WIN), maxLevel=_STAB_LK_LEVELS,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.01))
    good = (st.ravel() == 1) if st is not None else np.zeros(len(corners), bool)
    if pts is not None:
        q = pts.reshape(-1, 2)
        edge = margin / 2.0
        good &= np.isfinite(q).all(axis=1)
        good &= ((q[:, 0] >= edge) & (q[:, 0] <= tgt.shape[1] - 1 - edge)
                 & (q[:, 1] >= edge) & (q[:, 1] <= tgt.shape[0] - 1 - edge))
    if good.sum() < _STAB_MIN_CORNERS:
        rec["reason"] = "tracking_failed"
        return None, rec
    src = pts.reshape(-1, 2)[good].astype(np.float32)
    dst = corners.reshape(-1, 2)[good].astype(np.float32)
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=_STAB_RANSAC_PX,
                                         maxIters=500, confidence=0.995, refineIters=10)
    if M is None or inl is None:
        rec["reason"] = "estimate_failed"
        return None, rec
    inl = inl.ravel().astype(bool)
    proj = src[inl] @ M[:, :2].T + M[:, 2]
    resid = float(np.sqrt(np.mean(np.sum((proj - dst[inl]) ** 2, axis=1)))) if inl.any() else float("inf")
    rec["inliers"] = int(inl.sum())
    rec["residual_px"] = resid if math.isfinite(resid) else None
    if inl.sum() < _STAB_MIN_INLIERS or inl.mean() < _STAB_MIN_INLIER_RATIO or not resid <= _STAB_MAX_RESIDUAL_PX:
        rec["reason"] = "low_quality"
        return None, rec
    return M.astype(np.float64), rec


def _transform_record(M: np.ndarray, h: int, w: int) -> Dict[str, float]:
    a, b = M[0, 0], M[1, 0]
    c = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    shift = M[:, :2] @ c + M[:, 2] - c
    return {"dx": float(M[0, 2]), "dy": float(M[1, 2]), "angle_deg": float(math.degrees(math.atan2(b, a))),
            "scale": float(math.hypot(a, b)), "shift_x": float(shift[0]), "shift_y": float(shift[1]),
            "shift_px": float(np.hypot(*shift))}


def _is_identity(M: np.ndarray) -> bool:
    return bool(np.array_equal(M, _identity()))


def _warp_image(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    if _is_identity(M):
        return img.copy()
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _warp_mask(mask: np.ndarray, M: np.ndarray) -> np.ndarray:
    if _is_identity(M) or not mask.any():
        return mask.copy()
    return cv2.warpAffine(mask.astype(np.uint8), M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0


# =============================================================================
# Masks and analysis frame
# =============================================================================


def _persistent_filter(masks: np.ndarray) -> Tuple[np.ndarray, List[int], int]:
    """Keep a component only if it overlaps the dilated mask of the previous or next observation."""
    n, h, w = masks.shape
    r = max(3, int(round(_PERSIST_DILATE_FRAC * min(h, w))))
    if n < 2:
        return masks.copy(), [0] * n, r
    kernel = np.ones((2 * r + 1, 2 * r + 1), np.uint8)           # square: separable, fast
    dilated = [cv2.dilate(m.view(np.uint8), kernel) > 0 if m.any() else m for m in masks]
    out = np.zeros_like(masks)
    removed = [0] * n
    for t in range(n):
        if not masks[t].any():
            continue
        count, labels = cv2.connectedComponents(masks[t].view(np.uint8), connectivity=8)
        neighbours = np.zeros((h, w), dtype=bool)
        if t > 0:
            neighbours |= dilated[t - 1]
        if t < n - 1:
            neighbours |= dilated[t + 1]
        hit = np.bincount(labels[neighbours], minlength=count) > 0
        hit[0] = False
        out[t] = hit[labels]
        removed[t] = int(count - 1 - hit.sum())
    return out, removed, r


def _weighted_extent(weights: np.ndarray, lo: float = 0.02, hi: float = 0.98) -> Tuple[int, int]:
    """Index range holding the central lo..hi share of a 1D weight profile (end-exclusive)."""
    c = np.cumsum(weights, dtype=np.float64)
    total = c[-1]
    a = int(np.searchsorted(c, lo * total, side="left"))
    b = int(np.searchsorted(c, hi * total, side="left")) + 1
    return a, max(b, a + 1)


def _roi_box(counts: np.ndarray, cfg: FeatureConfig) -> Tuple[Tuple[int, int, int, int], str]:
    """(x0, y0, x1, y1) of the analysis frame, end-exclusive, native px; and why.

    `counts` is the number of frames in which each pixel is inside a
    candidate mask. The box spans the central 96 % of those mask pixels (so a
    stray false-positive blob far from the region cannot inflate it), then
    is expanded by `roi_scale`.
    """
    h, w = counts.shape
    if cfg.roi == "full":
        return (0, 0, w, h), "full_config"
    if not counts.any():
        return (0, 0, w, h), "full_no_mask"
    by0, by1 = _weighted_extent(counts.sum(axis=1))
    bx0, bx1 = _weighted_extent(counts.sum(axis=0))
    side = max(cfg.roi_scale * max(bx1 - bx0, by1 - by0), float(cfg.roi_min_px))
    side_w = int(min(w, max(1, round(side))))
    side_h = int(min(h, max(1, round(side))))
    cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    x0 = int(min(max(round(cx - side_w / 2.0), 0), w - side_w))
    y0 = int(min(max(round(cy - side_h / 2.0), 0), h - side_h))
    reason = "union_bbox" if (side_w == round(side) and side_h == round(side)) else "union_bbox_clipped"
    return (x0, y0, x0 + side_w, y0 + side_h), reason


def _work_shape(h: int, w: int, work_size: int) -> Tuple[int, int]:
    s = work_size / float(max(h, w))
    return max(1, int(round(h * s))), max(1, int(round(w * s)))


def _to_work(img: np.ndarray, M: np.ndarray, box, work_shape) -> np.ndarray:
    """Stabilise (warp straight into the ROI) and resize to the analysis frame."""
    x0, y0, x1, y1 = box
    if _is_identity(M):
        crop = img[y0:y1, x0:x1]
    else:
        Mo = M.copy()
        Mo[:, 2] -= (x0, y0)
        crop = cv2.warpAffine(img, Mo, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return _resize(crop, work_shape)


def _resize(img: np.ndarray, work_shape) -> np.ndarray:
    hh, ww = work_shape
    if img.shape == (hh, ww):
        return np.ascontiguousarray(img)
    interp = cv2.INTER_AREA if hh < img.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(img, (ww, hh), interpolation=interp)


def _mask_to_work(mask: np.ndarray, work_shape) -> np.ndarray:
    """Area-weighted (down) or bilinear (up) resize of the 0/255 mask, kept where >= 50 %."""
    if not mask.any():
        return np.zeros(work_shape, dtype=bool)
    return _resize(mask.astype(np.uint8) * np.uint8(255), work_shape) >= 128


# =============================================================================
# Geometry
# =============================================================================


def _centroid(mask: np.ndarray) -> np.ndarray:
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])


def _roughness(mask: np.ndarray, min_px: int) -> float:
    """P / (2 sqrt(pi A)) - 1 minus the digital-disc bias (0 for a disc); ROI-border segments excluded."""
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    area = int(m.sum())
    if area < max(min_px, 1):
        return 0.0
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = _perimeter(contours, m.shape)
    value = perim / (2.0 * math.sqrt(math.pi * area)) - 1.0 - _disc_bias(int(round(4 * math.sqrt(area / math.pi))))
    return max(0.0, value)


def _perimeter(contours, shape) -> float:
    """Closed contour length (as cv2.arcLength) without segments that run along the ROI border."""
    h, w = shape
    total = 0.0
    for c in contours:
        p = c.reshape(-1, 2).astype(np.float64)
        if len(p) < 2:
            continue
        q = np.roll(p, -1, axis=0)
        seg = np.hypot(q[:, 0] - p[:, 0], q[:, 1] - p[:, 1])
        on_border = (((p[:, 0] == 0) & (q[:, 0] == 0)) | ((p[:, 0] == w - 1) & (q[:, 0] == w - 1))
                     | ((p[:, 1] == 0) & (q[:, 1] == 0)) | ((p[:, 1] == h - 1) & (q[:, 1] == h - 1)))
        total += float(seg[~on_border].sum())
    return total


@lru_cache(maxsize=1024)
def _disc_bias(radius_q4: int) -> float:
    """Roughness value of a digital disc of radius radius_q4 / 4 px (the zero point)."""
    r = max(radius_q4 / 4.0, 0.5)
    size = int(2 * math.ceil(r) + 7)
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    disc = (((xx - c) ** 2 + (yy - c) ** 2) <= r * r).astype(np.uint8)
    area = int(disc.sum())
    if area == 0:
        return 0.0
    contours, _ = cv2.findContours(disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(cn, True) for cn in contours)
    return perim / (2.0 * math.sqrt(math.pi * area)) - 1.0


def _bottom_rows(mask: np.ndarray) -> np.ndarray:
    """(W,) lowest occupied row per column (image rows grow downward); -1 for empty columns."""
    h = mask.shape[0]
    occupied = mask.any(axis=0)
    last = h - 1 - np.argmax(mask[::-1], axis=0)
    return np.where(occupied, last, -1)


def _anchor_stability(b_t: np.ndarray, b_p: np.ndarray, h: int, L: float, band_px: int) -> float:
    """exp(-median |dz_bottom| / (0.02 L)) * IoU of the bottom-band column sets of t and t-1.

    The bottom band of a frame is the set of columns whose lowest row lies
    within `band_px` of the region's lowest row: the source end of a plume,
    the base of a fog layer. 0 when more than half of either band lies on the
    ROI's bottom edge (the anchor is not observed).
    """
    sets = []
    for b in (b_t, b_p):
        occ = b >= 0
        if not occ.any():
            return 0.0
        band = occ & (b >= b[occ].max() - band_px)
        if np.mean(b[band] == h - 1) > 0.5:
            return 0.0
        sets.append(band)
    inter = sets[0] & sets[1]
    if not inter.any():
        return 0.0
    iou = inter.sum() / float((sets[0] | sets[1]).sum())
    med = float(np.median(np.abs(b_t[inter] - b_p[inter])))
    return float(math.exp(-med / (_ANCHOR_DZ_SCALE * L)) * iou)


# =============================================================================
# Optical flow
# =============================================================================


class _FlowEngine:
    """One flow method for the whole clip (DIS PRESET_MEDIUM by default)."""

    def __init__(self, method: str):
        self.method = method
        self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM) if method == "dis" else None
        self._dis_init = None                    # DIS that refines a given start (created on first use)

    def __call__(self, a: np.ndarray, b: np.ndarray, start: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Flow (h, w, 2) float32 in px: a(p) ~ b(p + flow(p)); `start` a uniform initial flow (dx, dy)."""
        if start is None:
            if self._dis is not None:
                return self._dis.calc(a, b, None)
            return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = np.empty(a.shape + (2,), dtype=np.float32)
        flow[..., 0], flow[..., 1] = start
        if self._dis is None:
            return cv2.calcOpticalFlowFarneback(a, b, flow, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_USE_INITIAL_FLOW)
        if self._dis_init is None:
            self._dis_init = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        return self._dis_init.calc(a, b, flow)     # DIS refines a passed CV_32FC2 flow of the same size as its start


def _flow_crop(masks: Sequence[np.ndarray], margin: int, shape) -> Tuple[int, int, int, int]:
    """Box (x0, y0, x1, y1) around the masks plus margin, at least _FLOW_MIN_SIDE_PX, inside the frame."""
    h, w = shape
    union = np.zeros(shape, dtype=bool)
    for m in masks:
        union |= m
    rows = np.flatnonzero(union.any(axis=1))
    cols = np.flatnonzero(union.any(axis=0))
    box = []
    for lo, hi, size in ((cols[0], cols[-1] + 1, w), (rows[0], rows[-1] + 1, h)):
        lo, hi = max(0, lo - margin), min(size, hi + margin)
        if hi - lo < _FLOW_MIN_SIDE_PX:
            c = (lo + hi) / 2.0
            lo = int(max(0, min(size - _FLOW_MIN_SIDE_PX, round(c - _FLOW_MIN_SIDE_PX / 2.0))))
            hi = min(size, lo + _FLOW_MIN_SIDE_PX)
        box += [int(lo), int(hi)]
    return box[0], box[2], box[1], box[3]


def _winsorized(values: np.ndarray) -> Tuple[float, float]:
    """Mean and std after clipping to the 5th / 95th percentile (order statistics, no interpolation)."""
    n = values.size
    k_lo = int(round(_WINSOR_PCT[0] / 100.0 * (n - 1)))
    k_hi = int(round(_WINSOR_PCT[1] / 100.0 * (n - 1)))
    part = np.partition(values, (k_lo, k_hi))
    c = np.clip(values, part[k_lo], part[k_hi])
    return float(c.mean()), float(c.std())


@lru_cache(maxsize=64)
def _pixel_grid(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    gx.flags.writeable = False
    gy.flags.writeable = False
    return gx, gy


def _affine_divergence(vel: np.ndarray, region: np.ndarray) -> float:
    """du/dx + dv/dy of the least-squares fit vel ~ c + J (x, y) over the region (normal equations)."""
    ys, xs = np.nonzero(region)
    x = xs - xs.mean()
    y = ys - ys.mean()
    uv = vel[region].astype(np.float64)
    # The centred design decouples the constant term; the 2x2 system gives the Jacobian rows.
    sxx, sxy, syy = float(x @ x), float(x @ y), float(y @ y)
    det = sxx * syy - sxy * sxy
    if not det > 1e-9 * max(sxx * syy, 1.0):
        return float("nan")                              # degenerate region (a line)
    bu = np.array([x @ uv[:, 0], y @ uv[:, 0]])
    bv = np.array([x @ uv[:, 1], y @ uv[:, 1]])
    du_dx = (syy * bu[0] - sxy * bu[1]) / det
    dv_dy = (sxx * bv[1] - sxy * bv[0]) / det
    return float(du_dx + dv_dy)


def _flow_stats(prev: np.ndarray, cur: np.ndarray, mask: np.ndarray, crop_masks, margin: int,
                cfg: FeatureConfig, engine: _FlowEngine,
                guesses: Sequence[Tuple[float, float]] = ()) -> Dict[str, Any]:
    """Flow statistics of one pair (prev -> cur) on the eroded mask of `cur`, in working px.

    The forward flow prev -> cur (lag -> observation) is sampled where each
    observation pixel came from and averaged with the negated backward flow,
    so the velocity field lives on the observation grid, where the mask is
    defined; their disagreement is the forward-backward error.

    `guesses` are extra uniform starts (dx, dy) in working px (consecutive
    mode: the centroid displacement). Each start gives a candidate flow; one
    that passes the forward-backward check replaces the zero start only if it
    reconstructs `cur` from `prev` with a photometric residual at least
    `_GUESS_MIN_GAIN` lower (a consistent zero flow is not proof of no motion:
    DIS also returns one when the true displacement is beyond its reach).
    Returns {"status", "region_px", "fb_px", "start" ("zero" | "centroid"),
    "stats": (u, v, std_u, std_v, div, |curl|)} with displacements per pair
    (the caller divides by the elapsed time).
    """
    x0, y0, x1, y1 = _flow_crop(crop_masks, margin, mask.shape)
    region = mask[y0:y1, x0:x1].astype(np.uint8)
    e = int(cfg.flow_erode_px)
    if e > 0:
        region = cv2.erode(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1)))
    region = region > 0
    n_px = int(region.sum())
    out: Dict[str, Any] = {"status": "ok", "region_px": n_px, "fb_px": None, "start": None, "stats": None}
    if n_px < _MIN_FLOW_REGION_PX:
        out["status"] = "region_too_small"
        return out
    a = np.ascontiguousarray(prev[y0:y1, x0:x1])
    b = np.ascontiguousarray(cur[y0:y1, x0:x1])
    gx, gy = _pixel_grid(*b.shape)
    # Large regions: the robust statistics use every other pixel in each direction
    # (same estimates, a quarter of the cost); |curl| still averages every pixel.
    sample = region
    if n_px > _STATS_SUBSAMPLE_ABOVE_PX:
        sample = np.zeros_like(region)
        sample[::2, ::2] = region[::2, ::2]
    candidates = []
    for start in (None,) + tuple(guesses):
        try:
            fwd = engine(a, b, start)
            bwd = engine(b, a, None if start is None else (-start[0], -start[1]))
        except cv2.error:
            continue
        fwd_at_src = cv2.remap(fwd, gx + bwd[..., 0], gy + bwd[..., 1], cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        fb = np.hypot(*(bwd + fwd_at_src)[sample].T)
        cand = {"start": "zero" if start is None else "centroid", "fb": float(np.median(fb)),
                "vel": 0.5 * (fwd_at_src - bwd)}          # displacement prev -> cur at cur's pixels
        if guesses:
            cand["resid"] = _photometric_residual(a, b, cand["vel"], sample, gx, gy)
        candidates.append(cand)
    if not candidates:
        out["status"] = "flow_failed"
        return out
    passing = [c for c in candidates if c["fb"] <= cfg.fb_max_px]
    if passing:
        best = passing[0]
        for c in passing[1:]:
            if c["resid"] < (1.0 - _GUESS_MIN_GAIN) * best["resid"]:
                best = c
    else:
        best = min(candidates, key=lambda c: c["fb"] if math.isfinite(c["fb"]) else math.inf)
    fb_med = best["fb"]
    out["fb_px"] = fb_med if math.isfinite(fb_med) else None
    out["start"] = best["start"]
    if not (fb_med <= cfg.fb_max_px):
        out["status"] = "fb_inconsistent"
        return out
    vel = best["vel"]
    u, v = vel[..., 0], vel[..., 1]
    mu, su = _winsorized(u[sample])
    mv, sv = _winsorized(v[sample])
    div = _affine_divergence(vel, sample)               # trace of the affine fit's Jacobian
    # vorticity: mean |dv/dx - du/dy| from pixel derivatives (sign-free, so no flip needed)
    curl = np.gradient(v, axis=1) - np.gradient(u, axis=0)
    out["stats"] = (mu, mv, su, sv, div, float(np.mean(np.abs(curl[region]))))
    return out


def _photometric_residual(prev: np.ndarray, cur: np.ndarray, vel: np.ndarray, sample: np.ndarray,
                          gx: np.ndarray, gy: np.ndarray) -> float:
    """Mean |cur(p) - prev(p - vel(p))| over the sample (gray levels): how well a flow explains the pair."""
    warped = cv2.remap(prev, gx - vel[..., 0], gy - vel[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return float(np.mean(np.abs(warped.astype(np.float32) - cur.astype(np.float32))[sample]))


# =============================================================================
# Timing
# =============================================================================


def _opt_float(v) -> Optional[float]:
    v = float(v)
    return v if math.isfinite(v) else None


def _timing_meta(ts: np.ndarray, lag_ts: Optional[np.ndarray], cfg: FeatureConfig, flow_mode: str) -> Dict[str, Any]:
    """Achieved intervals and whether they match what the model was trained on.

    Each observation may sit `max_offset_s` from its target, so an interval is
    compatible within 2 x max_offset_s of `expected_interval_s` (far-field
    profile: 60 +- 20 s), a lag within max_offset_s of `flow_lag_s`. The flow
    mode must be the configuration's: consecutive flow under a lag
    configuration is flagged "consecutive_flow"; under a consecutive
    configuration it is the expected mode and `lag_compatible` is None (not
    applicable). `expected_flow_lag_s` is 0.0 for a consecutive configuration.
    """
    intervals = np.diff(ts)
    reasons = []
    interval_ok = bool(np.all(np.abs(intervals - cfg.expected_interval_s) <= 2 * cfg.max_offset_s + 1e-9))
    if not interval_ok:
        reasons.append("observation_interval")
    expected_mode = cfg.flow_mode
    lag_ok = None
    lags = None
    if lag_ts is not None:
        lags = ts - lag_ts
        lag_ok = bool(np.all(np.abs(lags - cfg.flow_lag_s) <= cfg.max_offset_s + 1e-9))
        if not lag_ok:
            reasons.append("flow_lag")
    mode_ok = flow_mode == expected_mode
    if flow_mode == "consecutive" and not mode_ok:
        reasons.append("consecutive_flow")
    return {
        "intervals_s": [_opt_float(v) for v in intervals],
        "flow_lags_s": None if lags is None else [_opt_float(v) for v in lags],
        "expected_interval_s": float(cfg.expected_interval_s),
        "expected_flow_lag_s": float(cfg.flow_lag_s) if expected_mode == "lag" else 0.0,
        "expected_flow_mode": expected_mode,
        "max_offset_s": float(cfg.max_offset_s),
        "interval_compatible": interval_ok,
        "lag_compatible": lag_ok,
        "flow_mode_compatible": mode_ok,
        "compatible": interval_ok and mode_ok and lag_ok is not False,
        "reasons": reasons,
    }


def _nearest_index(ts: np.ndarray, targets: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(ts, targets), 1, len(ts) - 1)
    left = ts[idx - 1]
    right = ts[idx]
    return np.where(np.abs(targets - left) <= np.abs(right - targets), idx - 1, idx).astype(np.int64)


# =============================================================================
# Loading footage
# =============================================================================


def _container_timestamps_usable(path: Path) -> bool:
    """True when the backend reports increasing per-frame times (probe of the first frames)."""
    cap = cv2.VideoCapture(str(path))
    try:
        stamps = []
        while len(stamps) < 3 and cap.grab():
            stamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    finally:
        cap.release()
    return len(stamps) < 2 or bool(np.all(np.diff(stamps) > 0))


def _load_video(path: Path, times, timestamps) -> Tuple[np.ndarray, np.ndarray]:
    """Single pass: grab every frame, convert only the best candidate for each requested time."""
    use_container = timestamps is None and _container_timestamps_usable(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if timestamps is None and not use_container and fps <= 0:
        cap.release()
        raise IncompatibleFrameTiming(f"{path}: no container timestamps and no frame rate")
    override = None if timestamps is None else np.asarray(timestamps, dtype=np.float64).ravel()
    want = None if times is None else np.asarray(times, dtype=np.float64).ravel()
    frames: List[np.ndarray] = []
    stamps: List[float] = []
    if want is not None:
        best_err = np.full(len(want), np.inf)
        best_frame: List[Optional[np.ndarray]] = [None] * len(want)
        best_ts = np.full(len(want), np.nan)
    spacing = 1.0 / fps if fps > 0 else 0.0      # largest frame spacing seen so far
    i = 0
    try:
        while cap.grab():
            if override is not None:
                if i >= len(override):
                    break
                t = float(override[i])
            elif use_container:
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            else:
                t = i / fps
            if stamps:
                if not t > stamps[-1]:
                    raise IncompatibleFrameTiming(f"{path}: frame times stop increasing at frame {i}")
                spacing = max(spacing, t - stamps[-1])
            stamps.append(t)
            i += 1
            if want is None:
                ok, img = cap.retrieve()
                if not ok:
                    raise ValueError(f"{path}: cannot decode frame {i - 1}")
                frames.append(_to_gray_u8(img[None])[0][0])
                continue
            # The nearest frame to a target is the last one before it or the first one
            # after it; only frames inside that window are converted.
            err = np.abs(want - t)
            take = (err < best_err) & (t >= want - 1.5 * spacing)
            if take.any():
                ok, img = cap.retrieve()
                if ok:
                    gray = _to_gray_u8(img[None])[0][0]
                    for k in np.flatnonzero(take):
                        best_err[k], best_frame[k], best_ts[k] = err[k], gray, t
    finally:
        cap.release()
    if not stamps:
        raise ValueError(f"no frames decoded from {path}")
    if override is not None and len(stamps) != len(override):
        raise ValueError(f"{path}: {len(override)} timestamps for {len(stamps)} frames")
    if want is None:
        return np.stack(frames), np.asarray(stamps, dtype=np.float64)
    if any(f is None for f in best_frame):
        raise ValueError(f"{path}: could not decode a frame for every requested time")
    return np.stack(best_frame), best_ts


def _load_image_folder(folder: Path, times, timestamps) -> Tuple[np.ndarray, np.ndarray]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"no images in {folder}")
    if timestamps is not None:
        ts = np.asarray(timestamps, dtype=np.float64).ravel()
        if len(ts) != len(files):
            raise ValueError(f"{len(ts)} timestamps for {len(files)} images in {folder}")
    else:
        files, ts = _folder_timestamps(folder, files)
    order = np.argsort(ts, kind="stable")
    files = [files[k] for k in order]
    ts = ts[order]
    if times is None:
        pick = np.arange(len(files))
    else:
        pick = _nearest_index(ts, np.asarray(times, dtype=np.float64).ravel()) if len(files) > 1 else \
            np.zeros(len(np.ravel(times)), dtype=np.int64)
    frames = []
    for k in pick:
        img = cv2.imread(str(files[k]), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cannot read image {files[k]}")
        frames.append(img)
    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
        raise ValueError(f"images in {folder} have different sizes: {sorted(shapes)}")
    (gray,) = _to_gray_u8(np.stack(frames))
    return gray, ts[pick]


def _folder_timestamps(folder: Path, files: List[Path]) -> Tuple[List[Path], np.ndarray]:
    for name in _SIDECAR_NAMES:
        sidecar = folder / name
        if sidecar.is_file():
            with open(sidecar, "r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None or not {"filename", "timestamp_s"} <= set(reader.fieldnames):
                    raise ValueError(f"{sidecar} must have the columns filename,timestamp_s")
                table = {row["filename"].strip(): float(row["timestamp_s"]) for row in reader}
            listed = [p for p in files if p.name in table]
            if not listed:
                raise IncompatibleFrameTiming(f"{sidecar} lists none of the images in {folder}")
            return listed, np.array([table[p.name] for p in listed], dtype=np.float64)
    parsed = [_FILENAME_TIME.search(p.stem) for p in files]
    if all(m is not None for m in parsed):
        return files, np.array([float(m.group(1)) for m in parsed], dtype=np.float64)
    raise IncompatibleFrameTiming(
        f"no timestamps for the images in {folder}: pass `timestamps`, add timestamps.csv "
        "(filename,timestamp_s) or name files *_t<seconds>s.<ext>")
