"""Synthetic grayscale camera frames, lag frames and candidate masks from a simulation.

The renderer turns simulated concentration fields into what a fixed
surveillance camera would record, so the network only ever sees images (and
the motion features extracted from them), never CFD fields. Every visual
choice is drawn from the same distributions for every class; the class only
decides which component group is exposure-normalised as "primary".

Optical model (docs/design.md section 6):

1. Per component c in {smoke, fog, tracer}: optical depth on the grid
   tau_c = beta_c * L_c, beta = mass extinction * concentration (fog:
   thermo.fog_extinction(LWC)), L_c an independent line-of-sight depth.
2. View: a zoom window inside the interior of the domain (sponge layers
   cropped) whose bottom edge is the simulated floor; a textured foreground
   ground band fills the image below it, terrain (sim.solid) is drawn as
   ground. Each frame has its own camera shake / rotation (and zoom jitter
   with a strong artefact), a stationary Ornstein-Uhlenbeck (AR(1)) process
   evaluated at the actual frame times: stationary std camera_shake_px /
   camera_rotation_deg / zoom_jitter, correlation time camera_shake_tau_s,
   so a lag frame 1 s before its observation shakes almost with it while
   frames a minute apart are independent. The frame's pixel -> world mapping
   is used for everything (background, fields, texture, occluders, shadow,
   cloud), so frames, opacities and masks all live in the same shaken
   coordinates.
3. tau is upsampled bilinearly (cv2.remap) and modulated by an ADVECTED
   TEXTURE: a tileable multi-octave noise pattern in metres is sampled at
   the advected coordinates sim.texcoords (pairs a and b blended with
   sim.tex_weights, normalised so the texture variance does not dip during
   a cross-fade), giving a factor exp(g n - g^2 / 2) with one gain g per
   sequence for all components. The texture moves exactly with the air.
   Where the simulation provides the subgrid (Smagorinsky) velocity scale
   sim.sgs_u, the sampling coordinates are further displaced by a smooth,
   zero-mean random field D(x, z, t) (random Fourier modes a few cells long
   whose phases follow integrated OU processes, one eddy clock per octave
   of u_sgs) with local velocity scale texture.sgs_gain * u_sgs and local
   correlation time ~ eddy size / (the cell's peak u_sgs over the
   sequence), so the texture also "boils" with
   the unresolved turbulence, wherever it is and whatever the class (see
   _sgs_displacement). Without sim.sgs_u, or with sgs_gain 0, nothing moves.
4. Exposure normalisation per group, on the untextured in-view peak tau of
   the observation frames (lag frames get the same scale): primary smoke /
   fog / neither-aerosol tracer -> render.peak_opacity, background-haze
   tracer -> render.background_haze_opacity; raw peak tau < 1e-6 is left
   alone; unlabelled smoke / fog keep their physical scale.
5. alpha = 1 - exp(-sum tau), I = (1 - alpha) B + alpha A with A the
   tau-weighted mean of per-component brightnesses drawn from ONE
   distribution. Then distant cloud (behind the aerosol), moving cloud
   shadow, occluders, global haze, contrast / brightness / exposure
   flicker, sensor noise and JPEG at the native size.
6. Candidate masks from the visible-contrast map (alpha |A - B| + cloud
   contrast + shadow brightness change) via masks.degrade_masks, boxes of
   those, or masks.detect_candidate_masks on the rendered frames.

Lag frames: a simulation with lag snapshots (sim.lag_times, `_lag` fields)
gets one lag frame per observation, rendered with the same parameters. A
simulation without them (flow_lag_s null or 0, e.g. the one-frame-per-minute
far-field profile) gets the observation frames only and
`RenderedSequence.lag_frames` / `lag_timestamps` are None; masks,
visibility and exposure are computed from the observations in both cases.

Randomness: `sample_render_params` draws every per-variant value (it never
sees the class; hints only switch nuisances on, and the stream is the same
with or without them; the version-2 keys camera_shake_tau_s and
texture.sgs_gain / sgs_seed are drawn last, so every earlier value is what
version 1 drew; stored version-1 params, which lack them, still render as
before: independent shake per frame, no subgrid displacement).
Background, texture, subgrid-turbulence, occluder, shadow
and cloud shapes come from seeds stored in the params. `render_sequence`
draws the per-frame values from its `rng` in a fixed order (camera: one
4-vector of shake innovations per frame, observation and lag frames
interleaved in time order; exposure flicker; sensor noise; then masks),
with F = 2T frames with lag frames and F = T without, so a rendering is a
deterministic function of (sim, params, rng seed), and two simulations
with the same frame times rendered with the same params and seed get the
same camera motion and sensor noise.

Where this fits: the dataset generator calls sample_render_params ->
render_sequence -> check_label_visibility for every rendered variant of a
valid SimulationResult; robustness suites pass `overrides`.
Calls into: numpy, OpenCV, src.cfd.thermo (fog optics), src.data.masks,
src.utils.config / io (default render block, background manifest).
Start reading at: render_sequence, then _sample_layers and _composite_frame.
"""

from __future__ import annotations

import copy
import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.cfd.results import SimulationResult
from src.cfd.thermo import fog_extinction
from src.data.masks import _false_positive_tracks, boxes_from_masks, degrade_masks, detect_candidate_masks
from src.utils.config import REPO_ROOT, load_simulation_config
from src.utils.io import read_json

COMPONENTS: Tuple[str, ...] = ("smoke", "fog", "tracer")
OPACITY_COMPONENTS: Tuple[str, ...] = ("smoke", "fog", "tracer", "cloud")
MASK_SOURCES: Tuple[str, ...] = ("degraded", "bbox", "detector")
HINT_KEYS: Tuple[str, ...] = ("shadow", "strong_artifact", "fp_boost", "distant_cloud")
PRIMARY_TRACER_REGIMES: Tuple[str, ...] = ("haze_drift", "dust_burst", "rain_shaft")   # the scene's own distractor
PEAK_OPACITY_FLOOR = 0.05      # lower floor of a primary group's target peak opacity
RAW_TAU_MIN = 1e-6             # a group with a smaller raw peak tau is not normalised
PARAMS_VERSION = 2             # 2: camera_shake_tau_s, texture.sgs_gain / sgs_seed
SHAKE_TAU_S_DEFAULT = (0.5, 4.0)   # render.camera_shake_tau_s when the config has none
SGS_GAIN_DEFAULT = (0.5, 1.5)      # render.texture.sgs_gain when the config has none

_MAX_OPACITY = 0.999
_PATTERN_SIZE = 384            # texture tile side, pattern pixels (period > the widest view)
_PATTERN_PX_PER_WAVELENGTH = 4.0
_MIN_WAVELENGTH_IMAGE_PX = 2.5  # shorter octaves would alias at the image resolution
_CANVAS_MARGIN = 40            # static scene canvas margin around the frame (shake headroom), px
_PEAK_BOX_PX = 5               # visibility peaks: max of a 5x5 box mean (ignores single-pixel speckle)
_DEGRADE_KEYS = ("threshold_contrast", "morph_px", "boundary_noise", "drop_component_prob",
                 "false_positive_prob", "frame_dropout_prob", "min_area_frac")
_SHAKE_CLIP = 3.0              # camera-shake process clipped at +-3 stationary std
# Subgrid texture turbulence (_sgs_displacement).
_SGS_MODES = 12                # random Fourier modes of the displacement field
_SGS_WAVELENGTH_CELLS = (3.0, 8.0)   # mode wavelengths, grid cells (log-uniform): scales with dx
_SGS_STRAIN = 0.3              # rms |grad D|: keeps the texture warp fold-free
_SGS_BANDS = 10                # velocity octaves below the peak u_sgs, each with its own eddy clock


@dataclass
class RenderedSequence:
    frames: np.ndarray          # (T, H, W) uint8 grayscale at the observation times
    lag_frames: Optional[np.ndarray]      # (T, H, W) uint8 at the lag times (t_k - flow_lag_s); None without lag
    masks: np.ndarray           # (T, H, W) bool candidate masks at the observation times
    timestamps: np.ndarray      # (T,) float64 s (= sim.times)
    lag_timestamps: Optional[np.ndarray]  # (T,) float64 s (= sim.lag_times); None without lag
    mask_source: str            # "degraded" | "bbox" | "detector"
    opacity: np.ndarray         # (T, H, W) float32 total visible aerosol opacity (diagnostic only)
    component_opacity: dict     # {"smoke","fog","tracer","cloud"} -> (T, H, W) float16 (diagnostic only)
    visibility: dict            # per-frame visibility statistics (JSON-able)
    render_params: dict


# ---------------------------------------------------------------------- configuration
@functools.lru_cache(maxsize=1)
def _config_render_defaults() -> Dict[str, Any]:
    """The render block of configs/simulation.yaml (fills keys a partial config lacks)."""
    try:
        return load_simulation_config().get("render", {}) or {}
    except (OSError, ValueError):
        return {}


def _merge_one_level(base: Mapping[str, Any], over: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Robustness-override semantics: keys replace, nested dicts (masks, shadow, ...) merge one level."""
    out = copy.deepcopy(dict(base))
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            merged = dict(out[key])
            merged.update(copy.deepcopy(value))
            out[key] = merged
        else:
            out[key] = copy.deepcopy(value)
    return out


def _render_block(cfg: Optional[Mapping[str, Any]], overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """`cfg` may be the full simulation config or its render block; overrides as in `robustness:`."""
    cfg = cfg or {}
    block = cfg["render"] if isinstance(cfg.get("render"), dict) else cfg
    return _merge_one_level(_merge_one_level(_config_render_defaults(), block), overrides)


def _u(rng: np.random.Generator, bounds: Any) -> float:
    """Uniform draw from [lo, hi] (a scalar is returned as is). Always consumes one draw."""
    u = float(rng.random())
    if isinstance(bounds, (list, tuple)):
        lo, hi = float(bounds[0]), float(bounds[1])
        return lo + (hi - lo) * u
    return float(bounds)


def _ui(rng: np.random.Generator, bounds: Any) -> int:
    """Uniform integer in [lo, hi] inclusive. Always consumes one draw."""
    u = float(rng.random())
    if isinstance(bounds, (list, tuple)):
        lo, hi = int(bounds[0]), int(bounds[1])
        return int(min(lo + math.floor((hi - lo + 1) * u), hi))
    return int(bounds)


def _seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


def _choose_source(rng: np.random.Generator, probs: Mapping[str, float]) -> str:
    u = float(rng.random())
    w = np.array([max(float(probs.get(s, 0.0)), 0.0) for s in MASK_SOURCES])
    if w.sum() <= 0.0:
        raise ValueError(f"render.masks.source_probs has no positive weight: {dict(probs)}")
    idx = int(np.searchsorted(np.cumsum(w / w.sum()), u, side="right"))
    return MASK_SOURCES[min(idx, len(MASK_SOURCES) - 1)]


def _normalise_hints(hints: Optional[Mapping[str, Any]]) -> Dict[str, bool]:
    hints = hints or {}
    return {k: bool(hints.get(k, False)) for k in HINT_KEYS}


@functools.lru_cache(maxsize=4)
def _real_backgrounds(directory: str, exclude: Tuple[str, ...]) -> Tuple[Tuple[str, str], ...]:
    """(path, camera_id) of the real background images listed in <directory>/manifest.json."""
    manifest = Path(directory) / "manifest.json"
    if not manifest.is_file():
        return ()
    doc = read_json(manifest)
    entries = doc.get("images", []) if isinstance(doc, dict) else doc
    out = []
    for e in entries:
        path = Path(directory) / str(e["file"])
        cam = str(e.get("camera_id", ""))
        if path.is_file() and cam not in exclude:
            out.append((str(path), cam))
    return tuple(out)


# ---------------------------------------------------------------------- parameter sampling
def sample_render_params(rng: np.random.Generator, cfg: Dict[str, Any], hints: Optional[Dict[str, Any]] = None,
                         overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Every per-variant rendering choice, as a JSON-able dict.

    Class-independent by construction: the function never sees the category.
    `hints` (params["render_hints"]) only switch nuisances on; their strengths
    are always drawn (so the random stream, and every other value, is the same
    with or without them). `overrides` replace render.* keys (ranges or
    scalars); nested blocks such as masks.* merge one level, exactly like the
    `robustness:` suites in configs/simulation.yaml.
    """
    r = _render_block(cfg, overrides)
    hints_n = _normalise_hints(hints)
    H, W = (int(v) for v in r["native_size"])
    p: Dict[str, Any] = {"version": PARAMS_VERSION, "native_size": [H, W],
                         "view_margin_cells": int(r.get("view_margin_cells", 6))}
    # Geometry: zoom window (fraction of the interior width), its horizontal position, ground band.
    p["zoom"] = _u(rng, r["zoom"])
    p["view_x_frac"] = _u(rng, (0.0, 1.0))
    p["foreground_band"] = _u(rng, r["foreground_band"])
    # Optics: independent line-of-sight depth and brightness per component, same distributions.
    p["line_of_sight_m"] = {c: _u(rng, r["line_of_sight_m"]) for c in COMPONENTS}
    p["smoke_mass_extinction_m2_mg"] = float(r["smoke_mass_extinction_m2_mg"])
    p["tracer_mass_extinction_m2_mg"] = float(r["tracer_mass_extinction_m2_mg"])
    p["aerosol_brightness"] = {c: _u(rng, r["aerosol_brightness"]) for c in COMPONENTS}
    p["peak_opacity"] = {c: min(max(PEAK_OPACITY_FLOOR, _u(rng, r["peak_opacity"])), _MAX_OPACITY)
                         for c in COMPONENTS}
    p["background_haze_opacity"] = min(_u(rng, r["background_haze_opacity"]), _MAX_OPACITY)
    tex = r["texture"]
    p["texture"] = {"gain": _u(rng, tex["gain"]),
                    "wavelengths_m": [float(tex["wavelengths_m"][0]), float(tex["wavelengths_m"][1])],
                    "seed": _seed(rng),
                    "shading": _u(rng, tex.get("shading", [0.0, 0.0]))}
    if tex.get("max_stretch"):              # a fixed setting, no draw (the random stream is unchanged)
        p["texture"]["max_stretch"] = float(tex["max_stretch"])
    # Background: procedural unless a real image is drawn AND available.
    bg = r["background"]
    want_real = float(rng.random()) >= float(bg.get("procedural_prob", 1.0))
    pick = float(rng.random())
    p["background"] = {"kind": "procedural", "requested_real": bool(want_real), "file": None, "camera_id": None,
                       "seed": _seed(rng), "sky_brightness": _u(rng, bg["sky_brightness"]),
                       "ground_brightness": _u(rng, bg["ground_brightness"])}
    if want_real:
        # Falls back to procedural (requested_real stays True) when no image is available.
        directory = str(bg.get("dir") or (REPO_ROOT / "data" / "backgrounds"))
        avail = _real_backgrounds(directory, tuple(str(c) for c in bg.get("exclude_cameras", ())))
        if avail:
            path, cam = avail[min(int(pick * len(avail)), len(avail) - 1)]
            p["background"].update(kind="real", file=path, camera_id=cam)
    # Photometry and camera.
    p["global_haze"] = _u(rng, r["global_haze"])
    p["contrast"] = _u(rng, r["contrast"])
    p["brightness_offset"] = _u(rng, r["brightness_offset"])
    p["noise_sigma"] = _u(rng, r["noise_sigma"])
    p["jpeg"] = bool(float(rng.random()) < float(r["jpeg_prob"]))
    p["jpeg_quality"] = _ui(rng, r["jpeg_quality"])
    p["exposure_flicker"] = _u(rng, r["exposure_flicker"])
    p["camera_shake_px"] = _u(rng, r["camera_shake_px"])
    p["camera_rotation_deg"] = _u(rng, r["camera_rotation_deg"])
    p["zoom_jitter"] = 0.0
    # Nuisances: strengths always drawn, presence decided by the hints only.
    sa, sh, dc = r["strong_artifact"], r["shadow"], r["distant_cloud"]
    strong = {k: _u(rng, sa[k]) for k in ("camera_shake_px", "zoom_jitter", "exposure_flicker")}
    shadow = {"strength": _u(rng, sh["strength"]), "speed_frac_s": _u(rng, sh["speed_frac_s"]),
              "direction": 1.0 if rng.random() < 0.5 else -1.0, "seed": _seed(rng)}
    cloud = {"brightness": _u(rng, dc["brightness"]), "size_frac": _u(rng, dc["size_frac"]),
             "speed_frac_s": _u(rng, dc["speed_frac_s"]),
             "direction": 1.0 if rng.random() < 0.5 else -1.0, "seed": _seed(rng)}
    if hints_n["strong_artifact"]:
        p["camera_shake_px"] = max(p["camera_shake_px"], strong["camera_shake_px"])
        p["zoom_jitter"] = strong["zoom_jitter"]
        p["exposure_flicker"] = max(p["exposure_flicker"], strong["exposure_flicker"])
    p["strong_artifact"] = strong if hints_n["strong_artifact"] else None
    p["shadow"] = shadow if hints_n["shadow"] else None
    p["distant_cloud"] = cloud if hints_n["distant_cloud"] else None
    p["fp_boost"] = hints_n["fp_boost"]
    # Occlusion.
    occ_on = float(rng.random()) < float(r["occlusion_prob"])
    occ = {"area_frac": _u(rng, r["occlusion_area_frac"]), "seed": _seed(rng)}
    p["occlusion"] = occ if occ_on else None
    # Candidate masks.
    mk = r["masks"]
    p["masks"] = {
        "source": _choose_source(rng, mk["source_probs"]),
        "threshold_contrast": _u(rng, mk["threshold_contrast"]),
        "morph_px": _ui(rng, mk["morph_px"]),
        "boundary_noise": _u(rng, mk["boundary_noise"]),
        "drop_component_prob": float(mk["drop_component_prob"]),
        "false_positive_prob": float(mk["false_positive_prob_boost"] if hints_n["fp_boost"]
                                     else mk["false_positive_prob"]),
        "frame_dropout_prob": float(mk["frame_dropout_prob"]),
        "min_area_frac": float(mk["min_area_frac"]),
        "bbox_jitter": float(mk.get("bbox_jitter", 0.1)),
    }
    if isinstance(mk.get("detector"), dict):
        p["masks"]["detector"] = copy.deepcopy(mk["detector"])
    # Version 2, drawn last so every value above keeps its place in the stream: correlation time
    # of the camera-shake process (0 = independent per frame, the version-1 model) and the
    # subgrid-turbulence displacement of the texture (gain 0 disables it).
    shake_tau = r.get("camera_shake_tau_s")
    p["camera_shake_tau_s"] = max(_u(rng, list(SHAKE_TAU_S_DEFAULT) if shake_tau is None else shake_tau), 0.0)
    sgs_gain = tex.get("sgs_gain")
    p["texture"]["sgs_gain"] = max(_u(rng, list(SGS_GAIN_DEFAULT) if sgs_gain is None else sgs_gain), 0.0)
    p["texture"]["sgs_seed"] = _seed(rng)
    p["hints"] = hints_n
    return p


# ---------------------------------------------------------------------- view geometry
@dataclass(frozen=True)
class _View:
    """Frame / scene-canvas / world geometry of one rendered variant.

    The static scene canvas is the unshaken frame plus `margin` pixels on
    every side; frame pixel (c, r) of an unshaken camera is canvas pixel
    (c + margin, r + margin). Canvas row v_floor is the first row below the
    simulated floor (z = 0); world x = x0 + (u - margin + 0.5) s and
    z = (v_floor - v - 0.5) s, with s metres per pixel.
    """

    H: int
    W: int
    H_sim: int
    margin: int
    s: float
    x0: float
    dx: float

    @property
    def Hc(self) -> int:
        return self.H + 2 * self.margin

    @property
    def Wc(self) -> int:
        return self.W + 2 * self.margin

    @property
    def v_floor(self) -> int:
        return self.margin + self.H_sim

    def canvas_x(self, u):
        return self.x0 + (np.asarray(u, dtype=np.float64) - self.margin + 0.5) * self.s

    def canvas_z(self, v):
        return (self.v_floor - np.asarray(v, dtype=np.float64) - 0.5) * self.s

    def canvas_to_grid(self) -> np.ndarray:
        """(2, 3) affine: [i, k] (continuous grid indices, k = 0 at the ground) from [u, v, 1]."""
        a = self.s / self.dx
        return np.array([[a, 0.0, (self.x0 + (0.5 - self.margin) * self.s) / self.dx - 0.5],
                         [0.0, -a, (self.v_floor - 0.5) * self.s / self.dx - 0.5]])


def _view(sim: SimulationResult, rp: Mapping[str, Any]) -> _View:
    H, W = (int(v) for v in rp["native_size"])
    nz, nx = sim.solid.shape
    dx = float(sim.dx)
    m = int(np.clip(int(rp.get("view_margin_cells", 6)), 0, max(0, (min(nx, nz) - 2) // 2)))
    w_int = (nx - 2 * m) * dx
    h_int = (nz - m) * dx
    h_sim = int(np.clip(round(H * (1.0 - float(rp["foreground_band"]))), 8, H))
    s = float(rp["zoom"]) * w_int / W
    if s * h_sim > h_int:
        s = h_int / h_sim
    x0 = m * dx + float(rp["view_x_frac"]) * max(w_int - s * W, 0.0)
    return _View(H=H, W=W, H_sim=h_sim, margin=_CANVAS_MARGIN, s=s, x0=x0, dx=dx)


def _compose(G: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Affine G o A for (2, 3) matrices acting on [x, y, 1]."""
    return np.concatenate([G[:, :2] @ A[:, :2], (G[:, :2] @ A[:, 2] + G[:, 2])[:, None]], axis=1)


# ---------------------------------------------------------------------- procedural noise
def _octave(rng: np.random.Generator, h: int, w: int, cell: int) -> np.ndarray:
    """Smooth random field with correlation length ~cell pixels (bicubic value noise)."""
    if cell <= 1:
        return rng.standard_normal((h, w)).astype(np.float32)
    nh, nw = h // cell + 3, w // cell + 3
    low = rng.standard_normal((nh, nw)).astype(np.float32)
    up = cv2.resize(low, (nw * cell, nh * cell), interpolation=cv2.INTER_CUBIC)
    oy, ox = (int(v) for v in rng.integers(0, cell, 2))
    return up[oy:oy + h, ox:ox + w]


def _fractal_noise(rng: np.random.Generator, h: int, w: int, cells: Sequence[int], exponent: float) -> np.ndarray:
    """Zero-mean, unit-std multi-scale value noise; the octave of cell size c is weighted c**exponent.

    Octaves of 8 px and more are summed at quarter resolution and upsampled
    once (they are smooth at that scale), which halves the cost.
    """
    out = np.zeros((h, w), dtype=np.float32)
    coarse = [c for c in cells if c >= 8]
    if coarse:
        hs, ws = h // 4 + 2, w // 4 + 2
        acc = np.zeros((hs, ws), dtype=np.float32)
        for cell in coarse:
            acc += (cell ** exponent) * _octave(rng, hs, ws, cell // 4)
        out += cv2.resize(acc, (ws * 4, hs * 4), interpolation=cv2.INTER_LINEAR)[:h, :w]
    for cell in (c for c in cells if c < 8):
        out += (cell ** exponent) * _octave(rng, h, w, cell)
    out -= out.mean()
    return out / (out.std() + 1e-8)


def _fractal_1d(rng: np.random.Generator, n: int, cells: Sequence[int]) -> np.ndarray:
    """Zero-mean, unit-std 1-D multi-scale noise with kinks (ridge lines)."""
    out = np.zeros(n)
    for cell in cells:
        k = n // cell + 3
        low = rng.standard_normal(k)
        x = (np.arange(n) + rng.uniform(0.0, cell)) / cell
        out += cell ** 0.8 * np.interp(x, np.arange(k), low)
    out -= out.mean()
    return out / (out.std() + 1e-8)


@functools.lru_cache(maxsize=4)
def _texture_pattern(seed: int, pixel_m: float, lam_lo: float, lam_hi: float) -> np.ndarray:
    """Tileable multi-octave noise (zero mean, unit std), a _PATTERN_SIZE^2 tile of pixel_m metres
    pixels, returned tiled 3 x 3.

    Octaves are log-Gaussian bands at wavelengths lam_hi, lam_hi/2, ... >=
    lam_lo; band amplitudes are compensated for the number of Fourier modes
    so the octave std is ~ (lam / lam_hi)^(1/3) (more energy at large scales).
    """
    n = _PATTERN_SIZE
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, n), dtype=np.float32)
    ky = np.fft.fftfreq(n, d=pixel_m).astype(np.float32)[:, None]
    kx = np.fft.rfftfreq(n, d=pixel_m).astype(np.float32)[None, :]
    k = np.maximum(np.hypot(kx, ky), np.float32(1e-12))
    log_k = np.log2(k)
    lams = []
    lam = lam_hi
    while lam >= 0.999 * lam_lo:
        lams.append(lam)
        lam *= 0.5
    if lams[-1] > 1.3 * lam_lo:
        lams.append(lam_lo)
    # A 2-D annulus at |k| holds ~k^2 dln(k) modes, so a log-band shaped g(k lam) / k carries a
    # variance independent of lam; the octave weights then give octave variance ~ lam^(2/3),
    # the Kolmogorov / Obukhov-Corrsin k^(-5/3) scalar spectrum.
    filt = np.zeros_like(k)
    for lam in lams:     # log-Gaussian band of half an octave around k = 1 / lam
        filt += np.float32((lam / lam_hi) ** (1.0 / 3.0)) * np.exp(np.float32(-2.0) * (log_k + np.float32(math.log2(lam))) ** 2)
    filt /= k
    filt[0, 0] = 0.0
    pat = np.fft.irfft2(np.fft.rfft2(white) * filt, s=(n, n))
    pat = ((pat - pat.mean()) / (pat.std() + 1e-12)).astype(np.float32)
    # Tiled 3 x 3: coordinates shifted by whole periods into [0, 3n) need no border handling.
    tiled = np.ascontiguousarray(np.tile(pat, (3, 3)))
    tiled.flags.writeable = False
    return tiled


# ---------------------------------------------------------------------- static scene
def _procedural_background(rng: np.random.Generator, view: _View, sky_b: float, ground_b: float):
    """Sky gradient + 1-3 textured ridges + textured foreground ground band; returns (image, sky mask)."""
    Hc, Wc, vf, hs = view.Hc, view.Wc, view.v_floor, view.H_sim
    v = np.arange(Hc, dtype=np.float32)[:, None]
    frac = np.clip(v / max(vf, 1), 0.0, 1.0)
    horizon = rng.uniform(-0.05, 0.25)          # > 0: sky brighter towards the horizon
    sky_line = sky_b * (1.0 - horizon * (1.0 - frac))
    img = sky_line + 0.03 * sky_b * _fractal_noise(rng, Hc, Wc, (32, 64, 128), 0.8)
    sky = np.ones((Hc, Wc), dtype=np.float32)
    tex = _fractal_noise(rng, Hc, Wc, (2, 4, 8, 16, 32, 64), 0.35)
    n_ridges = int(rng.integers(1, 4))
    tops = np.sort(rng.uniform(0.1, 0.7, n_ridges))[::-1]     # far ridges highest
    for j, top in enumerate(tops):
        depth = (j + 1) / n_ridges                             # 1 = nearest
        line_noise = _fractal_1d(rng, Wc, (6, 12, 24, 48, 96))
        height = np.clip(top + rng.uniform(0.02, 0.1) * line_noise, 0.02, 0.95)
        line = (vf - hs * height).astype(np.float32)[None, :]
        perspective = 0.65 * (1.0 - depth) + 0.12 * rng.random()   # far ridges fade towards the sky
        level = ground_b + (sky_b - ground_b) * perspective
        shade = 1.0 - 0.15 * np.clip((v - line) / max(hs, 1), 0.0, 1.0)
        flips = (tex, tex[::-1], tex[:, ::-1])[j % 3]          # flipped copies: no seams, uncorrelated
        ridge = level * shade * (1.0 + (0.06 + 0.14 * depth) * flips)
        cover = np.clip(v - line + 0.5, 0.0, 1.0)
        img = img + cover * (ridge - img)
        sky *= 1.0 - cover
    band = (v >= vf).astype(np.float32)
    grad = np.clip((v - vf) / max(Hc - vf, 1), 0.0, 1.0)
    fg_tex = 0.6 * _fractal_noise(rng, Hc, Wc, (2, 3, 6), 0.3) + 0.4 * _fractal_noise(rng, Hc, Wc, (12, 24), 0.3)
    fg = ground_b * (1.0 + rng.uniform(-0.3, 0.3) * grad) * (1.0 + 0.25 * fg_tex)
    img = img + band * (fg - img)
    sky *= 1.0 - band
    return np.clip(img, 0.02, 0.98).astype(np.float32), sky


def _real_background(path: str, view: _View, rng: np.random.Generator):
    """A random crop of a real grayscale image, resized to the canvas; None if unreadable."""
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return None, None
    img = raw.astype(np.float32) / 255.0
    h, w = img.shape
    aspect = view.Wc / view.Hc
    ch = max(8, int(min(h, w / aspect) * rng.uniform(0.6, 1.0)))
    cw = max(8, int(ch * aspect))
    y0 = int(rng.integers(0, max(h - ch, 0) + 1))
    x0 = int(rng.integers(0, max(w - cw, 0) + 1))
    crop = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (view.Wc, view.Hc), interpolation=cv2.INTER_AREA)
    v = np.arange(view.Hc, dtype=np.float32)[:, None]
    sky = np.clip(1.0 - (v - view.margin) / (0.35 * view.H_sim), 0.0, 1.0) * np.ones((1, view.Wc), np.float32)
    return crop, sky


def _terrain_cover(sim: SimulationResult, view: _View):
    """(terrain, air) on the canvas: terrain = ground inside sim.solid (piecewise-linear surface)."""
    solid = np.asarray(sim.solid, dtype=bool)
    nz, nx = solid.shape
    fluid = ~solid
    k_ground = np.where(fluid.any(axis=0), np.argmax(fluid, axis=0), nz)
    x = view.canvas_x(np.arange(view.Wc))
    z = view.canvas_z(np.arange(view.Hc))
    zt = np.interp(x, (np.arange(nx) + 0.5) * view.dx, k_ground * view.dx)
    above = (np.arange(view.Hc) < view.v_floor)[:, None]
    terr = np.clip((zt[None, :] - z[:, None]) / view.s + 0.5, 0.0, 1.0) * above
    air = (1.0 - terr) * above
    return terr.astype(np.float32), air.astype(np.float32)


def _occluder_layer(rng: np.random.Generator, view: _View, area_frac: float):
    """A static foreground occluder (tree, pole, wall edge or bush) covering ~area_frac of the frame."""
    Hc, Wc, M, H, W = view.Hc, view.Wc, view.margin, view.H, view.W
    kind = ("tree", "pole", "edge", "bush")[int(rng.integers(0, 4))]
    xc = M + rng.uniform(0.1, 0.9) * W
    yc = M + rng.uniform(0.2, 0.6) * H
    left = bool(rng.random() < 0.5)
    lobes = rng.uniform(-0.6, 0.6, (6, 2))
    radii = rng.uniform(0.5, 0.95, 6)
    jag = rng.uniform(-0.15, 0.15, 8)
    bar = rng.uniform(0.1, 0.4)

    def draw(scale: float) -> np.ndarray:
        img = np.zeros((Hc, Wc), dtype=np.uint8)
        if kind == "tree":
            R = scale * 0.35 * W
            cv2.rectangle(img, (int(xc - 0.12 * R), int(yc)), (int(xc + 0.12 * R), Hc - 1), 1, -1)
            for (ox, oy), rr in zip(lobes, radii):
                cv2.circle(img, (int(xc + ox * R), int(yc + oy * R)), max(1, int(rr * R)), 1, -1)
        elif kind == "pole":
            half = max(1, int(scale * 0.05 * W))
            cv2.rectangle(img, (int(xc) - half, 0), (int(xc) + half, Hc - 1), 1, -1)
            yb = int(M + bar * H)
            cv2.rectangle(img, (int(xc) - 6 * half, yb - half), (int(xc) + 6 * half, yb + half), 1, -1)
        elif kind == "edge":
            width = scale * 0.5 * W
            xs = np.linspace(0, width, len(jag))
            tops = yc + jag * H
            xs = (M + xs) if left else (M + W - xs)
            edge_x = 0 if left else Wc - 1
            pts = [(edge_x, Hc - 1), (edge_x, int(tops[0]))] + \
                  [(int(x), int(y)) for x, y in zip(xs, tops)] + [(int(xs[-1]), Hc - 1)]
            cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], 1)
        else:  # bush along the bottom edge
            R = scale * 0.3 * W
            for (ox, _), rr in zip(lobes, radii):
                cv2.circle(img, (int(xc + ox * 1.5 * R), M + H), max(1, int(rr * R)), 1, -1)
        return img

    target = float(np.clip(area_frac, 0.0, 0.95))
    lo, hi = 0.0, 3.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if draw(mid)[M:M + H, M:M + W].mean() < target:
            lo = mid
        else:
            hi = mid
    occ = cv2.GaussianBlur(draw(hi).astype(np.float32), (0, 0), 0.7)
    value = rng.uniform(0.04, 0.45) * (1.0 + 0.15 * _fractal_noise(rng, Hc, Wc, (2, 4, 8, 16), 0.4))
    return occ, np.clip(value, 0.0, 1.0).astype(np.float32)


def _scene_canvas(sim: SimulationResult, rp: Mapping[str, Any], view: _View):
    """Static scene on the canvas: (4, Hc, Wc) [background, air, occluder, sky], airlight, occluded?"""
    bgp = rp["background"]
    rng = np.random.default_rng(int(bgp["seed"]))
    sky_b, ground_b = float(bgp["sky_brightness"]), float(bgp["ground_brightness"])
    img = sky = None
    if bgp.get("kind") == "real" and bgp.get("file"):
        img, sky = _real_background(str(bgp["file"]), view, rng)
    if img is None:
        img, sky = _procedural_background(rng, view, sky_b, ground_b)
        airlight = sky_b
    else:
        airlight = float(np.percentile(img[:view.v_floor], 90))
    terr, air = _terrain_cover(sim, view)
    if terr.any():
        tex = _fractal_noise(rng, view.Hc, view.Wc, (2, 4, 8, 16, 32), 0.4)
        ground = (ground_b + 0.15 * (sky_b - ground_b)) * (1.0 + 0.18 * tex)
        img = img + terr * (ground - img)
        sky = sky * (1.0 - terr)
    occ = np.zeros_like(img)
    if rp.get("occlusion"):
        occ, value = _occluder_layer(np.random.default_rng(int(rp["occlusion"]["seed"])), view,
                                     float(rp["occlusion"]["area_frac"]))
        img = img + occ * (value - img)
    scene = np.ascontiguousarray(np.stack([img, air, occ, sky]).astype(np.float32))
    return scene, float(airlight), bool(rp.get("occlusion"))


# ---------------------------------------------------------------------- camera and nuisances
def _shake_process(rng: np.random.Generator, times: np.ndarray, tau_s: float) -> np.ndarray:
    """(n, 4) unit-variance camera-shake coordinates (x, y, rotation, zoom) at `times` (frame order).

    A stationary Ornstein-Uhlenbeck process sampled exactly at the frame
    times: in time order, z_0 = e_0 and z_j = rho z_{j-1} + sqrt(1 - rho^2) e_j
    with rho = exp(-(t_j - t_{j-1}) / tau_s), so every frame has unit
    variance and two frames dt apart correlate as exp(-dt / tau_s). The
    innovations e are drawn in time order (n x 4 standard normals, the same
    number of draws as before). tau_s <= 0 gives the version-1 model:
    independent per frame, drawn in frame order. Clipped at +-3.
    """
    t = np.asarray(times, dtype=np.float64).ravel()
    e = rng.standard_normal((t.size, 4))
    if not tau_s > 0.0:
        return np.clip(e, -_SHAKE_CLIP, _SHAKE_CLIP)
    order = np.argsort(t, kind="stable")
    z = np.empty_like(e)
    z[order[0]] = e[0]
    for j in range(1, t.size):
        f, prev = order[j], order[j - 1]
        rho = math.exp(-max(t[f] - t[prev], 0.0) / tau_s)
        z[f] = rho * z[prev] + math.sqrt(max(1.0 - rho * rho, 0.0)) * e[j]
    return np.clip(z, -_SHAKE_CLIP, _SHAKE_CLIP)


def _camera_affines(rng: np.random.Generator, rp: Mapping[str, Any], times: np.ndarray, view: _View) -> np.ndarray:
    """(n, 2, 3) frame-pixel -> canvas affines at the n frame `times`: shake (px), rotation (deg) and
    zoom jitter about the centre, each the stationary std of the correlated process _shake_process."""
    z = _shake_process(rng, times, float(rp.get("camera_shake_tau_s") or 0.0))
    n = z.shape[0]
    shake = float(rp["camera_shake_px"])
    rot = math.radians(float(rp["camera_rotation_deg"]))
    zj = float(rp.get("zoom_jitter", 0.0))
    c0 = np.array([(view.W - 1) / 2.0, (view.H - 1) / 2.0])
    out = np.empty((n, 2, 3))
    for f in range(n):
        th = rot * z[f, 2]
        sc = 1.0 + zj * z[f, 3]
        R = sc * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        out[f, :, :2] = R
        out[f, :, 2] = c0 + shake * z[f, :2] + view.margin - R @ c0
    return out


def _sprite_affine(centre, extent, shape) -> np.ndarray:
    """(2, 3) canvas -> sprite-pixel affine for a sprite of `shape` (h, w) covering `extent` (w, h) px."""
    h, w = shape
    ax, ay = w / extent[0], h / extent[1]
    return np.array([[ax, 0.0, w / 2.0 - 0.5 - centre[0] * ax],
                     [0.0, ay, h / 2.0 - 0.5 - centre[1] * ay]])


def _soft_blob(rng: np.random.Generator, h: int, w: int, edge: float) -> np.ndarray:
    """A soft blob with a wavy boundary in a (h, w) sprite, 0 at the sprite border."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xx = (xx + 0.5) / w * 2.0 - 1.0
    yy = (yy + 0.5) / h * 2.0 - 1.0
    r = np.hypot(xx, yy)
    ang = np.arctan2(yy, xx)
    boundary = 0.72 + sum(rng.uniform(0.0, 0.08) * np.cos(m * ang + rng.uniform(0, 2 * np.pi)) for m in (2, 3, 5, 7))
    return np.clip((boundary - r) / edge + 0.5, 0.0, 1.0)


def _nuisance_layers(rp: Mapping[str, Any], view: _View, times: np.ndarray):
    """Moving cloud shadow and distant cloud sprites with their linear tracks (None when absent)."""
    t_mid = 0.5 * (float(times.min()) + float(times.max()))
    shadow = cloud = None
    sh = rp.get("shadow")
    if sh:
        rng = np.random.default_rng(int(sh["seed"]))
        n = 64
        blob = _soft_blob(rng, n, n, 0.25)
        patch = np.clip(0.8 + 0.3 * _fractal_noise(rng, n, n, (4, 8, 16), 0.5), 0.0, 1.0)
        size = rng.uniform(0.7, 1.5) * max(view.H, view.W)
        speed = float(sh["speed_frac_s"]) * view.W
        shadow = {
            "sprite": (blob * patch).astype(np.float32), "extent": (size, size * rng.uniform(0.6, 1.0)),
            "centre": (view.margin + rng.uniform(0.15, 0.85) * view.W, view.margin + rng.uniform(0.15, 0.85) * view.H_sim),
            "velocity": (float(sh["direction"]) * speed, speed * rng.uniform(-0.3, 0.3)),
            "t_mid": t_mid, "strength": float(sh["strength"]),
        }
    dc = rp.get("distant_cloud")
    if dc:
        rng = np.random.default_rng(int(dc["seed"]))
        h, w = 40, 80
        blob = _soft_blob(rng, h, w, 0.35)
        tex = np.clip(0.6 + 0.4 * _fractal_noise(rng, h, w, (3, 6, 12), 0.6), 0.0, 1.0)
        width = float(dc["size_frac"]) * view.W
        speed = float(dc["speed_frac_s"]) * view.W
        cloud = {
            "sprite": (rng.uniform(0.6, 0.95) * blob * tex).astype(np.float32),
            "extent": (width, width * rng.uniform(0.3, 0.6)),
            "centre": (view.margin + rng.uniform(0.15, 0.85) * view.W, view.margin + rng.uniform(0.06, 0.3) * view.H_sim),
            "velocity": (float(dc["direction"]) * speed, 0.0),
            "t_mid": t_mid, "brightness": float(dc["brightness"]),
        }
    return shadow, cloud


# ---------------------------------------------------------------------- per-frame sampling
_WARP_FLAGS = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP


def _warp(src: np.ndarray, M: np.ndarray, view: _View, border: int) -> np.ndarray:
    """dst(c, r) = src(M [c, r, 1]), bilinear, at the frame size."""
    return cv2.warpAffine(src, M, (view.W, view.H), flags=_WARP_FLAGS, borderMode=border, borderValue=0)


def _optical_depth_grid(sim: SimulationResult, rp: Mapping[str, Any], lag: bool) -> np.ndarray:
    """(T, 3, nz, nx) float32 raw optical depth [smoke, fog, tracer] on the grid."""
    smoke = np.asarray(sim.field_at("smoke", lag), dtype=np.float64)
    lwc = np.asarray(sim.field_at("lwc", lag), dtype=np.float64)
    tracer = np.asarray(sim.field_at("tracer", lag), dtype=np.float64)
    los = rp["line_of_sight_m"]
    out = np.empty((smoke.shape[0], 3) + smoke.shape[1:], dtype=np.float32)
    out[:, 0] = np.maximum(smoke, 0.0) * (float(rp["smoke_mass_extinction_m2_mg"]) * float(los["smoke"]))
    out[:, 1] = fog_extinction(lwc) * float(los["fog"])
    out[:, 2] = np.maximum(tracer, 0.0) * (float(rp["tracer_mass_extinction_m2_mg"]) * float(los["tracer"]))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _stretch_attenuation(tc: np.ndarray, dx: float, max_stretch: Optional[float]) -> Optional[np.ndarray]:
    """(T, 2, nz, nx) float32 texture attenuation per coordinate pair, or None when disabled.

    The advected coordinates of a pair are stretched by the shear they have
    seen since their last reset (largest singular value s of d(X, Z)/d(x, z)).
    Real aerosol is mixed smooth where it is strongly sheared instead of being
    drawn out into streaks, so a pair's texture is faded by min(1, max_stretch / s).
    """
    if not max_stretch:
        return None
    tc = np.asarray(tc, dtype=np.float64)
    out = np.empty((tc.shape[0], 2) + tc.shape[2:], dtype=np.float32)
    for k, (ix, iz) in enumerate(((0, 1), (2, 3))):
        dXdz, dXdx = np.gradient(tc[:, ix], dx, axis=(1, 2))
        dZdz, dZdx = np.gradient(tc[:, iz], dx, axis=(1, 2))
        half = 0.5 * (dXdx ** 2 + dXdz ** 2 + dZdx ** 2 + dZdz ** 2)
        det = dXdx * dZdz - dXdz * dZdx
        s = np.sqrt(half + np.sqrt(np.maximum(half ** 2 - det ** 2, 0.0)))
        out[:, k] = np.minimum(1.0, float(max_stretch) / np.maximum(s, 1e-9))
    return out


def _texcoord_grid(sim: SimulationResult, lag: bool, inv_p: float, offset_b: Tuple[float, float],
                   disp: Optional[np.ndarray] = None, max_stretch: Optional[float] = None):
    """Texture coordinates in pattern pixels on the grid, (T, nz, nx, 4) [Xa, Za, Xb, Zb], and blend weights.

    Pair b is read at a fixed offset of about half a tile, so pairs a and b
    sample unrelated texture even where their coordinates coincide (then the
    variance-preserving blend weights w / |w| are exact). Without
    sim.texcoords the texture is static (identity coordinates). `disp`
    ((T, nz, nx, 2) [D_x, D_z] metres, see _sgs_displacement) is subtracted
    from both pairs: the label of the air at x is that of x - D, so the
    texture appears displaced by +D. With `max_stretch` (render.texture.max_stretch)
    a fourth return value gives each pair's stretch attenuation (see
    _stretch_attenuation), else None.
    """
    n_t = int(sim.n_frames)
    nz, nx = sim.solid.shape
    tc = getattr(sim, "texcoords_lag", None) if lag else sim.texcoords
    wts = getattr(sim, "tex_weights_lag", None) if lag else sim.tex_weights
    advected = tc is not None
    if tc is None:
        X, Z = np.meshgrid((np.arange(nx) + 0.5) * float(sim.dx), (np.arange(nz) + 0.5) * float(sim.dx))
        tc = np.broadcast_to(np.stack([X, Z, X, Z])[None], (n_t, 4, nz, nx))
        wts = None
    att = _stretch_attenuation(tc, float(sim.dx), max_stretch) if advected else None
    out = np.asarray(tc, dtype=np.float64) * inv_p
    if disp is not None:
        d = np.asarray(disp, dtype=np.float64).transpose(0, 3, 1, 2) * inv_p      # (T, 2, nz, nx)
        out[:, 0:2] -= d
        out[:, 2:4] -= d
    out[:, 2] += offset_b[0]
    out[:, 3] += offset_b[1]
    # Whole-period shifts (exact for a periodic pattern) keep lookups inside the 3 x 3 tile.
    out -= _PATTERN_SIZE * np.floor(out.min(axis=(2, 3), keepdims=True) / _PATTERN_SIZE)
    out = np.ascontiguousarray(out.transpose(0, 2, 3, 1), dtype=np.float32)
    w = np.tile([1.0, 0.0], (n_t, 1)) if wts is None else np.clip(np.asarray(wts, dtype=np.float64), 0.0, None)
    norm = np.sqrt((w ** 2).sum(axis=1, keepdims=True))
    coef = np.divide(w, norm, out=np.zeros_like(w), where=norm > 1e-12)
    return out, coef, advected, att


def _sgs_velocity(sim: SimulationResult, has_lag: bool) -> Optional[np.ndarray]:
    """(F, nz, nx) subgrid velocity scale u_sgs (m/s, >= 0) at the observation then lag frames, or None.

    Lag frames use sim.sgs_u_lag, or the observation's own sim.sgs_u when the
    simulation stored none (u_sgs changes little over the lag).
    """
    obs = getattr(sim, "sgs_u", None)
    if obs is None:
        return None
    shape = (int(sim.n_frames),) + tuple(sim.solid.shape)
    parts = [obs]
    if has_lag:
        lag = getattr(sim, "sgs_u_lag", None)
        parts.append(obs if lag is None else lag)
    out = []
    for name, a in zip(("sgs_u", "sgs_u_lag"), parts):
        a = np.asarray(a, dtype=np.float32)
        if a.shape != shape:
            raise ValueError(f"simulation {sim.sim_id}: {name} has shape {a.shape}, expected {shape}")
        out.append(a)
    u = np.concatenate(out)
    return np.clip(np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)


def _sgs_displacement(u_sgs: Optional[np.ndarray], times: np.ndarray, dx: float, texture: Mapping[str, Any]):
    """Subgrid-turbulence displacement of the texture, (F, nz, nx, 2) [D_x, D_z] metres (or None), and a
    JSON-able record.

    Eddies of size ~A turn over in ~A / u_sgs, and u_sgs spans orders of magnitude within one
    scene (wall shear in the first cell row, a plume, quiet air), so the field is built from
    velocity octaves: band b has the velocity u_b = u_max 2^-b (b < _SGS_BANDS, u_max the peak
    u_sgs of the sequence) and its own eddy time theta_b(t) = g u_b (t - t_first) / A, and carries
    the displacement per unit velocity
      h_b(x, t) = A f_b(x, theta_b(t)) / u_b   (seconds; rms |dh_b/dt| = g),
      f_b = sum_m c_m e_m sin(k_m . x + phi_bm(theta_b)): _SGS_MODES random Fourier modes, shared by
          the bands, wavelengths _SGS_WAVELENGTH_CELLS grid cells (so they scale with dx), random
          directions, each mode divergence-free (e_m perpendicular to k_m), c_m = sqrt(2 / M), so
          rms |f| = 1;
      phi_bm = phi0_bm + integral of omega_bm, omega_bm a unit-variance OU process in eddy time
          with unit correlation, so rms |df/dtheta| = 1, and f_b decorrelates over ~1.5 eddy
          times (phase variance 2 (dtheta - 1 + exp(-dtheta))); the pair (omega, phi) is advanced
          exactly between the frame times (no time stepping);
      A = _SGS_STRAIN / k_rms, the rms displacement that keeps the texture warp fold-free.
    Every cell blends the two bands bracketing its reference velocity u_ref(x) = max over all
    frames (observation and lag) of u_sgs(x, t), linearly in log2 u_ref with variance-preserving
    weights w / |w|: h = sum_b w_b h_b, with bands and weights fixed for the whole sequence. D is
    h integrated against the local velocity, frame by frame in time order (observation and lag
    frames interleaved; trapezoidal in u_sgs):
      D(t_0) = u_sgs(t_0) h(t_0),
      D(t_j) = D(t_{j-1}) + (u_sgs(t_j) + u_sgs(t_{j-1})) / 2 (h(t_j) - h(t_{j-1})),
    so with u_sgs constant in time D = u_sgs h, and when u_sgs changes between frames D does not
    jump: the part already displaced stays and only the further change scales with the new u_sgs.
    The velocity is g u_sgs (rms, vector magnitude) everywhere and at every frame. Where u_sgs ~
    u_ref >= u_max 2^-(B-1), D is ~A in size and decorrelates over ~1.5 A / (g u_ref) (eddy size
    over eddy velocity, within a factor 2); where u_sgs < u_ref the eddies keep the u_ref clock and
    the amplitude shrinks by u_sgs / u_ref; where u_ref < u_max 2^-(B-1) the slowest band's clock
    caps the correlation time at ~1.5 A / (g u_max 2^-(B-1)) and the amplitude is ~A u_sgs /
    (u_max 2^-(B-1)). D is zero-mean (random phases; no net drift) and vanishes where u_sgs is 0 at
    every frame. A lag frame 1 s before its observation is therefore displaced by ~ g u_sgs x 1 s
    relative to it; frames much further apart than the correlation time by ~ sqrt(2) A (at u_sgs ~
    u_ref). Class-independent: it acts wherever u_sgs is, smoke, fog or clear air alike.
    Randomness: np.random.default_rng(texture["sgs_seed"]): modes, initial phases and rates, then
    the innovations of every frame step (fixed counts).
    """
    gain = float(texture.get("sgs_gain", 0.0) or 0.0)
    record: Dict[str, Any] = {"applied": False, "gain": gain}
    if u_sgs is None or gain <= 0.0:
        return None, record
    u = np.asarray(u_sgs, dtype=np.float64)
    u_max = float(u.max(initial=0.0))
    if not u_max > 0.0:
        return None, record
    seed = texture.get("sgs_seed")
    rng = np.random.default_rng(int(seed) if seed is not None else [int(texture.get("seed", 0)), 0x5C5])
    M, B = _SGS_MODES, _SGS_BANDS
    lam = dx * np.exp(rng.uniform(math.log(_SGS_WAVELENGTH_CELLS[0]), math.log(_SGS_WAVELENGTH_CELLS[1]), M))
    ang = rng.uniform(0.0, math.pi, M)
    k = 2.0 * math.pi / lam
    amp = _SGS_STRAIN / math.sqrt(float(np.mean(k ** 2)))
    u_band = u_max * 2.0 ** -np.arange(B)
    # Phases of every band and mode at every frame: the integrated unit OU process (omega, phi),
    # stepped exactly over each eddy-time interval d (innovation covariance [[1 - e^-2d, (1 - e^-d)^2],
    # [., 2 d - 4 (1 - e^-d) + 1 - e^-2d]]), in time order.
    t = np.asarray(times, dtype=np.float64)
    n_t = t.size
    phi = rng.uniform(0.0, 2.0 * math.pi, (B, M))
    omega = rng.standard_normal((B, M))
    eps = rng.standard_normal((max(n_t - 1, 0), B, M, 2))
    order = np.argsort(t, kind="stable")
    phase = np.empty((n_t, B, M))
    phase[order[0]] = phi
    rate = (gain * u_band / amp)[:, None]                      # eddy times per second, (B, 1)
    for j in range(1, n_t):
        d = max(t[order[j]] - t[order[j - 1]], 0.0) * rate
        a = -np.expm1(-d)                                       # 1 - e^-d
        var_w = -np.expm1(-2.0 * d)
        var_i = np.where(d < 1e-3, 2.0 * d ** 3 / 3.0, 2.0 * d - 4.0 * a + var_w)
        l11 = np.sqrt(var_w)
        l21 = np.divide(a * a, l11, out=np.zeros_like(d), where=l11 > 0.0)
        l22 = np.sqrt(np.maximum(var_i - l21 ** 2, 0.0))
        x1, x2 = eps[j - 1, ..., 0], eps[j - 1, ..., 1]
        phi = phi + a * omega + l21 * x1 + l22 * x2
        omega = (1.0 - a) * omega + l11 * x1
        phase[order[j]] = phi
    # Every band's field f_b at every frame, (F, B, 2, N): sin(k.x + b) = sin(k.x) cos b + cos(k.x) sin b,
    # i.e. 2M basis fields times per-frame coefficients (one matrix product for all bands).
    _, nz, nx = u.shape
    N = nz * nx
    X = (np.arange(nx) + 0.5) * dx
    Z = (np.arange(nz) + 0.5) * dx
    arg = (k * np.cos(ang))[:, None, None] * X[None, None, :] + (k * np.sin(ang))[:, None, None] * Z[None, :, None]
    basis = np.concatenate([np.sin(arg), np.cos(arg)]).reshape(2 * M, N).astype(np.float32)
    e_perp = math.sqrt(2.0 / M) * np.stack([-np.sin(ang), np.cos(ang)])   # c_m e_m, e_m perpendicular to k_m
    coef = np.concatenate([np.cos(phase)[:, :, None, :] * e_perp, np.sin(phase)[:, :, None, :] * e_perp], axis=-1)
    f = (coef.reshape(-1, 2 * M).astype(np.float32) @ basis).reshape(n_t, B, 2, N)
    # Band blend per cell, once for the whole sequence: p = log2(u_max / u_ref) in [0, B - 1], bands
    # floor(p) and floor(p) + 1 (u_ref = 0 lands in the last band; D is 0 there), giving the
    # displacement per unit velocity h = sum_b w_b A f_b / u_b, (F, 2, N).
    uf = u.reshape(n_t, N)
    u_ref = uf.max(axis=0)
    p = np.clip(np.log2(u_max / np.maximum(u_ref, u_max * 2.0 ** -(B + 1))), 0.0, B - 1.0)
    lo = np.minimum(np.floor(p).astype(np.int64), B - 2)
    frac = p - lo
    norm = np.hypot(1.0 - frac, frac)
    h = np.zeros((n_t, 2, N), dtype=np.float32)
    for band, w in ((lo, (1.0 - frac) / norm), (lo + 1, frac / norm)):
        g = np.take_along_axis(f, np.broadcast_to(band[None, None, None, :], (n_t, 1, 2, N)), axis=1)[:, 0]
        h += (w * amp / u_band[band]).astype(np.float32) * g
    # D = integral of u dh in time order (trapezoidal in u): no jump where u_sgs changes between frames.
    disp = np.empty((n_t, 2, N), dtype=np.float32)
    acc = uf[order[0]] * h[order[0]].astype(np.float64)
    disp[order[0]] = acc
    for j in range(1, n_t):
        i0, i1 = order[j - 1], order[j]
        acc = acc + (0.5 * (uf[i0] + uf[i1])) * (h[i1].astype(np.float64) - h[i0])
        disp[i1] = acc
    record.update(applied=True, u_max_ms=u_max, amplitude_m=float(amp),
                  correlation_time_s_at_u_max=float(1.5 * amp / (gain * u_max)))
    return np.ascontiguousarray(disp.reshape(n_t, 2, nz, nx).transpose(0, 2, 3, 1)), record


def _sample_layers(view: _View, scene: np.ndarray, tau_grid: np.ndarray, tc_grid: np.ndarray, coef: np.ndarray,
                   pattern: np.ndarray, affines: np.ndarray, times: np.ndarray, shadow: Optional[dict],
                   cloud: Optional[dict], active: Sequence[int], occluded: bool,
                   att_grid: Optional[np.ndarray] = None) -> Dict[str, List[Any]]:
    """Pass 1: every layer of every frame, sampled through that frame's camera mapping.

    Only the layers a variant needs are warped (single-channel and 4-channel
    warps take OpenCV's fast paths); unneeded scene layers are shared zeros.
    """
    G = view.canvas_to_grid()
    zeros = np.zeros((view.H, view.W), dtype=np.float32)
    need = {"bg": True, "air": bool(active), "occ": occluded, "sky": shadow is not None or cloud is not None}
    out: Dict[str, List[Any]] = {k: [] for k in ("bg", "air", "occ", "sky", "tau", "noise", "shadow", "cloud")}
    reflect, replicate = cv2.BORDER_REFLECT_101, cv2.BORDER_REPLICATE
    for f in range(affines.shape[0]):
        A = affines[f]
        for j, name in enumerate(("bg", "air", "occ", "sky")):
            out[name].append(_warp(scene[j], A, view, reflect) if need[name] else zeros)
        if active:
            Gf = _compose(G, A)
            out["tau"].append({j: _warp(tau_grid[f, j], Gf, view, replicate) for j in active})
            xa, za, xb, zb = cv2.split(_warp(tc_grid[f], Gf, view, replicate))
            noise = np.zeros((view.H, view.W), dtype=np.float32)
            att = None if att_grid is None else cv2.split(_warp(att_grid[f], Gf, view, replicate))
            for j, (mx, mz) in enumerate(((xa, za), (xb, zb))):
                if coef[f, j] > 0.0:
                    sample = cv2.remap(pattern, mx, mz, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                    if att is not None:
                        sample *= att[j]
                    noise += np.float32(coef[f, j]) * sample
            out["noise"].append(noise)
        for name, layer in (("shadow", shadow), ("cloud", cloud)):
            if layer is None:
                continue
            dt = float(times[f]) - layer["t_mid"]
            centre = (layer["centre"][0] + layer["velocity"][0] * dt, layer["centre"][1] + layer["velocity"][1] * dt)
            S = _sprite_affine(centre, layer["extent"], layer["sprite"].shape)
            out[name].append(_warp(layer["sprite"], _compose(S, A), view, cv2.BORDER_CONSTANT))
    return out


@functools.lru_cache(maxsize=1)
def _gaussian_lut() -> np.ndarray:
    """65536 standard-normal quantiles at (k + 0.5) / 65536 (Acklam's inverse normal CDF).

    Sensor noise is drawn as uniform 16-bit integers mapped through this
    table: Gaussian to well below one grey level, ~2.5x cheaper than
    Generator.standard_normal for the ~1e6 samples of a sequence.
    """
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    p = (np.arange(65536) + 0.5) / 65536.0
    x = np.empty_like(p)
    lo = p < 0.02425
    hi = p > 1.0 - 0.02425
    mid = ~(lo | hi)
    q = p[mid] - 0.5
    r = q * q
    x[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
             (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    for sel, sign, pp in ((lo, 1.0, p[lo]), (hi, -1.0, 1.0 - p[hi])):
        q = np.sqrt(-2.0 * np.log(pp))
        x[sel] = sign * (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                 ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    out = (x / x.std()).astype(np.float32)
    out.flags.writeable = False
    return out


# ---------------------------------------------------------------------- exposure and compositing
def _group_roles(sim: SimulationResult) -> Dict[str, str]:
    smoke_l, fog_l = (int(v) for v in sim.labels)
    params = sim.params or {}
    distractor = params.get("distractor_regime")
    if distractor is None and isinstance(params.get("distractor"), dict):
        distractor = params["distractor"].get("regime")
    tracer_primary = sim.category == "neither" and distractor in PRIMARY_TRACER_REGIMES
    return {"smoke": "primary" if smoke_l else "unlabelled",
            "fog": "primary" if fog_l else "unlabelled",
            "tracer": "primary" if tracer_primary else "background"}


def _exposure_scales(sim: SimulationResult, rp: Mapping[str, Any], layers: Dict[str, List[Any]], n_obs: int):
    """Per-component tau scale factors (3,) and a JSON-able record of the group normalisation.

    The reference optical depth is the 90th percentile of the untextured tau
    over the aerosol-occupied pixels in view (tau above 2 % of its maximum,
    above the floor, outside terrain) over the observation frames: the body
    of the plume or layer, not its near-source spike. Normalising the maximum
    instead let a dense source region (tau > 100 for a hot plume) push the
    rest of the plume below visibility; now the source may saturate, as dense
    smoke does, while the target opacity describes what most of it looks like.
    """
    roles = _group_roles(sim)
    groups: Dict[str, Any] = {}
    scales = np.ones(3, dtype=np.float64)
    for j, c in enumerate(COMPONENTS):
        raw_peak = 0.0
        if layers["tau"] and j in layers["tau"][0]:
            frames_tau = [layers["tau"][f][j] * layers["air"][f] for f in range(n_obs)]
            tau_max = max(float(t.max()) for t in frames_tau)
            if tau_max > 0.0:
                occupied = np.concatenate([t[t > 0.02 * tau_max] for t in frames_tau])
                raw_peak = float(np.percentile(occupied, 90.0)) if occupied.size else 0.0
        role = roles[c]
        target = None
        if role == "primary":
            target = float(rp["peak_opacity"][c])
        elif role == "background":
            target = float(rp["background_haze_opacity"])
        normalised = target is not None and raw_peak >= RAW_TAU_MIN
        if normalised:
            scales[j] = -math.log(1.0 - min(max(target, 0.0), _MAX_OPACITY)) / raw_peak
        groups[c] = {"role": role, "raw_peak_tau": raw_peak, "target_opacity": target,
                     "scale": float(scales[j]), "normalised": bool(normalised)}
    return scales, groups


def _composite_frame(L: Dict[str, List[Any]], f: int, rp: Mapping[str, Any], scales: np.ndarray,
                     active: Sequence[int], shadow: Optional[dict], cloud: Optional[dict]):
    """Scene radiance of frame f (before photometry) and its optical quantities.

    Returns (image, visible contrast, {component: textured tau}, alpha, cloud opacity, background
    behind the aerosol).
    """
    B, air, occ, sky = L["bg"][f], L["air"][f], L["occ"][f], L["sky"][f]
    taus: Dict[int, np.ndarray] = {}
    alpha = A = None
    if active:
        g = float(rp["texture"]["gain"])
        w = np.exp(np.float32(g) * np.clip(L["noise"][f], -3.5, 3.5) - np.float32(0.5 * g * g))
        w *= air
        tot = np.zeros_like(B)
        num = np.zeros_like(B)
        for j in active:
            t = L["tau"][f][j] * (w * np.float32(scales[j]))
            taus[j] = t
            tot += t
            num += np.float32(rp["aerosol_brightness"][COMPONENTS[j]]) * t
        alpha = 1.0 - np.exp(-tot)
        A = num / np.maximum(tot, np.float32(1e-12))
        # Self-shadowing: the 3D structure of dense smoke or fog makes its brightness vary
        # with the same advected texture, so even an opaque region shows moving structure
        # (with radiance alone, saturated regions were flat and optical flow read ~0 there).
        sh = float(rp["texture"].get("shading", 0.0))
        if sh > 0.0:
            A = np.clip(A * (1.0 + np.float32(sh) * np.clip(L["noise"][f], -2.5, 2.5)), 0.0, 1.0)
    Bc = B
    a_cloud = None
    contrast = None
    if cloud is not None:
        C = np.float32(cloud["brightness"])
        a_cloud = L["cloud"][f] * sky                  # a distant cloud is only seen against the sky
        Bc = B + a_cloud * (C - B)
        contrast = a_cloud * np.abs(C - B)
    if alpha is not None:
        img = Bc + alpha * (A - Bc)
        vis = alpha * np.abs(A - Bc)
        contrast = vis if contrast is None else contrast + vis
    else:
        img = Bc.copy()
    if shadow is not None:
        # A cloud shadow darkens lit surfaces and the aerosol inside it, not the sky.
        lit = Bc * (1.0 - sky)
        if alpha is not None:
            lit = (1.0 - alpha) * lit + alpha * A
        change = np.float32(shadow["strength"]) * L["shadow"][f] * lit
        img -= change
        contrast = change if contrast is None else contrast + change
    img += occ * (B - img)                             # the occluder value is baked into B under it
    if contrast is None:
        contrast = np.zeros_like(B)
    contrast *= 1.0 - occ
    return img, contrast, taus, alpha, a_cloud, Bc


def _jpeg(frame: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if ok else frame


def _candidate_masks(rp: Mapping[str, Any], detect: np.ndarray, occ: np.ndarray, frames: np.ndarray,
                     timestamps: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mk = rp["masks"]
    source = mk["source"]
    if source == "detector":
        masks = detect_candidate_masks(frames, timestamps, mk.get("detector"))
        if rp.get("fp_boost"):
            h, w = frames.shape[1:]
            _false_positive_tracks(masks, rng, float(mk["false_positive_prob"]),
                                   max(4, int(round(float(mk["min_area_frac"]) * h * w))))
        return masks
    masks = degrade_masks(detect, rng, {k: mk[k] for k in _DEGRADE_KEYS}, occluder=occ)
    if source == "bbox":
        masks = boxes_from_masks(masks, rng, float(mk.get("bbox_jitter", 0.1)))
    elif source != "degraded":
        raise ValueError(f"unknown mask source {source!r}")
    return masks


def _box_peak(img: np.ndarray) -> float:
    """Maximum of a 5x5 box mean: an in-view peak that ignores single-pixel texture speckle."""
    return float(cv2.blur(img, (_PEAK_BOX_PX, _PEAK_BOX_PX)).max())


def _visibility(comp: np.ndarray, total: np.ndarray, masks: np.ndarray, detect: np.ndarray) -> Dict[str, Any]:
    """Per frame and component: in-view peak opacity (max of a 5x5 box mean) and the share of
    that component's opacity inside the candidate mask. comp is (4, T, H, W)."""
    T = total.shape[0]
    peaks: Dict[str, List[float]] = {}
    share: Dict[str, List[float]] = {}
    layers = [(c, comp[j]) for j, c in enumerate(OPACITY_COMPONENTS)] + [("total", total)]
    for name, op in layers:
        if not op.any():
            peaks[name] = [0.0] * T
            share[name] = [0.0] * T
            continue
        peaks[name] = [_box_peak(op[t]) for t in range(T)]
        s = []
        for t in range(T):
            whole = float(op[t].sum(dtype=np.float64))
            s.append(float(op[t][masks[t]].sum(dtype=np.float64)) / whole if whole > 1e-9 else 0.0)
        share[name] = s
    return {
        "peak_opacity": peaks,
        "mask_share": share,
        "mask_area_frac": [float(m.mean()) for m in masks],
        "peak_contrast": [float(d.max()) for d in detect],
    }


# ---------------------------------------------------------------------- public API
def render_sequence(sim: SimulationResult, render_params: Dict[str, Any], rng: np.random.Generator) -> RenderedSequence:
    """Render observation frames, lag frames (when the simulation has lag snapshots) and candidate
    masks of one variant (see module docstring)."""
    rp = render_params
    T = int(sim.n_frames)
    has_lag = getattr(sim, "lag_times", None) is not None
    if has_lag:
        for name in ("smoke_lag", "lwc_lag", "tracer_lag"):
            if getattr(sim, name, None) is None:
                raise ValueError(f"simulation {sim.sim_id} has lag_times but no {name}: lag frames cannot be rendered")
    declared = (sim.params or {}).get("render_hints") or {}
    if declared and _normalise_hints(declared) != _normalise_hints(rp.get("hints")):
        # Nuisance presence is decided by the simulation's hints only (an illumination_change
        # negative without its shadow would be an empty scene).
        raise ValueError(f"render params were sampled with hints {rp.get('hints')} but simulation "
                         f"{sim.sim_id} declares {_normalise_hints(declared)}; pass sim.params['render_hints']")
    view = _view(sim, rp)
    H, W = view.H, view.W
    lags = (True,) if has_lag else ()
    F = (1 + len(lags)) * T                           # observation frames, then their lag frames (if any)
    times = np.concatenate([np.asarray(sim.times, dtype=np.float64)]
                           + [np.asarray(sim.lag_times, dtype=np.float64) for _ in lags])
    tau_grid = np.concatenate([_optical_depth_grid(sim, rp, lag) for lag in (False,) + lags])
    active = [j for j in range(3) if tau_grid[:, j].max(initial=0.0) > 0.0]

    # Advected texture: octaves from max(lam_lo, 2.5 image px) to lam_hi on a 384 px tile.
    lam_lo, lam_hi = (float(v) for v in rp["texture"]["wavelengths_m"])
    lam_lo = max(lam_lo, _MIN_WAVELENGTH_IMAGE_PX * view.s)
    lam_hi = max(lam_hi, 2.0 * lam_lo)
    pixel_m = lam_lo / _PATTERN_PX_PER_WAVELENGTH
    pattern = _texture_pattern(int(rp["texture"]["seed"]), round(pixel_m, 6), round(lam_lo, 6), round(lam_hi, 6))
    off = (0.5 * _PATTERN_SIZE + 37.3, 0.5 * _PATTERN_SIZE + 71.9)
    # Subgrid turbulence (None without sim.sgs_u or with sgs_gain 0): extra texture displacement.
    disp, sgs_record = _sgs_displacement(_sgs_velocity(sim, has_lag), times, float(sim.dx), rp["texture"])
    max_stretch = rp["texture"].get("max_stretch")
    tex_parts = [_texcoord_grid(sim, lag, 1.0 / pixel_m, off,
                                None if disp is None else disp[i * T:(i + 1) * T], max_stretch)
                 for i, lag in enumerate((False,) + lags)]
    advected = tex_parts[0][2]
    att_grid = (None if tex_parts[0][3] is None else
                np.ascontiguousarray(np.concatenate([p[3] for p in tex_parts]).transpose(0, 2, 3, 1)))

    scene, airlight, occluded = _scene_canvas(sim, rp, view)
    shadow, cloud = _nuisance_layers(rp, view, times)
    # The only draws from `rng`, in this order: camera, exposure flicker, sensor noise, masks.
    affines = _camera_affines(rng, rp, times, view)
    gains = 1.0 + float(rp["exposure_flicker"]) * np.clip(rng.standard_normal(F), -3.0, 3.0)
    noise_idx = np.frombuffer(rng.bytes(2 * F * H * W), dtype="<u2").reshape(F, H, W)

    L = _sample_layers(view, scene, tau_grid, np.concatenate([p[0] for p in tex_parts]),
                       np.concatenate([p[1] for p in tex_parts]), pattern, affines, times, shadow, cloud, active,
                       occluded, att_grid)
    scales, groups = _exposure_scales(sim, rp, L, T)

    haze, con, offset = float(rp["global_haze"]), float(rp["contrast"]), float(rp["brightness_offset"])
    noise_lut = _gaussian_lut() * np.float32(255.0 * float(rp["noise_sigma"]))
    frames = np.empty((F, H, W), dtype=np.uint8)
    detect = np.empty((T, H, W), dtype=np.float32)
    occ_obs = np.empty((T, H, W), dtype=np.float32)
    comp = np.zeros((len(OPACITY_COMPONENTS), T, H, W), dtype=np.float32)
    comp_contrast = {c: [0.0] * T for c in OPACITY_COMPONENTS}
    k_vis = (1.0 - haze) * con
    total = np.zeros((T, H, W), dtype=np.float32)
    for f in range(F):
        img, contrast, taus, alpha, a_cloud, Bc = _composite_frame(L, f, rp, scales, active, shadow, cloud)
        # Photometry: global haze (airlight), contrast, brightness offset, exposure flicker, noise.
        k1 = 255.0 * (1.0 - haze) * con * gains[f]
        k0 = 255.0 * ((haze * airlight - 0.5) * con + 0.5 + offset) * gains[f] + 0.5
        y = img * np.float32(k1)
        y += np.float32(k0)
        y += np.take(noise_lut, noise_idx[f])
        frames[f] = np.clip(y, 0.0, 255.0).astype(np.uint8)
        if rp.get("jpeg"):
            frames[f] = _jpeg(frames[f], int(rp["jpeg_quality"]))
        if f < T:
            # Visible contrast in final-image units: haze and contrast scale what a detector sees.
            detect[f] = contrast * np.float32(k_vis)
            occ_obs[f] = L["occ"][f]
            visible = 1.0 - L["occ"][f]
            for j, t in taus.items():
                comp[j, f] = (1.0 - np.exp(-t)) * visible
                # The contrast this component alone would show against what is behind it.
                c_j = comp[j, f] * np.abs(np.float32(rp["aerosol_brightness"][COMPONENTS[j]]) - Bc)
                comp_contrast[COMPONENTS[j]][f] = _box_peak(c_j) * k_vis
            if a_cloud is not None:
                comp[3, f] = a_cloud * visible
                c_c = comp[3, f] * np.abs(np.float32(cloud["brightness"]) - L["bg"][f])
                comp_contrast["cloud"][f] = _box_peak(c_c) * k_vis
            if alpha is not None or a_cloud is not None:
                clear = (1.0 - alpha) if alpha is not None else 1.0
                total[f] = (1.0 - clear * (1.0 - (a_cloud if a_cloud is not None else 0.0))) * visible

    obs_frames, lag_frames = frames[:T], (frames[T:] if has_lag else None)
    masks = _candidate_masks(rp, detect, occ_obs, obs_frames, np.asarray(sim.times), rng)
    visibility = _visibility(comp, total, masks, detect)
    visibility["peak_contrast_by_component"] = comp_contrast
    visibility["groups"] = groups
    visibility["texture_advected"] = bool(advected)
    visibility["sgs_texture"] = sgs_record
    visibility["metres_per_pixel"] = float(view.s)
    visibility["floor_row"] = int(view.H_sim)
    return RenderedSequence(
        frames=obs_frames,
        lag_frames=lag_frames,
        masks=masks.astype(bool),
        timestamps=np.asarray(sim.times, dtype=np.float64).copy(),
        lag_timestamps=np.asarray(sim.lag_times, dtype=np.float64).copy() if has_lag else None,
        mask_source=str(rp["masks"]["source"]),
        opacity=total,
        component_opacity={c: (comp[j].astype(np.float16) if comp[j].any() else np.zeros(comp[j].shape, np.float16))
                           for j, c in enumerate(OPACITY_COMPONENTS)},
        visibility=visibility,
        render_params=copy.deepcopy(dict(rp)),
    )


def check_label_visibility(rendered: RenderedSequence, labels: Tuple[int, int], cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Accept a rendered variant only if what the camera shows supports its labels.

    Every labelled component of (smoke, fog) needs an in-view peak opacity >=
    render.label_min_opacity in at least render.label_min_frames observation
    frames; every unlabelled one must stay <= render.absent_max_opacity in
    every frame. Returns (accepted, reasons for rejection).

    Optional (off unless `render.label_min_contrast` is configured): the
    labelled component must also reach that visible contrast
    (opacity x |A - B| in final-image units) in those frames, which rejects
    aerosol that is opaque but as bright as the terrain behind it.
    """
    r = _render_block(cfg)
    min_op = float(r.get("label_min_opacity", 0.08))
    min_frames = int(r.get("label_min_frames", 4))
    absent = float(r.get("absent_max_opacity", 0.03))
    min_contrast = r.get("label_min_contrast")
    peaks = rendered.visibility["peak_opacity"]
    contrasts = rendered.visibility.get("peak_contrast_by_component", {})
    reasons: List[str] = []
    for name, lab in zip(("smoke", "fog"), labels):
        p = np.asarray(peaks[name], dtype=np.float64)
        if int(lab):
            ok = p >= min_op
            if min_contrast is not None and name in contrasts:
                ok &= np.asarray(contrasts[name], dtype=np.float64) >= float(min_contrast)
            n = int(ok.sum())
            if n < min_frames:
                what = f"peak opacity >= {min_op}" + (f" and contrast >= {min_contrast}" if min_contrast is not None else "")
                reasons.append(f"{name} labelled but {what} in only {n} of {p.size} frames (need {min_frames})")
        elif p.size and float(p.max()) > absent:
            reasons.append(f"{name} not labelled but visible: peak opacity {float(p.max()):.3f} > {absent}")
    return (not reasons), reasons
