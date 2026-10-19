# Design and interface contract

This document is the contract between the modules of the physics expert.
Each module owner implements exactly the public API listed here; anything
else is private (leading underscore). If an implementation needs a contract
change, the change is made here first.

Status: synthetic pipeline only. Nothing in this repository has been
validated on real camera footage.

## 0. Conventions

* Python >= 3.10, numpy, numba (`@njit(cache=True)`, never `parallel=True`),
  OpenCV (`cv2`), PyTorch (CPU), PyYAML, matplotlib (reports only), pytest.
  No scipy, no scikit-learn, no pandas.
* Everything runs on ONE CPU thread. `import src` sets the thread env vars;
  scripts call `src.utils.runtime.configure_single_thread()` at start.
* Randomness: functions take an explicit `np.random.Generator` (`rng`). The
  global numpy RNG is never used. Per-sample streams come from
  `src.utils.runtime.child_rng(base_seed, *keys)`.
* 2D fields are `(nz, nx)`, row `k = 0` at the GROUND, `z = (k + 0.5) dx`,
  column `i = 0` at the left, `x = (i + 0.5) dx`. Images are `(H, W)` with
  row 0 at the TOP; the renderer flips rows.
* Units: SI. Smoke and tracer concentration in mg/m^3; `q_v`, `q_l` in kg/kg;
  liquid water content (LWC) in g/m^3; temperature scalar is POTENTIAL
  temperature theta (K); absolute temperature `T = theta * Exner(p(z))`.
* Constants live in `src/constants.py` (labels, categories, regimes) and
  `src/data/feature_spec.py` (feature layout, gates). Do not duplicate them.
* Module docstrings say where the module fits, what calls it and what it
  calls (see `src/utils/runtime.py`).

## 1. Physics model (summary)

Incompressible flow with Boussinesq buoyancy in a vertical x-z section,
solved with a D2Q9 BGK lattice-Boltzmann method (weakly compressible,
low Mach), Guo forcing, Smagorinsky subgrid viscosity (still a BGK
single-relaxation-time collision, with a local tau).

Buoyancy (per unit mass, +z up), relative to the horizontally uniform
ambient state at rest:

    b = g * [ (theta - theta_ref(z)) / theta0 + 0.608 (q_v - q_v_ref(z)) - (q_l - q_l_ref(z)) ]

Transported scalars: the anomalies theta_l' = theta_l - theta_l_ref(z) and
q_t' = q_t - q_t_ref(z) of the phase-change-conserved variables
(theta_l = theta - L_v/(c_p Exner) q_l, q_t = q_v + q_l), smoke C, tracer D
and four texture coordinates. They are advected by the LBM velocity in
finite-volume advective form (flux divergence minus C div u, see
scalars.py) and diffused with an eddy diffusivity K (sampled independently
of the LBM viscosity):

    dC/dt + u . grad C = div(K grad C) + S

with background-advection sources -w d theta_l_ref/dz and -w d q_t_ref/dz
for the anomalies, so the ambient state is exactly steady.

Moisture: theta, q_v and q_l are diagnosed every step by saturation
adjustment from (theta_l, q_t); latent heat is included through theta_l.

Boundaries:
* Bottom: no-slip wall (half-way bounce-back) at z = 0, plus optional
  terrain given by a `solid` mask. An ambient-maintaining body force keeps
  the ambient wind profile steady over the wall (lbm.py). Scalars: zero
  flux, except prescribed surface fluxes and settling deposition.
* Left / right: velocity inlet (feq(rho_node, u_amb)) where the ambient
  wind enters; pressure outlet (feq(1, u_node)) where it leaves or when the
  ambient wind is calm. Top: pressure outlet. Sponges relax towards the
  ambient state at a rate in 1/s (the top sponge never damps w and relaxes
  buoyancy instead). Scalars: where the face velocity points into the domain
  the ghost value is the prescribed inflow profile, otherwise zero-gradient.
  Wind can blow from either side, so there is no mirroring.
* Forcing: buoyancy, gust acceleration dU/dt applied everywhere, the
  ambient-maintaining force and a divergence-free stochastic turbulence
  forcing advected with the wind. A source-free precursor spins up the flow
  before the scenario starts.

Lattice units: `dx` (m), `dt` (s); `u_lat = u dt/dx`,
`nu_lat = nu dt/dx^2`, `tau = 0.5 + 3 nu_lat`, `a_lat = a dt^2/dx`.
The starting `dt` (0.1 s) is halved before the run when the expected peak
speed would exceed `target_lattice_speed`, and the run is restarted with a
halved `dt` (up to `max_dt_halvings`) if the measured lattice speed ever
exceeds `max_lattice_speed`. A run that still fails, or fails any check in
`src/cfd/validation.py`, is rejected and never rendered.

## 2. `src/cfd/lbm.py`  (owner: LBM)

```python
CS2: float = 1/3
E: np.ndarray       # (9, 2) int64, [ex, ez]: 0:(0,0) 1:(1,0) 2:(0,1) 3:(-1,0) 4:(0,-1) 5:(1,1) 6:(-1,1) 7:(-1,-1) 8:(1,-1)
W: np.ndarray       # (9,) float64 weights 4/9, 1/9 x4, 1/36 x4
OPP: np.ndarray     # (9,) int64 opposite directions

@dataclass(frozen=True)
class LatticeUnits:
    dx: float                    # m
    dt: float                    # s
    def velocity_to_lattice(self, u): ...      # u dt/dx (works on arrays)
    def velocity_to_physical(self, u_lat): ...
    def viscosity_to_lattice(self, nu): ...    # nu dt/dx^2
    def acceleration_to_lattice(self, a): ...  # a dt^2/dx
    def tau(self, nu) -> float: ...            # 0.5 + 3 nu_lat

@dataclass
class LBMConfig:
    nx: int
    nz: int
    units: LatticeUnits
    nu: float                       # base kinematic viscosity, m^2/s (eddy, not molecular)
    smagorinsky_cs: float = 0.12
    sponge_width: int = 6           # cells
    sponge_strength: float = 0.08
    lateral: str = "open"           # "open" | "periodic" | "wall"
    top: str = "open"               # "open" | "wall"

class LBMSolver:
    def __init__(self, cfg: LBMConfig, solid: np.ndarray | None = None,
                 u_ambient: np.ndarray | None = None): ...
        # solid: (nz, nx) bool terrain (the z=0 wall is implicit, not in the mask)
        # u_ambient: (nz,) horizontal ambient wind, m/s; flow initialised to it
        # (zero inside solid), density 1.
    cfg: LBMConfig
    solid: np.ndarray               # (nz, nx) bool
    fluid: np.ndarray               # (nz, nx) bool, ~solid
    f: np.ndarray                   # (nz, nx, 9) float64 populations (cell-contiguous)
    rho: np.ndarray                 # (nz, nx) lattice density
    ux_lat: np.ndarray              # (nz, nx) lattice velocity (half-force corrected)
    uz_lat: np.ndarray
    def set_ambient(self, u_ambient: np.ndarray) -> None   # (nz,) m/s, may change every step (gusts)
    def step(self, ax: np.ndarray | None, az: np.ndarray | None) -> None
        # one collide + stream + boundary update with body acceleration (m/s^2, (nz, nx))
    def velocity(self) -> tuple[np.ndarray, np.ndarray]    # physical (ux, uz) m/s, 0 in solid
    def max_lattice_speed(self) -> float                    # over fluid cells
    def max_mach(self) -> float                             # max_lattice_speed / sqrt(CS2)
    def total_mass(self) -> float                           # sum of rho over fluid cells
    def is_finite(self) -> bool
```

Required tests (`tests/test_lbm.py`): D2Q9 moment identities; rest state
stays at rest with zero force (open and closed); closed box conserves mass
to ~1e-12 relative; body-force-driven channel flow (periodic x, wall top
and bottom) matches the parabolic profile within 2 % (validates the
viscosity mapping and Guo forcing); a warm (positive `az`) blob rises;
uniform ambient wind passes through the open domain for 2000 steps stably
with interior velocity close to the ambient profile; terrain obstacle flow
stays finite; a `@pytest.mark.slow` convergence check.

## 3. `src/cfd/scalars.py`  (owner: scalars)

```python
@dataclass(frozen=True)
class ScalarSpec:
    name: str
    diffusivity: float               # m^2/s
    nonnegative: bool = True         # theta is the only one with False
    settling_velocity: float = 0.0   # m/s, >0 falls (dust); deposits at the ground/terrain

@dataclass
class MassBudget:                    # per scalar, cumulative, in field-units * m^2 (per metre of depth)
    initial: float = 0.0
    sources: float = 0.0             # added by add_source (positive or negative)
    inflow: float = 0.0              # entered through open boundaries
    outflow: float = 0.0             # left through open boundaries (positive number)
    deposited: float = 0.0           # settled out at walls (positive number)
    conversion: float = 0.0          # phase change in (+) / out (-) (q_v <-> q_l)
    clipped: float = 0.0             # mass added by clipping negatives to 0 (positive number)
    def expected(self) -> float      # initial + sources + inflow - outflow - deposited + conversion + clipped
    def residual(self, current: float) -> float  # (current - expected) / max(scale, tiny)

class ScalarTransport:
    def __init__(self, nx: int, nz: int, units: LatticeUnits, solid: np.ndarray | None,
                 specs: list[ScalarSpec], scheme: str = "muscl",
                 lateral: str = "open", top: str = "open"): ...
    budgets: dict[str, MassBudget]
    def start_budget(self, fields: dict[str, np.ndarray]) -> None   # record initial masses
    def step(self, fields: dict[str, np.ndarray], ux: np.ndarray, uz: np.ndarray,
             inflow: dict[str, dict[str, np.ndarray]] | None = None) -> None
        # advance every field one dt IN PLACE: conservative FV advection (face velocity =
        # mean of the two adjacent cell-centre velocities, zero on wall/solid faces) +
        # explicit diffusion (no diffusive flux through walls or open boundaries) +
        # settling. inflow[name] = {"left": (nz,), "right": (nz,), "top": (nx,)} ghost values
        # used where the face velocity points INTO the domain (missing keys -> zero-gradient).
        # Velocities physical m/s at cell centres. Updates budgets.
    def add_source(self, name: str, field: np.ndarray, rate: np.ndarray) -> None
        # field += dt * rate on fluid cells; rate in field-units/s; budget.sources updated
    def record_conversion(self, name: str, delta: np.ndarray) -> None
        # account an in-place change made by the caller (phase change), field-units
    def total(self, name: str, field: np.ndarray) -> float           # sum over fluid * dx^2
    def max_cfl(self, ux, uz) -> float
```

Explicit stability: the class raises `ValueError` if `max(|u|) dt/dx` > 0.5
or `K dt/dx^2` > 0.2 for any spec at construction/step time.

Required tests (`tests/test_scalars.py`): uniform field stays uniform under
uniform flow; pure advection of a blob in uniform flow translates it with
the right speed and stays non-negative; budget residual < 1e-10 with open
inflow/outflow, sources and settling; closed box conserves exactly;
diffusion of a Gaussian matches the analytic variance growth; theta
(nonnegative=False) allowed negative anomalies; MUSCL is less diffusive
than upwind.

## 4. `src/cfd/thermo.py`  (owner: scalars)

```python
G = 9.81; CP = 1004.0; R_D = 287.04; R_V = 461.5; L_V = 2.5e6; EPS = 0.622; P0 = 100000.0
def pressure_profile(z: np.ndarray, p_surface: float = 101325.0, theta0: float = 288.0) -> np.ndarray  # Pa
def exner(p: np.ndarray) -> np.ndarray                 # (p / P0) ** (R_D / CP)
def saturation_vapor_pressure(T: np.ndarray) -> np.ndarray   # Pa, Bolton (1980) over liquid water
def saturation_mixing_ratio(T: np.ndarray, p: np.ndarray) -> np.ndarray  # kg/kg
def relative_humidity(qv, T, p) -> np.ndarray
def saturation_adjustment(theta, qv, ql, p, exner_p, iterations: int = 3) -> np.ndarray
    # IN PLACE on theta, qv, ql (nz, nx); returns condensed amount dq (kg/kg, + = condensation).
    # Conserves qv + ql to round-off, keeps qv, ql >= 0, latent heating L_V/(CP*exner) * dq.
def lwc_from_ql(ql, rho_air: float = 1.2) -> np.ndarray   # g/m^3
def fog_extinction(lwc) -> np.ndarray                     # 1/m, Kunkel (1984): 0.1447 * LWC^0.88
```

Required tests (`tests/test_scalars.py`, thermo section): q_s increases
with T; supersaturated air condenses to saturation (RH = 1 within 1e-3) and
warms; sub-saturated air with liquid evaporates and cools; total water
conserved; no negatives; already-saturated fog is a fixed point.

## 5. Scenarios and the coupled run (`smoke.py`, `fog.py`, `scenarios.py`, `simulation.py`, `validation.py`)  (owner: CFD integration — implemented)

```python
# scenarios.py
SPLITS = ("train", "val_select", "val_calib", "test_iid", "test_ood")
HOLDOUT_TAGS = ("fast_wind", "elevated_source")
def scenario_seed(cfg, split, category, index, attempt=0) -> int    # derive_seed(base, split, category, index, attempt)
def sample_scenario(category, split, index, cfg, attempt=0) -> dict  # JSON-able params, deterministic
def build_spec(params, cfg, dt=None, halvings=0) -> SimulationSpec
def run_scenario(params, cfg) -> SimulationResult    # dt halving retries; raises SimulationUnstable
# simulation.py
def run_simulation(spec, cfg) -> SimulationResult
```

Context-first sampling (see the `scenarios.py` docstring): wind speed
(0-7 m/s for every category), direction, gusts, mixing, terrain (P = 0.35,
same shapes for every category), atmosphere, spin-up, background haze and
the camera nuisances are class-independent; a compatible regime is then
chosen. `params` keys used downstream:

* `sim_id`, `category`, `split`, `index`, `attempt`, `seed`, `labels`,
  `regime`, `smoke_regime`, `fog_regime`, `distractor_regime`, `holdout_tags`
* `wind` (`speed_ms`, `sign`, `gust_fraction`, ...), `terrain` (`kind`, `height_m`),
  `mixing`, `atmosphere`, `timing`
* `render_hints`: `{"shadow": bool, "strong_artifact": bool, "fp_boost": bool,
  "distant_cloud": bool}` — class-independent nuisance flags the renderer
  MUST honour (it samples their strengths, not their presence)
* `background_haze`: dict or None (tracer as faint background haze);
  `distractor`: for neither (`haze_drift` / `dust_burst` put aerosol in the
  tracer field; `illumination_change` / `camera_artifact` / `distant_cloud`
  have no aerosol and rely on the corresponding nuisance)

Held-out recipes (`test_ood` only, every category, alternating by index):
`fast_wind` (7-9 m/s) and `elevated_source` (a 45-80 m hill; smoke sources
on the hilltop). Validity (`validation.py`) includes label support: the
labelled aerosol(s) visible in view, unlabelled ones absent.

## 6. `src/data/render.py` and `src/data/masks.py`  (owner: render)

```python
@dataclass
class RenderedSequence:
    frames: np.ndarray          # (T, H, W) uint8 grayscale at the observation times
    lag_frames: np.ndarray      # (T, H, W) uint8 at the lag times (t_k - flow_lag_s)
    masks: np.ndarray           # (T, H, W) bool candidate masks at the observation times
    timestamps: np.ndarray      # (T,) float64 s (= sim.times)
    lag_timestamps: np.ndarray  # (T,) float64 s (= sim.lag_times)
    mask_source: str            # "degraded" | "bbox" | "detector"
    opacity: np.ndarray         # (T, H, W) float32 total aerosol opacity (diagnostic only)
    component_opacity: dict     # {"smoke","fog","tracer","cloud"} -> (T, H, W) float16 (diagnostic only)
    visibility: dict            # per-frame visibility statistics, see below
    render_params: dict

def sample_render_params(rng, cfg: dict, hints: dict | None = None,
                         overrides: dict | None = None) -> dict
def render_sequence(sim: SimulationResult, render_params: dict, rng) -> RenderedSequence
def check_label_visibility(rendered: RenderedSequence, labels: tuple[int, int], cfg: dict) -> tuple[bool, list[str]]
```

Optical model: for each component `c` in {smoke, fog, tracer}, extinction
`beta_c = sigma_c * field_c` (fog: `thermo.fog_extinction(LWC)`) times its own
line-of-sight depth L_c (drawn independently from one distribution) gives
optical depth `tau_c(x, z)` on the grid. It is upsampled (bilinear) to the
camera view and MODULATED BY ADVECTED TEXTURE: a fixed high-resolution
tileable noise pattern (several octaves, wavelengths ~3-40 m, one class-
independent random gain per sequence, identical for all components) is
sampled at the upsampled texture coordinates `sim.texcoords` (pair a and pair
b blended with `sim.tex_weights`), so the texture moves exactly with the
simulated air. Without this step a quasi-steady concentration field produces
no optical flow at all (the reviewers measured flow/fluid ratios of
0.001-0.02). Then `alpha = 1 - exp(-sum_c tau_c)` and
`I = (1 - alpha) B + alpha A`, with ONE aerosol-brightness distribution for
all aerosols. Lag frames are rendered with the same parameters from the
`_lag` fields.

Native resolution: frames are rendered at `render.native_size` (default
192 x 192), noise and JPEG are applied at that size (not at the 128 px
working size), exactly like real footage that is compressed natively and
downscaled later by the feature extractor.

Exposure normalisation, per component GROUP (never jointly):
* primary groups — smoke if labelled, fog if labelled, tracer if the
  category is `neither` and the distractor is `haze_drift` / `dust_burst` —
  each get their own sampled target peak in-view opacity from
  `render.peak_opacity` (lower floor 0.05); their tau is scaled so the
  sequence peak opacity matches it;
* background haze (tracer in any other case) gets a target from
  `render.background_haze_opacity` ([0.02, 0.2]);
* a group whose raw peak tau is below 1e-6 is left untouched (no division).

Camera: per-frame shake/rotation is applied to frames AND masks (masks are in
the same, shaken, coordinates as the frames, as real detector masks are). A
foreground ground band (0-20 % of the image height, textured) is rendered below
the simulated floor so the domain floor is not the image edge; occluders,
zoom and backgrounds (procedural sky gradient + multi-scale ridge and ground
texture, so stabilisation has something to lock onto) are sampled identically
for every category.

Nuisances: `params["render_hints"]` decides PRESENCE of shadow / strong
camera artefact / detector false-positive boost / distant cloud; strengths
come from config ranges identical for every category. `distant_cloud`
renders a bright soft textured cloud in the sky band drifting slowly.

Masks: the class-agnostic detectability map is VISIBLE CONTRAST,
`alpha * |A - B|` (plus the brightness-change magnitude of a moving shadow),
not raw opacity. `mask_source` is sampled per variant from
`render.masks.source_probs` (default degraded 0.5, bbox 0.2, detector 0.3):
`degraded` thresholds and degrades the detectability map with `degrade_masks`;
`bbox` turns the degraded mask into per-frame jittered bounding boxes of its
components (YOLO-like); `detector` runs `detect_candidate_masks` on the
rendered frames. Boundary noise is kept small ([0, 0.15]).

Visibility statistics (`rendered.visibility`): per frame and component, the
in-view peak opacity and the share of the total opacity inside the candidate
mask. `check_label_visibility` accepts a sequence only if every labelled
component has peak opacity >= `render.label_min_opacity` in at least
`render.label_min_frames` frames and every unlabelled component of {smoke,
fog} stays below `render.absent_max_opacity`.

```python
# masks.py
def degrade_masks(detectability: np.ndarray, rng, params: dict, occluder: np.ndarray | None = None) -> np.ndarray
    # (T, H, W) float -> (T, H, W) bool: random threshold, morphology, boundary noise,
    # fragmentation / dropped components, false-positive blobs (more with fp_boost),
    # frame dropouts, occluded pixels removed. Same statistics for every class.
def boxes_from_masks(masks: np.ndarray, rng, jitter: float = 0.1) -> np.ndarray
def detect_candidate_masks(frames: np.ndarray, timestamps: np.ndarray | None = None,
                           params: dict | None = None) -> np.ndarray
    # image-only, grayscale, class-agnostic fallback detector for footage without masks
```

Clean simulation fields or masks are never used at real-world inference.

## 7. `src/data/features.py`  (owner: features)

```python
class IncompatibleFrameTiming(ValueError): ...

@dataclass
class FeatureConfig:
    work_size: int = 128            # longer side of the analysis frame (ROI) after resizing
    roi: str = "union_bbox"         # "union_bbox" | "full"
    roi_scale: float = 2.0          # ROI side = roi_scale * max(extent w, h) (synthetic: drawn from [1.5, 2.5])
    roi_min_px: int = 32            # minimum ROI side in native pixels
    stabilize: bool = True
    min_mask_frac: float = 0.002    # of the analysis frame
    flow_method: str = "dis"        # "dis" (default) | "farneback"
    flow_erode_px: int = 2          # flow statistics on the mask eroded by this (working px)
    fb_max_px: float = 1.0          # forward-backward consistency (median, working px)
    anchor_band_px: int = 6
    min_roughness_px: int = 100
    expected_interval_s: float = 5.0
    flow_lag_s: float = 1.0
    max_offset_s: float = 0.5

def extract_features(frames, timestamps, masks, lag_frames=None, lag_timestamps=None,
                     cfg: FeatureConfig | None = None) -> np.ndarray      # (T, 14) float32
def extract_features_with_meta(...same...) -> tuple[np.ndarray, dict]
    # meta: roi box, stabilisation per frame (used / skipped + reason, quality),
    # flow mode ("lag" | "consecutive"), achieved intervals, valid frame counts
def stabilize_sequence(frames, masks, lag_frames=None) -> tuple[...]
def select_window(timestamps, n=12, interval_s=5.0, start=None, max_offset_s=0.5,
                  flow_lag_s=1.0) -> dict
    # {"obs_idx": (n,), "lag_idx": (n,), "offsets": ..., "native_dt": ...}; raises
    # IncompatibleFrameTiming when a selected frame is > max_offset_s from its target,
    # indices repeat, the clip is too short, or the native frame spacing exceeds
    # flow_lag_s / 2 (e.g. one-minute HPWREN/FIgLib imagery is refused, not stretched)
def load_frames(path, times=None, timestamps=None) -> tuple[np.ndarray, np.ndarray]
    # video file (timestamps from the container) or image folder (timestamps from a
    # sidecar CSV or a filename pattern; raises if there are none); decodes only `times`
```

Pipeline (identical for synthetic and real input):

1. Convert every input (uint8, float in [0, 1], BGR) to one uint8 grayscale
   representation in a single private function.
2. Stabilise: a bounded similarity transform of every observation and lag
   frame onto observation frame 0, estimated ONLY from pixels outside the
   dilated union of candidate masks (>= 20 % textured background required),
   warped with BORDER_REPLICATE; masks follow; a frame whose estimate fails
   or exceeds the bound gets `flow_valid = 0`. Skipped entirely (and
   recorded) when the background is insufficient, so a frame-filling fog
   bank is never "stabilised away".
3. Persistent-component filter: a mask component is kept only if, dilated
   by a few pixels, it overlaps a component in the previous or next
   observation (drops single-frame false positives; keeps multi-object
   scenes such as smoke + fog).
4. ROI (analysis frame): square of side `roi_scale * max(bbox w, h)` (at
   least `roi_min_px`) centred on the bounding box of the union of filtered
   masks over the window, shifted to stay inside the image, clipped when
   larger than the image, resized so its longer side is `work_size`. All
   lengths are in units of `L = sqrt(H_roi * W_roi)` of the resized ROI and
   all rates per second. With no mask in any frame, the full image is used.
5. Flow: OpenCV DIS (PRESET_MEDIUM) from the lag frame to the observation
   frame (`lag_frames` given) — displacements stay small and texture stays
   correlated — or, without lag frames, between consecutive observations
   (flagged as `flow_mode = "consecutive"`; the inference API refuses it
   unless explicitly allowed). DIS recovers smooth-field displacements that
   Farneback underestimates 2-5x. `flow_valid[t] = 1` only if t has a flow
   pair, both masks are valid, stabilisation succeeded for the pair, and
   the median forward-backward error over the eroded region is <= `fb_max_px`.
6. Per-frame features (layout and gates in `feature_spec.py`):
   flow mean / std are winsorised (5-95 %) statistics on the mask eroded by
   `flow_erode_px` (fallback: closed flow gate when < 20 px remain);
   divergence is the trace of a least-squares affine fit of the flow over
   that region; vorticity is the mean |curl| from pixel derivatives there;
   all in 1/s after dividing by the pair's elapsed time.
   Area, area growth (d ln area / dt between observations), centroid
   velocity (between observations, +z up), roughness (perimeter from
   `cv2.arcLength` of the external contours after a 1-px open/close,
   excluding segments on the ROI border or occluder; `P / (2 sqrt(pi A)) - 1`
   minus the digital-disc bias; 0 when the region is < `min_roughness_px`),
   anchor stability (column-wise: for columns occupied at t and t-1, the
   bottom-boundary row; `exp(-median |dz_bottom| / (0.02 L))` times the IoU
   of the two column sets; 0 when more than half of the bottom boundary lies
   on the ROI border).
   First observation: `mask_pair` features 0 (no previous observation); its
   flow is available when a lag frame exists.

## 8. `src/data/normalization.py`  (owner: training)

```python
class FeatureNormalizer:
    def __init__(self, clip: float = 5.0)
    def fit(self, X) -> FeatureNormalizer          # X (N, 12, 14) TRAIN ONLY; gated stats
    def transform(self, X) -> np.ndarray           # float32; closed gates -> 0; clip; flags untouched
    def clip_fraction(self, X) -> np.ndarray       # (N,) fraction of open-gate entries hitting the clip
    def to_dict / from_dict / save / load
```

## 9. `src/models/physics_gru.py`  (owner: training)

```python
@dataclass
class ModelConfig: input_dim=14, seq_len=12, hidden_size=32, num_layers=1, embedding_dim=16,
                   dropout=0.1, output_names=("smoke", "fog")
class PhysicsEncoder(nn.Module):   # GRU(14->32) -> last hidden -> Dropout(0.1) -> Linear(32,16) -> LayerNorm -> GELU
class PhysicsGRU(nn.Module):       # encoder + head Linear(16, 2)
    def forward(x) -> logits (B, 2); def encode(x) -> (B, 16); def predict(x) -> {"logits", "probs"}
def save_model_artifacts(model, out_dir, normalizer, extra_config: dict) -> dict[str, Path]
def load_model(artifact_dir) -> (PhysicsGRU, FeatureNormalizer, dict)
def load_encoder(artifact_dir) -> (PhysicsEncoder, FeatureNormalizer, dict)
```

## 10. `src/data/dataset.py`  (owner: pipeline)

```python
@dataclass
class FeatureSet:
    X: (N, 12, 14) float32 raw features; y: (N, 2) float32
    seq_ids, sim_ids, groups, splits, categories, regimes, tags, mask_sources: (N,) str arrays
    meta: list[dict]       # per sequence: params subset, render params, feature meta
    source: str            # "synthetic" | "real"
    save(path) / load(path) / subset(mask) / split(name)
def save_sequence(path, rendered, sim, seq_id, extra=None) -> Path
def load_sequence(path) -> dict
def assert_no_group_leakage(fs) -> None   # groups and seeds never span two splits
class SequenceTensorDataset(torch.utils.data.Dataset)
```

## 11. Training, evaluation, inference (owner: training)

Splits: `train` (fit), `val_select` (early stopping, seed choice),
`val_calib` (temperature, thresholds, abstain band), `test_iid`,
`test_ood`, robustness re-renders of test_iid. Normaliser on train only.

* Training: BCEWithLogits (+`pos_weight` when a label ratio exceeds 1.5),
  AdamW (lr 1e-3, wd 1e-4), batch 64, <= 50 epochs, early stopping
  (patience 7) on val_select loss, grad clip 1.0, three seeds, one thread,
  deterministic. Report test metrics as mean ± sd over the seeds; export
  the val_select-best seed and say so.
* Calibration: per-label temperature on val_calib, bounded to [0.5, 5];
  if val_calib is separable (fit hits a bound), T = 1 and a flag is stored.
* Thresholds, stored with `fit_source = "synthetic_val_calib"` and the
  calibration-set prevalence: `balanced` (max Youden's J per label;
  max F1 degenerated to "always positive" on weak scores) and
  `low_false_alarm` (smoke threshold giving <= 5 % false-smoke rate on
  fog_only val_calib sequences). Full ROC / PR curves are reported.
* Abstention (`decision = -1`): logit within a margin of the threshold
  (margin in logit space, floor 0.25, widened on val_calib until the
  confident error <= 5 % with abstention <= 30 %), OR input-validity
  failure: fewer than 3 valid-mask frames, fewer than 2 valid-flow frames,
  more than 20 % of open-gate normalised entries clipped, or an
  incompatible frame interval. `predict()` returns `calibration_source`.
* Metrics (numpy only): per-label precision / recall / F1 / AUROC / AUPRC,
  4-class confusion from the two decisions, smoke/fog confusion,
  false-smoke rate on fog_only, Brier, ECE with equal-mass bins, abstain
  rate and selective accuracy; 95 % CIs from a grouped bootstrap over
  sim_id (camera id for real data).
* Baselines: A = heuristic (fixed functional form on time-pooled flow_v_mean,
  centroid_dz, anchor_stability, area_growth; coefficients fitted on train
  only; thresholds from val_calib); time-pooled logistic regression (mean
  and std of every feature over time); flags-only probe (features 12-13)
  and per-feature univariate AUCs as a shortcut audit (flag when the
  flags-only AUC > 0.6 for any label).
* Experiments: C (CFD-pretrained GRU) on synthetic splits; leave-one-
  regime-out (LORO) generalisation; robustness suites; mask-source shift
  (detector masks). B and D need real labelled footage and are recorded
  as `null` with a reason until it exists. A "synthetic transfer proxy"
  (pretrained vs scratch on a small shifted synthetic domain) may be run
  and is stored under `synthetic_proxy` with `not_real_world_evidence: true`.

* Diagnostics (never model inputs): `scripts/qa_dataset.py` (gate
  validity per category, univariate feature and parameter AUROCs, optical
  flow vs CFD velocity), `scripts/oracle_diagnostic.py` (the twelve motion
  features recomputed from the CFD fields, compared with the camera
  features per feature group to localise information loss), and
  `scripts/summarize_results.py` (evaluation.json -> reports/results.md).

`evaluation.json` top level: `{"synthetic": {...}, "real": {"status":
"pending", "reason": ...}, "ablations": {"A": ..., "B": null, "C": ...,
"D": null, "reasons": {...}}, "synthetic_proxy": ..., "shortcut_audit":
..., "robustness": ..., "loro": ..., "data": {counts, acceptance}}`.

Real data (Stage B): group = camera/location id; camera-disjoint
train/val/test; B (scratch) uses `training.lr`, D (fine-tune) uses
`finetune.lr`; identical epochs, patience, seeds, splits and normaliser
policy; report counts of real sequences, videos and cameras.

```python
# src/inference/predict.py — imports only torch, numpy, yaml, cv2 and src.models / src.data
class PhysicsExpert:
    @classmethod
    def load(cls, artifact_dir) -> "PhysicsExpert"
    def features_from_frames(self, frames, timestamps, masks=None) -> tuple[np.ndarray, dict]
    def encode(self, features) -> np.ndarray     # raw (B, 12, 14) or (12, 14) -> (B, 16)
    def predict(self, features, operating_point="balanced", meta=None) -> dict
        # logits, probs, decisions (1 / 0 / -1 abstain), abstain_reasons, calibration_source
```

## 12. Data splits and leakage rules

* A simulation and all its rendered variants share one split (group = sim_id).
* Seeds come from (base, split, category, index, attempt): no random stream
  is shared across splits; `assert_no_group_leakage` checks groups, seeds
  and parameter vectors.
* `test_ood` simulations come only from the held-out recipes; train / val /
  test_iid never sample them.
* The normaliser is fitted on train only; early stopping and seed choice on
  val_select; temperatures, thresholds and the abstain band on val_calib.
  Test splits are evaluated once per trained model.
* Real footage: group = camera/location id; one original video never spans
  two splits; evaluation cameras are unseen in training. Real background
  images used for rendering (if any) must come from cameras disjoint from
  real validation/test cameras.
