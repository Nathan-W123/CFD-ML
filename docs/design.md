# Design and interface contract

This document is the contract between the modules of the physics expert.
Each module owner implements exactly the public API listed here; anything
else is private (leading underscore). If an implementation needs a contract
change, the change is made here first.

Status: synthetic pipeline only. Nothing in this repository has been
validated on real camera footage.

## Profiles

A profile is a complete simulation config (one schema for all);
`scripts/generate_dataset.py --config` selects it. A model is trained on
one profile and exported with its timing.

| | near field (default) | far field |
|---|---|---|
| config | `configs/simulation.yaml` | `configs/simulation_far.yaml` |
| domain | 320 m, 64 x 64 cells of 5 m | 3200 m, 64 x 64 cells of 50 m |
| observations | 12, 5 s apart (55 s) | 12, 60 s apart (11 min) |
| optical flow | lag pairs: a frame 1 s before each observation (`time.flow_lag_s: 1`) | consecutive observations, no lag frames (`time.flow_lag_s: null`) |
| frame-time tolerance | `time.max_offset_s: 0.5` | `time.max_offset_s: 10` |
| starting dt | 0.1 s | 0.5 s |
| wind (held out) | 0-7 m/s (7-9) | 0-12 m/s (12-15) |
| terrain (held out) | hill, valley, slope, 5-40 m (45-80 m hill) | also ridge, 50-600 m (700-1000 m hill) |
| opt-in regimes | off (weight 0) | ridge_cloud (fog), rain_shaft (neither) |
| footage it serves | surveillance video at >= 2 frames/s | wildfire camera networks at one frame per minute (HPWREN, AlertWildfire) |

The CFD code is scale-agnostic: physical constants (sizes, heights,
depths, periods, sponge rates, heat cap, texture period, inversion offset,
wind reference height) come from the config, with the near-field values as
code defaults; numerical widths (edge ramps, tanh fronts, layer softness)
are multiples of `dx`; schedule times defined relative to the observation
window (onsets, switch-offs, activity ramps) scale with
`frame_interval_s / 5 s` (`scenario_util.time_scale`). The extractor's
timing comes from the profile's `time` block (`pipeline.feature_timing`),
is stored with every row and exported with the model
(`preprocessing.feature_config`, `training_flow_modes`), so inference
measures what training measured.

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
solved with a D2Q9 lattice-Boltzmann method (weakly compressible, low
Mach): a regularized single-relaxation-time (BGK) collision — the raw
non-equilibrium stress projected on the second-order Hermite term (Latt &
Chopard) plus the first-order -F/2 term that Guo's half-force velocity
implies — with Guo forcing and a Smagorinsky local tau (strain estimated
from the half-force-corrected stress). Up to second order it equals
BGK-Guo (tested); it only filters the ghost moments, which plain BGK at the
low viscosities used here does not survive.

Buoyancy (per unit mass, +z up), relative to the horizontally uniform
ambient state at rest:

    b = g * [ (theta - theta_ref(z)) / theta0 + 0.608 (q_v - q_v_ref(z)) - (q_l - q_l_ref(z)) ]

Transported scalars: the anomalies theta_l' = theta_l - theta_l_ref(z) and
q_t' = q_t - q_t_ref(z) of the phase-change-conserved variables
(theta_l = theta - L_v/(c_p Exner) q_l, q_t = q_v + q_l), smoke C, tracer D,
in rain-shaft scenes a precipitation scalar P with its own fast settling
velocity (so background haze in D is not rained out with it; reported to the
renderer as part of the tracer), and four texture coordinates. They are
advected by the LBM velocity in
finite-volume advective form (flux divergence minus C div u, see
scalars.py) and diffused with an eddy diffusivity K (sampled independently
of the LBM viscosity):

    dC/dt + u . grad C = div(K grad C) + S

with background-advection sources -w d theta_l_ref/dz and -w d q_t_ref/dz
for the anomalies, so the ambient state is exactly steady.

Moisture: theta, q_v and q_l are diagnosed every step by saturation
adjustment from (theta_l, q_t); latent heat is included through theta_l.

Subgrid velocity: at every snapshot the run stores u_sgs = C_s dx |S| (m/s),
|S| the resolved strain-rate magnitude estimated exactly as the collision's
Smagorinsky model does (half-force-corrected non-equilibrium stress, local
tau). It is computed only at snapshot times; the renderer uses it to move
the texture with the unresolved eddies (section 6).

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
* Forcing: buoyancy, gust acceleration dU/dt applied everywhere and the
  ambient-maintaining force. (A stochastic turbulence forcing was tried and
  removed: in 2D it fed the inverse cascade and destabilised runs.) A
  source-free precursor spins up the flow before the scenario starts.
* Drifting features (fog patches, fog-bank fronts, advected smoke layers)
  are placed by their position at the first frame (`drift_start`; a rain
  shaft's cloud by its position at mid-window); a part that starts upwind of
  the domain enters through the inflow boundary only once its front reaches
  it. Precipitation has zero ghost values on every open side (a
  zero-gradient top would feed falling rain back in).
* The ambient wind may accelerate over the observation window (`ramp_factor`,
  drawn for every category alike, end speed inside the split's wind range);
  fog_dissipating can use such a context as its clearing mechanism.

Lattice units: `dx` (m), `dt` (s); `u_lat = u dt/dx`,
`nu_lat = nu dt/dx^2`, `tau = 0.5 + 3 nu_lat`, `a_lat = a dt^2/dx`.
The starting `dt` (`time.dt_s`: 0.1 s near field, 0.5 s far field) is
halved before the run when the expected peak speed would exceed
`target_lattice_speed`, and the run is restarted with a halved `dt` (up to
`max_dt_halvings`) if the measured lattice speed ever exceeds
`max_lattice_speed`. A run that still fails, or fails any check in
`src/cfd/validation.py`, is rejected and never rendered. On 5 m cells with
winds to 7 m/s and hot plumes most runs need it: in the engine v1 near-field
dataset 25 % of accepted runs used 0.1 s, 44 % 0.05 s and 31 % 0.025 s
(frame times are physical and identical for all).

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
    def strain_rate(self) -> np.ndarray                     # |S| = sqrt(2 S:S), 1/s, from the non-equilibrium stress; 0 in solid
    def sgs_velocity(self) -> np.ndarray                    # u_sgs = C_s dx |S|, m/s (recomputed on demand)
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
def regime_weight(regime, weights) -> float   # from the category's block (missing -> 0); no block: 1, opt-in 0
# simulation.py
def run_simulation(spec, cfg) -> SimulationResult
def has_lag_frames(flow_lag_s) -> bool        # False for None / 0 (no lag snapshots)
# results.py
@dataclass
class SimulationResult:
    sim_id, category, regime, labels, seed, params, dx, valid, validity, extra
    times: (T,)                        # s since the first observation
    smoke, lwc, tracer: (T, nz, nx)    # mg/m^3, g/m^3, mg/m^3 (tracer includes a rain shaft's precipitation)
    texcoords: (T, 4, nz, nx); tex_weights: (T, 2); solid: (nz, nx) bool
    sgs_u: (T, nz, nx) | None          # subgrid velocity scale C_s dx |S|, m/s (None in files older than v2)
    lag_times, smoke_lag, lwc_lag, tracer_lag, texcoords_lag, tex_weights_lag, sgs_u_lag
                                       # at times - flow_lag_s; ALL None for a profile without lag frames
    theta_anom, qv, ux, uz             # diagnostics only, never model inputs
    has_lag: bool                      # property: lag snapshots were stored
    def field_at(name, lag=False)      # a lag field of a run without lag frames raises MissingLagFields
    def save(path, diagnostics=True) / load(path)
class MissingLagFields(ValueError)
```

Context-first sampling (see the `scenarios.py` docstring): wind speed
(near 0-7 m/s, far 0-12 m/s, for every category), direction, gusts,
mixing, terrain (near P = 0.35 of hill / valley / slope, far P = 0.5 with
ridges too; the same shapes for every category), atmosphere, spin-up,
background haze and the camera nuisances are class-independent; a
compatible regime is then chosen with the profile's
`scenarios.regime_weights`, renormalised over the compatible regimes. A
regime missing from a category's block, or weighted 0, is never sampled;
the opt-in regimes (`constants.OPT_IN_REGIMES`: ridge_cloud, rain_shaft)
need a positive weight, which only the far profile gives them. Regimes:

* smoke: hot_plume, smoldering, crosswind, pulsing, advected_smoke;
* fog: radiation_fog, advected_fog_bank, terrain_fog (valley), fog_patch,
  fog_dissipating, steam_fog (flat ground), and ridge_cloud: an orographic
  cap cloud. The humid inflow is set so that air lifted by `lcl_frac` of
  the ridge height saturates, and every level stays sub-saturated at rest,
  so no liquid is prescribed: the cloud forms during the spin-up where the
  ridge lifts the air, stays over the crest while the air moves through it,
  evaporates in the lee descent and may trail downstream aloft. Needs a hill
  or ridge of at least `min_terrain_height_m` and wind;
* neither: haze_drift, dust_burst, illumination_change, camera_artifact,
  distant_cloud, and rain_shaft: precipitation generated in a band at cloud
  base under a cloud drifting with a fraction of the mid-window wind there
  (capped so it stays in view for `label_min_frames` frames), falling at
  3-8 m/s, intermittent, optionally evaporating on the way down (virga). It
  is parameterised by its in-shaft concentration (`shaft_mg_m3`: optical
  depth 0.4-6 over 1 km, independent of dx); its onset is in literal
  seconds per profile. The three look-alikes without aerosol need their
  class-independent nuisance; their weights are set so that each is at
  least ~12 % of neither scenes in both profiles (2-5 % in v1).

`params` keys used downstream:

* `sim_id`, `category`, `split`, `index`, `attempt`, `seed`, `labels`,
  `regime`, `smoke_regime`, `fog_regime`, `distractor_regime`, `holdout_tags`
* `wind` (`speed_ms`, `sign`, `gust_fraction`, ...), `terrain` (`kind`, `height_m`),
  `mixing`, `atmosphere`, `timing`
* `render_hints`: `{"shadow": bool, "strong_artifact": bool, "fp_boost": bool,
  "distant_cloud": bool}` — class-independent nuisance flags the renderer
  MUST honour (it samples their strengths, not their presence)
* `background_haze`: dict or None (tracer as faint background haze);
  `distractor`: for neither (`haze_drift` / `dust_burst` put aerosol in the
  tracer field, `rain_shaft` in the precipitation scalar;
  `illumination_change` / `camera_artifact` / `distant_cloud` have no
  aerosol and rely on the corresponding nuisance)

Held-out recipes (`test_ood` only, every category, alternating by index):
`fast_wind` (near 7-9 m/s, far 12-15 m/s) and `elevated_source` (a hill of
45-80 m near, 700-1000 m far; smoke sources on the hilltop). Validity
(`validation.py`, observation snapshots only) includes label support: the
labelled aerosol(s) visible in view, unlabelled ones absent.

A profile without lag frames (`time.flow_lag_s` null or 0) stores no lag
snapshots; everything else in the run is the same.

## 6. `src/data/render.py` and `src/data/masks.py`  (owner: render)

```python
@dataclass
class RenderedSequence:
    frames: np.ndarray          # (T, H, W) uint8 grayscale at the observation times
    lag_frames: np.ndarray | None      # (T, H, W) uint8 at the lag times (t_k - flow_lag_s); None without lag
    masks: np.ndarray           # (T, H, W) bool candidate masks at the observation times
    timestamps: np.ndarray      # (T,) float64 s (= sim.times)
    lag_timestamps: np.ndarray | None  # (T,) float64 s (= sim.lag_times); None without lag
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
tileable noise pattern (several octaves, wavelengths `render.texture.wavelengths_m`:
3-40 m near, 30-400 m far; one class-independent random gain per sequence,
identical for all components) is
sampled at the upsampled texture coordinates `sim.texcoords` (pair a and pair
b blended with `sim.tex_weights`), so the texture moves exactly with the
simulated air. Without this step a quasi-steady concentration field produces
no optical flow at all (the reviewers measured flow/fluid ratios of
0.001-0.02). Then `alpha = 1 - exp(-sum_c tau_c)` and
`I = (1 - alpha) B + alpha A`, with ONE aerosol-brightness distribution for
all aerosols. Lag frames are rendered with the same parameters from the
`_lag` fields; a simulation without lag snapshots gets the observation
frames only (`lag_frames` None), and masks, visibility and exposure come
from the observation frames in both cases.

Subgrid texture turbulence: the texture also moves with the unresolved
eddies. Where `sim.sgs_u` exists, the texture coordinates are displaced by
a smooth, zero-mean, divergence-free random field D (12 random Fourier
modes of 3-8 cells, so it scales with dx; rms strain 0.3, so the warp never
folds) whose local velocity scale is `texture.sgs_gain * u_sgs`
(`render.texture.sgs_gain`, [0.5, 1.5] in both profiles) and whose local
correlation time is about eddy size / u_sgs. The field is built from
velocity octaves below the sequence's peak u_sgs, each with its own eddy
clock; every cell blends the two octaves bracketing its peak u_sgs over the
sequence, a band that is fixed for the whole sequence. D is integrated from
velocity-scaled increments in time order (observation and lag frames
interleaved), so it does not jump where u_sgs changes between frames. It is
class-independent (it acts wherever u_sgs is, clear air included) and
deterministic from `texture.sgs_seed`; `sgs_gain` 0 (robustness suite
`sgs_off`) or a simulation without `sgs_u` disables it. A lag frame 1 s
before its observation is displaced by about `sgs_gain * u_sgs * 1 s`
relative to it; frames a minute apart see decorrelated sub-grid texture.

Stretch fade (far field): over a 480 s texture period, km-scale shear
stretches the advected texture coordinates into streaks a real plume or
cloud does not show (P90 stretch about 2, up to about 5). With
`render.texture.max_stretch` set (1.5 in the far profile, unset in the near
profile), each texture pair's modulation is faded towards 1 by
`min(1, max_stretch / s)`, where s is the largest singular value of the
Jacobian of that pair's texture coordinates on the grid (coordinates and
grid both in metres, so unstretched air has s = 1). The weight is warped to the image with the
texture, so texture fades only where it is sheared, the opacity field
itself is unchanged, and it applies to every class alike.

Far-profile imaging ranges (`ground_brightness`, `global_haze`, `contrast`,
`brightness_offset`, `exposure_flicker`, `camera_shake_px`,
`camera_rotation_deg`) were set from `scripts/image_stats.py` on the FIgLib
train split only (114 sequences); the held-out FIgLib cameras are compared
afterwards in `reports/far/image_stats_figlib.md`. The near profile was not
tuned to FIRESENSE, its only real footage, because that footage is also its
zero-shot check (`reports/image_stats_firesense.md` flags its contrast and
JPEG blocking as slightly beyond the synthetic range).

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

Camera shake: shake (x, y), rotation and zoom jitter are a stationary
Ornstein-Uhlenbeck (AR(1)) process sampled at the actual frame times
(observation and lag frames interleaved): stationary std `camera_shake_px` /
`camera_rotation_deg` / `zoom_jitter`, correlation time
`render.camera_shake_tau_s` (near 0.5-4 s, so a lag frame 1 s earlier
shakes almost with its observation; far 1-30 s, so frames a minute apart are
practically independent), clipped at 3 std; tau 0 is the v1 model
(independent per frame). Render params are version 2: the v2 keys
(`camera_shake_tau_s`, `texture.sgs_gain`, `texture.sgs_seed`) are drawn
last, so every earlier value is what v1 drew, and stored v1 params still
render as before.

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

Robustness suites (`robustness:` in each profile; re-renders of the test_iid
simulations, merged into the render config): `nominal`, `occlusion_heavy`,
`shake_heavy` (near field: also a short shake correlation time, 0.2-1 s, so
the 1 s lag pairs get nearly independent shake), `render_shift`, `zoom_out`,
`detector_masks`, `bbox_masks` and `sgs_off` (no subgrid texture motion).

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
    require_stabilization: bool = True  # no camera-motion estimate -> flow gate closed
    min_mask_frac: float = 0.002    # of the analysis frame
    flow_method: str = "dis"        # "dis" (default) | "farneback"
    flow_erode_px: int = 2          # flow statistics on the mask eroded by this (working px)
    fb_max_px: float = 1.0          # forward-backward consistency (median, working px)
    anchor_band_px: int = 6
    min_roughness_px: int = 100
    expected_interval_s: float = 5.0        # far-field profile: 60
    flow_lag_s: float | None = 1.0          # None or 0: consecutive-observation flow (far field)
    max_offset_s: float = 0.5               # per frame; observation intervals get 2x (far field: 10)
    flow_mode: str                          # property: "lag" | "consecutive"

def extract_features(frames, timestamps, masks, lag_frames=None, lag_timestamps=None,
                     cfg: FeatureConfig | None = None) -> np.ndarray      # (T, 14) float32
def extract_features_with_meta(...same...) -> tuple[np.ndarray, dict]
    # meta: roi box, stabilisation per frame (used / skipped + reason, quality),
    # flow_mode ("lag" | "consecutive"), lag_frames_ignored, per-frame flow status /
    # dt / fb error / start ("zero" | "centroid"), valid frame counts, and
    # timing: intervals_s, flow_lags_s, expected_interval_s, expected_flow_lag_s
    # (0 for a consecutive configuration), expected_flow_mode, max_offset_s,
    # interval_compatible, lag_compatible (None when not applicable),
    # flow_mode_compatible, compatible, reasons
def stabilize_sequence(frames, masks, lag_frames=None) -> tuple[...]
def select_window(timestamps, n=12, interval_s=5.0, start=None, max_offset_s=0.5,
                  flow_lag_s=1.0) -> dict
    # {"obs_idx": (n,), "lag_idx": (n,) | None, "offsets": ..., "native_dt": ...,
    # "flow_mode": "lag" | "consecutive"}; raises IncompatibleFrameTiming when a
    # selected frame is > max_offset_s from its target, indices repeat, the clip is
    # too short, or (lag mode only) the native frame spacing exceeds flow_lag_s / 2
    # (one-minute imagery cannot give 1 s pairs and is refused, not stretched).
    # flow_lag_s None / 0: no lag frames, first target at the first frame, any frame
    # rate that puts a frame within max_offset_s of every target (one-minute
    # HPWREN / FIgLib imagery included)
def load_frames(path, times=None, timestamps=None) -> tuple[np.ndarray, np.ndarray]
    # video file (timestamps from the container) or image folder (timestamps from a
    # sidecar CSV or a filename pattern; raises if there are none); decodes only `times`
```

Pipeline (identical for synthetic and real input):

1. Convert every input (uint8, float in [0, 1], BGR) to one uint8 grayscale
   representation in a single private function.
2. Stabilise: a bounded similarity transform of every observation and lag
   frame onto observation frame 0, estimated ONLY from pixels outside the
   dilated union of candidate masks (>= 3 % of the image textured background
   and >= 12 corners; inlier and residual checks guard each estimate),
   warped with BORDER_REPLICATE; masks follow; a frame whose estimate fails
   or exceeds the bound gets `flow_valid = 0`. Skipped entirely (and
   recorded) when the background is insufficient, unreliable or drifting,
   so a frame-filling fog bank is never "stabilised away"; a skipped clip
   then has its flow gate closed (`stabilization_unavailable`) unless
   `FeatureConfig.require_stabilization = False` (rigid static mounts
   only), because camera shake would otherwise be read as aerosol motion
   (with the old 20 % rule about 45 % of clips kept shake in their flow).
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
5. Flow: OpenCV DIS (PRESET_MEDIUM), in the configuration's flow mode.
   `"lag"` (near field): from the lag frame to the observation frame —
   displacements stay small and texture stays correlated; without lag
   frames a lag configuration falls back to consecutive flow from a zero
   start and flags it (`timing.reasons` "consecutive_flow"; the inference
   API refuses it unless explicitly allowed). `"consecutive"` (far field,
   `flow_lag_s` None / 0): flow between consecutive observations is the
   first-class mode; lag frames, if passed, are ignored (recorded), and
   because a minute's displacement can exceed what DIS tracks from rest,
   each pair is also run from a uniform start at the mask-centroid
   displacement (when it is >= 3 working px); among the starts that pass
   the forward-backward check the zero start is kept unless the centroid
   start reconstructs the frame with a photometric residual >= 5 % lower
   (`meta["flow"]["start"]`). The centroid start exists only in the
   consecutive configuration; lag configurations are unchanged. DIS
   recovers smooth-field displacements that Farneback underestimates 2-5x.
   `flow_valid[t] = 1` only if t has a flow pair, both masks are valid,
   stabilisation succeeded for the pair, and the median forward-backward
   error over the eroded region is <= `fb_max_px`.
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
   First observation: `mask_pair` features 0 (no previous observation). No
   extra flag marks this: the 12 x 14 layout has only `mask_valid` and
   `flow_valid`, both of which describe t itself. The missingness is
   implicit, because timestep 0 is always the GRU's first step and, for
   t >= 1, the `mask_pair` gate is mask_valid[t] AND mask_valid[t-1], both
   visible to the network. In the lag mode flow at t = 0 comes from its own
   1 s lag frame, so flow_valid[0] can legitimately be 1; in the
   consecutive mode t = 0 has no pair, so flow_valid[0] = 0 and at most 11
   steps carry flow.
7. Timing record (`meta["timing"]`): an observation interval is compatible
   within 2 x `max_offset_s` of `expected_interval_s` (near 5 +- 1 s, far
   60 +- 20 s), a lag within `max_offset_s` of `flow_lag_s`, and the flow
   mode must be the configuration's.

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
    X: (N, 12, 14) float32 raw features
    y: (N, 2) float32 in [0, 1]; NaN = unknown (e.g. real footage whose fog is not annotated)
    seq_ids, sim_ids, groups, splits, categories, regimes, tags, mask_sources: (N,) str arrays
    meta: list[dict]       # per sequence: params subset, render params, feature meta
    source: str            # "synthetic" | "real"
    known: (N, 2) bool     # property: target is not NaN
    save(path) / load(path) / subset(mask) / split(name)   # NaN round-trips
PARTIAL_CATEGORIES         # category of a row with a NaN target, e.g. "smoke_fog_unknown"
def label_category(smoke, fog) -> str      # one of CATEGORIES, or of PARTIAL_CATEGORIES
def parse_label(value) -> float            # manifest cell: 0 / 1, or NaN for "", "nan", "unknown", "?", "-"
def save_sequence(path, rendered, sim, seq_id, extra=None) -> Path   # no lag arrays without lag frames
def load_sequence(path) -> dict            # lag_frames / lag_timestamps None when none were stored
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
  the val_select-best seed and say so. The export records what inference
  must reproduce: `preprocessing.feature_config` (the extractor settings
  of the train rows, timing included) and `training_flow_modes` (counts of
  "lag" / "consecutive" rows).
* Unknown labels (NaN targets, e.g. real footage with smoke annotated and
  fog not): the loss is masked (per label over its known entries, then
  averaged over the labels that have any; identical to plain BCE when
  everything is known), early stopping uses the masked val_select loss,
  and every calibration fit uses only the rows where its label is known.
* Calibration: per-label temperature on val_calib, bounded to [0.5, 5];
  if val_calib is separable, T = 1 and a flag is stored; any other optimum
  at a bound uses the bound (flagged `clipped`). A label with too few
  known val_calib rows (`calibration.min_known_per_label`, default 10, or
  one class) keeps the pretrained model's temperature, thresholds and
  margins when fine-tuning (D); otherwise it is fitted on what there is
  (`fitted_on_insufficient_labels`). A label with no known row and nothing
  to keep (`default_unfitted`: T = 1, threshold logit 0) is listed in
  `calibration.untrained_labels` and gets an abstain margin larger than any
  logit, so the model always abstains on it. `calibration.label_support`
  says, per label, how it was calibrated.
* Thresholds, stored with `fit_source` (`"synthetic_val_calib"`, or
  `"real_val_calib"` after Stage B) and the calibration-set prevalence:
  `balanced` (max Youden's J per label; max F1 degenerated to "always
  positive" on weak scores) and `low_false_alarm` (smoke threshold giving
  <= 5 % false-smoke rate on fog_only val_calib sequences). Full ROC / PR
  curves are reported.
* Abstention (`decision = -1`): logit within a margin of the threshold
  (margin in logit space, floor 0.25, widened on val_calib until the
  confident error <= 5 % with abstention <= 30 %), OR input-validity
  failure: fewer than 3 valid-mask frames, fewer than 2 valid-flow frames,
  more than 20 % of open-gate normalised entries clipped, an incompatible
  frame interval (judged against the model's exported timing: interval
  within 2 x `max_offset_s`, lag within `max_offset_s`), or a flow mode the
  model was not trained on: consecutive flow for a lag-trained model
  (`consecutive_flow`, unless `allow_consecutive`), lag-pair flow for a
  model trained only on consecutive flow (`lag_flow`). `predict()` returns
  `calibration_source`.
* Metrics (numpy only): per-label precision / recall / F1 / AUROC / AUPRC,
  4-class confusion from the two decisions, smoke/fog confusion,
  false-smoke rate on fog_only, Brier, ECE with equal-mass bins, abstain
  rate and selective accuracy; 95 % CIs from a grouped bootstrap over
  sim_id (camera id for real data). Each uses only the rows where its
  label is known (the 4-class confusion and fog_only need both); a label
  with no known row gets null metrics and status "no_labels", never 0.
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
  features per feature group to localise information loss),
  `scripts/image_stats.py` (real vs synthetic frame statistics: noise,
  contrast, brightness, sharpness, JPEG blockiness, flicker, camera motion;
  says which `render:` ranges to widen), and `scripts/summarize_results.py`
  (evaluation.json -> reports/results.md).

`evaluation.json` top level: `{"synthetic": {...}, "real": {...},
"ablations": {"A": ..., "B": null, "C": ..., "D": null, "reasons": {...}},
"synthetic_proxy": ..., "shortcut_audit": ..., "robustness": ..., "loro":
..., "data": {counts, acceptance}}`. `real.status` is `"pending"` (with
the reason) until Stage B runs, then `"complete"` (every label evaluated),
`"partial"` (some label has no known target on the real test split, e.g.
smoke-only supervision) or `"no_labels"`; `real.label_status` gives, per
label, `"evaluated"` or `"no_labels"` with the reason, and the counts of
known targets per split. With a complete or partial result, ablations B and
D are filled and carry `real_label_status`.

Real data (Stage B): group = camera/location id; camera-disjoint
train/val/test; B (scratch) uses `training.lr`, D (fine-tune) uses
`finetune.lr`; identical epochs, patience, seeds, splits and normaliser
policy; report counts of real sequences, videos and cameras (and known
labels per split). Smoke-only labels are enough: with fog unknown, D keeps
the pretrained fog calibration and B always abstains on fog. Real
manifests are prepared by `scripts/prepare_firesense.py` (near field;
testpos03 / 04 / 05 excluded with the reason: decoder crash, edited
footage, container clock ~27 % slow), `scripts/prepare_mivia.py` (near
field; labels in `configs/real/mivia_smoke_labels.csv`) and
`scripts/prepare_figlib.py` (far field: HPWREN FIgLib, one frame per
minute, imagery courtesy of HPWREN, credit https://www.hpwren.ucsd.edu/;
12-frame windows entirely before the plume's first visible frame are
smoke = 0, windows starting >= 60 s after it smoke = 1, straddling windows
are never written; fog is unknown). `scripts/extract_features.py` takes
the extractor settings from `--artifacts` (the exported model), else
`--profile` (a profile's `time` block); timing flags that contradict the
model need `--force-timing`. `scripts/evaluate_real_zero_shot.py` runs the
exported model unchanged on real windows (not Stage B).

Real fog at the far-field timing (`scripts/prepare_hpwren_fog.py`): no
public footage at one frame per minute carries fog labels, so the set is
built from the HPWREN camera archive, which keeps every frame for about three
months.

1. **Proposing windows.** Airport METARs (IEM) propose morning windows. A
   fog morning is one where an airport inside the camera's view reports
   FG, visibility of 1 mile or less, BR with visibility of 3 miles or less,
   or a ceiling at or below 1000 ft between 03:00 and 10:00 local. Clear
   mornings are matched to the same cameras and time slots, and only slots
   with the sun at least 5° up are used.
2. **Labelling.** The labels come from looking at the frames, not from the
   METARs:
   * contact sheets with the camera banner blanked and opaque ids;
   * labelled by the rules in `configs/real/hpwren_fog_labeling_guide.md`;
   * two independent votes per sheet, and a third where the first two
     disagree or one is unsure;
   * consolidated by majority, with Cohen's kappa reported.
3. **Splits.** Windows take the camera-mount split of `prepare_figlib.py`,
   so FIgLib and HPWREN windows of one camera share a split, and
   `scripts/merge_features.py` (`FeatureSet.concat` plus the leakage check)
   combines the two sets.

Two tests use these windows:

* **Full chain.** Masks come from the fallback detector, as the model would
  run today.
* **Region level.** Masks come from per-frame boxes drawn around the fog,
  or, in fog-free windows, around the most fog- or smoke-like region
  (`scripts/region_boxes.py`). The fallback detector finds change against
  the window's median frame, so it never proposes a static fog sea, and the
  model was trained with masks around the aerosol (box masks among them).

All labels and boxes are from AI vision labellers, not people; their
agreement is reported, and a human spot check is recommended before the
numbers are relied on.

```python
# src/inference/predict.py — imports only torch, numpy, yaml, cv2 and src.models / src.data
class PhysicsExpert:
    @classmethod
    def load(cls, artifact_dir) -> "PhysicsExpert"
    def features_from_frames(self, frames, timestamps, masks=None, lag_frames=None,
                             lag_timestamps=None, feature_cfg=None) -> tuple[np.ndarray, dict]
        # extractor rebuilt from the exported feature_config; masks=None runs the fallback detector
    def encode(self, features) -> np.ndarray     # raw (B, 12, 14) or (12, 14) -> (B, 16)
    def predict(self, features, operating_point="low_false_alarm", meta=None,
                allow_consecutive=False) -> dict
        # logits, probs, decisions (1 / 0 / -1 abstain), abstain_reasons, calibration_source
    training_flow_modes: tuple[str, ...]          # ("lag",), ("consecutive",) or both
    consecutive_native: bool                      # trained only on consecutive flow (far field)
```

`predict` defaults to `low_false_alarm`: in the engine v1 and v2
evaluations the balanced point called smoke on 70-92 % of the fog_only
clips it decided on (near and far field), so `balanced` is for uses where a
missed fire costs more than a fog false alarm. A model trained only on consecutive flow takes consecutive input as
native (lag frames passed to `features_from_frames` are ignored).

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
  two splits; evaluation cameras are unseen in training. FIgLib splits are
  assigned per camera mount (crc32 of the site-direction, e.g. `stgo-s`),
  so the colour and mono cameras of one mount, and every fire seen from it,
  share a split; groups stay camera ids. Real background images used for
  rendering (if any) must come from cameras disjoint from real
  validation/test cameras.

## 13. Real-footage detector (`src/detector`, `scripts/embed_windows.py`, `scripts/train_detector.py`)

The physics expert alone does not recognise real fog (section "Real
footage" of the README), so the far-field detector that is evaluated on real
footage puts an appearance (vision) expert next to it. One window is 12
frames one minute apart, selected exactly as `extract_features.py` selects
them.

* **Vision.** A frozen CLIP ViT-B/32 image encoder (`openai/clip-vit-base-
  patch32`), CPU only. Coarse crops: frames 0, 6 and 11, each as the whole
  frame and a 2 x 3 tile grid (`clip.npz`, (F, 7, 512)). Fine crops: frames 0
  and 11 as a 4 x 6 tile grid (`clip_fine.npz`, (2, 24, 512)), so a tile of a
  1024-px frame is about the 224-px CLIP input and a plume a few dozen pixels
  wide is still a few patches. Text prompts (8 per head) give similarity
  features in the same space. The camera banner is blanked.
* **Physics.** The motion expert's 12 x 14 window features, pooled over time
  (feature group `motion`); used when model selection picks it.
* **Weather.** Not an input. The fog windows were found through METAR fog
  reports and the smoke windows through CAL FIRE incident reports, so a
  weather input would partly learn how the windows were selected. A weather
  prior has to be validated on windows not selected by weather.
* **Fusion.** Each head (smoke, fog) is a standardiser plus an L2 logistic
  regression on concatenated feature groups (`full`, `tiles`, `sims`,
  `fine`, `motion`; `src/detector/features.py`). Smoke also has a tile head:
  the same kind of model, fitted separately, applied to every fine tile (`tile_instances`:
  last-frame embedding, its change since frame 0, prompt similarities and
  their change), every tile of a smoke window labelled smoke at fit time,
  and the window scored by its best tile. A two-weight logistic stack
  combines the window head's and the tile head's logits. All weights are
  plain arrays (`detector.json` + `detector_weights.npz`); inference needs
  numpy plus the CLIP encoder.
* **Selection and thresholds.** Dev cameras only (every split but
  `test_iid`). Feature groups and C by 5-fold cross-validation grouped by
  camera mount (out-of-fold AUROC); the tile head's C the same way; the
  stack on the out-of-fold logits, scored by a second grouped
  cross-validation. Thresholds from out-of-fold probabilities: smoke
  `deploy` lets at most 2.5 % of the dev fog windows (fog 1, smoke 0) raise
  a smoke alarm (half the 5 % goal, as a margin for new cameras); smoke
  `strict` lets at most 5 % of all dev smoke-free windows raise one (the
  deploy point budgets fog only, and clear scenes minutes before a fire
  alarm far more often); `balanced` is the maximum-Youden point. Model
  selection maximises the within-set AUROC (below). The `test_iid` cameras
  are only scored (`--final`), never used to fit or choose anything.
* **Reporting.** Pooled smoke AUROC mixes sets: a no-smoke window from a set
  without smoke (the fog sets) is an easy negative, so the report also gives
  the AUROC over smoke / no-smoke pairs from the same set ("within sets")
  and each set's own AUROC. Intervals resample camera mounts.
* **Labels.** FIgLib smoke labels come from the library (ignition time);
  the HPWREN 2026 fog and smoke labels come from blind AI vision labelling
  of contact sheets (two votes per sheet, a third where either was unsure or
  they disagree), not from people.
