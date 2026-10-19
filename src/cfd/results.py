"""The simulation output record handed from the CFD stage to the renderer.

Where this fits: `simulation.run_simulation` produces a `SimulationResult`;
`src.data.render.render_sequence` consumes it; `dataset` stores it so test
simulations can be re-rendered for robustness suites without re-simulating.

Field layout: arrays are (T, nz, nx) with row k = 0 at the GROUND (z grows
with k) and column i = 0 at the left (x grows with i). Images use the
opposite row order (row 0 at the top); the renderer flips.

Every observation time t_k has a partner "lag" snapshot at t_k - flow_lag_s
(1 s in the near-field profile): optical flow is measured over that short
lag, while the 12 observations are 5 s apart (see docs/design.md section 7).
Lag fields carry the `_lag` suffix. A profile without lag frames
(`time.flow_lag_s` null or 0, e.g. the far-field one-frame-per-minute
profile, whose flow is measured between consecutive observations) stores
none: every `_lag` field and `lag_times` are None, `has_lag` is False, and
`field_at(name, lag=True)` raises.

Subgrid velocity: `sgs_u` (and `sgs_u_lag`) is the Smagorinsky turbulent
velocity scale u_sgs = C_s dx |S| in m/s at each snapshot (|S| the resolved
strain-rate magnitude, see lbm.LBMSolver.sgs_velocity); the renderer uses it
to displace the advected texture by unresolved eddies. None in files written
before it existed.

Texture coordinates: the CFD run advects two pairs of passive coordinate
fields (X_a, Z_a) and (X_b, Z_b) with the flow (no diffusion), each reset to
the identity on a staggered cycle. The renderer evaluates a high-resolution
noise pattern at these coordinates, so sub-grid texture moves with the
simulated air, as turbulent texture does in real smoke and fog; the
weights blend the two pairs so neither is shown when it is reset.

The velocity, temperature and vapour fields are diagnostics. They must never
reach the neural network: the model sees only motion features extracted from
rendered frames, exactly as it would from a real camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.utils.io import PathLike, load_npz, save_npz

RENDER_FIELDS = ("smoke", "lwc", "tracer", "texcoords")
LAG_FIELDS = ("smoke_lag", "lwc_lag", "tracer_lag", "texcoords_lag")
# Every array that exists only when the simulation stored lag snapshots.
ALL_LAG_FIELDS = ("lag_times",) + LAG_FIELDS + ("tex_weights_lag", "sgs_u_lag")
DIAGNOSTIC_FIELDS = ("theta_anom", "qv", "ux", "uz")
SGS_FIELDS = ("sgs_u", "sgs_u_lag")


class MissingLagFields(ValueError):
    """A lag snapshot was requested from a simulation stored without lag frames."""


@dataclass
class SimulationResult:
    sim_id: str
    category: str                 # one of src.constants.CATEGORIES
    regime: str                   # e.g. "hot_plume", "radiation_fog", "hot_plume+terrain_fog" (both)
    labels: Tuple[int, int]       # (smoke, fog)
    seed: int
    params: Dict[str, Any]        # every sampled scenario parameter (JSON-able)
    dx: float                     # grid spacing, m (square cells)
    times: np.ndarray             # (T,) s since the first output frame (0, 5, ..., 55)
    smoke: np.ndarray             # (T, nz, nx) smoke concentration, mg/m^3
    lwc: np.ndarray               # (T, nz, nx) fog liquid water content, g/m^3
    tracer: np.ndarray            # (T, nz, nx) distractor aerosol (haze/dust), mg/m^3
    solid: np.ndarray             # (nz, nx) bool terrain mask
    valid: bool
    validity: Dict[str, Any]      # numerical checks, see src.cfd.validation
    texcoords: np.ndarray         # (T, 4, nz, nx) advected (X_a, Z_a, X_b, Z_b), m
    tex_weights: np.ndarray       # (T, 2) blend weights of pair a and pair b (sum 1)
    # Lag snapshots: all None when the profile stores no lag frames (flow_lag_s null / 0).
    lag_times: Optional[np.ndarray] = None        # (T,) = times - flow_lag_s
    smoke_lag: Optional[np.ndarray] = None        # (T, nz, nx) at lag_times
    lwc_lag: Optional[np.ndarray] = None
    tracer_lag: Optional[np.ndarray] = None
    texcoords_lag: Optional[np.ndarray] = None    # (T, 4, nz, nx)
    tex_weights_lag: Optional[np.ndarray] = None  # (T, 2)
    theta_anom: Optional[np.ndarray] = None  # (T, nz, nx) theta - theta_ref(z), K
    qv: Optional[np.ndarray] = None          # (T, nz, nx) vapour mixing ratio, kg/kg
    ux: Optional[np.ndarray] = None          # (T, nz, nx) m/s, +right
    uz: Optional[np.ndarray] = None          # (T, nz, nx) m/s, +up
    sgs_u: Optional[np.ndarray] = None       # (T, nz, nx) subgrid velocity C_s dx |S|, m/s (float32)
    sgs_u_lag: Optional[np.ndarray] = None   # (T, nz, nx) at lag_times, or None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.lag_times is None:
            present = [name for name in ALL_LAG_FIELDS if getattr(self, name) is not None]
            if present:
                raise ValueError(f"simulation {self.sim_id}: lag_times is None but {present} are given")
        else:
            missing = [name for name in ("smoke_lag", "lwc_lag", "tracer_lag") if getattr(self, name) is None]
            if missing:
                raise ValueError(f"simulation {self.sim_id}: lag_times given but {missing} are None")

    @property
    def n_frames(self) -> int:
        return int(self.times.shape[0])

    @property
    def has_lag(self) -> bool:
        """True when lag snapshots (t_k - flow_lag_s) were stored."""
        return self.lag_times is not None

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.solid.shape)  # (nz, nx)

    def save(self, path: PathLike, diagnostics: bool = True) -> Path:
        """Compressed .npz; diagnostics and sgs_u are stored as float16 to save space.

        Absent lag fields (no lag frames) are simply not written; `load` turns
        them back into None.
        """
        meta = {
            "sim_id": self.sim_id,
            "category": self.category,
            "regime": self.regime,
            "labels": list(self.labels),
            "seed": int(self.seed),
            "params": self.params,
            "dx": float(self.dx),
            "valid": bool(self.valid),
            "validity": self.validity,
            "extra": self.extra,
        }
        arrays: Dict[str, Any] = {
            "times": self.times.astype(np.float64),
            "solid": self.solid.astype(bool),
        }
        if self.lag_times is not None:
            arrays["lag_times"] = self.lag_times.astype(np.float64)
        for name in ("tex_weights", "tex_weights_lag") + RENDER_FIELDS + LAG_FIELDS:
            value = getattr(self, name)
            if value is not None:
                arrays[name] = value.astype(np.float32)
        for name in SGS_FIELDS:
            value = getattr(self, name)
            if value is not None:
                arrays[name] = value.astype(np.float16)
        if diagnostics:
            for name in DIAGNOSTIC_FIELDS:
                value = getattr(self, name)
                if value is not None:
                    arrays[name] = value.astype(np.float16)
        return save_npz(path, meta=meta, **arrays)

    @classmethod
    def load(cls, path: PathLike) -> "SimulationResult":
        data = load_npz(path)
        meta = data["meta"]
        kwargs: Dict[str, Any] = {}
        for name in DIAGNOSTIC_FIELDS + SGS_FIELDS:
            if name in data:
                kwargs[name] = data[name].astype(np.float32)
        return cls(
            sim_id=meta["sim_id"],
            category=meta["category"],
            regime=meta["regime"],
            labels=tuple(int(v) for v in meta["labels"]),
            seed=int(meta["seed"]),
            params=meta["params"],
            dx=float(meta["dx"]),
            times=data["times"],
            smoke=data["smoke"],
            lwc=data["lwc"],
            tracer=data["tracer"],
            solid=data["solid"].astype(bool),
            valid=bool(meta["valid"]),
            validity=meta["validity"],
            texcoords=data.get("texcoords"),
            tex_weights=data.get("tex_weights"),
            lag_times=data.get("lag_times"),
            smoke_lag=data.get("smoke_lag"),
            lwc_lag=data.get("lwc_lag"),
            tracer_lag=data.get("tracer_lag"),
            texcoords_lag=data.get("texcoords_lag"),
            tex_weights_lag=data.get("tex_weights_lag"),
            extra=meta.get("extra", {}),
            **kwargs,
        )

    def field_at(self, name: str, lag: bool = False) -> Optional[np.ndarray]:
        """`name` in {"smoke", "lwc", "tracer", "texcoords", "tex_weights", "sgs_u"} at the
        observation or lag times. Asking for a lag field of a simulation stored without
        lag frames raises MissingLagFields (sgs_u may be None in old files)."""
        if lag and not self.has_lag:
            raise MissingLagFields(
                f"simulation {self.sim_id} has no lag snapshots (stored with flow_lag_s null / 0): "
                f"{name}_lag is unavailable; measure flow between consecutive observations instead")
        return getattr(self, f"{name}_lag" if lag else name)
