"""The simulation output record handed from the CFD stage to the renderer.

Where this fits: `simulation.run_simulation` produces a `SimulationResult`;
`src.data.render.render_sequence` consumes it; `dataset` stores it so test
simulations can be re-rendered for robustness suites without re-simulating.

Field layout: arrays are (T, nz, nx) with row k = 0 at the GROUND (z grows
with k) and column i = 0 at the left (x grows with i). Images use the
opposite row order (row 0 at the top); the renderer flips.

Every observation time t_k has a partner "lag" snapshot at t_k - flow_lag_s
(1 s by default): optical flow is measured over that short lag, while the
12 observations are 5 s apart (see docs/design.md section 7). Lag fields
carry the `_lag` suffix.

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
DIAGNOSTIC_FIELDS = ("theta_anom", "qv", "ux", "uz")


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
    lag_times: np.ndarray         # (T,) = times - flow_lag_s
    smoke_lag: np.ndarray         # (T, nz, nx) at lag_times
    lwc_lag: np.ndarray
    tracer_lag: np.ndarray
    texcoords_lag: np.ndarray     # (T, 4, nz, nx)
    tex_weights_lag: np.ndarray   # (T, 2)
    theta_anom: Optional[np.ndarray] = None  # (T, nz, nx) theta - theta_ref(z), K
    qv: Optional[np.ndarray] = None          # (T, nz, nx) vapour mixing ratio, kg/kg
    ux: Optional[np.ndarray] = None          # (T, nz, nx) m/s, +right
    uz: Optional[np.ndarray] = None          # (T, nz, nx) m/s, +up
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.times.shape[0])

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.solid.shape)  # (nz, nx)

    def save(self, path: PathLike, diagnostics: bool = True) -> Path:
        """Compressed .npz; diagnostics are stored as float16 to save space."""
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
            "lag_times": self.lag_times.astype(np.float64),
            "solid": self.solid.astype(bool),
            "tex_weights": self.tex_weights.astype(np.float32),
            "tex_weights_lag": self.tex_weights_lag.astype(np.float32),
        }
        for name in RENDER_FIELDS + LAG_FIELDS:
            arrays[name] = getattr(self, name).astype(np.float32)
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
        for name in DIAGNOSTIC_FIELDS:
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
            texcoords=data["texcoords"],
            tex_weights=data["tex_weights"],
            lag_times=data["lag_times"],
            smoke_lag=data["smoke_lag"],
            lwc_lag=data["lwc_lag"],
            tracer_lag=data["tracer_lag"],
            texcoords_lag=data["texcoords_lag"],
            tex_weights_lag=data["tex_weights_lag"],
            extra=meta.get("extra", {}),
            **kwargs,
        )

    def field_at(self, name: str, lag: bool = False) -> np.ndarray:
        """`name` in {"smoke", "lwc", "tracer", "texcoords"} at the observation or lag times."""
        return getattr(self, f"{name}_lag" if lag else name)
