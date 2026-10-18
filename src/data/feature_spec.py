"""The 14-feature motion representation shared by synthetic and real footage.

This module is the single source of truth for the model input layout:
12 timesteps x 14 features. `features.py` produces it, `normalization.py`
standardises it, the GRU consumes it. Nothing here depends on OpenCV.

Gates: a feature whose gate is closed at a timestep is exactly 0.0 in the raw
array and stays 0.0 after normalisation; its statistics are fitted only over
timesteps where the gate is open.

    "flow"      open when flow_valid[t] == 1: the short-lag optical-flow pair
                (t - flow_lag_s, t) exists, both masks are valid, camera
                stabilisation succeeded and the flow passed the
                forward-backward consistency check
    "mask"      open when mask_valid[t] == 1
    "mask_pair" open when mask_valid[t] == 1 and mask_valid[t-1] == 1 (t >= 1):
                the temporal derivatives between consecutive observations
    None        always open (the validity flags themselves; never normalised)

Lengths are in units of L = sqrt(H * W) of the analysis frame (the region of
interest around the candidate masks, see docs/design.md section 7), rates
per second, vertical components positive UP.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

N_STEPS = 12
N_FEATURES = 14

FEATURE_NAMES: Tuple[str, ...] = (
    "flow_u_mean",          # 0  mean horizontal flow, +right, frame-scale units / s
    "flow_v_mean",          # 1  mean vertical flow, +UP, frame-scale units / s
    "flow_u_std",           # 2  std of horizontal flow in the region
    "flow_v_std",           # 3  std of vertical flow in the region
    "flow_divergence",      # 4  mean du/dx + dv/dz in the region, 1/s
    "flow_vorticity_abs",   # 5  mean |dv/dx - du/dz| in the region, 1/s
    "region_area",          # 6  mask area / image area
    "area_growth",          # 7  d ln(area) / dt, 1/s
    "centroid_dx",          # 8  centroid x velocity, frame-scale units / s, +right
    "centroid_dz",          # 9  centroid vertical velocity, frame-scale units / s, +UP
    "boundary_roughness",   # 10 perimeter / (2 sqrt(pi * area)) - 1 (0 for a disc)
    "anchor_stability",     # 11 column-wise stability of the region's bottom boundary, t vs t-1, in [0, 1]
    "mask_valid",           # 12 candidate mask non-empty (>= min area) at t
    "flow_valid",           # 13 optical flow available and valid at t
)

FEATURE_GATES: Tuple[Optional[str], ...] = (
    "flow", "flow", "flow", "flow", "flow", "flow",
    "mask", "mask_pair", "mask_pair", "mask_pair", "mask", "mask_pair",
    None, None,
)

MASK_VALID_INDEX = 12
FLOW_VALID_INDEX = 13
VALIDITY_INDICES: Tuple[int, int] = (MASK_VALID_INDEX, FLOW_VALID_INDEX)

FEATURE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}

assert len(FEATURE_NAMES) == N_FEATURES and len(FEATURE_GATES) == N_FEATURES


def gate_masks(x: np.ndarray) -> np.ndarray:
    """Boolean array, same shape as `x` (..., T, 14): True where a feature's gate is open."""
    x = np.asarray(x)
    if x.shape[-1] != N_FEATURES:
        raise ValueError(f"expected last dimension {N_FEATURES}, got shape {x.shape}")
    mask_valid = x[..., MASK_VALID_INDEX] > 0.5
    flow_valid = x[..., FLOW_VALID_INDEX] > 0.5
    pair = np.zeros_like(mask_valid)
    pair[..., 1:] = mask_valid[..., 1:] & mask_valid[..., :-1]
    out = np.zeros(x.shape, dtype=bool)
    for i, gate in enumerate(FEATURE_GATES):
        if gate == "flow":
            out[..., i] = flow_valid
        elif gate == "mask":
            out[..., i] = mask_valid
        elif gate == "mask_pair":
            out[..., i] = pair
        else:
            out[..., i] = True
    return out
