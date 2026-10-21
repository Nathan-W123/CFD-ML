"""Windows ending at every frame of a sequence, assembled from per-frame embeddings (scripts/embed_frames.py).

A deployed camera scores a window ending every minute. Each frame is embedded once (coarse: whole frame + 2 x 3
tiles, fine: 4 x 6 tiles); the window ending at frame i takes as frame 11 that frame, as frames 0 and 6 the
frames nearest 11 and 5 intervals earlier, and as its reference the frame nearest REF_LAG_MIN minutes before
frame 0 (each within a tolerance), exactly the crops the detector was trained on (scripts/embed_windows.py,
scripts/reference_frames.py): coarse of frames 0, 6, 11; fine of frames 0 and 11; fine of the reference.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

INTERVAL_S = 60
REF_LAG_MIN = 14
TOL_S = 30


def _nearest(unix: np.ndarray, targets: np.ndarray, tol: float) -> np.ndarray:
    """Index of the frame nearest each target time (unix sorted), or -1 when none is within tol."""
    j = np.clip(np.searchsorted(unix, targets), 1, len(unix) - 1) if len(unix) > 1 else np.zeros(len(targets), int)
    best = np.where(np.abs(unix[j - 1] - targets) <= np.abs(unix[j] - targets), j - 1, j) if len(unix) > 1 else j
    return np.where(np.abs(unix[best] - targets) <= tol, best, -1)


def window_index(unix, interval: float = INTERVAL_S, tol: float = TOL_S, ref_lag_min: float = REF_LAG_MIN) -> np.ndarray:
    """(F, 4): for each frame taken as frame 11, the indices of frames 0, 6, 11 and of the reference (-1: none)."""
    unix = np.asarray(unix, np.int64)
    t11 = unix.astype(np.float64)
    i0 = _nearest(unix, t11 - 11 * interval, tol)
    i6 = _nearest(unix, t11 - 5 * interval, tol)
    iref = _nearest(unix, t11 - 11 * interval - ref_lag_min * 60, tol)
    return np.stack([i0, i6, np.arange(len(unix)), iref], 1)


def assemble(coarse: np.ndarray, fine: np.ndarray, idx: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
    """Detector inputs of the windows idx (rows of window_index with frames 0 and 6 present): E (N, 3, 7, D),
    fine (N, 2, 24, D) and ref (N, 24, D) when every row has a reference (else None)."""
    idx = np.asarray(idx)
    if (idx[:, :2] < 0).any():
        raise ValueError("windows without frame 0 or 6 cannot be assembled")
    E = np.asarray(coarse, np.float32)[idx[:, [0, 1, 2]]]
    Fe = np.asarray(fine, np.float32)[idx[:, [0, 2]]]
    ref = np.asarray(fine, np.float32)[idx[:, 3]] if (idx[:, 3] >= 0).all() else None
    return {"E": E, "fine": Fe, "ref": ref}
