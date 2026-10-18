"""Example figures: rendered frames, candidate masks and the CFD fields behind them.

Where this fits: `scripts/generate_dataset.py --examples` writes one figure
per category (smoke, fog, both, neither) into reports/figures/; nothing in
training or inference imports this module (it needs matplotlib).
Calls into: matplotlib (Agg), src.data.dataset.load_sequence, src.cfd.results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.cfd.results import SimulationResult
from src.utils.io import PathLike


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def sequence_figure(seq: dict, sim: Optional[SimulationResult], out_path: PathLike,
                    title: str = "", frames: Sequence[int] = (0, 3, 6, 9, 11)) -> Path:
    """Rows: rendered frames; frames with the candidate mask outlined; CFD smoke / fog fields."""
    plt = _plt()
    import cv2

    n = len(frames)
    rows = 3 if sim is not None else 2
    fig, axes = plt.subplots(rows, n, figsize=(2.3 * n, 2.4 * rows), squeeze=False)
    imgs = seq["frames"]
    masks = seq["masks"].astype(bool)
    times = seq["timestamps"]
    for j, f in enumerate(frames):
        ax = axes[0, j]
        ax.imshow(imgs[f], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"t = {times[f]:.0f} s", fontsize=8)
        rgb = np.repeat(imgs[f][..., None], 3, axis=2).copy()
        contours, _ = cv2.findContours(masks[f].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, contours, -1, (255, 60, 60), 1)
        axes[1, j].imshow(rgb)
        if sim is not None:
            k = min(f, sim.smoke.shape[0] - 1)
            smoke = sim.smoke[k] / max(float(sim.smoke.max()), 1e-9)
            fog = sim.lwc[k] / max(float(sim.lwc.max()), 1e-9)
            tracer = sim.tracer[k] / max(float(sim.tracer.max()), 1e-9)
            field = np.stack([smoke, fog, tracer], axis=-1)[::-1]        # row 0 at the top
            field[sim.solid[::-1]] = 0.35
            axes[2, j].imshow(np.clip(field, 0, 1), interpolation="nearest")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("frame", fontsize=8)
    axes[1, 0].set_ylabel("candidate mask", fontsize=8)
    if sim is not None:
        axes[2, 0].set_ylabel("CFD: smoke R, fog G,\nhaze/dust B", fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
