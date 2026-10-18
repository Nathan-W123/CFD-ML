"""Single-thread execution and reproducible seeding.

Where this fits: called once at the start of every script and test session.
Calls into: torch, cv2 and numba when they are importable; numpy always.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

import numpy as np

# Random-stream registry: the second key of child_rng(sim_seed, KEY, ...).
# Scenario streams (1-5) live in src/cfd/scenarios.py; render streams here.
STREAM_RENDER_PARAMS = 101
STREAM_RENDER_NOISE = 102
STREAM_MASKS = 103
STREAM_DETECTOR = 104
STREAM_ROBUSTNESS = 105


def configure_single_thread(strict: bool = False) -> Dict[str, Optional[int]]:
    """Limit torch, OpenCV, numba and BLAS to one thread; return the effective counts.

    The environment variables are already forced to 1 by src/__init__; this
    repeats the runtime calls they cannot cover and caps any BLAS / OpenMP pool
    numpy already loaded. With strict=True a count above one raises.
    """
    counts: Dict[str, Optional[int]] = {"torch_intra": None, "torch_interop": None,
                                        "cv2": None, "numba": None, "blas": None}
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Interop threads can only be set before any parallel work ran.
            pass
        counts["torch_intra"] = int(torch.get_num_threads())
        counts["torch_interop"] = int(torch.get_num_interop_threads())
    except ImportError:
        pass
    try:
        import cv2

        cv2.setNumThreads(1)
        counts["cv2"] = int(cv2.getNumThreads())
    except ImportError:
        pass
    try:
        import numba

        numba.set_num_threads(1)
        counts["numba"] = int(numba.get_num_threads())
    except (ImportError, ValueError):
        pass
    try:
        from threadpoolctl import threadpool_info, threadpool_limits

        threadpool_limits(1)
        pools = [int(info.get("num_threads", 1)) for info in threadpool_info()]
        counts["blas"] = max(pools) if pools else 1
    except ImportError:
        pass
    if strict:
        over = {k: v for k, v in counts.items() if v is not None and v > 1}
        if over:
            raise RuntimeError(f"more than one thread in use: {over}")
    return counts


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed python, numpy's legacy global RNG and torch; make torch deterministic.

    Library code never uses the global numpy RNG (it takes a Generator), but
    seeding it guards third-party code that does. Seeds are never derived from
    hash() (salted per process); see derive_seed.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def _entropy(base_seed: int, keys) -> list:
    # The key count is part of the entropy: SeedSequence([s, k]) and [s, k, 0]
    # would otherwise produce the same stream.
    return [int(base_seed), len(keys), *[int(k) for k in keys]]


def child_rng(base_seed: int, *keys: int) -> np.random.Generator:
    """A Generator derived deterministically from a base seed and integer keys.

    e.g. `child_rng(sim_seed, stream_key)` gives every component of a
    simulation its own independent stream, so regenerating one sample never
    depends on how many others were generated before it.
    """
    return np.random.default_rng(np.random.SeedSequence(_entropy(base_seed, keys)))


def derive_seed(base_seed: int, *keys: int) -> int:
    """A 32-bit integer seed derived like `child_rng`, for storing in metadata."""
    ss = np.random.SeedSequence(_entropy(base_seed, keys))
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def torch_threads() -> Optional[int]:
    """The current torch intra-op thread count, or None without torch."""
    try:
        import torch

        return int(torch.get_num_threads())
    except ImportError:
        return None
