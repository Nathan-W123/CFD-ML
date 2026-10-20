"""Window features for the detector heads, from CLIP embeddings of a window's frames.

A window's embeddings E have shape (F, 7, D): frames F (0, 6 and 11 of the
12-frame window, scripts/embed_windows.py), crops 7 (crop 0 the whole frame,
crops 1-6 a 2 x 3 tile grid) and D = 512 (L2-normalised CLIP image
embeddings). Text prompts are embedded once with the same CLIP model; a
"similarity" is 100 x the cosine between a crop and a prompt (CLIP's logit
scale).

Feature groups (names used in the detector config):

* full: whole-frame embedding averaged over the frames (D);
* tiles: tile embeddings of the last frame, averaged and max-pooled over the
  tiles, and their change over the window (|last - first| averaged, last -
  first max-pooled) (4 D);
* sims: prompt similarities of every crop averaged over the frames, the
  max over tiles averaged over the frames, and the max over tiles of the
  change in similarity from the first to the last frame (7 P + 2 P);
* motion: the motion expert's pooled 12 x 14 features (time mean and std of
  each feature over its valid steps, src/training/baselines.pooled_features),
  when a FeatureSet row is available (28);
* fine: from the finer embeddings Fe of a window, shape (2, 24, D): frames 0
  and 11, a 4 x 6 tile grid without the whole frame (embed_windows.py --grid
  4x6 --frames 0,11 --no-full), so a tile of a 1024-px frame is about the
  CLIP input size: per tile the mean prompt similarity in the last frame, its
  change from the first frame and the size of the embedding change, each as
  its top 3 over the tiles and its mean; the max over tiles of every prompt's
  last-frame similarity and change; and the change (max-pooled and |.|
  averaged) and last-frame embeddings (max-pooled) over the tiles
  (12 + 2 P + 3 D).

tile_instances gives the same fine tiles one by one ([last, last - first,
prompt similarities last, their change], (N, 24, 2 D + 2 P)) for the tile
(multiple-instance) smoke head, which scores a window by its highest tile;
zoom_instances does the same for the full-resolution zoom crops (frames 0
and 11, 2 x 2 quarters of the window's top tiles: scripts/zoom_windows.py).

Scalar stack inputs (no fitting), from the smoke prompts' contrast per fine
tile, s = mean similarity to the smoke prompts (the first SMOKE_POSITIVE of
PROMPTS["smoke"]) minus mean similarity to the others:

* change_scores: c = s(frame 11) - s(frame 0) per tile; the window's max over
  tiles and its robust z ((max - median) / MAD);
* reference_scores: r = s(frame 11) - s(reference) per tile, the reference
  being the same camera 14 min before frame 0 (scripts/reference_frames.py);
  max and robust z.
"""

from typing import Dict, Iterable, Optional, Sequence

import numpy as np

MODEL_ID = "openai/clip-vit-base-patch32"
PROMPTS: Dict[str, Sequence[str]] = {
    "fog": ("a sea of fog below the mountains", "fog filling the valleys", "a camera view inside thick fog",
            "low clouds covering the hills", "a clear view of mountains and valleys",
            "a hazy horizon on a clear day", "clouds in the sky above clear land",
            "wildfire smoke rising from the hills"),
    "smoke": ("a plume of wildfire smoke rising from a hillside", "a thin column of grey smoke in the distance",
              "smoke from a brush fire drifting over hills", "a clear landscape with no smoke",
              "white clouds in a blue sky", "a hazy horizon", "fog in the valleys", "a sunny mountain view"),
}


def text_embeddings(prompts: Iterable[str], model_id: str = MODEL_ID) -> np.ndarray:
    """L2-normalised CLIP text embeddings (P, D) of the prompts (needs transformers and the model weights)."""
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    model = CLIPModel.from_pretrained(model_id).eval()
    tok = CLIPTokenizer.from_pretrained(model_id)
    with torch.no_grad():
        t = model.get_text_features(**tok(list(prompts), padding=True, return_tensors="pt"))
        t = t / t.norm(dim=-1, keepdim=True)
    return t.numpy().astype(np.float32)


def _check(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=np.float32)
    if E.ndim == 3:
        E = E[None]
    if E.ndim != 4 or E.shape[2] != 7:
        raise ValueError(f"embeddings must be (N, F, 7, D) or (F, 7, D), got {E.shape}")
    return E


def group_features(E: np.ndarray, T: np.ndarray, group: str) -> np.ndarray:
    """One feature group (see the module docstring) for N windows: (N, d)."""
    E = _check(E)
    n = len(E)
    if group == "full":
        return E[:, :, 0].mean(1)
    if group == "tiles":
        tiles = E[:, :, 1:]
        d = tiles[:, -1] - tiles[:, 0]
        return np.concatenate([tiles[:, -1].mean(1), tiles[:, -1].max(1), np.abs(d).mean(1), d.max(1)], 1)
    if group == "sims":
        S = np.einsum("nfcd,pd->nfcp", E, np.asarray(T, np.float32)) * 100.0
        return np.concatenate([S.mean(1).reshape(n, -1), S[:, :, 1:].max(2).mean(1),
                               (S[:, -1] - S[:, 0])[:, 1:].max(1)], 1)
    raise ValueError(f"unknown feature group {group!r}")


FINE_TILES = 24                     # 4 x 6
FINE_TOP = 3
SMOKE_POSITIVE = 3                  # PROMPTS["smoke"][:3] describe smoke, the rest look-alikes


def _check_fine(Fe: np.ndarray) -> np.ndarray:
    Fe = np.asarray(Fe, dtype=np.float32)
    if Fe.ndim == 3:
        Fe = Fe[None]
    if Fe.ndim != 4 or Fe.shape[1] != 2 or Fe.shape[2] != FINE_TILES:
        raise ValueError(f"fine embeddings must be (N, 2, {FINE_TILES}, D) or (2, {FINE_TILES}, D), got {Fe.shape}")
    return Fe


def fine_features(Fe: np.ndarray, T: np.ndarray) -> np.ndarray:
    """The "fine" group (see the module docstring) for N windows: (N, 12 + 2 P + 3 D)."""
    Fe = _check_fine(Fe)
    S = np.einsum("nftd,pd->nftp", Fe, np.asarray(T, np.float32)) * 100.0     # N, 2, 24, P
    dS = S[:, 1] - S[:, 0]
    sm, dsm = S[:, 1].mean(-1), dS.mean(-1)                                    # N, 24
    d = Fe[:, 1] - Fe[:, 0]
    nrm = np.linalg.norm(d, axis=-1)

    def top(a):
        return -np.sort(-a, axis=1)[:, :FINE_TOP]

    return np.concatenate([top(sm), top(dsm), top(nrm), sm.mean(1, keepdims=True), dsm.mean(1, keepdims=True),
                           nrm.mean(1, keepdims=True), S[:, 1].max(1), dS.max(1),
                           d.max(1), np.abs(d).mean(1), Fe[:, 1].max(1)], 1)


def _instances(Fe: np.ndarray, T: np.ndarray) -> np.ndarray:
    S = np.einsum("nftd,pd->nftp", Fe, np.asarray(T, np.float32)) * 100.0
    return np.concatenate([Fe[:, 1], Fe[:, 1] - Fe[:, 0], S[:, 1], S[:, 1] - S[:, 0]], -1).astype(np.float32)


def tile_instances(Fe: np.ndarray, T: np.ndarray) -> np.ndarray:
    """The fine tiles as instances of the tile smoke head: (N, 24, 2 D + 2 P)."""
    return _instances(_check_fine(Fe), T)


def zoom_instances(Z: np.ndarray, T: np.ndarray) -> np.ndarray:
    """The zoom crops (N, 2, K, D) as instances of the zoom smoke head: (N, K, 2 D + 2 P)."""
    Z = np.asarray(Z, dtype=np.float32)
    if Z.ndim == 3:
        Z = Z[None]
    if Z.ndim != 4 or Z.shape[1] != 2:
        raise ValueError(f"zoom embeddings must be (N, 2, K, D) or (2, K, D), got {Z.shape}")
    return _instances(Z, T)


def _contrast(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Smoke-prompt contrast of embeddings X (..., D): mean similarity to the smoke prompts minus the others'."""
    S = np.einsum("...d,pd->...p", np.asarray(X, np.float32), np.asarray(T, np.float32)) * 100.0
    return S[..., :SMOKE_POSITIVE].mean(-1) - S[..., SMOKE_POSITIVE:].mean(-1)


def _max_and_robust_z(c: np.ndarray) -> np.ndarray:
    mx = c.max(1)
    med = np.median(c, 1)
    mad = np.median(np.abs(c - med[:, None]), 1) + 1e-3
    return np.stack([mx, (mx - med) / mad], 1).astype(np.float32)


def change_scores(Fe: np.ndarray, T: np.ndarray) -> np.ndarray:
    """(N, 2): max over fine tiles of the contrast change from frame 0 to frame 11, and its robust z."""
    Fe = _check_fine(Fe)
    c = _contrast(Fe, T)
    return _max_and_robust_z(c[:, 1] - c[:, 0])


def reference_scores(Fe: np.ndarray, Fref: np.ndarray, T: np.ndarray) -> np.ndarray:
    """(N, 2): max over fine tiles of the contrast of frame 11 minus the reference frame's (N, 24, D), robust z."""
    Fe = _check_fine(Fe)
    Fref = np.asarray(Fref, np.float32)
    if Fref.ndim == 2:
        Fref = Fref[None]
    if Fref.shape[:2] != (len(Fe), FINE_TILES):
        raise ValueError(f"reference fine embeddings must be (N, {FINE_TILES}, D), got {Fref.shape}")
    return _max_and_robust_z(_contrast(Fe[:, 1], T) - _contrast(Fref, T))


def window_features(E: np.ndarray, T: np.ndarray, groups: Sequence[str],
                    motion: Optional[np.ndarray] = None, fine: Optional[np.ndarray] = None) -> np.ndarray:
    """Concatenated feature groups for N windows; "motion" takes the given (N, 28) pooled motion features,
    "fine" the (N, 2, 24, D) fine embeddings."""
    parts = []
    for g in groups:
        if g == "motion":
            if motion is None:
                raise ValueError("feature group 'motion' needs the pooled motion features")
            parts.append(np.asarray(motion, np.float32))
        elif g == "fine":
            if fine is None:
                raise ValueError("feature group 'fine' needs the fine (4 x 6 tile) embeddings")
            parts.append(fine_features(fine, T))
        else:
            parts.append(group_features(E, T, g))
    return np.concatenate(parts, 1).astype(np.float32)
