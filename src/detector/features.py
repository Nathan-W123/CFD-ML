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
  when a FeatureSet row is available (28).
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


def window_features(E: np.ndarray, T: np.ndarray, groups: Sequence[str],
                    motion: Optional[np.ndarray] = None) -> np.ndarray:
    """Concatenated feature groups for N windows; "motion" takes the given (N, 28) pooled motion features."""
    parts = []
    for g in groups:
        if g == "motion":
            if motion is None:
                raise ValueError("feature group 'motion' needs the pooled motion features")
            parts.append(np.asarray(motion, np.float32))
        else:
            parts.append(group_features(E, T, g))
    return np.concatenate(parts, 1).astype(np.float32)
