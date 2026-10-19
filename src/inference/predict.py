"""Deployment API of the physics expert: artifacts in, 16-d embeddings and smoke / fog decisions out.

    expert = PhysicsExpert.load("artifacts")
    X, meta = expert.features_from_frames(frames, timestamps, masks, lag_frames, lag_timestamps)
    emb = expert.encode(X)                       # (B, 16) for the fusion network
    out = expert.predict(X, operating_point="balanced", meta=meta)
    out["decisions"]   # (B, 2) per label: 1 present, 0 absent, -1 abstain

`predict` applies, in order: the fitted normaliser (train-split statistics),
the GRU, the per-label temperature, the operating-point threshold (both
fitted on the calibration split named by `calibration_source`) and two
abstention rules:

* input validity (both labels abstain): fewer than `min_mask_frames` frames
  with a valid candidate mask, fewer than `min_flow_frames` frames with valid
  optical flow, more than `max_clip_fraction` of the open-gate normalised
  entries at the clip, non-finite features, or (when the extractor's `meta`
  is passed) frame timing incompatible with training or flow measured
  between consecutive observations instead of short lag pairs;
* margin (per label): the calibrated logit lies within the fitted abstain
  margin of the threshold.

The thresholds come from SYNTHETIC validation data until real labelled
footage is available (`calibration_source` says which); they are not
validated real-world operating points.

This module imports only numpy, torch, PyYAML (via src.utils.io), src.models
and src.data (OpenCV only inside `features_from_frames`). It never imports
src.training or matplotlib, so a deployment needs no training code.

CLI:
    python -m src.inference.predict --artifacts artifacts --features x.npy [--operating-point low_false_alarm]
x.npy holds (12, 14) or (B, 12, 14) raw features; prints JSON.

Where this fits: the integration point for the multimodal system;
`scripts/benchmark_cpu.py` times it; `src.training.evaluate` uses it so
evaluation applies exactly the deployed decision rules.
Calls into: src.models.physics_gru (artifacts), src.data.normalization,
src.data.features / src.data.masks (lazily, for raw frames).
Start reading at: PhysicsExpert.predict.
"""

from __future__ import annotations

if __package__ in (None, ""):          # `python src/inference/predict.py`: make `src` importable
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import src  # noqa: E402,F401  (pins every numeric library to one thread before numpy / torch load)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from dataclasses import fields as _dc_fields  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.feature_spec import FLOW_VALID_INDEX, MASK_VALID_INDEX, N_FEATURES  # noqa: E402
from src.data.normalization import FeatureNormalizer  # noqa: E402
from src.models.physics_gru import PhysicsGRU, load_model  # noqa: E402
from src.utils.io import PathLike, dumps_json  # noqa: E402

ABSTAIN = -1
_DEFAULT_VALIDITY = {"min_mask_frames": 3, "min_flow_frames": 2, "max_clip_fraction": 0.2}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class PhysicsExpert:
    """A loaded physics expert: normaliser + GRU + calibrated decision rules."""

    def __init__(self, model: PhysicsGRU, normalizer: FeatureNormalizer, config: Mapping[str, Any],
                 artifact_dir: Optional[PathLike] = None):
        self.model = model.eval()
        self.normalizer = normalizer
        self.config = dict(config)
        self.artifact_dir = None if artifact_dir is None else Path(artifact_dir)
        self.label_names: Tuple[str, ...] = tuple(model.config.output_names)
        self.seq_len = int(model.config.seq_len)

    @classmethod
    def load(cls, artifact_dir: PathLike) -> "PhysicsExpert":
        model, normalizer, config = load_model(artifact_dir)
        return cls(model, normalizer, config, artifact_dir)

    # ------------------------------------------------------------- calibration
    @property
    def calibration_source(self) -> str:
        return str((self.config.get("calibration") or {}).get("fit_source", "uncalibrated"))

    @property
    def operating_points(self) -> List[str]:
        return list((self.config.get("thresholds") or {}).keys()) or ["default"]

    def _temperatures(self) -> np.ndarray:
        temps = self.config.get("temperatures") or {}
        return np.array([float(temps.get(n, 1.0)) for n in self.label_names])

    def _thresholds(self, operating_point: str) -> Tuple[np.ndarray, np.ndarray]:
        """(threshold logits, abstain margins) per label for an operating point."""
        thresholds = self.config.get("thresholds") or {}
        if not thresholds:
            return np.zeros(len(self.label_names)), np.zeros(len(self.label_names))
        if operating_point not in thresholds:
            raise KeyError(f"unknown operating point {operating_point!r}; available: {sorted(thresholds)}")
        thr = np.array([float(thresholds[operating_point][n]["logit"]) for n in self.label_names])
        margins = ((self.config.get("abstain") or {}).get("margins") or {}).get(operating_point, {})
        return thr, np.array([float(margins.get(n, 0.0)) for n in self.label_names])

    def _validity_rules(self) -> Dict[str, float]:
        rules = dict(_DEFAULT_VALIDITY)
        rules.update(((self.config.get("abstain") or {}).get("input_validity") or {}))
        return rules

    # ---------------------------------------------------------------- features
    def _batch(self, features) -> np.ndarray:
        X = np.asarray(features, dtype=np.float32)
        if X.ndim == 2:
            X = X[None]
        if X.ndim != 3 or X.shape[1:] != (self.seq_len, N_FEATURES):
            raise ValueError(f"expected features (B, {self.seq_len}, {N_FEATURES}) or ({self.seq_len}, "
                             f"{N_FEATURES}), got {X.shape}")
        return X

    def features_from_frames(self, frames, timestamps, masks=None, lag_frames=None, lag_timestamps=None,
                             feature_cfg=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """(12, 14) raw features and the extraction record from one clip (the training extractor).

        frames / timestamps: the 12 observations (~5 s apart); lag_frames /
        lag_timestamps: a frame ~1 s before each (strongly recommended: without
        them flow is measured between observations and `predict` abstains
        unless allow_consecutive=True). masks: candidate masks from a
        detector; None runs the image-only fallback detector (recorded in
        meta["mask_source"]). Clean simulation masks must never be passed here.
        """
        from src.data.features import FeatureConfig, extract_features_with_meta

        mask_source = "provided"
        if masks is None:
            from src.data.masks import detect_candidate_masks

            masks = detect_candidate_masks(frames, timestamps)
            mask_source = "fallback_detector"
        if feature_cfg is None:
            stored = ((self.config.get("preprocessing") or {}).get("feature_config") or {})
            known = {f.name for f in _dc_fields(FeatureConfig)}
            feature_cfg = FeatureConfig(**{k: v for k, v in stored.items() if k in known})
        X, meta = extract_features_with_meta(frames, timestamps, masks, lag_frames, lag_timestamps, feature_cfg)
        meta["mask_source"] = mask_source
        return X, meta

    # ------------------------------------------------------------------ outputs
    @torch.no_grad()
    def encode(self, features) -> np.ndarray:
        """(B, 16) float32 physics embedding of raw features (B, 12, 14) or (12, 14)."""
        X = self._batch(features)
        X = np.where(np.isfinite(X), X, 0.0)
        Z = torch.from_numpy(self.normalizer.transform(X))
        return self.model.encode(Z).numpy().astype(np.float32)

    @torch.no_grad()
    def predict(self, features, operating_point: str = "balanced", meta: Union[None, Mapping, Sequence] = None,
                allow_consecutive: bool = False) -> Dict[str, Any]:
        """Calibrated probabilities and decisions for raw features (B, 12, 14) or (12, 14).

        meta: the extractor's record (one dict, or one per sequence) for the
        timing checks; omitted, timing is assumed compatible. Returns
        logits (raw), calibrated_logits, probs (calibrated), hard_decisions
        (threshold only), decisions (with abstention, -1), abstain_reasons
        (list per sequence), valid_input, clip_fraction, operating_point,
        thresholds (probabilities), calibration_source, label_names.
        """
        X = self._batch(features)
        n = len(X)
        finite = np.isfinite(X).all(axis=(1, 2))
        clip_frac = self.normalizer.clip_fraction(X)
        Xs = np.where(np.isfinite(X), X, 0.0)
        logits = self.model(torch.from_numpy(self.normalizer.transform(Xs))).numpy().astype(np.float64)
        z = logits / self._temperatures()
        thr, margin = self._thresholds(operating_point)
        hard = (z >= thr).astype(np.int64)
        decisions = hard.copy()

        rules = self._validity_rules()
        metas = self._metas(meta, n)
        reasons: List[List[str]] = [[] for _ in range(n)]
        n_mask = (X[:, :, MASK_VALID_INDEX] > 0.5).sum(axis=1)
        n_flow = (X[:, :, FLOW_VALID_INDEX] > 0.5).sum(axis=1)
        for i in range(n):
            r = reasons[i]
            if not finite[i]:
                r.append("nonfinite_features")
            if n_mask[i] < rules["min_mask_frames"]:
                r.append("few_valid_mask_frames")
            if n_flow[i] < rules["min_flow_frames"]:
                r.append("few_valid_flow_frames")
            if clip_frac[i] > rules["max_clip_fraction"]:
                r.append("clipped_inputs")
            m = metas[i]
            if m:
                timing = m.get("timing") or {}
                if timing.get("interval_compatible") is False or timing.get("lag_compatible") is False \
                        or m.get("incompatible_timing"):
                    r.append("incompatible_frame_interval")
                if m.get("flow_mode") == "consecutive" and not allow_consecutive:
                    r.append("consecutive_flow")
        valid = np.array([not r for r in reasons])
        decisions[~valid] = ABSTAIN
        near = np.abs(z - thr) < margin
        for i in range(n):
            for j, name in enumerate(self.label_names):
                if valid[i] and near[i, j]:
                    decisions[i, j] = ABSTAIN
                    reasons[i].append(f"{name}_near_threshold")
        return {
            "logits": logits,
            "calibrated_logits": z,
            "probs": _sigmoid(z),
            "hard_decisions": hard,
            "decisions": decisions,
            "abstain_reasons": reasons,
            "valid_input": valid,
            "clip_fraction": clip_frac,
            "operating_point": operating_point,
            "thresholds": {n: float(_sigmoid(t)) for n, t in zip(self.label_names, thr)},
            "calibration_source": self.calibration_source,
            "label_names": list(self.label_names),
        }

    @staticmethod
    def _metas(meta, n: int) -> List[Optional[Mapping[str, Any]]]:
        if meta is None:
            return [None] * n
        if isinstance(meta, Mapping):
            return [meta] * n
        metas = list(meta)
        if len(metas) != n:
            raise ValueError(f"{len(metas)} meta records for {n} sequences")
        return metas


# =============================================================================
# CLI
# =============================================================================


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Physics expert inference on precomputed 12 x 14 features.")
    ap.add_argument("--artifacts", required=True, help="artifact directory (physics_classifier.pt, ...)")
    ap.add_argument("--features", required=True, help=".npy file with (12, 14) or (B, 12, 14) raw features")
    ap.add_argument("--meta", default=None, help="optional JSON file: extractor meta (one dict or a list)")
    ap.add_argument("--operating-point", default="balanced")
    ap.add_argument("--embedding", action="store_true", help="also print the 16-d embeddings")
    ap.add_argument("--allow-consecutive", action="store_true", help="accept flow between consecutive frames")
    args = ap.parse_args(argv)
    from src.utils.runtime import configure_single_thread

    configure_single_thread()
    expert = PhysicsExpert.load(args.artifacts)
    X = np.load(args.features, allow_pickle=False)
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8")) if args.meta else None
    out = expert.predict(X, args.operating_point, meta=meta, allow_consecutive=args.allow_consecutive)
    doc = {k: v for k, v in out.items()}
    if args.embedding:
        doc["embedding"] = expert.encode(X)
    sys.stdout.write(dumps_json(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
