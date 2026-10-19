"""The physics expert network: a small GRU encoder (16-d embedding) and a two-label head.

    x (B, 12, 14) normalised features
      -> GRU(14 -> 32, one layer, unidirectional) -> last hidden state (B, 32)
      -> Dropout(0.1) -> Linear(32, 16) -> LayerNorm(16) -> GELU      = embedding (B, 16)
      -> Linear(16, 2)                                                 = logits (smoke, fog)

The embedding is the product: the multimodal fusion network consumes
`encode(x)`, which never touches the classification head. The two logits are
independent binary decisions (smoke present, fog present); training uses
them with BCEWithLogits, inference applies the per-label temperature and
threshold stored in model_config.yaml (see `src.inference.predict`).

Artifacts (`save_model_artifacts`):
    physics_encoder.pt        encoder state_dict (embedding only)
    physics_classifier.pt     full model state_dict (encoder + head)
    feature_normalizer.json   gated normaliser fitted on the train split
    model_config.yaml         ModelConfig, feature layout, preprocessing,
                              temperatures, thresholds, abstain margins and
                              their provenance (calibration set, fit source)
State dicts are loaded with `torch.load(weights_only=True)`, so a checkpoint
can never execute code.

Where this fits: `src.training.train` builds and trains PhysicsGRU and calls
`save_model_artifacts`; `src.inference.predict.PhysicsExpert` calls
`load_model`; the fusion network can call `load_encoder` alone.
Calls into: torch, `src.data.normalization`, `src.data.feature_spec`, `src.utils.io`.
Never imports src.training (inference must load without training code).
Start reading at: PhysicsEncoder.forward.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn

from src.data.feature_spec import FEATURE_GATES, FEATURE_NAMES, N_FEATURES, N_STEPS
from src.data.normalization import FeatureNormalizer
from src.utils.io import PathLike, read_yaml, write_yaml

ENCODER_FILE = "physics_encoder.pt"
CLASSIFIER_FILE = "physics_classifier.pt"
NORMALIZER_FILE = "feature_normalizer.json"
CONFIG_FILE = "model_config.yaml"
ARTIFACT_FORMAT_VERSION = 1


@dataclass
class ModelConfig:
    input_dim: int = N_FEATURES
    seq_len: int = N_STEPS
    hidden_size: int = 32
    num_layers: int = 1
    embedding_dim: int = 16
    dropout: float = 0.1
    output_names: Tuple[str, ...] = ("smoke", "fog")

    def __post_init__(self):
        self.output_names = tuple(self.output_names)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["output_names"] = list(self.output_names)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown ModelConfig keys: {sorted(unknown)}")
        return cls(**dict(d))


def _as_tensor(x) -> torch.Tensor:
    """float32 tensor (B, T, F) from an array / tensor of shape (B, T, F) or (T, F)."""
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    t = t.to(torch.float32)
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.dim() != 3:
        raise ValueError(f"expected features of shape (B, T, F) or (T, F), got {tuple(t.shape)}")
    return t


class PhysicsEncoder(nn.Module):
    """GRU -> last hidden state -> Dropout -> Linear -> LayerNorm -> GELU: the 16-d physics embedding."""

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.config = cfg
        self.gru = nn.GRU(input_size=cfg.input_dim, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
                          batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.proj = nn.Linear(cfg.hidden_size, cfg.embedding_dim)
        self.norm = nn.LayerNorm(cfg.embedding_dim)
        self.act = nn.GELU()

    def forward(self, x) -> torch.Tensor:
        x = _as_tensor(x)
        if x.shape[-1] != self.config.input_dim:
            raise ValueError(f"expected {self.config.input_dim} features per timestep, got {x.shape[-1]}")
        _, h = self.gru(x)              # h: (num_layers, B, hidden)
        z = self.dropout(h[-1])         # last layer's final hidden state
        return self.act(self.norm(self.proj(z)))


class PhysicsGRU(nn.Module):
    """Encoder + Linear(16, 2) head producing (smoke, fog) logits."""

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.config = cfg
        self.encoder = PhysicsEncoder(cfg)
        self.head = nn.Linear(cfg.embedding_dim, len(cfg.output_names))

    def forward(self, x) -> torch.Tensor:
        return self.head(self.encoder(x))

    def encode(self, x) -> torch.Tensor:
        """(B, 16) embedding; the head is not evaluated."""
        return self.encoder(x)

    @torch.no_grad()
    def predict(self, x) -> Dict[str, torch.Tensor]:
        """Raw logits and sigmoid probabilities (B, 2) in eval mode (no temperature, no threshold)."""
        was_training = self.training
        self.eval()
        try:
            logits = self.forward(x)
        finally:
            self.train(was_training)
        return {"logits": logits, "probs": torch.sigmoid(logits)}


# =============================================================================
# Artifacts
# =============================================================================


def _default_preprocessing(normalizer: FeatureNormalizer) -> Dict[str, Any]:
    return {
        "n_steps": N_STEPS,
        "n_features": N_FEATURES,
        "feature_names": list(FEATURE_NAMES),
        "feature_gates": [g if g is not None else "none" for g in FEATURE_GATES],
        "validity_flags": ["mask_valid", "flow_valid"],
        "normalizer_file": NORMALIZER_FILE,
        "normalizer_fit_split": "train",
        "normalizer_clip": normalizer.clip,
        "closed_gate_value": 0.0,
    }


def save_model_artifacts(model: PhysicsGRU, out_dir: PathLike, normalizer: FeatureNormalizer,
                         extra_config: Optional[Mapping[str, Any]] = None) -> Dict[str, Path]:
    """Write the four model artifacts; return {"encoder", "classifier", "normalizer", "config"} -> path.

    `extra_config` is merged into model_config.yaml (thresholds, temperatures,
    abstain margins, calibration and fit provenance, training summary). Its
    "preprocessing" mapping is merged over the default feature-layout record.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    extra = copy.deepcopy(dict(extra_config or {}))
    paths = {
        "encoder": out / ENCODER_FILE,
        "classifier": out / CLASSIFIER_FILE,
        "normalizer": out / NORMALIZER_FILE,
        "config": out / CONFIG_FILE,
    }
    torch.save({k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}, paths["encoder"])
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, paths["classifier"])
    normalizer.save(paths["normalizer"])
    preprocessing = _default_preprocessing(normalizer)
    preprocessing.update(extra.pop("preprocessing", {}) or {})
    doc: Dict[str, Any] = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "model": model.config.to_dict(),
        "feature_names": list(FEATURE_NAMES),
        "preprocessing": preprocessing,
        "files": {k: p.name for k, p in paths.items()},
        "torch_version": str(torch.__version__),
    }
    for key, value in extra.items():
        if key in ("model", "files", "artifact_format_version"):
            raise KeyError(f"extra_config may not override {key!r}")
        doc[key] = value
    write_yaml(paths["config"], doc)
    return paths


def _read_config(artifact_dir: Path) -> Dict[str, Any]:
    path = artifact_dir / CONFIG_FILE
    if not path.exists():
        raise FileNotFoundError(f"no {CONFIG_FILE} in {artifact_dir}")
    cfg = read_yaml(path)
    if int(cfg.get("artifact_format_version", 1)) > ARTIFACT_FORMAT_VERSION:
        raise ValueError(f"artifact format {cfg['artifact_format_version']} is newer than this code")
    if tuple(cfg.get("feature_names", FEATURE_NAMES)) != FEATURE_NAMES:
        raise ValueError("artifact feature layout does not match src.data.feature_spec")
    return cfg


def _load_state(path: Path) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model(artifact_dir: PathLike) -> Tuple[PhysicsGRU, FeatureNormalizer, Dict[str, Any]]:
    """(PhysicsGRU in eval mode, FeatureNormalizer, model_config dict) from an artifact directory."""
    d = Path(artifact_dir)
    cfg = _read_config(d)
    model = PhysicsGRU(ModelConfig.from_dict(cfg["model"]))
    model.load_state_dict(_load_state(d / CLASSIFIER_FILE))
    model.eval()
    return model, FeatureNormalizer.load(d / NORMALIZER_FILE), cfg


def load_encoder(artifact_dir: PathLike) -> Tuple[PhysicsEncoder, FeatureNormalizer, Dict[str, Any]]:
    """(PhysicsEncoder in eval mode, FeatureNormalizer, model_config dict); the head is not loaded."""
    d = Path(artifact_dir)
    cfg = _read_config(d)
    encoder = PhysicsEncoder(ModelConfig.from_dict(cfg["model"]))
    encoder.load_state_dict(_load_state(d / ENCODER_FILE))
    encoder.eval()
    return encoder, FeatureNormalizer.load(d / NORMALIZER_FILE), cfg
