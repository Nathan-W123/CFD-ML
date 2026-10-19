"""Training of the physics GRU: Stage A (CFD pretraining) and Stage B (real-footage fine-tuning).

Stage A (`run_pretraining`, experiment C), on a synthetic FeatureSet:

1. check split leakage; fit the gated normaliser on the TRAIN split only;
2. for each seed in `training.seeds`: build PhysicsGRU, `train_model`
   (BCEWithLogits, + pos_weight when a label is imbalanced, AdamW, batch 64,
   gradient clipping, early stopping on val_select loss with patience,
   best-epoch weights restored), then fit the decision rules on val_calib
   with `fit_calibration`: per-label temperature (bounded to [0.5, 5]; T = 1
   and a flag when val_calib is separable), thresholds for the `balanced` and
   `low_false_alarm` operating points, and the logit-space abstain band;
3. export the seed with the lowest val_select loss as the model artifacts
   (and every seed under seeds/seed_<k>/ for the mean +- sd evaluation),
   with per-seed CSV / JSON histories and a training-curve figure in logs/.

Stage B (`run_finetune`, experiments B and D), on a real FeatureSet whose
groups are camera ids: the same procedure with the camera-disjoint splits of
that set; D starts from the pretrained artifacts with `finetune.lr`, B from
random weights with `training.lr`; epochs, patience, seeds, splits and the
normaliser policy (refit on the real train split) are identical.

`run_loro` trains one model per held-out regime (leave-one-regime-out) and
reports how the unseen regime is handled.

Nothing in val_calib or the test splits influences the weights; nothing in
the test splits influences anything.

Where this fits: `scripts/train_physics.py` calls run_pretraining /
run_finetune; `scripts/evaluate_physics.py` may call run_loro.
Calls into: torch, numpy, src.models.physics_gru, src.data.{dataset,normalization},
src.training.metrics, matplotlib (Agg, figures only).
Start reading at: run_pretraining, then train_model.
"""

from __future__ import annotations

import copy
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from src.constants import CATEGORIES, LABEL_NAMES
from src.data.dataset import FeatureSet, assert_no_group_leakage
from src.data.normalization import FeatureNormalizer
from src.models.physics_gru import ModelConfig, PhysicsGRU, load_model, save_model_artifacts
from src.training import metrics as M
from src.utils.io import PathLike, write_json
from src.utils.runtime import seed_everything

BALANCE_RATIO = 1.5           # "auto" balancing kicks in above this label ratio
EVAL_BATCH = 4096


@dataclass
class TrainResult:
    model: PhysicsGRU
    history: List[Dict[str, Any]]
    best_epoch: int
    best_val_loss: float
    epochs_run: int
    stopped_early: bool
    pos_weight: Optional[List[float]]
    balance: str
    seconds: float
    settings: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# One model
# =============================================================================


def _label_balance(y: np.ndarray, mode: str) -> Tuple[Optional[np.ndarray], str]:
    """pos_weight per label (n_neg / n_pos) or None, and the balancing actually used.

    "auto": pos_weight for the labels whose positive:negative ratio exceeds
    1.5 either way (others 1.0), nothing when every label is balanced;
    "pos_weight": always; "sampler": category-balanced sampling instead;
    "none": nothing.
    """
    if mode not in ("auto", "pos_weight", "sampler", "none"):
        raise ValueError(f"unknown training.balance {mode!r}")
    if mode in ("none", "sampler"):
        return None, mode
    n_pos = y.sum(axis=0).astype(np.float64)
    n_neg = len(y) - n_pos
    weight = np.maximum(n_neg, 1.0) / np.maximum(n_pos, 1.0)
    imbalanced = (weight > BALANCE_RATIO) | (weight < 1.0 / BALANCE_RATIO)
    if mode == "pos_weight":
        return weight.astype(np.float32), "pos_weight"
    if not imbalanced.any():
        return None, "none"
    return np.where(imbalanced, weight, 1.0).astype(np.float32), "pos_weight"


def _sampler_weights(y: np.ndarray) -> torch.Tensor:
    """Per-sample weights giving every (smoke, fog) category equal total weight."""
    code = (y[:, 0] > 0.5).astype(int) * 2 + (y[:, 1] > 0.5).astype(int)
    counts = np.bincount(code, minlength=4).astype(np.float64)
    return torch.as_tensor(1.0 / counts[code], dtype=torch.float64)


@torch.no_grad()
def _logits(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    was = model.training
    model.eval()
    out = torch.cat([model(X[i:i + EVAL_BATCH]) for i in range(0, len(X), EVAL_BATCH)]) if len(X) else \
        torch.zeros((0, 2))
    model.train(was)
    return out


def train_model(model: PhysicsGRU, train: Tuple[np.ndarray, np.ndarray], val: Tuple[np.ndarray, np.ndarray],
                tcfg: Mapping[str, Any], seed: int, lr: Optional[float] = None,
                max_epochs: Optional[int] = None, patience: Optional[int] = None,
                freeze_encoder_epochs: int = 0, log: Optional[Callable[[str], None]] = None) -> TrainResult:
    """Train in place on NORMALISED features; early-stop on the val loss; restore the best epoch.

    train / val: (X (N, 12, 14) normalised float32, y (N, 2)). `tcfg` is the
    `training` section of training.yaml; lr / max_epochs / patience override it.
    Deterministic for a given seed on one thread (shuffling uses its own
    generator; dropout draws from torch's global generator, seeded by the caller).
    """
    lr = float(tcfg.get("lr", 1e-3) if lr is None else lr)
    max_epochs = int(tcfg.get("max_epochs", 50) if max_epochs is None else max_epochs)
    patience = int(tcfg.get("early_stopping_patience", 7) if patience is None else patience)
    batch = int(tcfg.get("batch_size", 64))
    clip = float(tcfg.get("grad_clip_norm", 1.0))
    Xtr = torch.as_tensor(np.ascontiguousarray(train[0]), dtype=torch.float32)
    ytr_np = np.asarray(train[1], dtype=np.float32)
    ytr = torch.as_tensor(ytr_np)
    Xva = torch.as_tensor(np.ascontiguousarray(val[0]), dtype=torch.float32)
    yva = torch.as_tensor(np.asarray(val[1], dtype=np.float32))
    if len(Xtr) == 0 or len(Xva) == 0:
        raise ValueError(f"empty training ({len(Xtr)}) or validation ({len(Xva)}) set")

    pos_weight, balance = _label_balance(ytr_np, str(tcfg.get("balance", "auto")))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=None if pos_weight is None else torch.as_tensor(pos_weight))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(tcfg.get("weight_decay", 1e-4)))
    gen = torch.Generator().manual_seed(int(seed))
    sample_w = _sampler_weights(ytr_np) if balance == "sampler" else None

    history: List[Dict[str, Any]] = []
    best_loss, best_epoch, wait = float("inf"), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    stopped = False
    t_start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        frozen = epoch <= freeze_encoder_epochs
        for p in model.encoder.parameters():
            p.requires_grad_(not frozen)
        model.train()
        if sample_w is not None:
            order = torch.multinomial(sample_w, len(Xtr), replacement=True, generator=gen)
        else:
            order = torch.randperm(len(Xtr), generator=gen)
        total, grad_norms = 0.0, []
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            grad_norms.append(float(nn.utils.clip_grad_norm_(model.parameters(), clip)))
            opt.step()
            total += float(loss.detach()) * len(idx)
        train_loss = total / len(order)
        val_logits = _logits(model, Xva)
        val_loss = float(loss_fn(val_logits, yva))
        if not (np.isfinite(train_loss) and np.isfinite(val_loss)):
            raise FloatingPointError(f"training diverged at epoch {epoch} (seed {seed}): "
                                     f"train {train_loss}, val {val_loss}")
        probs = torch.sigmoid(val_logits).numpy()
        rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr,
               "grad_norm_mean": float(np.mean(grad_norms)), "encoder_frozen": frozen,
               "seconds": time.perf_counter() - t0}
        for j, name in enumerate(LABEL_NAMES):
            rec[f"val_auroc_{name}"] = M.auroc(probs[:, j], yva.numpy()[:, j])
        history.append(rec)
        if val_loss < best_loss - 1e-9:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if log:
            log(f"  epoch {epoch:3d} train {train_loss:.4f} val {val_loss:.4f}"
                + (" *" if best_epoch == epoch else ""))
        if wait >= patience:
            stopped = True
            break
    for p in model.encoder.parameters():
        p.requires_grad_(True)
    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model=model, history=history, best_epoch=best_epoch, best_val_loss=best_loss,
                       epochs_run=len(history), stopped_early=stopped,
                       pos_weight=None if pos_weight is None else [float(v) for v in pos_weight],
                       balance=balance, seconds=time.perf_counter() - t_start,
                       settings={"lr": lr, "max_epochs": max_epochs, "patience": patience, "batch_size": batch,
                                 "weight_decay": float(tcfg.get("weight_decay", 1e-4)), "grad_clip_norm": clip,
                                 "freeze_encoder_epochs": int(freeze_encoder_epochs), "seed": int(seed),
                                 "loss": "bce_with_logits", "optimizer": "adamw"})


# =============================================================================
# Calibration on val_calib
# =============================================================================


def model_logits(model: PhysicsGRU, normalizer: FeatureNormalizer, X: np.ndarray) -> np.ndarray:
    """Raw (uncalibrated) logits (N, 2) of a model on RAW features."""
    if len(X) == 0:
        return np.zeros((0, 2))
    return _logits(model, torch.from_numpy(normalizer.transform(X))).numpy().astype(np.float64)


def fit_calibration(logits: np.ndarray, fs_calib: FeatureSet, cfg: Mapping[str, Any],
                    fit_source: str) -> Dict[str, Any]:
    """Temperatures, operating-point thresholds and abstain margins from val_calib logits.

    Every count is per simulation (group), not per rendered variant. Returns
    the dict merged into model_config.yaml.
    """
    z_raw = np.asarray(logits, dtype=np.float64)
    y = fs_calib.y
    w = M.per_group_weights(fs_calib.groups)
    ccfg = cfg.get("calibration", {})
    bounds = tuple(ccfg.get("temperature_bounds", (0.5, 5.0)))
    temps, temp_fit = {}, {}
    for j, name in enumerate(LABEL_NAMES):
        if ccfg.get("temperature_scaling", True):
            T, info = M.fit_temperature(z_raw[:, j], y[:, j], bounds=bounds, weights=w)
        else:
            T, info = 1.0, {"fallback": True, "reason": "disabled"}
        temps[name] = float(T)
        temp_fit[name] = info
    z = z_raw / np.array([temps[n] for n in LABEL_NAMES])
    ops = M.fit_operating_points(z, y, fs_calib.categories, fs_calib.groups, cfg)

    acfg = cfg.get("abstain", {})
    margins: Dict[str, Dict[str, float]] = {}
    margin_fit: Dict[str, Dict[str, Any]] = {}
    for op, per_label in ops["thresholds"].items():
        margins[op], margin_fit[op] = {}, {}
        for j, name in enumerate(LABEL_NAMES):
            m, info = M.fit_abstain_margin(z[:, j], y[:, j], per_label[name]["logit"],
                                           floor=float(acfg.get("logit_margin_floor", 0.25)),
                                           target_error=float(acfg.get("target_error", 0.05)),
                                           max_rate=float(acfg.get("max_rate", 0.3)), weights=w)
            margins[op][name] = m
            margin_fit[op][name] = info

    min_sims = int(cfg.get("thresholds", {}).get("min_calibration_sims_per_category", 20))
    sims_per_cat = {c: int(len(np.unique(fs_calib.groups[fs_calib.categories == c]))) for c in CATEGORIES}
    insufficient = [c for c, n in sims_per_cat.items() if n < min_sims]
    prevalence = {name: {"per_sequence": float(y[:, j].mean()) if len(y) else None,
                         "per_simulation": float(np.sum(w * y[:, j]) / np.sum(w)) if len(y) else None}
                  for j, name in enumerate(LABEL_NAMES)}
    return {
        "temperatures": temps,
        "thresholds": ops["thresholds"],
        "abstain": {
            "margins": margins,
            "margin_space": "calibrated_logit",
            "input_validity": {
                "min_mask_frames": int(acfg.get("min_mask_frames", 3)),
                "min_flow_frames": int(acfg.get("min_flow_frames", 2)),
                "max_clip_fraction": float(acfg.get("max_clip_fraction", 0.2)),
                "incompatible_frame_interval": True,
            },
        },
        "calibration": {
            "fit_source": fit_source,
            "split": "val_calib",
            "weighting": "per_simulation",
            "n_sequences": int(len(y)),
            "n_groups": int(len(np.unique(fs_calib.groups))),
            "groups_per_category": sims_per_cat,
            "min_groups_per_category": min_sims,
            "insufficient_calibration_data": bool(insufficient),
            "insufficient_categories": insufficient,
            "prevalence": prevalence,
            "temperature_fit": temp_fit,
            # T = 1 instead of a fit ("separable", "single_class", "disabled") or None
            "temperature_fallback": {n: temp_fit[n].get("reason") if temp_fit[n].get("fallback") else None
                                     for n in LABEL_NAMES},
            # T clipped to a bound ("optimum_at_upper_bound" / "optimum_at_lower_bound") or None
            "temperature_clipped": {n: temp_fit[n].get("reason") if temp_fit[n].get("clipped") else None
                                    for n in LABEL_NAMES},
            "threshold_fit": ops["fit"],
            "abstain_fit": margin_fit,
        },
    }


# =============================================================================
# Multi-seed runs
# =============================================================================


def _counts(fs: FeatureSet) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for split in sorted(set(fs.splits)):
        sub = fs.split(split)
        out[split] = {"sequences": len(sub), "groups": int(len(np.unique(sub.groups))),
                      "originals": int(len(np.unique(sub.sim_ids))),
                      "per_category": {c: int(np.sum(sub.categories == c)) for c in CATEGORIES}}
    return out


def _preprocessing_record(fs: FeatureSet, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """What inference must reproduce: extraction timing and the flow mode the model saw."""
    flow_modes: Dict[str, int] = {}
    roi_scales = []
    for m in fs.meta:
        mode = (m.get("feature_meta") or {}).get("flow_mode")
        if mode:
            flow_modes[mode] = flow_modes.get(mode, 0) + 1
        if "roi_scale" in m:
            roi_scales.append(float(m["roi_scale"]))
    fcfg = dict(cfg.get("features", {}) or {})
    fcfg.setdefault("expected_interval_s", 5.0)
    fcfg.setdefault("flow_lag_s", 1.0)
    return {
        "feature_config": fcfg,
        "training_flow_modes": flow_modes,
        "training_roi_scale_range": [min(roi_scales), max(roi_scales)] if roi_scales else None,
        "training_data_source": fs.source,
        "training_mask_sources": {s: int(np.sum(fs.mask_sources == s)) for s in sorted(set(fs.mask_sources))},
    }


def _write_history(path_base: Path, history: List[Dict[str, Any]]) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    write_json(path_base.with_suffix(".json"), history)
    keys = list(history[0]) if history else []
    with open(path_base.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in history:
            w.writerow({k: ("" if row[k] is None else row[k]) for k in keys})


def _plot_training_curves(histories: Dict[int, List[Dict[str, Any]]], chosen: int, path: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, (seed, hist) in enumerate(sorted(histories.items())):
        c = colors[k % len(colors)]
        ep = [h["epoch"] for h in hist]
        lw = 2.0 if seed == chosen else 1.0
        axes[0].plot(ep, [h["train_loss"] for h in hist], color=c, lw=lw, ls="--", label=f"seed {seed} train")
        axes[0].plot(ep, [h["val_loss"] for h in hist], color=c, lw=lw, label=f"seed {seed} val_select")
        for ax, name in ((axes[1], "smoke"), (axes[2], "fog")):
            vals = [np.nan if h.get(f"val_auroc_{name}") is None else h[f"val_auroc_{name}"] for h in hist]
            ax.plot(ep, vals, color=c, lw=lw, label=f"seed {seed}")
        best = min(hist, key=lambda h: h["val_loss"])["epoch"]
        axes[0].axvline(best, color=c, lw=0.6, ls=":")
    axes[0].set_title("BCE loss (dashed: train, solid: val_select)")
    axes[1].set_title("val_select AUROC, smoke")
    axes[2].set_title("val_select AUROC, fog")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    fig.suptitle(f"{title} (exported: seed {chosen}, thick lines)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _run_seeds(fs: FeatureSet, cfg: Mapping[str, Any], out_dir: PathLike, *, stage: str, experiment: str,
               make_model: Callable[[int], PhysicsGRU], normalizer: FeatureNormalizer, lr: float,
               max_epochs: int, patience: int, freeze_encoder_epochs: int, fit_source: str,
               provenance: Mapping[str, Any], log: Optional[Callable[[str], None]]) -> Dict[str, Any]:
    out = Path(out_dir)
    logs = out / "logs"
    train, vsel, vcal = fs.split("train"), fs.split("val_select"), fs.split("val_calib")
    for name, part in (("train", train), ("val_select", vsel), ("val_calib", vcal)):
        if len(part) == 0:
            raise ValueError(f"the FeatureSet has no {name} sequences")
    Xtr, Xvs = normalizer.transform(train.X), normalizer.transform(vsel.X)
    tcfg = dict(cfg.get("training", {}))
    seeds = [int(s) for s in tcfg.get("seeds", [0, 1, 2])]
    preprocessing = _preprocessing_record(train, cfg)  # what the model was fitted on (train split only)
    counts = _counts(fs)
    results: Dict[int, Dict[str, Any]] = {}
    histories: Dict[int, List[Dict[str, Any]]] = {}
    models: Dict[int, PhysicsGRU] = {}
    calibs: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        if log:
            log(f"[{experiment}] seed {seed}: {len(train)} train / {len(vsel)} val_select sequences")
        seed_everything(seed)
        model = make_model(seed)
        res = train_model(model, (Xtr, train.y), (Xvs, vsel.y), tcfg, seed, lr=lr, max_epochs=max_epochs,
                          patience=patience, freeze_encoder_epochs=freeze_encoder_epochs, log=log)
        calib = fit_calibration(model_logits(res.model, normalizer, vcal.X), vcal, cfg, fit_source)
        histories[seed], models[seed], calibs[seed] = res.history, res.model, calib
        results[seed] = {"best_epoch": res.best_epoch, "best_val_select_loss": res.best_val_loss,
                         "epochs_run": res.epochs_run, "stopped_early": res.stopped_early,
                         "pos_weight": res.pos_weight, "balance": res.balance, "train_seconds": res.seconds,
                         "settings": res.settings}
        _write_history(logs / f"history_seed{seed}", res.history)
    chosen = min(seeds, key=lambda s: (results[s]["best_val_select_loss"], seeds.index(s)))

    def extra(seed: int, exported: bool) -> Dict[str, Any]:
        doc = copy.deepcopy(calibs[seed])
        doc["preprocessing"] = preprocessing
        doc["operating_points"] = ["balanced", "low_false_alarm"]
        doc["fit"] = {
            "stage": stage, "experiment": experiment, "seed": seed, "seeds": seeds,
            "exported_seed": chosen, "is_exported_model": exported,
            "selection": "lowest val_select loss (early-stopping split) among seeds",
            "val_select_loss_per_seed": {str(s): results[s]["best_val_select_loss"] for s in seeds},
            **{k: results[seed][k] for k in ("best_epoch", "epochs_run", "stopped_early", "pos_weight", "balance")},
            "training": results[seed]["settings"],
            "normalizer_fit": "train split only",
            "data_source": fs.source,
            "data_counts": counts,
            "torch_threads": int(torch.get_num_threads()),
            **dict(provenance),
        }
        return doc

    seed_dirs = {}
    for seed in seeds:
        seed_dirs[seed] = out / "seeds" / f"seed_{seed}"
        save_model_artifacts(models[seed], seed_dirs[seed], normalizer, extra(seed, seed == chosen))
    paths = save_model_artifacts(models[chosen], out, normalizer, extra(chosen, True))
    curves = _plot_training_curves(histories, chosen, logs / "training_curves.png",
                                   f"{experiment}: {stage} ({fs.source})")
    with open(logs / "seeds.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "best_epoch", "epochs_run", "stopped_early", "best_val_select_loss",
                    "train_seconds", "temperature_smoke", "temperature_fog", "exported"])
        for s in seeds:
            r = results[s]
            w.writerow([s, r["best_epoch"], r["epochs_run"], r["stopped_early"], f"{r['best_val_select_loss']:.6f}",
                        f"{r['train_seconds']:.2f}", calibs[s]["temperatures"]["smoke"],
                        calibs[s]["temperatures"]["fog"], s == chosen])
    summary = {
        "stage": stage, "experiment": experiment, "data_source": fs.source,
        "exported_seed": chosen, "seeds": {str(s): results[s] for s in seeds},
        "artifacts": {k: str(v) for k, v in paths.items()},
        "seed_artifacts": {str(s): str(d) for s, d in seed_dirs.items()},
        "training_curves": str(curves),
        "calibration": {k: calibs[chosen][k] for k in ("temperatures", "thresholds")},
        "abstain_margins": calibs[chosen]["abstain"]["margins"],
        "insufficient_calibration_data": calibs[chosen]["calibration"]["insufficient_calibration_data"],
        "data_counts": counts,
        "provenance": dict(provenance),
    }
    write_json(logs / "training_summary.json", summary)
    return summary


def run_pretraining(fs: FeatureSet, cfg: Mapping[str, Any], out_dir: PathLike,
                    log: Optional[Callable[[str], None]] = print) -> Dict[str, Any]:
    """Stage A / experiment C on a synthetic FeatureSet; writes the artifacts into out_dir."""
    assert_no_group_leakage(fs)
    normalizer = FeatureNormalizer(clip=float(cfg.get("normalization", {}).get("clip", 5.0)))
    normalizer.fit(fs.split("train").X)
    mcfg = ModelConfig.from_dict(cfg.get("model", {}))
    tcfg = cfg.get("training", {})
    return _run_seeds(
        fs, cfg, out_dir, stage="A_cfd_pretraining", experiment="C",
        make_model=lambda seed: PhysicsGRU(mcfg), normalizer=normalizer,
        lr=float(tcfg.get("lr", 1e-3)), max_epochs=int(tcfg.get("max_epochs", 50)),
        patience=int(tcfg.get("early_stopping_patience", 7)), freeze_encoder_epochs=0,
        fit_source=f"{fs.source}_val_calib", provenance={"init": "random"}, log=log)


def run_finetune(features: FeatureSet, cfg: Mapping[str, Any], init_artifacts: Optional[PathLike],
                 out_dir: PathLike, from_scratch: bool = False,
                 log: Optional[Callable[[str], None]] = print) -> Dict[str, Any]:
    """Stage B on (real) labelled footage: D = fine-tune the pretrained GRU, B = same data from scratch.

    Groups must be camera ids (camera-disjoint splits). Identical for B and D:
    splits, seeds, max epochs and patience (`finetune` section), the
    normaliser policy (refitted on this set's train split) and the
    calibration procedure; only the initial weights and the learning rate
    (B: training.lr, D: finetune.lr) differ.
    """
    fs = features
    assert_no_group_leakage(fs)
    ft = dict(cfg.get("finetune", {}))
    tcfg = dict(cfg.get("training", {}))
    normalizer = FeatureNormalizer(clip=float(cfg.get("normalization", {}).get("clip", 5.0)))
    normalizer.fit(fs.split("train").X)
    init_cfg: Dict[str, Any] = {}
    if init_artifacts is not None:
        _, _, init_cfg = load_model(init_artifacts)
    if not from_scratch and init_artifacts is None:
        raise ValueError("fine-tuning (experiment D) needs the pretrained artifacts")
    mcfg = ModelConfig.from_dict(init_cfg["model"]) if init_cfg else ModelConfig.from_dict(cfg.get("model", {}))

    def make_model(seed: int) -> PhysicsGRU:
        if from_scratch:
            return PhysicsGRU(mcfg)
        model, _, _ = load_model(init_artifacts)
        model.train()
        return model

    sources = {}
    for split in sorted(set(fs.splits)):
        sub = fs.split(split)
        sources[split] = {"sequences": len(sub), "videos": int(len(np.unique(sub.sim_ids))),
                          "cameras": int(len(np.unique(sub.groups)))}
    provenance = {
        "init": "random" if from_scratch else f"pretrained:{Path(init_artifacts)}",
        "pretrained_fit": (init_cfg.get("fit", {}) or {}).get("stage") if init_cfg else None,
        "normalizer_policy": "refit on this FeatureSet's train split (identical for B and D)",
        "real_counts": sources,
        "groups_are": "camera ids" if fs.source == "real" else "as given",
        "not_real_world_evidence": fs.source != "real",
    }
    lr = float(tcfg.get("lr", 1e-3)) if from_scratch else float(ft.get("lr", 1e-4))
    return _run_seeds(
        fs, cfg, out_dir, stage="B_real_finetuning", experiment="B" if from_scratch else "D",
        make_model=make_model, normalizer=normalizer, lr=lr,
        max_epochs=int(ft.get("max_epochs", tcfg.get("max_epochs", 50))),
        patience=int(ft.get("early_stopping_patience", tcfg.get("early_stopping_patience", 7))),
        freeze_encoder_epochs=0 if from_scratch else int(ft.get("freeze_encoder_epochs", 0)),
        fit_source=f"{fs.source}_val_calib", provenance=provenance, log=log)


# =============================================================================
# Leave-one-regime-out
# =============================================================================


def _regime_parts(regime: str) -> List[str]:
    return [p for p in str(regime).split("+") if p]


def run_loro(fs: FeatureSet, cfg: Mapping[str, Any], regimes: Optional[Sequence[str]] = None,
             seed: Optional[int] = None, eval_split: str = "test_iid",
             log: Optional[Callable[[str], None]] = print) -> Dict[str, Any]:
    """Leave-one-regime-out: for each regime, train without it (train / val_select / val_calib) and
    measure, on `eval_split`, how its held-out sequences are classified.

    A regime whose sequences are positive for a label reports that label's
    recall and the AUROC of the held-out positives against the split's
    negatives; a regime negative for a label reports the false-positive rate
    and the AUROC of the split's positives against the held-out sequences.
    One seed (the first in training.seeds by default); balanced thresholds
    refitted on the reduced val_calib.
    """
    tcfg = dict(cfg.get("training", {}))
    seed = int(tcfg.get("seeds", [0])[0] if seed is None else seed)
    parts = [_regime_parts(r) for r in fs.regimes]
    all_regimes = sorted({p for ps in parts for p in ps})
    regimes = list(regimes) if regimes is not None else all_regimes
    mcfg = ModelConfig.from_dict(cfg.get("model", {}))
    out: Dict[str, Any] = {"seed": seed, "eval_split": eval_split, "regimes": {}}
    for regime in regimes:
        held = np.array([regime in ps for ps in parts])
        reduced = fs.subset(~held | ~np.isin(fs.splits, ["train", "val_select", "val_calib"]))
        train, vsel, vcal = reduced.split("train"), reduced.split("val_select"), reduced.split("val_calib")
        ev = fs.split(eval_split)
        ev_held = np.array([regime in _regime_parts(r) for r in ev.regimes])
        if ev_held.sum() == 0 or len(train) == 0 or len(vsel) == 0 or len(vcal) == 0:
            out["regimes"][regime] = {"status": "skipped", "n_eval_held_out": int(ev_held.sum())}
            continue
        if log:
            log(f"[LORO] without {regime}: {len(train)} train, {int(ev_held.sum())} held-out {eval_split}")
        norm = FeatureNormalizer(clip=float(cfg.get("normalization", {}).get("clip", 5.0))).fit(train.X)
        seed_everything(seed)
        model = PhysicsGRU(mcfg)
        res = train_model(model, (norm.transform(train.X), train.y), (norm.transform(vsel.X), vsel.y),
                          tcfg, seed, log=None)
        calib = fit_calibration(model_logits(res.model, norm, vcal.X), vcal, cfg, "synthetic_val_calib_loro")
        z = model_logits(res.model, norm, ev.X) / np.array([calib["temperatures"][n] for n in LABEL_NAMES])
        rec: Dict[str, Any] = {"n_train": len(train), "n_eval_held_out": int(ev_held.sum()),
                               "best_epoch": res.best_epoch, "labels": {}}
        for j, name in enumerate(LABEL_NAMES):
            thr = calib["thresholds"]["balanced"][name]["logit"]
            yh = ev.y[ev_held, j] > 0.5
            others = ~ev_held
            if yh.all():
                negs = others & (ev.y[:, j] < 0.5)
                rec["labels"][name] = {
                    "role": "positive", "recall": float(np.mean(z[ev_held, j] >= thr)),
                    "auroc_vs_split_negatives": M.auroc(np.r_[z[ev_held, j], z[negs, j]],
                                                        np.r_[np.ones(ev_held.sum()), np.zeros(negs.sum())])}
            elif (~yh).all():
                poss = others & (ev.y[:, j] > 0.5)
                rec["labels"][name] = {
                    "role": "negative", "false_positive_rate": float(np.mean(z[ev_held, j] >= thr)),
                    "auroc_split_positives_vs_held_out": M.auroc(
                        np.r_[z[poss, j], z[ev_held, j]], np.r_[np.ones(poss.sum()), np.zeros(ev_held.sum())])}
            else:
                rec["labels"][name] = {"role": "mixed", "accuracy": float(np.mean((z[ev_held, j] >= thr) == yh))}
        out["regimes"][regime] = rec
    return out
