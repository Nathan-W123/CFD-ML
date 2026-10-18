"""Synthetic dataset generation: simulate -> render -> extract features, with rejection logging.

One "slot" is (split, category, index). For a slot the generator draws
scenario parameters with attempt key 0, runs the CFD, and on a numerical or
label-support rejection re-draws with attempt 1, 2, ... (logging every
rejected attempt with its parameters, so acceptance can be audited per
regime and per parameter bin). An accepted simulation is rendered into
several camera variants; a variant failing the label-visibility check is
re-rendered with a fresh render seed (also logged). Every random stream is
derived from the slot's seed, so the output does not depend on how many
worker processes ran or in which order.

Where this fits: `scripts/generate_dataset.py` (orchestration, CLI) calls
`process_slot`; `scripts/evaluate_physics.py` calls `render_and_extract` to
re-render test simulations for the robustness suites.
Calls into: src.cfd.scenarios, src.data.render, src.data.features, runtime seeding.
Start reading at: process_slot.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from src.cfd.results import SimulationResult
from src.cfd.scenarios import run_scenario, sample_scenario
from src.cfd.simulation import SimulationUnstable
from src.utils.runtime import STREAM_RENDER_PARAMS, child_rng, derive_seed

ROI_SCALE_RANGE = (1.5, 2.5)   # synthetic ROI scale; inference uses the fixed FeatureConfig default
_CFD_DIR = Path(__file__).resolve().parents[1] / "cfd"


def cfd_fingerprint(cfg: Mapping[str, Any]) -> str:
    """Hash of the CFD source code and every config section that affects a simulation.

    Cached simulations are reused only when this matches, so re-rendering
    after a renderer change never mixes in simulations from older physics.
    """
    h = hashlib.sha256()
    for path in sorted(_CFD_DIR.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    sim_cfg = {k: v for k, v in cfg.items() if k not in ("render", "robustness", "dataset")}
    sim_cfg["view_margin_cells"] = cfg.get("render", {}).get("view_margin_cells")
    h.update(json.dumps(sim_cfg, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def split_counts(n: int, fractions: Mapping[str, float]) -> Dict[str, int]:
    """Integer counts per split summing to n (largest-remainder rounding)."""
    names = list(fractions)
    total = float(sum(fractions.values()))
    raw = [n * fractions[k] / total for k in names]
    base = [int(math.floor(r)) for r in raw]
    order = sorted(range(len(names)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[: n - sum(base)]:
        base[i] += 1
    return dict(zip(names, base))


def context_summary(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Flat, class-independent context of a scenario (for rejection logs and shortcut audits)."""
    hints = params.get("render_hints", {})
    return {
        "wind_speed_ms": float(params["wind"]["speed_ms"]),
        "wind_sign": float(params["wind"]["sign"]),
        "gust_fraction": float(params["wind"]["gust_fraction"]),
        "terrain_kind": params["terrain"]["kind"],
        "terrain_height_m": float(params["terrain"].get("height_m", 0.0)),
        "spinup_s": float(params["timing"]["spinup_s"]),
        "background_haze": params.get("background_haze") is not None,
        "nu_m2s": float(params["mixing"]["nu_m2s"]),
        "diffusivity_m2s": float(params["mixing"]["diffusivity_m2s"]),
        "theta_ground_k": float(params["atmosphere"]["theta_ground_k"]),
        "lapse_dtheta_dz": float(params["atmosphere"]["lapse_dtheta_dz"]),
        "hint_shadow": bool(hints.get("shadow", False)),
        "hint_strong_artifact": bool(hints.get("strong_artifact", False)),
        "hint_fp_boost": bool(hints.get("fp_boost", False)),
        "hint_distant_cloud": bool(hints.get("distant_cloud", False)),
    }


def render_summary(rp: Mapping[str, Any]) -> Dict[str, Any]:
    """Scalar render parameters worth auditing for label shortcuts."""
    out = {}
    for key, value in rp.items():
        if isinstance(value, (bool, int, float, str)) and not key.endswith("seed"):
            out[f"render_{key}"] = value
    return out


def simulate_slot(category: str, split: str, index: int, cfg: Dict[str, Any],
                  max_attempts: int) -> Tuple[Optional[SimulationResult], List[Dict[str, Any]]]:
    """Run attempts 0, 1, ... until one simulation is valid; return it and the rejection log."""
    rejected: List[Dict[str, Any]] = []
    for attempt in range(max_attempts):
        params = sample_scenario(category, split, index, cfg, attempt=attempt)
        t0 = time.perf_counter()
        try:
            sim = run_scenario(params, cfg)
        except SimulationUnstable as exc:
            rejected.append(_rejection(params, "simulation", [f"unstable: {exc}"], time.perf_counter() - t0))
            continue
        if not sim.valid:
            rejected.append(_rejection(params, "simulation", sim.validity["reasons"], time.perf_counter() - t0))
            continue
        sim.extra["runtime_s"] = time.perf_counter() - t0
        return sim, rejected
    return None, rejected


def _rejection(params, stage, reasons, runtime) -> Dict[str, Any]:
    return {
        "stage": stage, "sim_id": params["sim_id"], "split": params["split"],
        "category": params["category"], "regime": params["regime"],
        "smoke_regime": params.get("smoke_regime"), "fog_regime": params.get("fog_regime"),
        "distractor_regime": params.get("distractor_regime"), "attempt": params["attempt"],
        "holdout_tags": ";".join(params.get("holdout_tags", [])), "reasons": " | ".join(reasons),
        "runtime_s": round(runtime, 3), **context_summary(params),
    }


def render_and_extract(sim: SimulationResult, cfg: Dict[str, Any], variant: int, suite: str = "nominal",
                       suite_index: int = 0, overrides: Optional[Dict[str, Any]] = None,
                       max_retries: int = 4, feature_cfg=None):
    """Render one camera variant of `sim` (re-drawing on label-visibility failure) and extract features.

    Returns (features (12, 14), info dict, rendered sequence or None, list of rejection dicts).
    """
    from src.data.features import FeatureConfig, extract_features_with_meta
    from src.data.render import check_label_visibility, render_sequence, sample_render_params

    rejected = []
    for retry in range(max_retries):
        render_seed = derive_seed(sim.seed, STREAM_RENDER_PARAMS, suite_index + 1, variant + 1, retry)
        rng_params = child_rng(render_seed, 1)
        rng_noise = child_rng(render_seed, 2)
        rp = sample_render_params(rng_params, cfg, hints=sim.params.get("render_hints", {}), overrides=overrides)
        rendered = render_sequence(sim, rp, rng_noise)
        ok, reasons = check_label_visibility(rendered, tuple(sim.labels), cfg)
        if not ok:
            rejected.append({"stage": "render", "sim_id": sim.sim_id, "split": sim.params.get("split"),
                             "category": sim.category, "regime": sim.regime, "variant": variant,
                             "retry": retry, "suite": suite, "reasons": " | ".join(reasons),
                             **context_summary(sim.params), **render_summary(rp)})
            continue
        if feature_cfg is None:
            roi_scale = float(child_rng(render_seed, 3).uniform(*ROI_SCALE_RANGE))
            fcfg = FeatureConfig(roi_scale=roi_scale,
                                 flow_lag_s=float(cfg.get("time", {}).get("flow_lag_s", 1.0)),
                                 expected_interval_s=float(cfg.get("time", {}).get("frame_interval_s", 5.0)))
        else:
            fcfg = feature_cfg
        feats, fmeta = extract_features_with_meta(
            rendered.frames, rendered.timestamps, rendered.masks,
            rendered.lag_frames, rendered.lag_timestamps, fcfg)
        info = {
            "render_seed": int(render_seed), "variant": int(variant), "retry": int(retry), "suite": suite,
            "mask_source": rendered.mask_source, "render_params": rp,
            "feature_config": asdict(fcfg), "feature_meta": fmeta,
            "visibility": rendered.visibility,
        }
        return feats, info, rendered, rejected
    return None, None, None, rejected


def cached_simulation(out_dir: Path, category: str, split: str, index: int, cfg: Dict[str, Any],
                      diagnostics: bool = True):
    """simulate_slot with an on-disk cache (slots/<split>-<category>-<index>.json + sims/...)."""
    slot_path = out_dir / "slots" / f"{split}-{category}-{index:04d}.json"
    fp = cfd_fingerprint(cfg)
    if slot_path.exists():
        rec = json.loads(slot_path.read_text())
        if rec.get("fingerprint") == fp:
            if rec["sim_id"] is None:
                return None, rec["rejected"], None
            sim_path = out_dir / rec["sim_path"]
            if sim_path.exists():
                return SimulationResult.load(sim_path), rec["rejected"], sim_path
    sim, rejected = simulate_slot(category, split, index, cfg, int(cfg.get("dataset", {}).get("max_attempts", 6)))
    sim_path = None
    if sim is not None:
        sim_path = out_dir / "sims" / split / f"{sim.sim_id}.npz"
        sim.save(sim_path, diagnostics=diagnostics)
    slot_path.parent.mkdir(parents=True, exist_ok=True)
    from src.utils.io import dumps_json

    slot_path.write_text(dumps_json({
        "fingerprint": fp, "sim_id": None if sim is None else sim.sim_id,
        "sim_path": None if sim_path is None else str(sim_path.relative_to(out_dir)),
        "rejected": rejected,
    }))
    return sim, rejected, sim_path


def process_slot(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Simulate one slot, render its variants, extract features; write sim and sequences to disk."""
    cfg = task["cfg"]
    out_dir = Path(task["out_dir"])
    category, split, index = task["category"], task["split"], int(task["index"])
    ds = cfg.get("dataset", {})
    sim, rejected, sim_path = cached_simulation(out_dir, category, split, index, cfg,
                                                bool(ds.get("store_diagnostics", True)))
    result: Dict[str, Any] = {"slot": (split, category, index), "rejected": list(rejected), "rows": [],
                              "sim_id": None, "sim_runtime_s": None}
    if sim is None or task.get("sims_only"):
        result["sim_id"] = None if sim is None else sim.sim_id
        return result
    from src.data.dataset import save_sequence

    result["sim_id"] = sim.sim_id
    result["sim_runtime_s"] = sim.extra.get("runtime_s")
    n_var = int(ds.get("variants", {}).get(split, 1))
    for v in range(n_var):
        t0 = time.perf_counter()
        feats, info, rendered, rej = render_and_extract(
            sim, cfg, v, max_retries=int(ds.get("max_variant_retries", 4)))
        result["rejected"].extend(rej)
        if feats is None:
            continue
        seq_id = f"{sim.sim_id}-v{v}"
        seq_path = out_dir / "seqs" / split / f"{seq_id}.npz"
        keep_diag = ds.get("store_render_diagnostics", "subset") == "all" or (
            ds.get("store_render_diagnostics", "subset") == "subset" and (split != "train" or v == 0 and index < 2))
        save_sequence(seq_path, rendered, sim, seq_id, extra={"render_info": info, "diagnostics": keep_diag})
        result["rows"].append({
            "seq_id": seq_id, "sim_id": sim.sim_id, "split": split, "category": category,
            "labels": list(sim.labels), "regime": sim.regime,
            "smoke_regime": sim.params.get("smoke_regime"), "fog_regime": sim.params.get("fog_regime"),
            "distractor_regime": sim.params.get("distractor_regime"),
            "holdout_tags": list(sim.params.get("holdout_tags", [])),
            "seed": int(sim.seed), "attempt": int(sim.params.get("attempt", 0)),
            "features": feats, "info": info, "context": context_summary(sim.params),
            "render": render_summary(info["render_params"]),
            "sim_path": str(sim_path.relative_to(out_dir)), "seq_path": str(seq_path.relative_to(out_dir)),
            "render_runtime_s": time.perf_counter() - t0,
        })
    return result
