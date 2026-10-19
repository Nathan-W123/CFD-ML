"""Single-core CPU benchmarks: simulation, rendering + features, training, inference.

Usage:
    python scripts/benchmark_cpu.py --out reports/benchmark_cpu.json [--sims-per-category 3]
        [--features data/synthetic/features.npz] [--artifacts artifacts]
    python scripts/benchmark_cpu.py --config configs/simulation_far.yaml --out reports/benchmark_cpu_far.json

Everything runs in this one process with every library pinned to one
thread (the effective thread counts are recorded). --config selects the
simulation profile (default configs/simulation.yaml); rendering and feature
extraction use its frames and feature timing. Writes JSON plus a short
Markdown table next to it.
"""

import _bootstrap  # noqa: F401

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np

from src.constants import CATEGORIES
from src.utils.config import load_simulation_config, load_training_config
from src.utils.io import write_json


def bench_simulation(cfg, per_category):
    from src.data.pipeline import simulate_slot

    out = {}
    all_times = []
    for c in CATEGORIES:
        times, dts, steps = [], [], []
        for i in range(per_category):
            t0 = time.perf_counter()
            sim, rejected = simulate_slot(c, "train", 10_000 + i, cfg, max_attempts=3)
            dt_wall = time.perf_counter() - t0
            if sim is None:
                continue
            times.append(dt_wall)
            dts.append(sim.validity["dt_used"])
            steps.append((float(sim.params["timing"]["spinup_s"]) + float(sim.times[-1])
                          + float(cfg["time"].get("precursor_s", 30.0))) / sim.validity["dt_used"])
        all_times += times
        out[c] = {"n": len(times), "mean_s": float(np.mean(times)) if times else None,
                  "max_s": float(np.max(times)) if times else None,
                  "dt_used": dts, "steps_mean": float(np.mean(steps)) if steps else None}
    mean = float(np.mean(all_times)) if all_times else None
    out["overall"] = {"mean_s_per_accepted_sim": mean, "sims_per_hour": 3600.0 / mean if mean else None}
    return out


def bench_render_features(cfg, n=4):
    from src.data.pipeline import feature_timing, render_and_extract, simulate_slot

    sim, _ = simulate_slot("smoke_only", "train", 20_000, cfg, max_attempts=3)
    if sim is None:
        return {"error": "no valid simulation"}
    times = []
    for v in range(n):
        t0 = time.perf_counter()
        render_and_extract(sim, cfg, v)
        times.append(time.perf_counter() - t0)
    from src.data.features import FeatureConfig, extract_features
    from src.data.render import render_sequence, sample_render_params
    from src.utils.runtime import child_rng

    rp = sample_render_params(child_rng(1, 1), cfg, hints=sim.params.get("render_hints", {}))
    t0 = time.perf_counter()
    rendered = render_sequence(sim, rp, child_rng(1, 2))
    t_render = time.perf_counter() - t0
    t0 = time.perf_counter()
    extract_features(rendered.frames, rendered.timestamps, rendered.masks, rendered.lag_frames,
                     rendered.lag_timestamps, FeatureConfig(**feature_timing(cfg)))
    t_feat = time.perf_counter() - t0
    return {"render_plus_features_mean_s": float(np.mean(times)), "render_s": t_render, "features_s": t_feat,
            "frame_shape": list(rendered.frames.shape), "lag_frames": rendered.lag_frames is not None}


def bench_training(tcfg, features_path, epochs=3):
    import torch

    from src.data.normalization import FeatureNormalizer
    from src.models.physics_gru import ModelConfig, PhysicsGRU

    if features_path and Path(features_path).exists():
        from src.data.dataset import FeatureSet

        fs = FeatureSet.load(features_path)
        X = fs.X[fs.splits == "train"]
        y = fs.y[fs.splits == "train"]
    else:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(2000, 12, 14)).astype(np.float32)
        X[..., 12:] = 1.0
        y = rng.integers(0, 2, size=(2000, 2)).astype(np.float32)
    norm = FeatureNormalizer().fit(X)
    Xn = torch.from_numpy(norm.transform(X))
    yt = torch.from_numpy(y)
    torch.manual_seed(0)
    model = PhysicsGRU(ModelConfig())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    bs = int(tcfg["training"]["batch_size"])
    times = []
    for _ in range(epochs):
        t0 = time.perf_counter()
        perm = torch.randperm(len(Xn))
        for i in range(0, len(Xn), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xn[idx]), yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        times.append(time.perf_counter() - t0)
    return {"n_train_sequences": int(len(Xn)), "epoch_s_mean": float(np.mean(times)),
            "parameters": int(sum(p.numel() for p in model.parameters()))}


def bench_inference(artifacts):
    import torch

    from src.inference.predict import PhysicsExpert

    out = {}
    if artifacts and (Path(artifacts) / "physics_classifier.pt").exists():
        expert = PhysicsExpert.load(artifacts)
    else:
        return {"error": f"no artifacts in {artifacts}"}
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 12, 14)).astype(np.float32)
    x[..., 12:] = 1.0
    for _ in range(5):
        expert.predict(x)
    t0 = time.perf_counter()
    for _ in range(200):
        expert.encode(x)
    out["encode_single_ms"] = (time.perf_counter() - t0) / 200 * 1e3
    t0 = time.perf_counter()
    for _ in range(200):
        expert.predict(x)
    out["predict_single_ms"] = (time.perf_counter() - t0) / 200 * 1e3
    xb = np.repeat(x, 256, axis=0)
    t0 = time.perf_counter()
    for _ in range(20):
        expert.predict(xb)
    out["predict_batch256_ms"] = (time.perf_counter() - t0) / 20 * 1e3
    out["torch_threads"] = torch.get_num_threads()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None, help="simulation YAML (default configs/simulation.yaml)")
    ap.add_argument("--out", default="reports/benchmark_cpu.json")
    ap.add_argument("--sims-per-category", type=int, default=3)
    ap.add_argument("--features", default=None)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--skip", default="", help="comma list of sections to skip: simulation,render,training,inference")
    args = ap.parse_args(argv)
    skip = set(filter(None, args.skip.split(",")))
    cfg = load_simulation_config(args.config)
    tcfg = load_training_config()
    report = {"machine": {"platform": platform.platform(), "processor": platform.processor(),
                          "python": sys.version.split()[0]},
              "threads": _bootstrap.THREADS, "config": args.config or "configs/simulation.yaml",
              "grid": cfg["grid"], "time": cfg["time"]}
    if "simulation" not in skip:
        report["simulation"] = bench_simulation(cfg, args.sims_per_category)
        print("simulation", report["simulation"]["overall"], flush=True)
    if "render" not in skip:
        report["render_features"] = bench_render_features(cfg)
        print("render+features", report["render_features"], flush=True)
    if "training" not in skip:
        report["training"] = bench_training(tcfg, args.features)
        print("training", report["training"], flush=True)
    if "inference" not in skip:
        report["inference"] = bench_inference(args.artifacts)
        print("inference", report["inference"], flush=True)
    out = Path(args.out)
    write_json(out, report)
    lines = ["| Stage | Single-core time |", "|---|---|"]
    sim = report.get("simulation", {}).get("overall", {})
    if sim.get("mean_s_per_accepted_sim"):
        lines.append(f"| CFD simulation (accepted, incl. precursor) | {sim['mean_s_per_accepted_sim']:.1f} s/sim ({sim['sims_per_hour']:.0f} sims/h) |")
    rf = report.get("render_features", {})
    if "render_s" in rf:
        n_obs = rf["frame_shape"][0]
        frames = f"{n_obs} + {n_obs} lag frames" if rf["lag_frames"] else f"{n_obs} frames (no lag frames)"
        lines.append(f"| Render {frames} | {rf['render_s'] * 1e3:.0f} ms |")
        lines.append(f"| Feature extraction ({n_obs} x 14) | {rf['features_s'] * 1e3:.0f} ms |")
    tr = report.get("training", {})
    if "epoch_s_mean" in tr:
        lines.append(f"| GRU training epoch ({tr['n_train_sequences']} sequences) | {tr['epoch_s_mean']:.2f} s |")
    inf = report.get("inference", {})
    if "predict_single_ms" in inf:
        lines.append(f"| Inference, one sequence (features -> probabilities) | {inf['predict_single_ms']:.2f} ms |")
        lines.append(f"| Inference, batch of 256 | {inf['predict_batch256_ms']:.1f} ms |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
