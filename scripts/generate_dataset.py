"""Generate the synthetic CFD dataset: simulations, rendered sequences, features, manifests.

Usage:
    python scripts/generate_dataset.py --out data/synthetic_debug --n-per-category 50
    python scripts/generate_dataset.py --out data/synthetic --n-per-category 250 --ood-per-category 50
    python scripts/generate_dataset.py ... --set render.zoom=[0.5,1.0] --workers 3

Output layout (under --out):
    config_used.yaml      the merged configuration (with its sha256 in summary.json)
    sims/<split>/*.npz    SimulationResult per accepted simulation (fields, texcoords, validity)
    seqs/<split>/*.npz    rendered sequences: 12 frames + 12 lag frames, masks, timestamps,
                          labels, parameters, seeds, validity metadata
    features.npz          FeatureSet (12 x 14 raw features per sequence + provenance)
    manifest.csv          one row per sequence
    rejected.csv          every rejected simulation attempt and render retry, with parameters
    summary.json          counts, acceptance per category / regime / wind bin, timing

One process is the reference (the system targets one CPU core); --workers N
runs N single-threaded processes over independent slots for throughput only:
the output is identical for any N because every random stream derives from
the slot's (split, category, index, attempt) seed.
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.cfd.scenarios import SPLITS
from src.constants import CATEGORIES
from src.data.pipeline import process_slot, split_counts
from src.utils.config import load_simulation_config
from src.utils.io import dumps_json, write_json, write_yaml


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None, help="simulation YAML (default configs/simulation.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override (repeatable)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n-per-category", type=int, default=None, help="independent simulations per category (train+val+test_iid)")
    ap.add_argument("--ood-per-category", type=int, default=None, help="extra test_ood simulations per category")
    ap.add_argument("--splits", default=",".join(SPLITS), help="comma-separated subset of splits to generate")
    ap.add_argument("--workers", type=int, default=1, help="worker processes (throughput only; default 1)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N slots (smoke test)")
    ap.add_argument("--sims-only", action="store_true", help="only run (and cache) the CFD simulations")
    ap.add_argument("--examples", default=None, help="write example figures (one per category and per regime) here")
    return ap.parse_args(argv)


def write_examples(out, rows, fig_dir):
    """One figure per category (first train sequence) and per regime, for inspection."""
    from src.cfd.results import SimulationResult
    from src.data.dataset import load_sequence
    from src.data.visualize import sequence_figure

    fig_dir = Path(fig_dir)
    done_cat, done_regime = set(), set()
    for r in sorted(rows, key=lambda r: (r["split"] != "train", r["seq_id"])):
        key_r = r["regime"]
        if r["category"] in done_cat and key_r in done_regime:
            continue
        seq = load_sequence(out / r["seq_path"])
        sim = SimulationResult.load(out / r["sim_path"])
        title = (f"{r['category']} | {r['regime']} | wind {r['context']['wind_speed_ms']:.1f} m/s | "
                 f"terrain {r['context']['terrain_kind']} | mask {r['info']['mask_source']} | {r['seq_id']}")
        if r["category"] not in done_cat:
            sequence_figure(seq, sim, fig_dir / f"examples_{r['category']}.png", title)
            done_cat.add(r["category"])
        if key_r not in done_regime:
            sequence_figure(seq, sim, fig_dir / "regimes" / f"{r['category']}__{key_r}.png", title)
            done_regime.add(key_r)


def build_tasks(cfg, out_dir, splits):
    ds = cfg.get("dataset", {})
    n = int(ds.get("n_per_category", 50))
    counts = split_counts(n, ds.get("split_fractions", {"train": 0.6, "val_select": 0.1, "val_calib": 0.15, "test_iid": 0.15}))
    counts["test_ood"] = int(ds.get("ood_per_category", 25))
    tasks = []
    for split in SPLITS:
        if split not in splits:
            continue
        for category in CATEGORIES:
            for index in range(counts.get(split, 0)):
                tasks.append({"cfg": cfg, "out_dir": str(out_dir), "split": split, "category": category, "index": index})
    return tasks, counts


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def wind_bin(speed):
    return f"{int(min(speed, 8.99) // 1.5) * 1.5:.1f}-{int(min(speed, 8.99) // 1.5) * 1.5 + 1.5:.1f}"


def main(argv=None):
    args = parse_args(argv)
    overrides = list(args.set)
    if args.n_per_category is not None:
        overrides.append(f"dataset.n_per_category={args.n_per_category}")
    if args.ood_per_category is not None:
        overrides.append(f"dataset.ood_per_category={args.ood_per_category}")
    cfg = load_simulation_config(args.config, overrides)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_yaml(out / "config_used.yaml", cfg)
    cfg_sha = hashlib.sha256(dumps_json(cfg).encode()).hexdigest()
    splits = set(args.splits.split(","))
    tasks, counts = build_tasks(cfg, out, splits)
    if args.limit:
        tasks = tasks[: args.limit]
    for t in tasks:
        t["sims_only"] = args.sims_only
    print(f"{len(tasks)} slots ({counts}), workers={args.workers}", flush=True)

    t0 = time.perf_counter()
    results = []
    if args.workers > 1:
        with mp.get_context("fork").Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(process_slot, tasks, chunksize=1)):
                results.append(res)
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{len(tasks)} slots, {time.perf_counter() - t0:.0f} s", flush=True)
    else:
        for i, task in enumerate(tasks):
            results.append(process_slot(task))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(tasks)} slots, {time.perf_counter() - t0:.0f} s", flush=True)
    wall = time.perf_counter() - t0
    results.sort(key=lambda r: (SPLITS.index(r["slot"][0]), CATEGORIES.index(r["slot"][1]), r["slot"][2]))
    if args.sims_only:
        n_ok = sum(1 for r in results if r["sim_id"])
        print(f"simulations cached: {n_ok}/{len(results)} slots in {wall:.0f} s", flush=True)
        return 0

    from src.data.dataset import FeatureSet

    rows = [row for r in results for row in r["rows"]]
    rejected = [rej for r in results for rej in r["rejected"]]
    if not rows:
        print("no sequences were accepted", file=sys.stderr)
        return 1
    X = np.stack([row["features"] for row in rows]).astype(np.float32)
    y = np.array([row["labels"] for row in rows], dtype=np.float32)
    fs = FeatureSet(
        X=X, y=y,
        seq_ids=np.array([r["seq_id"] for r in rows]),
        sim_ids=np.array([r["sim_id"] for r in rows]),
        groups=np.array([r["sim_id"] for r in rows]),
        splits=np.array([r["split"] for r in rows]),
        categories=np.array([r["category"] for r in rows]),
        regimes=np.array([r["regime"] for r in rows]),
        tags=np.array([";".join(r["holdout_tags"]) for r in rows]),
        mask_sources=np.array([r["info"]["mask_source"] for r in rows]),
        meta=[{
            "seed": r["seed"], "attempt": r["attempt"], "smoke_regime": r["smoke_regime"],
            "fog_regime": r["fog_regime"], "distractor_regime": r["distractor_regime"],
            "render_seed": r["info"]["render_seed"], "variant": r["info"]["variant"],
            "context": r["context"], "render": r["render"],
            "feature_meta": {k: v for k, v in r["info"]["feature_meta"].items() if k in ("roi", "flow_mode", "stabilization_used", "n_mask_valid", "n_flow_valid")},
            "roi_scale": r["info"]["feature_config"]["roi_scale"],
            "sim_path": r["sim_path"], "seq_path": r["seq_path"],
        } for r in rows],
        source="synthetic",
    )
    fs.save(out / "features.npz")

    manifest = [{
        "seq_id": r["seq_id"], "sim_id": r["sim_id"], "split": r["split"], "category": r["category"],
        "smoke": r["labels"][0], "fog": r["labels"][1], "regime": r["regime"],
        "smoke_regime": r["smoke_regime"], "fog_regime": r["fog_regime"], "distractor_regime": r["distractor_regime"],
        "holdout_tags": ";".join(r["holdout_tags"]), "variant": r["info"]["variant"],
        "render_seed": r["info"]["render_seed"], "mask_source": r["info"]["mask_source"],
        "seed": r["seed"], "attempt": r["attempt"], "wind_speed_ms": r["context"]["wind_speed_ms"],
        "seq_path": r["seq_path"], "sim_path": r["sim_path"],
    } for r in rows]
    write_csv(out / "manifest.csv", manifest)
    write_csv(out / "rejected.csv", rejected)

    # Acceptance per category / regime / wind bin (attempts = accepted + rejected simulations).
    sim_rows = {}
    for r in rows:
        sim_rows.setdefault(r["sim_id"], r)
    acc = defaultdict(lambda: Counter())
    for r in sim_rows.values():
        for key in (("category", r["category"]), ("regime", r["regime"]),
                    ("wind", f"{r['category']}:{wind_bin(r['context']['wind_speed_ms'])}")):
            acc[key[0]][key[1] + "|accepted"] += 1
    for rej in rejected:
        if rej["stage"] != "simulation":
            continue
        for key in (("category", rej["category"]), ("regime", rej["regime"]),
                    ("wind", f"{rej['category']}:{wind_bin(rej['wind_speed_ms'])}")):
            acc[key[0]][key[1] + "|rejected"] += 1
    acceptance = {}
    for level, counter in acc.items():
        names = sorted({k.split("|")[0] for k in counter})
        acceptance[level] = {n: {"accepted": counter[n + "|accepted"], "rejected": counter[n + "|rejected"],
                                 "rate": counter[n + "|accepted"] / max(1, counter[n + "|accepted"] + counter[n + "|rejected"])}
                             for n in names}
    sim_times = [r["sim_runtime_s"] for r in results if r["sim_runtime_s"] is not None]
    render_times = [row["render_runtime_s"] for row in rows]
    summary = {
        "config_sha256": cfg_sha,
        "slots": len(tasks),
        "slots_without_valid_simulation": sum(1 for r in results if r["sim_id"] is None),
        "simulations_accepted": len(sim_rows),
        "simulation_attempts_rejected": sum(1 for r in rejected if r["stage"] == "simulation"),
        "render_retries": sum(1 for r in rejected if r["stage"] == "render"),
        "sequences": len(rows),
        "sequences_per_split": dict(Counter(r["split"] for r in rows)),
        "sequences_per_category_split": {f"{c}/{s}": n for (c, s), n in Counter((r["category"], r["split"]) for r in rows).items()},
        "mask_sources": dict(Counter(r["info"]["mask_source"] for r in rows)),
        "acceptance": acceptance,
        "timing": {
            "wall_s": wall, "workers": args.workers,
            "sim_runtime_mean_s": float(np.mean(sim_times)) if sim_times else None,
            "render_extract_mean_s": float(np.mean(render_times)) if render_times else None,
        },
        "threads": _bootstrap.THREADS,
    }
    write_json(out / "summary.json", summary)
    if args.examples:
        write_examples(out, rows, args.examples)
    print(json.dumps({k: summary[k] for k in ("simulations_accepted", "simulation_attempts_rejected", "render_retries", "sequences", "slots_without_valid_simulation")}), flush=True)
    print(f"done in {wall:.0f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
