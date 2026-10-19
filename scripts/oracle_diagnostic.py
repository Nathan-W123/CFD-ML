"""CFD-oracle diagnostic: how much of the class signal survives the camera.

    python scripts/oracle_diagnostic.py --data data/synthetic --out reports/oracle_diagnostic.json

For every accepted simulation, the twelve motion features are recomputed
directly from the CFD fields over the camera view: the candidate region is the
cells whose total optical depth at the nominal line of sight
(validation.label_line_of_sight_m) reaches validation.label_tau_min, and the
velocity statistics are the exact CFD velocities inside it (m/s, 1/s). This is
a perfect measurement of the region's motion with no rendering, masks or
optical flow in between.

THE ORACLE FEATURES ARE NEVER MODEL INPUTS. They bound what the motion
representation could tell a camera that measured it perfectly, and the gap to
the camera-level features localises where information is lost:

* oracle "velocity" features (0-5) vs camera optical-flow features (0-5):
  optical-flow noise floor, stabilisation, region of interest;
* oracle "region" features (6-11) vs camera mask features (6-11):
  mask degradation, rendering (texture, exposure, occlusion).

For each label (smoke, fog) the report gives the univariate AUROC of every
oracle and camera feature (time mean over valid steps, train split, folded),
and a pooled logistic regression (per-feature mean and std over valid steps,
L2 = 1, fitted on the train split only) on each feature group, evaluated on
test_iid and test_ood with 95 % grouped-bootstrap intervals. The oracle is
evaluated once per simulation; the camera once per rendered sequence (each
simulation weighted once in the fit). Both use the same simulations.
"""

import _bootstrap  # noqa: F401

import argparse
import sys
from pathlib import Path

import numpy as np

from src.cfd import thermo
from src.cfd.results import SimulationResult
from src.cfd.validation import validation_settings, view_slices
from src.constants import LABEL_NAMES
from src.data.dataset import FeatureSet
from src.data.feature_spec import FEATURE_NAMES, N_STEPS
from src.training import metrics as M
from src.training.baselines import _LogisticRegression, pooled_features
from src.utils.config import load_simulation_config
from src.utils.io import write_json

ORACLE_NAMES = FEATURE_NAMES[:12]
GROUPS = {"velocity": list(range(0, 6)), "region": list(range(6, 12)), "all": list(range(0, 12))}
PAIR_FEATURES = (7, 8, 9, 11)          # temporal derivatives: need a valid region at t-1 too
MIN_CELLS = 4


def _perimeter_cells(m: np.ndarray) -> int:
    """Number of exposed cell edges of a boolean region (4-connectivity)."""
    p = np.pad(m, 1)
    return int((p[1:, :] != p[:-1, :]).sum() + (p[:, 1:] != p[:, :-1]).sum())


def _bottom_rows(m: np.ndarray) -> np.ndarray:
    """Per column, the lowest occupied row (rows count UP from the ground), -1 when empty."""
    occ = m.any(axis=0)
    return np.where(occ, np.argmax(m, axis=0), -1)


def oracle_features(sim: SimulationResult, settings, smoke_ext: float, tracer_ext: float, margin: int):
    """(T, 12) oracle features and (T, 2) validity [region valid, pair valid], physical units.

    Grid rows run upward (row 0 at the ground), so +w and +dz are UP like the
    camera features. Lengths in metres, velocities in m/s, rates in 1/s.
    """
    rows, cols = view_slices(*sim.solid.shape, margin)
    los = float(settings["label_line_of_sight_m"])
    dx = float(sim.dx)
    tau = (np.maximum(sim.smoke[:, rows, cols], 0.0) * smoke_ext * los
           + thermo.fog_extinction(np.maximum(sim.lwc[:, rows, cols], 0.0)) * los
           + np.maximum(sim.tracer[:, rows, cols], 0.0) * tracer_ext * los)
    fluid = ~sim.solid[rows, cols]
    times = np.asarray(sim.times, dtype=np.float64)
    T = min(len(times), N_STEPS)
    out = np.zeros((N_STEPS, 12))
    valid = np.zeros((N_STEPS, 2), dtype=bool)
    prev = None
    for t in range(T):
        m = (tau[t] >= float(settings["label_tau_min"])) & fluid
        if m.sum() < MIN_CELLS:
            prev = None
            continue
        u, w = sim.ux[t][rows, cols], sim.uz[t][rows, cols]
        dudx, dudz = np.gradient(u, dx, axis=1), np.gradient(u, dx, axis=0)
        dwdx, dwdz = np.gradient(w, dx, axis=1), np.gradient(w, dx, axis=0)
        zz, xx = np.nonzero(m)
        area = float(m.sum()) * dx * dx
        cx, cz = float(xx.mean()) * dx, float(zz.mean()) * dx
        bottom = _bottom_rows(m)
        out[t, 0:6] = [u[m].mean(), w[m].mean(), u[m].std(), w[m].std(),
                       (dudx + dwdz)[m].mean(), np.abs(dwdx - dudz)[m].mean()]
        out[t, 6] = float(m.mean())
        out[t, 10] = _perimeter_cells(m) * dx / (2.0 * np.sqrt(np.pi * area)) - 1.0
        valid[t, 0] = True
        if prev is not None:
            dt = times[t] - prev["time"]
            both = (bottom >= 0) & (prev["bottom"] >= 0)
            out[t, 7] = (np.log(area) - np.log(prev["area"])) / dt
            out[t, 8] = (cx - prev["cx"]) / dt
            out[t, 9] = (cz - prev["cz"]) / dt
            out[t, 11] = float((np.abs(bottom[both] - prev["bottom"][both]) <= 1).mean()) if both.any() else 0.0
            valid[t, 1] = True
        prev = {"time": times[t], "area": area, "cx": cx, "cz": cz, "bottom": bottom}
    return out, valid


def pooled_oracle(F: np.ndarray, V: np.ndarray) -> np.ndarray:
    """(N, 24): per feature, mean and std over the steps where it is valid (0 when none)."""
    g = np.repeat(V[:, :, :1], 12, axis=2).astype(np.float64)
    g[:, :, list(PAIR_FEATURES)] = V[:, :, 1:2]
    n = g.sum(axis=1)
    safe = np.maximum(n, 1.0)
    mean = (F * g).sum(axis=1) / safe
    var = (((F - mean[:, None, :]) ** 2) * g).sum(axis=1) / safe
    return np.concatenate([np.where(n > 0, mean, 0.0), np.where(n > 1, np.sqrt(np.maximum(var, 0.0)), 0.0)], axis=1)


def _columns(group):
    return list(group) + [12 + j for j in group]


def _camera_columns(group):
    return list(group) + [14 + j for j in group]


def _folded(a):
    return None if a is None else round(max(a, 1.0 - a), 3)


def _fit_eval(P_tr, y_tr, w_tr, parts, n_boot, seed):
    """Per label: logistic regression fitted on train, test AUROC with grouped-bootstrap 95 % CI."""
    out = {}
    for j, label in enumerate(LABEL_NAMES):
        model = _LogisticRegression(1.0).fit(P_tr, y_tr[:, j], w_tr)
        res = {}
        for name, (P, y, groups) in parts.items():
            if len(P) == 0:
                res[name] = None
                continue
            z = model.decision_function(P)
            t = y[:, j] > 0.5
            ci = M.grouped_bootstrap(lambda idx: {"auroc": M.auroc(z[idx], t[idx])}, groups,
                                     n_boot=n_boot, seed=seed)["auroc"]
            res[name] = {"auroc": M.auroc(z, t), "ci95": [ci["lo"], ci["hi"]], "n": int(len(P)),
                         "n_groups": int(len(np.unique(groups)))}
        out[label] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="dataset directory (features.npz + sims/)")
    ap.add_argument("--out", default="reports/oracle_diagnostic.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    base = Path(args.data)
    cfg = load_simulation_config()
    settings = validation_settings(cfg)
    render = cfg.get("render", {})
    smoke_ext = float(render.get("smoke_mass_extinction_m2_mg", 0.0076))
    tracer_ext = float(render.get("tracer_mass_extinction_m2_mg", 0.004))
    margin = int(render.get("view_margin_cells", 6))

    fs = FeatureSet.load(base / "features.npz")
    fs = fs.subset(np.isin(fs.splits, ["train", "val_select", "val_calib", "test_iid", "test_ood"]))

    # One oracle row per simulation.
    first = {}
    for i, sid in enumerate(fs.sim_ids):
        first.setdefault(sid, i)
    sims = list(first)
    F = np.zeros((len(sims), N_STEPS, 12))
    V = np.zeros((len(sims), N_STEPS, 2), dtype=bool)
    for k, sid in enumerate(sims):
        sim = SimulationResult.load(base / fs.meta[first[sid]]["sim_path"])
        F[k], V[k] = oracle_features(sim, settings, smoke_ext, tracer_ext, margin)
        if (k + 1) % 200 == 0:
            print(f"  oracle features: {k + 1}/{len(sims)} simulations", flush=True)
    rows = np.array([first[s] for s in sims])
    y_o, split_o, sims_o = fs.y[rows], fs.splits[rows], np.asarray(sims)
    P_o = pooled_oracle(F, V)
    P_c = pooled_features(fs.X)
    w_c = M.per_group_weights(fs.groups)

    tr_o, tr_c = split_o == "train", fs.splits == "train"
    doc = {
        "description": ("CFD-oracle diagnostic. Oracle features are computed from the CFD fields (exact "
                        "velocities, region from the optical depth at the nominal line of sight) and are "
                        "never model inputs; they bound what a perfect measurement of the same motion "
                        "representation could achieve. Synthetic data only."),
        "data": str(base),
        "region": {"line_of_sight_m": float(settings["label_line_of_sight_m"]),
                   "tau_min": float(settings["label_tau_min"]), "min_cells": MIN_CELLS, "view_margin_cells": margin},
        "n_simulations": {str(s): int((split_o == s).sum()) for s in np.unique(split_o)},
        "n_sequences": {str(s): int((fs.splits == s).sum()) for s in np.unique(fs.splits)},
        "oracle_region_valid_fraction": {
            str(cat): float(V[fs.categories[rows] == cat, :, 0].mean()) for cat in np.unique(fs.categories[rows])},
        "univariate_auroc_train": {},
        "pooled_logistic": {},
    }
    t_mean_o = pooled_oracle(F, V)[:, :12]
    t_mean_c = P_c[:, :14]
    for j, label in enumerate(LABEL_NAMES):
        doc["univariate_auroc_train"][label] = {
            name: {"oracle": _folded(M.auroc(t_mean_o[tr_o, k], y_o[tr_o, j] > 0.5)),
                   "camera": _folded(M.auroc(t_mean_c[tr_c, k], fs.y[tr_c, j] > 0.5))}
            for k, name in enumerate(ORACLE_NAMES)}
    for gname, group in GROUPS.items():
        co, cc = _columns(group), _camera_columns(group)
        parts_o = {s: (P_o[split_o == s][:, co], y_o[split_o == s], sims_o[split_o == s])
                   for s in ("test_iid", "test_ood")}
        parts_c = {s: (P_c[fs.splits == s][:, cc], fs.y[fs.splits == s], fs.groups[fs.splits == s])
                   for s in ("test_iid", "test_ood")}
        doc["pooled_logistic"][gname] = {
            "features": [ORACLE_NAMES[k] for k in group],
            "oracle": _fit_eval(P_o[tr_o][:, co], y_o[tr_o], None, parts_o, args.n_boot, args.seed),
            "camera": _fit_eval(P_c[tr_c][:, cc], fs.y[tr_c], w_c[tr_c], parts_c, args.n_boot, args.seed),
        }
    out = Path(args.out)
    write_json(out, doc)
    lines = ["# CFD-oracle diagnostic (synthetic only; oracle features are never model inputs)", "",
             f"Data: `{base}`; simulations per split: {doc['n_simulations']}.", "",
             "Pooled logistic regression, test AUROC [95 % grouped-bootstrap CI]:", "",
             "| features | label | oracle test_iid | camera test_iid | oracle test_ood | camera test_ood |",
             "|---|---|---|---|---|---|"]

    def cell(r):
        if not r or r["auroc"] is None:
            return "n/a"
        lo, hi = r["ci95"]
        ci = "" if lo is None else f" [{lo:.2f}, {hi:.2f}]"
        return f"{r['auroc']:.3f}{ci}"

    for gname, res in doc["pooled_logistic"].items():
        for label in LABEL_NAMES:
            o, c = res["oracle"][label], res["camera"][label]
            lines.append(f"| {gname} | {label} | {cell(o['test_iid'])} | {cell(c['test_iid'])} | "
                         f"{cell(o['test_ood'])} | {cell(c['test_ood'])} |")
    lines += ["", "Univariate AUROC on the train split (time mean over valid steps, folded):", "",
              "| feature | smoke oracle | smoke camera | fog oracle | fog camera |", "|---|---|---|---|---|"]
    for name in ORACLE_NAMES:
        s, f = doc["univariate_auroc_train"]["smoke"][name], doc["univariate_auroc_train"]["fog"][name]
        lines.append(f"| {name} | {s['oracle']} | {s['camera']} | {f['oracle']} | {f['camera']} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
