"""The real-footage detector (src/detector, scripts/train_detector.py) on toy embeddings, without CLIP."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from src.detector.features import group_features, window_features  # noqa: E402
from src.detector.model import Detector, fit_head  # noqa: E402

D = 16


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _windows(rng, n, smoke, fog):
    """Toy (n, 3, 7, D) embeddings: a 'smoke' direction grows in one tile over the window, 'fog' lifts the frame."""
    E = rng.normal(0, 1, (n, 3, 7, D))
    for i in range(n):
        if smoke[i] == 1:
            E[i, 1:, 3, 0] += 4.0                    # appears after the first frame, in tile 3
        if fog[i] == 1:
            E[i, :, 0, 1] += 3.0                     # whole frame
            E[i, :, 4:, 1] += 2.0                    # lower tiles
    return _unit(E).astype(np.float32)


def test_group_features_shapes_and_errors():
    rng = np.random.default_rng(0)
    E = _windows(rng, 5, [0] * 5, [0] * 5)
    T = _unit(rng.normal(0, 1, (3, D)))
    assert group_features(E, T, "full").shape == (5, D)
    assert group_features(E, T, "tiles").shape == (5, 4 * D)
    assert group_features(E, T, "sims").shape == (5, 7 * 3 + 2 * 3)
    assert group_features(E[0], T, "full").shape == (1, D)             # one window (F, 7, D)
    X = window_features(E, T, ("full", "motion"), motion=np.ones((5, 28)))
    assert X.shape == (5, D + 28) and X.dtype == np.float32
    with pytest.raises(ValueError):
        window_features(E, T, ("motion",))
    with pytest.raises(ValueError):
        group_features(E, T, "nope")
    with pytest.raises(ValueError):
        group_features(np.zeros((5, 3, 6, D)), T, "full")


def test_head_fit_save_load_predict(tmp_path):
    rng = np.random.default_rng(1)
    n = 120
    smoke = rng.integers(0, 2, n)
    fog = rng.integers(0, 2, n)
    E = _windows(rng, n, smoke, fog)
    T = {"smoke": _unit(rng.normal(0, 1, (4, D))), "fog": _unit(rng.normal(0, 1, (4, D)))}
    heads = {"smoke": fit_head(window_features(E, T["smoke"], ("tiles",)), smoke, ("tiles",), 1.0),
             "fog": fit_head(window_features(E, T["fog"], ("full",)), fog, ("full",), 1.0)}
    det = Detector(heads=heads, text=T, thresholds={"smoke": {"balanced": 0.5, "deploy": 0.8},
                                                    "fog": {"balanced": 0.5}}, info={"x": 1})
    p = det.proba(E)
    assert set(p) == {"smoke", "fog"} and p["smoke"].shape == (n,)
    assert np.mean((p["smoke"] > 0.5) == smoke) > 0.9 and np.mean((p["fog"] > 0.5) == fog) > 0.9
    det.save(tmp_path / "d")
    back = Detector.load(tmp_path / "d")
    q = back.proba(E)
    np.testing.assert_allclose(q["smoke"], p["smoke"], rtol=1e-9)
    out = back.predict(E, operating_point="deploy")
    assert np.array_equal(out["decisions"]["smoke"], (q["smoke"] >= 0.8).astype(int))
    assert np.array_equal(out["decisions"]["fog"], (q["fog"] >= 0.5).astype(int))   # no deploy point: balanced
    assert back.info == {"x": 1}


def test_train_detector_end_to_end(tmp_path, monkeypatch):
    import importlib

    sys.modules.pop("train_detector", None)
    TD = importlib.import_module("train_detector")
    rng = np.random.default_rng(2)
    monkeypatch.setattr(TD, "text_embeddings", lambda prompts: _unit(rng.normal(0, 1, (len(prompts), D))))
    monkeypatch.setattr(TD, "GROUP_SETS", {"fog": (("full",), ("full", "sims")), "smoke": (("tiles",),)})
    monkeypatch.setattr(TD, "C_GRID", (0.1, 1.0))
    splits = ["train", "train", "val_select", "val_calib", "test_iid"]

    def make_set(name, n_cams, per_cam, smoke_rule, fog_rule):
        rows, ids, embs = [], [], []
        for c in range(n_cams):
            cam = f"{name}{c}-s-mobo-c"
            split = splits[c % len(splits)]
            for k in range(per_cam):
                s, f = smoke_rule(k), fog_rule(k)
                wid = f"{name}-{c}-{k}"
                rows.append({"video_path": "x", "camera_id": cam, "video_id": wid, "window_id": wid, "split": split,
                             "smoke": "" if s is None else str(s), "fog": "" if f is None else str(f)})
                ids.append(wid)
                embs.append(_windows(rng, 1, [s or 0], [f or 0])[0])
        man = tmp_path / f"{name}.csv"
        with open(man, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        npz = tmp_path / f"{name}.npz"
        np.savez(npz, window_id=np.array(ids), seq_id=np.array([f"{i}-w0" for i in ids]), emb=np.stack(embs))
        return f"{name}:{man}:{npz}"

    fig = make_set("fig", 10, 8, lambda k: k % 2, lambda k: None)          # smoke labels, fog unknown
    hpf = make_set("hpf", 10, 8, lambda k: 0, lambda k: k % 2)             # fog windows, no smoke
    out, rep = tmp_path / "det", tmp_path / "rep.json"
    assert TD.main(["--set", fig, "--set", hpf, "--out", str(out), "--report", str(rep), "--final",
                    "--n-boot", "20"]) == 0
    doc = json.loads(rep.read_text())
    assert doc["selection"]["smoke"]["best"]["groups"] == ["tiles"]
    assert doc["dev_cv"]["smoke_auroc"] > 0.9 and doc["dev_cv"]["fog_auroc"] > 0.9
    assert doc["test"]["n"]["windows"] == 2 * 2 * 8                      # cameras 4 and 9 of each set
    assert set(doc["test"]["goal_met"]) == {"fog_auroc", "smoke_auroc", "deploy_point"}
    thr = doc["thresholds"]["smoke"]["deploy"]
    assert 0.0 <= thr <= 1.0
    det = Detector.load(out)
    assert det.thresholds["smoke"]["deploy"] == pytest.approx(thr)
    assert "Held-out test cameras" in rep.with_suffix(".md").read_text()


def test_embed_crops_grid_and_normalisation():
    import embed_windows as EW

    img = np.zeros((680, 1020, 3), np.uint8)
    img[510:, 850:] = 255                                        # exactly the bottom-right tile of a 4 x 6 grid
    img[:5] = 255                                                # banner rows: blanked before cropping
    default = EW.crops(img)
    assert len(default) == 7 and all(c.shape == (3, 224, 224) for c in default)
    fine = EW.crops(img, 4, 6, full=False)
    assert len(fine) == 24
    white = (1.0 - EW.CLIP_MEAN) / EW.CLIP_STD
    black = (0.0 - EW.CLIP_MEAN) / EW.CLIP_STD
    np.testing.assert_allclose(fine[-1].mean(axis=(1, 2)), white, atol=2e-3)
    np.testing.assert_allclose(fine[0].mean(axis=(1, 2)), black, atol=2e-3)     # banner blanked
    dino = EW.crops(img, 4, 6, False, EW.IMAGENET_MEAN, EW.IMAGENET_STD)
    np.testing.assert_allclose(dino[-1].mean(axis=(1, 2)), (1.0 - EW.IMAGENET_MEAN) / EW.IMAGENET_STD, atol=2e-3)


def test_train_detector_tile_head_and_stack(tmp_path, monkeypatch):
    import importlib

    sys.modules.pop("train_detector", None)
    TD = importlib.import_module("train_detector")
    rng = np.random.default_rng(3)
    monkeypatch.setattr(TD, "text_embeddings", lambda prompts: _unit(rng.normal(0, 1, (len(prompts), D))))
    monkeypatch.setattr(TD, "GROUP_SETS", {"fog": (("full",),), "smoke": (("full",), ("full", "fine"))})
    monkeypatch.setattr(TD, "C_GRID", (0.1,))
    monkeypatch.setattr(TD, "TILE_C_GRID", (0.01, 0.1))
    splits = ["train", "train", "val_select", "val_calib", "test_iid"]
    specs = []
    for name, smoke_rule, fog_rule in (("fig", lambda k: k % 2, lambda k: None), ("hpf", lambda k: 0, lambda k: k % 2)):
        rows, ids, embs, fines = [], [], [], []
        for c in range(10):
            for k in range(8):
                s, f = smoke_rule(k), fog_rule(k)
                wid = f"{name}-{c}-{k}"
                rows.append({"video_path": "x", "camera_id": f"{name}{c}-s-mobo-c", "video_id": wid, "window_id": wid,
                             "split": splits[c % 5], "smoke": str(s), "fog": "" if f is None else str(f)})
                ids.append(wid)
                embs.append(_windows(rng, 1, [0], [f or 0])[0])     # the coarse crops see no smoke
                fe = rng.normal(0, 1, (2, 24, D))
                if s == 1:
                    t0 = (c * 7 + k) % 18
                    fe[1, t0:t0 + 6, 0] += 6.0                   # a plume over 6 fine tiles, last frame
                fines.append(_unit(fe).astype(np.float32))
        man = tmp_path / f"{name}.csv"
        with open(man, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        np.savez(tmp_path / f"{name}.npz", window_id=np.array(ids), emb=np.stack(embs))
        np.savez(tmp_path / f"{name}_fine.npz", window_id=np.array(ids), emb=np.stack(fines))
        specs += ["--set", f"{name}:{man}:{tmp_path / f'{name}.npz'}::{tmp_path / f'{name}_fine.npz'}"]
    out, rep = tmp_path / "det", tmp_path / "rep.json"
    assert TD.main(specs + ["--out", str(out), "--report", str(rep), "--final", "--n-boot", "10"]) == 0
    doc = json.loads(rep.read_text())
    sel = doc["selection"]["smoke"]
    assert sel["tile"]["cv_auroc"] > 0.9 and sel["stacked"]["cv_auroc"] > 0.9
    assert doc["test"]["smoke_auroc"] > 0.9
    assert doc["dev_cv"]["smoke_auroc_within_sets"] == pytest.approx(doc["dev_cv"]["per_set"]["fig"]["smoke_auroc"])
    det = Detector.load(out)
    assert set(det.tile_heads) == {"smoke"} and len(det.stacks["smoke"]["coef"]) == 2
    E = np.stack([_windows(rng, 1, [0], [0])[0]] * 2)
    fe = _unit(rng.normal(0, 1, (2, 2, 24, D))).astype(np.float32)
    with pytest.raises(ValueError):
        det.proba(E)                                         # the tile head needs the fine embeddings
    fe[1, 1, 5:11, 0] += 6.0
    fe[1] = _unit(fe[1])
    p = det.proba(E, fine=fe)["smoke"]
    assert p[1] > p[0]
    assert "tile head" in rep.with_suffix(".md").read_text()
