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
    assert 0.0 <= doc["thresholds"]["smoke"]["strict"] <= 1.0
    assert doc["dev_cv"]["false_alarm_no_smoke_strict"] <= 0.05 + 1e-9
    assert doc["test"]["smoke_recall_strict"] is not None and "strict point" in rep.with_suffix(".md").read_text()
    det = Detector.load(out)
    assert det.thresholds["smoke"]["deploy"] == pytest.approx(thr)
    assert "Held-out test cameras" in rep.with_suffix(".md").read_text()


def test_fog_tile_and_change_scores():
    from src.detector.features import change_scores, fog_tile_scores, reference_scores

    rng = np.random.default_rng(5)
    Ts, Tf = _unit(rng.normal(0, 1, (8, D))), _unit(rng.normal(0, 1, (8, D)))
    fe = _unit(rng.normal(0, 1, (3, 2, 24, D))).astype(np.float32)
    ref = _unit(rng.normal(0, 1, (3, 24, D))).astype(np.float32)
    assert change_scores(fe, Ts).shape == (3, 2) and reference_scores(fe, ref, Ts).shape == (3, 2)
    f = fog_tile_scores(fe, None, Ts, Tf)
    assert f.shape == (3, 4) and np.isnan(f[:, 2:]).all() and np.isfinite(f[:, :2]).all()
    g = fog_tile_scores(fe, ref, Ts, Tf)
    np.testing.assert_allclose(g[:, :2], f[:, :2]) and np.testing.assert_equal(np.isfinite(g).all(), True)
    fe2 = fe.copy()
    fe2[0, 1, 7] = _unit(fe2[0, 1, 7] + 5 * Ts[:3].mean(0))          # tile 7 becomes smoke-like in frame 11
    assert change_scores(fe2, Ts)[0, 0] > change_scores(fe, Ts)[0, 0]


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


def _toy_tile_sets(tmp_path, rng, zoom_tiles=False):
    """Two toy sets with fine, zoom and reference embeddings: "fig" (smoke in every other window, a plume over 6
    fine tiles of the last frame and the first 6 zoom crops) and "hpf" (fog only). Returns the --set specs and
    {window_id: first plume tile}."""
    splits = ["train", "train", "val_select", "val_calib", "test_iid"]
    specs, plume = [], {}
    for name, smoke_rule, fog_rule in (("fig", lambda k: k % 2, lambda k: None), ("hpf", lambda k: 0, lambda k: k % 2)):
        rows, ids, embs, fines, zooms, refs = [], [], [], [], [], []
        for c in range(10):
            for k in range(8):
                s, f = smoke_rule(k), fog_rule(k)
                wid = f"{name}-{c}-{k}"
                rows.append({"video_path": "x", "camera_id": f"{name}{c}-s-mobo-c", "video_id": wid, "window_id": wid,
                             "split": splits[c % 5], "smoke": str(s), "fog": "" if f is None else str(f)})
                ids.append(wid)
                embs.append(_windows(rng, 1, [0], [f or 0])[0])     # the coarse crops see no smoke
                fe = rng.normal(0, 1, (2, 24, D))
                zm = rng.normal(0, 1, (2, 16, D))
                if s == 1:
                    t0 = (c * 7 + k) % 18
                    plume[wid] = t0
                    fe[1, t0:t0 + 6, 0] += 6.0                   # a plume over 6 fine tiles, last frame
                    zm[1, :6, 0] += 6.0                          # and in the zoom crops of the top tiles
                fines.append(_unit(fe).astype(np.float32))
                zooms.append(_unit(zm).astype(np.float32))
                refs.append(_unit(rng.normal(0, 1, (24, D))).astype(np.float32))
        man = tmp_path / f"{name}.csv"
        with open(man, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        np.savez(tmp_path / f"{name}.npz", window_id=np.array(ids), emb=np.stack(embs))
        np.savez(tmp_path / f"{name}_fine.npz", window_id=np.array(ids), emb=np.stack(fines))
        extra = {}
        if zoom_tiles:                                           # the zoom crops: quarters of tiles t0, t0 + 1, ...
            extra["tiles"] = np.array([[plume.get(w, 0) + j for j in range(4)] for w in ids])
        np.savez(tmp_path / f"{name}_zoom.npz", window_id=np.array(ids), emb=np.stack(zooms), **extra)
        keep = slice(8, None)                                    # the first camera has no reference frames
        np.savez(tmp_path / f"{name}_ref.npz", window_id=np.array(ids)[keep], fine=np.stack(refs)[keep],
                 emb=np.zeros((len(ids) - 8, 7, D), np.float32))
        specs += ["--set", f"{name}:{man}:{tmp_path / f'{name}.npz'}::{tmp_path / f'{name}_fine.npz'}:"
                           f"{tmp_path / f'{name}_zoom.npz'}:{tmp_path / f'{name}_ref.npz'}"]
    return specs, plume


def _toy_train_detector(monkeypatch, rng):
    import importlib

    sys.modules.pop("train_detector", None)
    TD = importlib.import_module("train_detector")
    monkeypatch.setattr(TD, "text_embeddings", lambda prompts: _unit(rng.normal(0, 1, (len(prompts), D))))
    monkeypatch.setattr(TD, "GROUP_SETS", {"fog": (("full",),), "smoke": (("full",), ("full", "fine"))})
    monkeypatch.setattr(TD, "C_GRID", (0.1,))
    monkeypatch.setattr(TD, "TILE_C_GRID", (0.01, 0.1))
    return TD


def test_train_detector_tile_head_and_stack(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    TD = _toy_train_detector(monkeypatch, rng)
    specs, _ = _toy_tile_sets(tmp_path, rng)
    out, rep = tmp_path / "det", tmp_path / "rep.json"
    assert TD.main(specs + ["--out", str(out), "--report", str(rep), "--final", "--n-boot", "10",
                            "--deploy-recall", "0.8", "--fog-veto", "0.95", "--stack-fog-weight", "3"]) == 0
    doc = json.loads(rep.read_text())
    sel = doc["selection"]["smoke"]
    assert sel["tile"]["cv_auroc"] > 0.9 and sel["zoom"]["cv_auroc"] > 0.9 and sel["stacked"]["cv_auroc"] > 0.9
    assert sel["stacked"]["rows_per_stack"]["head+tile+zoom+chg_max+chg_rz+fog_chg+fogd_chg"] == 16   # no ref
    assert doc["thresholds"]["smoke"]["fog_veto"] == 0.95 and "dev_cv_v1_stack" in doc and "test_v1_stack" in doc
    assert doc["dev_cv"]["smoke_recall"] >= 0.8 - 1e-9
    assert {tuple(u["stack"]) for u in doc["test_stacks_used"]} == {
        ("head", "tile", "zoom", "chg_max", "chg_rz", "ref_max", "ref_rz", "fog_chg", "fogd_chg", "fog_ref",
         "fogd_ref")}
    assert doc["rules"]["stack_fog_weight"] == 3.0
    assert doc["test"]["smoke_auroc"] > 0.9
    assert doc["dev_cv"]["smoke_auroc_within_sets"] == pytest.approx(doc["dev_cv"]["per_set"]["fig"]["smoke_auroc"])
    det = Detector.load(out)
    assert set(det.tile_heads) == {"smoke"} and set(det.zoom_heads) == {"smoke"}
    assert [st["inputs"][-1] for st in det.stacks["smoke"]] == ["fogd_ref", "fogd_chg", "fogd_ref", "fogd_chg"]
    E = np.stack([_windows(rng, 1, [0], [0])[0]] * 2)
    fe = _unit(rng.normal(0, 1, (2, 2, 24, D))).astype(np.float32)
    with pytest.raises(ValueError):
        det.proba(E)                                         # the tile head needs the fine embeddings
    fe[1, 1, 5:11, 0] += 6.0
    fe[1] = _unit(fe[1])
    p = det.proba(E, fine=fe)["smoke"]                     # fallback stack: no zoom, no reference
    assert p[1] > p[0]
    assert det.stack_used("smoke", det.stack_inputs("smoke", E, fine=fe)) == [
        "head", "tile", "chg_max", "chg_rz", "fog_chg", "fogd_chg"]
    assert det.select_zoom_tiles(fe, 4).shape == (2, 4)
    md = rep.with_suffix(".md").read_text()
    assert "tile head" in md and "zoom head" in md and "Deployed stack against the v1 stack" in md


def test_box_region_and_instance_labels(tmp_path):
    from src.detector.boxes import fine_regions, instance_labels, load_boxes, region_labels, zoom_regions

    fine = fine_regions()
    assert len(fine) == 24 and fine[7] == pytest.approx((1 / 6, 0.25, 2 / 6, 0.5))
    lab = region_labels(fine, (0.20, 0.30, 0.30, 0.45))               # a small plume inside tile 7
    assert lab[7] == 1 and (lab == 1).sum() == 1
    assert lab[23] == 0 and lab[6] == 0                                # far away / not within the margin
    edge = region_labels(fine, (0.20, 0.30, 0.34, 0.45))               # grazes tile 8 by 0.007 of the frame
    assert edge[7] == 1 and edge[8] == -1
    big = region_labels(fine, (0.0, 0.0, 1.0, 0.5))                    # a plume over the top half
    assert (big[:12] == 1).all() and (big[18:] == 0).all() and (big[12:18] == -1).all()
    zr = zoom_regions([7, 0])
    assert len(zr) == 8 and zr[0] == pytest.approx((1 / 6, 0.25, 1.5 / 6, 0.375))
    assert zr[4] == pytest.approx((0.0, 0.0, 1 / 12, 0.125))
    path = tmp_path / "boxes.csv"
    cols = ["window_id", "sheet_id", "target", "confidence", "what", "labeler"] + \
        [f"m{m}_{k}" for m in (0, 4, 8, 11) for k in ("visible", "x0", "y0", "x1", "y1")]
    rows = [("a", "smoke", "high", (0.20, 0.30, 0.30, 0.45)), ("b", "smoke", "low", (0.2, 0.3, 0.3, 0.45)),
            ("c", "none", "high", None), ("d", "smoke", "medium", None)]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for wid, target, conf, box in rows:
            r = {"window_id": wid, "sheet_id": wid, "target": target, "confidence": conf, "what": "", "labeler": ""}
            for m in (0, 4, 8, 11):
                vis = box is not None and m == 11
                r.update({f"m{m}_visible": int(vis), **{f"m{m}_{k}": (box[i] if vis else 0)
                                                        for i, k in enumerate(("x0", "y0", "x1", "y1"))}})
            w.writerow(r)
    boxes = load_boxes(path)
    assert boxes["a"]["boxes"][11] == pytest.approx((0.20, 0.30, 0.30, 0.45)) and boxes["a"]["boxes"][0] is None
    Y, W = instance_labels(["a", "b", "c", "d", "e", "f"], np.array([1, 1, 1, 1, 1, 0]), boxes, lambda i: fine)
    assert Y[0, 7] == 1 and Y[0].sum() == 1 and W[0].sum() == 24      # boxed: every tile used (none grazed)
    assert W[1:5].sum() == 0                                           # low confidence, none, invisible, no box
    assert W[5].sum() == 24 and Y[5].sum() == 0                        # smoke-free: every tile negative


def test_tile_head_on_instance_labels_and_zoom_selector(tmp_path):
    from src.detector.model import LinearHead, fit_tile_head, tile_logit

    rng = np.random.default_rng(5)
    n, k, d = 200, 24, 8
    TF = rng.normal(0, 1, (n, k, d))
    y = (np.arange(n) % 2).astype(float)
    Y = np.zeros((n, k))
    W = np.ones((n, k))
    for i in np.flatnonzero(y == 1):
        TF[i, i % k, 0] += 3.0                       # the plume tile
        TF[i, :, 1] += 2.0                           # a scene cue on every tile of a smoke window
        Y[i, i % k] = 1
    inst = fit_tile_head(TF, y, 1.0, Y, W, balance=True)
    mil = fit_tile_head(TF, y, 1.0)
    assert abs(inst.coef[0]) > abs(inst.coef[1])     # instance labels: the plume, not the scene cue
    assert abs(mil.coef[1]) > abs(mil.coef[0])       # every tile labelled: the scene cue dominates
    assert tile_logit(inst, TF).shape == (n,)
    sel = LinearHead(mean=np.zeros(2 * D + 14), scale=np.ones(2 * D + 14), coef=np.r_[1.0, np.zeros(2 * D + 13)],
                     intercept=0.0, groups=["tiles"], C=1.0)                  # scores frame 11's first dimension
    det = Detector(heads={"smoke": fit_head(rng.normal(0, 1, (20, 3)), np.arange(20) % 2, (), 1.0)},
                   text={"smoke": _unit(rng.normal(0, 1, (7, D)))}, thresholds={"smoke": {"balanced": 0.5}},
                   tile_heads={"smoke": LinearHead(mean=np.zeros(2 * D + 14), scale=np.ones(2 * D + 14),
                                                   coef=np.zeros(2 * D + 14), intercept=0.0, groups=["t"], C=1.0)},
                   zoom_selector={"smoke": sel})
    fe = np.zeros((1, 2, 24, D), np.float32)
    fe[0, 1, :, 1] = 1.0
    fe[0, 1, 9] = 0.0
    fe[0, 1, 9, 0] = 1.0
    assert det.select_zoom_tiles(fe, 1)[0, 0] == 9   # the selector, not the (all-zero) tile head, decides
    det.save(tmp_path / "d")
    back = Detector.load(tmp_path / "d")
    assert np.allclose(back.zoom_selector["smoke"].coef, sel.coef) and back.select_zoom_tiles(fe, 1)[0, 0] == 9


def test_train_detector_boxes_and_score_detector(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    TD = _toy_train_detector(monkeypatch, rng)
    specs, plume = _toy_tile_sets(tmp_path, rng, zoom_tiles=True)
    cols = ["window_id", "sheet_id", "target", "confidence", "what", "labeler"] + \
        [f"m{m}_{k}" for m in (0, 4, 8, 11) for k in ("visible", "x0", "y0", "x1", "y1")]
    boxes = tmp_path / "boxes.csv"
    with open(boxes, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i, (wid, t0) in enumerate(sorted(plume.items())):
            r, c = divmod(t0, 6)                                 # a box on the plume's first tile
            row = {"window_id": wid, "sheet_id": wid, "target": "smoke", "confidence": "high", "what": "",
                   "labeler": ""}
            for m in (0, 4, 8, 11):
                row.update({f"m{m}_visible": 1, f"m{m}_x0": (c + 0.2) / 6, f"m{m}_y0": (r + 0.2) / 4,
                            f"m{m}_x1": (c + 0.8) / 6, f"m{m}_y1": (r + 0.8) / 4})
            if i == 0:
                row["target"] = "none"                           # one window the labeller found no smoke in
            w.writerow(row)
    sel_dir = tmp_path / "sel"
    out, rep = tmp_path / "det", tmp_path / "rep.json"
    assert TD.main(specs + ["--out", str(sel_dir), "--report", str(tmp_path / "sel.json"), "--n-boot", "5"]) == 0
    assert TD.main(specs + ["--out", str(out), "--report", str(rep), "--final", "--n-boot", "10",
                            "--boxes", str(boxes), "--box-balance", "--zoom-selector", str(sel_dir),
                            "--ref-tile-head"]) == 0
    doc = json.loads(rep.read_text())
    sel = doc["selection"]["smoke"]
    n_smoke = sel["box_labels"]["smoke_windows"]
    assert sel["box_labels"]["smoke_windows_boxed"] == n_smoke - 1 == len(plume) - 1 - sum(
        1 for w in plume if int(w.split("-")[1]) % 5 == 4)     # test cameras are not dev rows
    assert sel["box_labels"]["positive_tiles"] == n_smoke - 1  # one tile per boxed window
    assert sel["box_labels_zoom"]["positive_tiles"] == 4 * (n_smoke - 1)   # the plume tile's 4 quarters
    assert sel["tile"]["cv_auroc"] > 0.9 and doc["test"]["smoke_auroc"] > 0.9
    assert doc["rules"]["box_balance"] is True and doc["rules"]["boxes"] == [str(boxes)]
    assert sel["tileref"]["n"] == 128 - 16 and sel["tileref"]["cv_auroc"] > 0.9    # rows with a reference
    det = Detector.load(out)
    assert np.allclose(det.zoom_selector["smoke"].coef, Detector.load(sel_dir).tile_heads["smoke"].coef)
    assert set(det.ref_tile_heads) == {"smoke"} and det.stacks["smoke"][0]["inputs"][-1] == "tileref"
    assert "tileref" not in det.stacks["smoke"][1]["inputs"]                       # the no-reference fallback
    assert "reference frame, best tile (box labels)" in rep.with_suffix(".md").read_text()
    import importlib

    SD = importlib.import_module("score_detector")
    monkeypatch.setattr(SD, "TD", TD)
    srep = tmp_path / "score.json"
    assert SD.main(["--detector", f"a={sel_dir}", "--detector", f"b={out}", "--report", str(srep),
                    "--n-boot", "10", "--matched-recall", "0.8"] + specs) == 0
    sc = json.loads(srep.read_text())
    assert set(sc["results"]) == {"a", "b"} and "b-a" in sc["paired"]
    assert sc["results"]["b"]["smoke_auroc"] == pytest.approx(doc["test"]["smoke_auroc"])
    assert sc["matched_recall"]["a"]["0.80"]["false_alarm_no_smoke"] is not None
    pv = sc["paired"]["b-a"]["value"]
    assert pv["smoke_auroc_within_sets"] == pytest.approx(sc["results"]["b"]["smoke_auroc_within_sets"]
                                                          - sc["results"]["a"]["smoke_auroc_within_sets"])
    assert np.load(srep.with_suffix(".probs.npz"))["p_smoke_b"].shape == (sc["results"]["b"]["n"]["windows"],)
