"""Per-minute windows from per-frame embeddings (src/detector/dense.py), alarm rules over them
(src/detector/temporal.py), and the scripts that score and tune them, on toy data."""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from src.detector.dense import assemble, window_index  # noqa: E402
from src.detector.temporal import Rule, evaluate, events, fires  # noqa: E402


def test_window_index_and_assemble():
    unix = np.r_[1000 + 60 * np.arange(30), 1000 + 60 * np.arange(31, 50) + 7]   # minute 30 missing, later +7 s
    idx = window_index(unix)
    assert (idx[:11, 0] == -1).all() and idx[11, 0] == 0 and idx[11, 1] == 6 and idx[11, 2] == 11
    assert idx[11, 3] == -1 and idx[25, 3] == 0                  # reference 25 frames before frame 11
    j = int(np.flatnonzero(unix == 1000 + 60 * 35 + 7)[0])     # frame 11 at minute 35: frame 6 would be minute 30
    assert idx[j, 1] == -1 and idx[j, 0] >= 0
    ok = (idx[:, :2] >= 0).all(1) & (idx[:, 3] >= 0)
    D = 4
    coarse = np.random.default_rng(0).normal(size=(len(unix), 7, D))
    fine = np.random.default_rng(1).normal(size=(len(unix), 24, D))
    x = assemble(coarse, fine, idx[ok])
    assert x["E"].shape == (ok.sum(), 3, 7, D) and x["fine"].shape == (ok.sum(), 2, 24, D)
    assert x["ref"].shape == (ok.sum(), 24, D)
    i = int(np.flatnonzero(ok)[0])
    assert np.allclose(x["fine"][0, 1], fine[i]) and np.allclose(x["ref"][0], fine[idx[i, 3]])
    with pytest.raises(ValueError):
        assemble(coarse, fine, idx[:5])


def test_rules_and_events():
    p = np.array([0.1, 0.5, 0.1, 0.5, 0.5, np.nan, 0.5, 0.9, 0.1])
    assert fires(Rule("single", 0.4), p).tolist() == [0, 1, 0, 1, 1, 0, 1, 1, 0]
    two_of_three = fires(Rule("persist", 0.4, k=2, m=3), p)
    assert two_of_three.tolist() == [0, 0, 0, 1, 1, 0, 1, 1, 1]          # minute 5 is missing, not fired
    bypass = fires(Rule("persist", 0.4, k=3, m=3, T2=0.8), p)
    assert bypass.tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 0]
    sm = fires(Rule("smooth", 0.4, m=2), p)
    assert sm[4] and not sm[2] and not sm[5]
    assert events([0, 5, 29, 30, 31, 70], holdoff=30) == 3


def test_evaluate_fire_delay_and_false_alarms():
    off = np.arange(-20, 21, dtype=float)
    p_fire = np.where(off >= 3, 0.9, 0.1)
    p_fire[2] = 0.95                                         # one false alarm before the plume
    p_fire[np.isclose(off, 10)] = np.nan
    seqs = [{"p": p_fire, "off": off, "kind": "fire"},
            {"p": np.full(41, 0.1), "off": off, "kind": "fire"},                     # never detected
            {"p": np.r_[np.full(30, 0.1), [0.9], np.full(29, 0.1)], "off": np.full(60, np.nan), "kind": "ordinary"}]
    r = evaluate(Rule("single", 0.5), seqs)
    # the false alarm at -18 starts an event; its 30-min hold-off holds back the fire's alarm until +12
    assert r["fires"] == 2 and r["fires_within_5"] == 0.0 and r["fires_within_15"] == 0.5
    assert r["median_delay_detected"] == 12 and r["fires_detected"] == 0.5
    assert r["fires_firing_within_5"] == 0.5                  # counting any firing after the plume: +3
    assert evaluate(Rule("single", 0.5), seqs, holdoff=10)["median_delay_detected"] == 3
    assert r["fire_pre_events"] == 1 and r["fire_pre_hours"] == pytest.approx(40 / 60)
    assert r["ordinary_events"] == 1 and r["ordinary_hours"] == pytest.approx(1.0)
    assert r["events_per_hour"] == pytest.approx(2 / (100 / 60))
    r2 = evaluate(Rule("persist", 0.5, k=2, m=2), seqs)
    assert r2["fire_pre_events"] == 0 and r2["ordinary_events"] == 0 and r2["median_delay_detected"] == 4
    assert math.isfinite(r2["events_per_hour"])


def _score_npz(path, kind, rows):
    cols = {k: [] for k in ("seq", "camera", "mount", "t_end", "off_end_s", "p_smoke", "p_fog", "fog_veto")}
    for r in rows:
        for k in cols:
            cols[k].append(r[k])
    np.savez(path, kind=np.array(kind), **{k: np.array(v) for k, v in cols.items()})


def test_alarm_rules_tune_and_evaluate(tmp_path):
    import alarm_rules

    rng = np.random.default_rng(0)
    rows_f, rows_o = [], []
    for s in range(12):
        mount = f"m{s % 6}"
        for minute in range(60):
            off = minute - 30
            p = 0.05 + 0.03 * rng.random()
            if off >= 2:
                p = 0.8
            if minute == 10 + s:
                p = 0.6                                          # a lone false-alarm spike before the plume
            rows_f.append({"seq": f"f{s}", "camera": "c", "mount": mount, "t_end": 1000 + 60 * minute,
                           "off_end_s": 60.0 * off, "p_smoke": p, "p_fog": 0.0, "fog_veto": 0.95})
        for minute in range(40):
            p = 0.05 if minute != 20 else 0.7
            rows_o.append({"seq": f"o{s}", "camera": "c", "mount": mount, "t_end": 5000 + 60 * minute,
                           "off_end_s": np.nan, "p_smoke": p, "p_fog": 0.99 if minute == 30 else 0.0,
                           "fog_veto": 0.95})
    _score_npz(tmp_path / "f.npz", "fire", rows_f)
    _score_npz(tmp_path / "o.npz", "ordinary", rows_o)
    seqs, _ = alarm_rules.load_sequences([tmp_path / "f.npz", tmp_path / "o.npz"])
    assert len(seqs) == 24 and seqs[0]["off"][0] == -30
    out = tmp_path / "tune.json"
    assert alarm_rules.main(["tune", "--scores", str(tmp_path / "f.npz"), "--scores", str(tmp_path / "o.npz"),
                             "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    b = doc["baselines"]["R0_deploy"]
    assert b["fire_pre_events"] == 12 and b["ordinary_events"] == 12            # every spike alarms
    ch = doc["chosen"]
    assert ch["events_per_hour"] == 0 and ch["fires_within_15"] >= b["fires_within_15"]
    ev = tmp_path / "eval.json"
    assert alarm_rules.main(["evaluate", "--chosen", str(out), "--scores", str(tmp_path / "f.npz"),
                             "--scores", str(tmp_path / "o.npz"), "--out", str(ev), "--n-boot", "20"]) == 0
    rep = json.loads(ev.read_text())
    assert rep["targets"]["all_met"] and "chosen - R0_deploy" in ev.with_suffix(".md").read_text()
    assert rep["paired_chosen_minus_deploy"]["value"]["events_per_hour"] < 0


def test_score_dense(tmp_path, monkeypatch):
    import score_dense

    class Stub:
        thresholds = {"smoke": {"deploy": 0.5, "fog_veto": 0.95}}

        def proba(self, E, motion, fine, zoom, ref):
            assert zoom is None and ref is not None and fine.shape[1:3] == (2, 24)
            return {"smoke": fine[:, 1, 0, 0].astype(float), "fog": np.zeros(len(E))}

        def stack_inputs(self, head, E, motion, fine, zoom, ref):
            return {"head": np.zeros(len(E)), "tileref": fine[:, 1, 0, 0].astype(float) - 0.5}

    monkeypatch.setattr(score_dense.Detector, "load", staticmethod(lambda d: Stub()))
    dense = tmp_path / "dense"
    dense.mkdir()
    unix = 1000 + 60 * np.arange(50)
    fine = np.zeros((50, 24, 3), np.float16)
    fine[:, 0, 0] = np.linspace(0, 1, 50)
    np.savez(dense / "seqA.npz", unix=unix, offset_s=60 * (np.arange(50) - 30), coarse=np.zeros((50, 7, 3)),
             fine=fine)
    man = tmp_path / "man.csv"
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["video_path", "camera_id", "camera_mount"])
        w.writeheader()
        w.writerow({"video_path": "x/seqA", "camera_id": "cam-s-mobo-c", "camera_mount": "cam-s"})
    out = tmp_path / "s.npz"
    assert score_dense.main(["--dense-dir", str(dense), "--kind", "fire", "--manifest", str(man),
                             "--detector", "x", "--out", str(out)]) == 0
    z = np.load(out)
    assert len(z["seq"]) == 50 - 25 and z["off_end_s"][0] == 60 * (25 - 30)         # first window with a reference
    assert np.allclose(z["p_smoke"], fine[25:, 0, 0].astype(float), atol=1e-3) and z["fog_veto"][0] == 0.95
    assert np.allclose(z["input_tileref"], z["p_smoke"] - 0.5) and len(z["input_head"]) == len(z["seq"])


def test_change_gate():
    p = np.array([0.9, 0.9, 0.9, 0.9])
    g = np.array([-3.0, 1.0, np.nan, 2.0])
    assert fires(Rule("single", 0.5, G=0.0), p, g).tolist() == [0, 1, 0, 1]
    assert fires(Rule("persist", 0.5, k=2, m=2, G=0.0), p, g).tolist() == [0, 0, 0, 0]
    with pytest.raises(ValueError):
        fires(Rule("single", 0.5, G=0.0), p)
    seqs = [{"p": p, "g": g, "off": np.array([-2.0, -1.0, 0.0, 1.0]), "kind": "fire"}]
    r = evaluate(Rule("single", 0.5, G=0.0), seqs, holdoff=1)
    assert r["fire_pre_events"] == 1 and r["median_delay_detected"] == 1
