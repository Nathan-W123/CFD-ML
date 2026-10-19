"""Training, calibration, metrics, baselines, leakage checks and evaluation.json (src.training, src.data.dataset)."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.constants import CATEGORIES, CATEGORY_LABELS
from src.data.dataset import (FeatureSet, SequenceTensorDataset, assert_no_group_leakage, load_sequence,
                              save_sequence)
from src.data.feature_spec import FEATURE_INDEX, gate_masks
from src.data.normalization import FeatureNormalizer
from src.models.physics_gru import ModelConfig, PhysicsGRU
from src.training import metrics as M
from src.training.baselines import HeuristicBaseline, flags_only_probe, pooled_features, univariate_aucs
from src.training.evaluate import run_evaluation
from src.training.train import fit_calibration, model_logits, run_finetune, run_pretraining, train_model
from src.utils.config import load_training_config
from src.utils.io import read_json, read_yaml, write_json
from src.utils.runtime import seed_everything

SIMS = {"train": 40, "val_select": 10, "val_calib": 20, "test_iid": 12, "test_ood": 6}


def make_feature_set(sims=SIMS, variants=2, seed=0, source="synthetic", flags_leak=False, groups=None):
    """Synthetic FeatureSet with a learnable but overlapping signal.

    Each simulation has latent strengths a_s ~ N(smoke, 0.8) and a_f ~ N(fog, 0.8):
    smoke rises (flow_v_mean, centroid_dz > 0) from a stable base (anchor
    stability), fog is large (region_area) and sinks slightly. Rendered
    variants share the latents; per-frame noise on top.
    """
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("X", "y", "seq", "sim", "grp", "split", "cat", "reg", "tag", "msrc", "meta")}
    for s_i, (split, n) in enumerate(sims.items()):
        for c in CATEGORIES:
            s_lab, f_lab = CATEGORY_LABELS[c]
            for i in range(n):
                sim_id = f"{split}-{c}-{i:04d}"
                a_s, a_f = s_lab + 0.8 * rng.normal(), f_lab + 0.8 * rng.normal()
                wind = float(rng.uniform(0, 9 if split == "test_ood" else 7))
                for v in range(variants):
                    x = rng.normal(size=(12, 14)) * 0.01
                    x[:, FEATURE_INDEX["flow_v_mean"]] += 0.02 * a_s - 0.01 * a_f
                    x[:, FEATURE_INDEX["centroid_dz"]] += 0.01 * a_s
                    x[:, FEATURE_INDEX["region_area"]] = np.abs(0.2 + 0.08 * a_f + x[:, 6])
                    x[:, FEATURE_INDEX["anchor_stability"]] = np.clip(0.4 + 0.15 * a_s + x[:, 11], 0, 1)
                    x[:, FEATURE_INDEX["flow_u_mean"]] += 0.01 * (wind - 3.5)
                    p_valid = 0.95 if not flags_leak else (0.95 if s_lab else 0.5)
                    x[:, 12] = (rng.random(12) < p_valid).astype(float)
                    x[:, 13] = x[:, 12] * (rng.random(12) < 0.9)
                    cols["X"].append(np.where(gate_masks(x), x, 0.0))
                    cols["y"].append([s_lab, f_lab])
                    cols["seq"].append(f"{sim_id}-v{v}")
                    cols["sim"].append(sim_id)
                    cols["grp"].append(sim_id if groups is None else groups(split, c, i))
                    cols["split"].append(split)
                    cols["cat"].append(c)
                    cols["reg"].append(f"{c}_{'ab'[i % 2]}")
                    cols["tag"].append("fast_wind" if split == "test_ood" else "")
                    cols["msrc"].append(("degraded", "bbox", "detector")[(i + v) % 3])
                    cols["meta"].append({"seed": 100000 * s_i + 1000 * CATEGORIES.index(c) + i,
                                         "context": {"wind_speed_ms": wind, "sim": sim_id},
                                         "render": {"render_contrast": float(rng.uniform(0.6, 1.3))}})
    return FeatureSet(X=np.array(cols["X"], np.float32), y=np.array(cols["y"], np.float32),
                      seq_ids=np.array(cols["seq"]), sim_ids=np.array(cols["sim"]), groups=np.array(cols["grp"]),
                      splits=np.array(cols["split"]), categories=np.array(cols["cat"]), regimes=np.array(cols["reg"]),
                      tags=np.array(cols["tag"]), mask_sources=np.array(cols["msrc"]), meta=cols["meta"],
                      source=source)


@pytest.fixture(scope="module")
def fs():
    return make_feature_set()


def _cfg(*overrides):
    return load_training_config(overrides=["training.seeds=[0]", "training.max_epochs=12",
                                           "evaluation.bootstrap_resamples=50", *overrides])


def _train_one(fs, seed=0, max_epochs=12, patience=7, y_train=None):
    tr, vs = fs.split("train"), fs.split("val_select")
    norm = FeatureNormalizer().fit(tr.X)
    seed_everything(seed)
    model = PhysicsGRU(ModelConfig())
    tcfg = _cfg()["training"]
    res = train_model(model, (norm.transform(tr.X), tr.y if y_train is None else y_train),
                      (norm.transform(vs.X), vs.y), tcfg, seed, max_epochs=max_epochs, patience=patience)
    return res, norm


# ============================================================================= training
def test_training_learns_the_signal_on_one_thread(fs):
    assert torch.get_num_threads() == 1
    res, norm = _train_one(fs)
    test = fs.split("test_iid")
    z = model_logits(res.model, norm, test.X)
    for j in range(2):
        assert M.auroc(z[:, j], test.y[:, j]) > 0.75          # chance is 0.5
    assert res.history[0]["val_loss"] > res.best_val_loss
    assert {"epoch", "train_loss", "val_loss", "val_auroc_smoke", "val_auroc_fog"} <= set(res.history[0])


def test_same_seed_gives_identical_weights(fs):
    a, _ = _train_one(fs, seed=3, max_epochs=4)
    b, _ = _train_one(fs, seed=3, max_epochs=4)
    c, _ = _train_one(fs, seed=4, max_epochs=4)
    sa, sb, sc = a.model.state_dict(), b.model.state_dict(), c.model.state_dict()
    for k in sa:
        assert torch.allclose(sa[k], sb[k], atol=1e-6, rtol=0), k
    assert any(not torch.allclose(sa[k], sc[k]) for k in sa)


def test_early_stopping_triggers_and_restores_the_best_epoch(fs):
    rng = np.random.default_rng(0)
    tr = fs.split("train")
    shuffled = tr.y[rng.permutation(len(tr.y))]          # no signal: val loss stops improving quickly
    res, norm = _train_one(fs, max_epochs=60, patience=2, y_train=shuffled)
    assert res.stopped_early and res.epochs_run < 60
    assert res.epochs_run == res.best_epoch + 2
    vs = fs.split("val_select")
    z = torch.from_numpy(model_logits(res.model, norm, vs.X)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, torch.from_numpy(vs.y))
    assert float(loss) == pytest.approx(res.best_val_loss, abs=1e-5)


def test_pos_weight_only_when_imbalanced(fs):
    from src.training.train import _label_balance

    pw, mode = _label_balance(fs.split("train").y, "auto")
    assert pw is None and mode == "none"
    y = np.array([[1, 0]] * 10 + [[0, 0]] * 30 + [[0, 1]] * 35, dtype=np.float32)   # fog 35:40 is balanced
    pw, mode = _label_balance(y, "auto")
    assert mode == "pos_weight" and pw[0] == pytest.approx(65 / 10) and pw[1] == 1.0


def test_calibration_and_thresholds_come_only_from_val_calib(tmp_path, fs):
    cfg = _cfg("training.max_epochs=6")
    run_pretraining(fs, cfg, tmp_path / "a", log=None)
    perturbed = copy.deepcopy(fs)
    test = np.isin(perturbed.splits, ["test_iid", "test_ood"])
    rng = np.random.default_rng(1)
    perturbed.X[test, :, :12] = rng.normal(size=perturbed.X[test, :, :12].shape).astype(np.float32)
    perturbed.y[test] = 1.0 - perturbed.y[test]
    run_pretraining(perturbed, cfg, tmp_path / "b", log=None)
    ca, cb = read_yaml(tmp_path / "a" / "model_config.yaml"), read_yaml(tmp_path / "b" / "model_config.yaml")
    for key in ("temperatures", "thresholds", "abstain"):
        assert ca[key] == cb[key], key
    assert ca["calibration"]["fit_source"] == "synthetic_val_calib"
    assert ca["calibration"]["n_sequences"] == int(np.sum(fs.splits == "val_calib"))
    na = read_json(tmp_path / "a" / "feature_normalizer.json")
    assert na == read_json(tmp_path / "b" / "feature_normalizer.json")
    assert na["n_sequences"] == int(np.sum(fs.splits == "train"))
    # ... while changing val_calib does change them
    shifted = copy.deepcopy(fs)
    vc = shifted.splits == "val_calib"
    shifted.y[vc, 0] = np.where(np.arange(vc.sum()) % 3 == 0, 1.0 - shifted.y[vc, 0], shifted.y[vc, 0])
    run_pretraining(shifted, cfg, tmp_path / "c", log=None)
    cc = read_yaml(tmp_path / "c" / "model_config.yaml")
    assert cc["thresholds"] != ca["thresholds"] or cc["temperatures"] != ca["temperatures"]


def test_fit_calibration_operating_points(fs):
    res, norm = _train_one(fs)
    vc = fs.split("val_calib")
    cal = fit_calibration(model_logits(res.model, norm, vc.X), vc, _cfg(), "synthetic_val_calib")
    for name in ("smoke", "fog"):
        assert 0.5 <= cal["temperatures"][name] <= 5.0
        bal, lfa = cal["thresholds"]["balanced"][name], cal["thresholds"]["low_false_alarm"][name]
        assert lfa["logit"] >= bal["logit"]
        for op in ("balanced", "low_false_alarm"):
            assert cal["abstain"]["margins"][op][name] >= 0.25
    z = model_logits(res.model, norm, vc.X)[:, 0] / cal["temperatures"]["smoke"]
    fog_only = vc.categories == "fog_only"
    w = M.per_group_weights(vc.groups)
    rate = np.sum(w[fog_only] * (z[fog_only] >= cal["thresholds"]["low_false_alarm"]["smoke"]["logit"])) \
        / np.sum(w[fog_only])
    assert rate <= 0.05 + 1e-9
    assert cal["calibration"]["groups_per_category"]["fog_only"] == 20
    assert cal["calibration"]["insufficient_calibration_data"] is False
    assert cal["calibration"]["category_support"] == {"status": "applicable"}
    assert cal["calibration"]["untrained_labels"] == []


# ============================================================================= metrics
def test_threshold_metrics_by_hand():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    p = np.array([1, 0, 1, 1, 0, 0, 0, 1])
    r = M.precision_recall_f1(y, p)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (3, 1, 1, 3)
    assert r["precision"] == pytest.approx(0.75) and r["recall"] == pytest.approx(0.75)
    assert r["f1"] == pytest.approx(0.75)
    assert M.precision_recall_f1([0, 0], [0, 0])["precision"] is None


def test_ranking_and_calibration_metrics_by_hand():
    s = np.array([0.1, 0.4, 0.35, 0.8])
    y = np.array([0, 0, 1, 1])
    assert M.auroc(s, y) == pytest.approx(0.75)
    assert M.auroc([0.5, 0.5, 0.5, 0.5], y) == pytest.approx(0.5)          # ties count one half
    assert M.auroc(s, [1, 1, 1, 1]) is None
    # average precision: thresholds 0.8 (P=1, R=.5), 0.4 (P=.5, R=.5), 0.35 (P=2/3, R=1)
    assert M.auprc(s, y) == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))
    prob = np.array([0.1, 0.2, 0.7, 0.9])
    assert M.brier_score(prob, y) == pytest.approx(np.mean([0.01, 0.04, 0.09, 0.01]))
    # equal-mass bins of two: (0.1, 0.2 | y 0, 0) and (0.7, 0.9 | y 1, 1)
    assert M.expected_calibration_error(prob, y, n_bins=2) == pytest.approx(0.5 * 0.15 + 0.5 * 0.2)
    bins = M.reliability_bins(np.linspace(0, 1, 10), np.zeros(10), n_bins=3)
    assert [b["n"] for b in bins] == [4, 3, 3]


def test_decision_metrics_by_hand():
    y = np.array([[1, 0], [0, 1], [0, 1], [0, 0], [1, 1], [0, 1]])
    d = np.array([[1, 0], [1, 0], [0, 1], [-1, 0], [1, 1], [1, 1]])
    conf = M.confusion_4class(y, d)
    m = np.array(conf["matrix"])
    assert conf["classes"] == list(CATEGORIES)
    assert m[0, 0] == 1 and m[1, 0] == 1 and m[1, 1] == 1 and m[1, 3] == 1 and m[3, 3] == 1
    assert conf["abstained"] == [0, 0, 1, 0] and m.sum() == 5
    sf = M.smoke_fog_confusion(conf)
    assert sf["fog_only_as_smoke_only"] == pytest.approx(1 / 3)
    assert sf["fog_only_with_smoke"] == pytest.approx(2 / 3)
    cats = np.array(["smoke_only", "fog_only", "fog_only", "neither", "both", "fog_only"])
    fsr = M.false_smoke_rate(d[:, 0], cats)
    assert fsr["rate"] == pytest.approx(2 / 3) and fsr["n_fog_only"] == 3
    ab = M.abstain_rate(d)
    assert ab["smoke"] == pytest.approx(1 / 6) and ab["fog"] == 0.0 and ab["any"] == pytest.approx(1 / 6)
    sel = M.selective_accuracy(y, d)
    assert sel["smoke"] == pytest.approx(3 / 5) and sel["fog"] == pytest.approx(5 / 6)
    assert sel["joint"] == pytest.approx(3 / 5)


def test_decision_rule_fitting():
    rng = np.random.default_rng(0)
    z = rng.normal(size=20000) * 3
    y = rng.random(20000) < M.sigmoid(z / 2.0)               # true temperature 2
    T, info = M.fit_temperature(z, y)
    assert T == pytest.approx(2.0, rel=0.08) and not info["fallback"]
    T, info = M.fit_temperature([-3, -2, 2, 3], [0, 0, 1, 1])  # separable: optimum at the bound
    assert T == 1.0 and info["fallback"] and info["at_bound"] == "lower" and info["reason"] == "separable"
    z = rng.normal(size=2000) * 20                              # confident logits, labels unrelated to them
    T, info = M.fit_temperature(z, rng.random(2000) < 0.5)
    assert T == 5.0 and info["at_bound"] == "upper" and info["reason"] == "optimum_at_upper_bound"
    assert info["clipped"] and not info["fallback"] and info["nll"] < info["nll_T1"]
    thr, info = M.threshold_max_f1([0.1, 0.2, 0.6, 0.7, 0.9], [0, 0, 1, 1, 1])
    assert 0.2 < thr < 0.6 and info["f1"] == pytest.approx(1.0)
    neg = np.array([1, 1, 1, 1, 0, 0], bool)
    thr, info = M.threshold_max_false_rate([0.1, 0.2, 0.3, 0.8, 0.5, 0.9], neg, max_rate=0.25)
    assert 0.3 < thr <= 0.5 and info["rate"] == pytest.approx(0.25)
    s = np.array([-3.0, -0.1, 0.05, 0.2, 2.0, 3.0])
    m, info = M.fit_abstain_margin(s, [0, 1, 0, 1, 1, 1], 0.0, floor=0.25, target_error=0.05, max_rate=0.5)
    assert m >= 0.25 and info["confident_error"] <= 0.05 and info["abstain_rate"] <= 0.5


def test_grouped_bootstrap_resamples_groups():
    y = np.array([1, 1, 0, 0, 1, 0])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    ci = M.grouped_bootstrap(lambda idx: {"mean": float(np.mean(y[idx]))}, groups, n_boot=300, seed=0)
    assert 0.0 <= ci["mean"]["lo"] <= 0.5 <= ci["mean"]["hi"] <= 1.0
    # with whole groups resampled, a mean can only take values k/6 for group-level sums
    vals = []
    M.grouped_bootstrap(lambda idx: vals.append(len(idx)) or {}, groups, n_boot=50, seed=0)
    assert set(vals) == {6}
    again = M.grouped_bootstrap(lambda idx: {"mean": float(np.mean(y[idx]))}, groups, n_boot=300, seed=0)
    assert again == ci


# ============================================================================= baselines
def test_baselines_and_shortcut_probes(fs):
    h = HeuristicBaseline().fit(fs.split("train")).calibrate(fs.split("val_calib"), _cfg())
    test = fs.split("test_iid")
    z = h.logits(test.X)
    assert M.auroc(z[:, 0], test.y[:, 0]) > 0.7
    assert set(h.describe()["coefficients"]["smoke"]) == {"intercept", "flow_v_mean_mean", "centroid_dz_mean",
                                                           "anchor_stability_mean", "area_growth_mean"}
    assert pooled_features(fs.X).shape == (len(fs), 28)
    assert not flags_only_probe(fs)["flag"]
    leaky = make_feature_set(flags_leak=True)
    probe = flags_only_probe(leaky)
    assert probe["flag"] and any(f["label"] == "smoke" for f in probe["flagged"])
    uni = univariate_aucs(fs)
    assert next(iter(uni["per_label"]["fog"])) in ("region_area_mean", "flow_v_mean_mean")


def test_baselines_and_audit_skip_without_train_or_val_calib(tmp_path, fs):
    from src.training.baselines import run_baselines
    from src.training.evaluate import shortcut_audit

    test_only = fs.split(["test_iid", "test_ood"])         # e.g. a test-only detector-mask re-extraction
    cfg = _cfg()
    base = run_baselines(test_only, cfg)
    assert set(base) == {"A_heuristic", "pooled_logistic"}
    assert all(b["status"] == "skipped" and "train" in b["reason"] for b in base.values())
    no_calib = fs.split(["train", "test_iid"])
    assert all(b["status"] == "skipped" and "val_calib" in b["reason"] for b in run_baselines(no_calib, cfg).values())
    audit = shortcut_audit(test_only, cfg)
    assert audit["flags_only"]["status"] == "skipped" and not audit["flags_only"]["flag"]
    assert audit["univariate_feature_aucs"]["status"] == "skipped"
    write_json(tmp_path / "baselines.json", {"baselines": base, "shortcut_audit": audit})   # finite JSON only


def test_preprocessing_record_takes_timing_from_the_extraction_meta(fs):
    from src.training.train import _preprocessing_record

    from dataclasses import asdict

    from src.data.features import FeatureConfig

    cfg = _cfg()
    tr = fs.split("train")
    defaults = {k: v for k, v in asdict(FeatureConfig()).items() if k != "roi_scale"}
    rec = _preprocessing_record(tr, cfg)                   # nothing recorded: the full code defaults
    assert rec["feature_config"] == defaults
    assert rec["feature_config_source"] == "code_defaults_at_training"
    # The timing tolerance max_offset_s is part of the timing record (far-field profile: 10 s).
    assert rec["feature_timing_source"] == {"expected_interval_s": "default", "flow_lag_s": "default",
                                            "max_offset_s": "default"}
    timed = copy.deepcopy(tr)
    for m in timed.meta:
        m["feature_meta"] = {"flow_mode": "lag", "timing": {"expected_interval_s": 4.0, "expected_flow_lag_s": 0.5}}
        m["feature_config"] = dict(defaults, flow_erode_px=3, roi_scale=1.7)
    rec = _preprocessing_record(timed, cfg)
    assert rec["feature_config"] == dict(defaults, flow_erode_px=3, expected_interval_s=4.0, flow_lag_s=0.5)
    assert rec["feature_config_source"] == "train_feature_meta" and "roi_scale" not in rec["feature_config"]
    assert set(rec["feature_timing_source"].values()) == {"train_feature_meta"}
    assert rec["training_flow_modes"] == {"lag": len(timed)}
    # The exported settings rebuild the extractor configuration at inference.
    FeatureConfig(**rec["feature_config"])
    # Consecutive-flow rows (far-field profile): lag None / 0 and the recorded timing tolerance are exported.
    far = copy.deepcopy(timed)
    for m in far.meta:
        m["feature_meta"] = {"flow_mode": "consecutive",
                             "timing": {"expected_interval_s": 60.0, "expected_flow_lag_s": 0.0, "max_offset_s": 10.0}}
        m["feature_config"] = dict(defaults, expected_interval_s=60.0, flow_lag_s=None, max_offset_s=10.0)
    rec = _preprocessing_record(far, cfg)
    assert rec["feature_config"]["flow_lag_s"] == 0.0 and rec["feature_config"]["max_offset_s"] == 10.0
    assert rec["feature_config"]["expected_interval_s"] == 60.0
    assert rec["training_flow_modes"] == {"consecutive": len(far)}
    timed.meta[0]["feature_config"] = dict(defaults, flow_erode_px=2)
    with pytest.raises(ValueError, match="feature configs"):
        _preprocessing_record(timed, cfg)
    timed.meta[0]["feature_config"] = dict(defaults, flow_erode_px=3)
    timed.meta[0]["feature_meta"]["timing"]["expected_flow_lag_s"] = 1.0
    with pytest.raises(ValueError, match="flow_lag_s"):
        _preprocessing_record(timed, cfg)


# ============================================================================= data
def test_leakage_check(fs):
    assert_no_group_leakage(fs)
    bad = copy.deepcopy(fs)
    i = int(np.flatnonzero(bad.splits == "test_iid")[0])
    j = int(np.flatnonzero(bad.splits == "train")[0])
    bad.groups[i] = bad.groups[j]
    with pytest.raises(ValueError, match="group"):
        assert_no_group_leakage(bad)
    bad = copy.deepcopy(fs)
    bad.meta[i] = dict(bad.meta[i], seed=bad.meta[j]["seed"])
    with pytest.raises(ValueError, match="seed"):
        assert_no_group_leakage(bad)
    bad = copy.deepcopy(fs)
    bad.meta[i] = dict(bad.meta[i], context=bad.meta[j]["context"])
    with pytest.raises(ValueError, match="parameter"):
        assert_no_group_leakage(bad)
    # robustness re-renders of test_iid simulations are not leakage
    rob = fs.split("test_iid")
    rob = FeatureSet(X=np.concatenate([rob.X, rob.X]), y=np.concatenate([rob.y, rob.y]),
                     seq_ids=np.r_[[s + "-a" for s in rob.seq_ids], [s + "-b" for s in rob.seq_ids]],
                     sim_ids=np.r_[rob.sim_ids, rob.sim_ids], groups=np.r_[rob.groups, rob.groups],
                     splits=np.r_[["robust:nominal"] * len(rob), ["robust:shake_heavy"] * len(rob)],
                     categories=np.r_[rob.categories, rob.categories], regimes=np.r_[rob.regimes, rob.regimes],
                     tags=np.r_[rob.tags, rob.tags], mask_sources=np.r_[rob.mask_sources, rob.mask_sources],
                     meta=rob.meta + rob.meta)
    assert_no_group_leakage(rob)


def test_featureset_roundtrip_subset_and_torch_dataset(tmp_path, fs):
    path = fs.save(tmp_path / "features.npz")
    back = FeatureSet.load(path)
    np.testing.assert_array_equal(back.X, fs.X)
    np.testing.assert_array_equal(back.splits, fs.splits)
    assert back.meta == fs.meta and back.source == "synthetic"
    tr = back.split("train")
    assert len(tr) == 2 * 4 * SIMS["train"] and set(tr.splits) == {"train"}
    assert len(back.split(["val_select", "val_calib"])) == 2 * 4 * (SIMS["val_select"] + SIMS["val_calib"])
    norm = FeatureNormalizer().fit(tr.X)
    ds = SequenceTensorDataset(tr, normalizer=norm)
    x, y = ds[0]
    assert x.shape == (12, 14) and y.shape == (2,) and len(ds) == len(tr)
    assert torch.allclose(x, torch.from_numpy(norm.transform(tr.X[0])))


def test_sequence_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    T, H, W = 12, 24, 30
    rendered = SimpleNamespace(
        frames=rng.integers(0, 255, (T, H, W), dtype=np.uint8), lag_frames=rng.integers(0, 255, (T, H, W), np.uint8),
        masks=rng.random((T, H, W)) > 0.7, timestamps=np.arange(T) * 5.0, lag_timestamps=np.arange(T) * 5.0 - 1.0,
        mask_source="bbox", opacity=rng.random((T, H, W)).astype(np.float32),
        component_opacity={"smoke": rng.random((T, H, W)).astype(np.float16)},
        visibility={"peak": [0.3] * T}, render_params={"zoom": 0.8, "seed": 7})
    sim = SimpleNamespace(sim_id="train-smoke_only-0000-a0", category="smoke_only", regime="hot_plume",
                          labels=(1, 0), seed=1234, params={"split": "train", "wind": {"speed_ms": 2.0}},
                          valid=True, validity={"max_mach": np.float64(0.1), "reasons": []}, dx=5.0)
    path = save_sequence(tmp_path / "seq", rendered, sim, "s-v0", extra={"render_info": {"variant": 0},
                                                                          "diagnostics": True})
    seq = load_sequence(path)
    np.testing.assert_array_equal(seq["frames"], rendered.frames)
    np.testing.assert_array_equal(seq["lag_frames"], rendered.lag_frames)
    np.testing.assert_array_equal(seq["masks"], rendered.masks)
    np.testing.assert_array_equal(seq["lag_timestamps"], rendered.lag_timestamps)
    assert seq["labels"] == (1, 0) and seq["seed"] == 1234 and seq["mask_source"] == "bbox"
    assert seq["params"]["wind"]["speed_ms"] == 2.0 and seq["validity"]["max_mach"] == pytest.approx(0.1)
    assert seq["render_params"]["zoom"] == 0.8 and seq["extra"]["render_info"]["variant"] == 0
    assert seq["opacity"].dtype == np.float16 and "smoke" in seq["component_opacity"]
    lean = load_sequence(save_sequence(tmp_path / "lean.npz", rendered, sim, "s-v1"))
    assert "opacity" not in lean


# ============================================================================= Stage B and evaluation.json
def test_finetune_and_scratch_differ_only_in_init_and_lr(tmp_path, fs):
    cfg = _cfg("training.max_epochs=4", "finetune.max_epochs=3")
    run_pretraining(fs, cfg, tmp_path / "pre", log=None)
    real = make_feature_set(sims={"train": 10, "val_select": 4, "val_calib": 5, "test_iid": 4}, seed=9,
                            source="real", groups=lambda split, c, i: f"cam-{split}-{i % 3}")
    d = run_finetune(real, cfg, tmp_path / "pre", tmp_path / "D", log=None)
    b = run_finetune(real, cfg, tmp_path / "pre", tmp_path / "B", from_scratch=True, log=None)
    cd, cb = read_yaml(tmp_path / "D" / "model_config.yaml"), read_yaml(tmp_path / "B" / "model_config.yaml")
    assert d["experiment"] == "D" and b["experiment"] == "B"
    assert cd["fit"]["training"]["lr"] == pytest.approx(1e-4) and cb["fit"]["training"]["lr"] == pytest.approx(1e-3)
    assert cd["fit"]["training"]["max_epochs"] == cb["fit"]["training"]["max_epochs"] == 3
    assert cd["calibration"]["fit_source"] == "real_val_calib"
    assert cd["fit"]["real_counts"]["train"]["cameras"] == 3
    assert cd["fit"]["init"].startswith("pretrained") and cb["fit"]["init"] == "random"


def test_evaluation_json_schema(tmp_path, fs):
    cfg = _cfg("training.seeds=[0,1]", "training.max_epochs=4")
    run_pretraining(fs, cfg, tmp_path / "art", log=None)
    rob = fs.split("test_iid")
    rob.splits = np.full(len(rob), "robust:nominal")
    doc = run_evaluation(fs, cfg, tmp_path / "art", tmp_path / "reports", robustness_fs=rob, log=None)
    assert set(doc) >= {"synthetic", "real", "ablations", "synthetic_proxy", "shortcut_audit", "robustness",
                        "loro", "data"}
    assert doc["real"]["status"] == "pending" and doc["real"]["reason"]
    ab = doc["ablations"]
    assert ab["B"] is None and ab["D"] is None and ab["reasons"]["B"] and ab["reasons"]["D"]
    assert ab["A"]["test"]["test_iid"]["balanced"]["smoke"]["f1"] is not None
    assert ab["C"]["test"]["test_ood"]["balanced"]["n"] == 2 * 4 * SIMS["test_ood"]
    syn = doc["synthetic"]
    assert syn["calibration_source"] == "synthetic_val_calib"
    ov = syn["test"]["test_iid"]["balanced"]["overall"]
    ci = ov["ci95"]["smoke.f1"]
    assert ci["lo"] <= ov["per_label"]["smoke"]["f1"] <= ci["hi"]
    assert set(syn["test"]["test_iid"]["balanced"]["slices"]) == {"category", "regime", "holdout_tag", "wind_bin",
                                                                   "mask_source"}
    roc = syn["curves"]["test_iid"]["smoke"]["roc"]
    assert roc["fpr"][0] == 0.0 and roc["tpr"][-1] == 1.0 and roc["thresholds"][0] is None
    ms = syn["seeds"]["mean_sd"]["test_iid.balanced.smoke.f1"]
    assert ms["n_seeds"] == 2 and ms["sd"] >= 0
    assert "flags_only" in doc["shortcut_audit"] and "parameters" in doc["shortcut_audit"]
    assert "nominal" in doc["robustness"]["suites"]
    assert doc["data"]["counts"]["train"]["simulations"] == 4 * SIMS["train"]
    on_disk = json.loads((tmp_path / "art" / "evaluation.json").read_text())
    assert on_disk["real"]["status"] == "pending"
    assert all((tmp_path / "reports" / f).exists() for f in ("confusion_test_iid.png", "reliability_test_ood.png",
                                                             "roc_pr_test_iid.png"))
    # a test-only FeatureSet (e.g. a detector-mask re-extraction): baselines skipped, finite JSON still written
    doc = run_evaluation(fs.split(["test_iid", "test_ood"]), cfg, tmp_path / "art", tmp_path / "reports_t",
                         n_boot=10, out_path=tmp_path / "eval_test_only.json", log=None)
    ab = read_json(tmp_path / "eval_test_only.json")["ablations"]
    assert ab["A"]["status"] == "skipped" and ab["extra_baselines"]["pooled_logistic"]["status"] == "skipped"
    assert doc["shortcut_audit"]["flags_only"]["status"] == "skipped"
    assert doc["synthetic"]["test"]["test_iid"]["balanced"]["overall"]["n"] == 2 * 4 * SIMS["test_iid"]


def test_loro_and_synthetic_proxy(tmp_path, fs):
    from src.training.evaluate import resplit_by_group, run_synthetic_proxy
    from src.training.train import run_loro

    cfg = _cfg("training.max_epochs=4", "finetune.max_epochs=3")
    loro = run_loro(fs, cfg, regimes=["smoke_only_a", "neither_b"], log=None)
    smoke = loro["regimes"]["smoke_only_a"]
    assert smoke["labels"]["smoke"]["role"] == "positive" and 0 <= smoke["labels"]["smoke"]["recall"] <= 1
    assert smoke["labels"]["fog"]["role"] == "negative"
    assert smoke["n_train"] < int(np.sum(fs.splits == "train"))
    assert loro["regimes"]["neither_b"]["labels"]["smoke"]["role"] == "negative"

    run_pretraining(fs, cfg, tmp_path / "pre", log=None)
    shifted = fs.split("test_ood")
    re = resplit_by_group(shifted, seed=0)
    assert_no_group_leakage(re)
    assert set(re.splits) == {"train", "val_select", "val_calib", "test_iid"}
    proxy = run_synthetic_proxy(shifted, cfg, tmp_path / "pre", tmp_path / "proxy", n_boot=20,
                                description="test_ood recipes", log=None)
    assert proxy["not_real_world_evidence"] is True
    assert {"C_zero_shot", "D_finetuned", "B_scratch"} <= set(proxy)
    assert proxy["D_finetuned"]["lr"] == pytest.approx(1e-4) and proxy["B_scratch"]["lr"] == pytest.approx(1e-3)


def test_youden_threshold_does_not_degenerate_on_weak_scores():
    from src.training.metrics import threshold_max_f1, threshold_max_youden

    rng = np.random.default_rng(0)
    y = np.r_[np.ones(200), np.zeros(200)]
    score = np.r_[rng.normal(0.3, 1.0, 200), rng.normal(0.0, 1.0, 200)]   # weak signal
    thr_f1, info_f1 = threshold_max_f1(score, y)
    thr_j, info_j = threshold_max_youden(score, y)
    assert info_f1["recall"] > 0.9                    # max F1 predicts (nearly) everything positive
    assert info_j["fpr"] < 0.8 and info_j["youden_j"] > 0.0
    # Separable scores: Youden puts the threshold between the classes.
    thr, info = threshold_max_youden(np.r_[np.full(10, 2.0), np.full(10, -2.0)], np.r_[np.ones(10), np.zeros(10)])
    assert -2.0 < thr <= 2.0 and info["youden_j"] == 1.0


# ============================================================================= unknown (NaN) labels, real Stage B
def _fog_unknown(fs):
    """Smoke-only supervision: every fog target NaN (unknown), categories as extract_features writes them."""
    from src.data.dataset import label_category

    out = copy.deepcopy(fs)
    out.y[:, 1] = np.nan
    out.categories = np.array([label_category(s, f) for s, f in out.y])
    return out


def _real_set(seed=9, fog_unknown=True):
    real = make_feature_set(sims={"train": 10, "val_select": 4, "val_calib": 5, "test_iid": 4}, seed=seed,
                            source="real", groups=lambda split, c, i: f"cam-{split}-{i % 3}")
    return _fog_unknown(real) if fog_unknown else real


def test_masked_loss_ignores_unknown_targets():
    from src.training.train import masked_bce_with_logits

    bce = torch.nn.functional.binary_cross_entropy_with_logits
    torch.manual_seed(0)
    z = torch.randn(8, 2)
    y = (torch.rand(8, 2) > 0.5).float()
    pw = torch.tensor([2.0, 0.5])
    assert torch.equal(masked_bce_with_logits(z, y), bce(z, y))              # all known: the plain loss
    assert torch.equal(masked_bce_with_logits(z, y, pw), bce(z, y, pos_weight=pw))
    part = y.clone()
    part[::2, 1] = float("nan")                                               # per label over its known entries
    want = 0.5 * (bce(z[:, 0], y[:, 0]) + bce(z[1::2, 1], y[1::2, 1]))
    assert float(masked_bce_with_logits(z, part)) == pytest.approx(float(want), rel=1e-6)
    fog_nan = y.clone()
    fog_nan[:, 1] = float("nan")                                              # a label never known: not averaged in
    zz = z.clone().requires_grad_(True)
    loss = masked_bce_with_logits(zz, fog_nan)
    assert float(loss.detach()) == pytest.approx(float(bce(z[:, 0], y[:, 0])), rel=1e-6)
    loss.backward()
    assert torch.isfinite(zz.grad).all() and torch.all(zz.grad[:, 1] == 0)
    none = masked_bce_with_logits(z.clone().requires_grad_(True), torch.full((8, 2), float("nan")))
    assert float(none.detach()) == 0.0 and none.requires_grad


def test_metrics_and_decision_rules_ignore_unknown_labels():
    rng = np.random.default_rng(0)
    n = 200
    y = np.stack([rng.random(n) < 0.5, rng.random(n) < 0.5], axis=1).astype(float)
    probs = np.clip(0.6 * y + 0.4 * rng.random((n, 2)), 0, 1)
    hard = (probs >= 0.5).astype(int)
    dec = np.where(rng.random((n, 2)) < 0.1, -1, hard)
    cats = np.array([CATEGORIES[k] for k in M.categories_from_decisions(y)])
    part = y.copy()
    part[::2, 1] = np.nan
    known = ~np.isnan(part[:, 1])
    doc = M.classification_metrics(part, probs, hard, dec, cats)
    ref = M.classification_metrics(y[known], probs[known], hard[known], dec[known], cats[known])
    for key in ("precision", "recall", "f1", "auroc", "auprc", "brier", "ece", "selective_accuracy"):
        assert doc["per_label"]["fog"][key] == pytest.approx(ref["per_label"]["fog"][key])
    assert doc["per_label"]["fog"]["n"] == known.sum() and doc["per_label"]["fog"]["n_unknown"] == n - known.sum()
    assert doc["per_label"]["smoke"]["n"] == n and doc["per_label"]["fog"]["status"] == "evaluated"
    assert doc["confusion_4class"]["n"] == known.sum() == ref["confusion_4class"]["n"]
    assert doc["confusion_4class"]["matrix"] == ref["confusion_4class"]["matrix"]
    assert doc["false_smoke_on_fog_only"] == ref["false_smoke_on_fog_only"]
    assert doc["n_known"] == {"smoke": n, "fog": int(known.sum()), "both": int(known.sum())}

    # Fog never annotated: every fog metric, the 4-class confusion and false smoke are null, never 0.
    none = y.copy()
    none[:, 1] = np.nan
    doc = M.classification_metrics(none, probs, hard, dec, cats)
    fog = doc["per_label"]["fog"]
    assert fog["status"] == "no_labels" and fog["n"] == 0
    assert all(fog[k] is None for k in ("precision", "recall", "f1", "auroc", "auprc", "brier", "ece",
                                        "selective_accuracy", "prevalence"))
    assert doc["per_label"]["smoke"]["auroc"] is not None
    assert doc["confusion_4class"]["n"] == 0 and doc["false_smoke_on_fog_only"]["rate"] is None
    assert doc["selective_accuracy"]["joint"] is None and doc["selective_accuracy"]["smoke"] is not None
    ci = M.bootstrap_classification_ci(none, probs, hard, dec, cats, np.arange(n) % 40, n_boot=30)
    assert ci["fog.auroc"]["lo"] is None and ci["smoke.auroc"]["lo"] is not None
    json.dumps(doc, allow_nan=False)

    # Decision rules: fitted on the known rows only, neutral when nothing is known.
    z = M.logit(probs[:, 1])
    assert M.fit_temperature(z, part[:, 1]) == M.fit_temperature(z[known], y[known, 1])
    T, info = M.fit_temperature(z, none[:, 1])
    assert T == 1.0 and info["reason"] == "no_known_labels"
    assert M.threshold_max_youden(z, part[:, 1]) == M.threshold_max_youden(z[known], y[known, 1])
    m, info = M.fit_abstain_margin(z, none[:, 1], 0.0, floor=0.25)
    assert m == 0.25 and info["reason"] == "no_known_labels"
    ops = M.fit_operating_points(M.logit(probs), none, np.array(["unknown"] * n), np.arange(n) % 40, _cfg())
    assert ops["fit"]["balanced"]["fog"]["reason"] == "no_known_labels"
    lfa = ops["fit"]["low_false_alarm"]["smoke"]
    assert lfa["constraint_fitted"] is False and lfa["reason"]
    assert ops["thresholds"]["low_false_alarm"]["smoke"] == ops["thresholds"]["balanced"]["smoke"]


def test_featureset_and_sequences_with_unknown_labels_or_no_lag_frames(tmp_path, fs):
    from src.data.dataset import label_category, parse_label

    real = _fog_unknown(fs.split(["train", "test_iid"]))
    real.source = "real"
    back = FeatureSet.load(real.save(tmp_path / "real.npz"))
    np.testing.assert_array_equal(np.isnan(back.y), np.isnan(real.y))
    assert np.isnan(back.y[:, 1]).all() and not np.isnan(back.y[:, 0]).any()
    assert back.known[:, 0].all() and not back.known[:, 1].any()
    assert set(back.categories) == {"smoke_fog_unknown", "no_smoke_fog_unknown"}
    assert_no_group_leakage(back)
    _, y = SequenceTensorDataset(back)[0]
    assert torch.isnan(y[1]) and not torch.isnan(y[0])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        FeatureSet(X=fs.X[:1], y=[[2.0, 0.0]], seq_ids=["a"], sim_ids=["a"], groups=["a"], splits=["train"],
                   categories=["smoke_only"], regimes=["r"], tags=[""], mask_sources=["degraded"])
    assert label_category(0, 1) == "fog_only" and label_category(1, np.nan) == "smoke_fog_unknown"
    assert np.isnan(parse_label("")) and np.isnan(parse_label("unknown")) and parse_label("1") == 1.0
    with pytest.raises(ValueError):
        parse_label("0.5")

    # A far-field sequence has no lag frames: stored without them, loaded back as None.
    rng = np.random.default_rng(0)
    T, H, W = 12, 16, 20
    rendered = SimpleNamespace(frames=rng.integers(0, 255, (T, H, W), dtype=np.uint8), lag_frames=None,
                               masks=rng.random((T, H, W)) > 0.7, timestamps=np.arange(T) * 60.0,
                               lag_timestamps=None, mask_source="degraded", visibility={}, render_params={})
    sim = SimpleNamespace(sim_id="s", category="fog_only", regime="ridge_cloud", labels=(0, 1), seed=1,
                          params={"split": "train"}, valid=True, validity={}, dx=50.0)
    seq = load_sequence(save_sequence(tmp_path / "far", rendered, sim, "s-v0"))
    assert seq["lag_frames"] is None and seq["lag_timestamps"] is None and seq["meta"]["has_lag_frames"] is False
    np.testing.assert_array_equal(seq["frames"], rendered.frames)
    np.testing.assert_array_equal(seq["masks"], rendered.masks)


def test_training_and_calibration_with_unknown_fog_labels(fs):
    from src.training.baselines import run_baselines
    from src.training.evaluate import shortcut_audit
    from src.utils.io import dumps_json

    nan_fs = _fog_unknown(fs)
    res, norm = _train_one(nan_fs, max_epochs=8)
    assert np.isfinite(res.best_val_loss)
    assert all(h["val_auroc_fog"] is None and h["val_auroc_smoke"] is not None for h in res.history)
    test = nan_fs.split("test_iid")
    z = model_logits(res.model, norm, test.X)
    assert M.auroc(z[:, 0], test.y[:, 0]) > 0.7               # the smoke signal is still learned
    vc = nan_fs.split("val_calib")
    cfg = _cfg()
    cal = fit_calibration(model_logits(res.model, norm, vc.X), vc, cfg, "real_val_calib")
    sup = cal["calibration"]["label_support"]
    assert sup["fog"]["n_known"] == 0 and sup["fog"]["calibration"] == "default_unfitted"
    assert sup["smoke"]["calibration"] == "fitted" and sup["smoke"]["n_known"] == len(vc)
    assert cal["temperatures"]["fog"] == 1.0 and cal["thresholds"]["balanced"]["fog"]["logit"] == 0.0
    assert cal["calibration"]["prevalence"]["fog"]["per_sequence"] is None
    assert cal["thresholds"]["low_false_alarm"]["smoke"] == cal["thresholds"]["balanced"]["smoke"]
    # Nothing known about fog and nothing to keep: always abstain on it, never a confident decision.
    from src.training.train import UNTRAINED_MARGIN

    assert cal["calibration"]["untrained_labels"] == ["fog"]
    for op in ("balanced", "low_false_alarm"):
        assert cal["abstain"]["margins"][op]["fog"] == UNTRAINED_MARGIN
        assert cal["calibration"]["abstain_fit"][op]["fog"]["reason"] == "untrained_label_always_abstain"
        assert 0.25 <= cal["abstain"]["margins"][op]["smoke"] < UNTRAINED_MARGIN
    # Smoke-only rows: the four (smoke, fog) categories are not applicable, not "insufficient".
    cs = cal["calibration"]["category_support"]
    assert cs["status"] == "not_applicable" and cs["n_unknown_label_rows"] == len(vc)
    assert set(cs["partial_categories"]) == {"smoke_fog_unknown", "no_smoke_fog_unknown"}
    assert sum(v["sequences"] for v in cs["partial_categories"].values()) == len(vc)
    assert cal["calibration"]["insufficient_categories"] == []
    assert cal["calibration"]["insufficient_calibration_data"] is False
    # Fine-tuning: a label without known calibration rows keeps the pretrained model's rules.
    pre = {"temperatures": {"smoke": 1.3, "fog": 2.2},
           "thresholds": {op: {"smoke": {"logit": 0.1}, "fog": {"logit": 0.7}} for op in ("balanced", "low_false_alarm")},
           "abstain": {"margins": {op: {"smoke": 0.3, "fog": 0.45} for op in ("balanced", "low_false_alarm")}}}
    kept = fit_calibration(model_logits(res.model, norm, vc.X), vc, cfg, "real_val_calib", fallback=pre)
    assert kept["temperatures"]["fog"] == 2.2 and kept["thresholds"]["balanced"]["fog"]["logit"] == 0.7
    assert kept["abstain"]["margins"]["low_false_alarm"]["fog"] == 0.45
    assert kept["calibration"]["label_support"]["fog"]["calibration"] == "kept_pretrained"
    assert kept["calibration"]["untrained_labels"] == []
    assert kept["temperatures"]["smoke"] == cal["temperatures"]["smoke"]      # known labels are still fitted
    dumps_json(kept)                                                          # finite, JSON / YAML-safe

    base = run_baselines(nan_fs, cfg, n_boot=10)
    a = base["A_heuristic"]
    assert a["skipped_labels"] == {"fog": "no known labels in the train split"} and a["coefficients"]["fog"] is None
    fog = a["results"]["test_iid"]["balanced"]["per_label"]["fog"]
    assert fog["status"] == "not_fitted" and fog["auroc"] is None
    assert a["results"]["test_iid"]["balanced"]["per_label"]["smoke"]["auroc"] > 0.6
    audit = shortcut_audit(nan_fs, cfg)
    assert audit["flags_only"]["pooled"]["auroc"]["fog"] is None
    assert audit["parameters"]["per_label"]["fog"] == {}
    dumps_json({"baselines": base, "audit": audit})


def test_stage_b_on_smoke_only_real_data(tmp_path, fs):
    import importlib
    import sys

    from src.training.evaluate import build_evaluation, compact, evaluate_real, update_real_section

    cfg = _cfg("training.max_epochs=3", "finetune.max_epochs=2")
    run_pretraining(fs, cfg, tmp_path / "pre", log=None)
    real = _real_set()
    d = run_finetune(real, cfg, tmp_path / "pre", tmp_path / "D", log=None)
    b = run_finetune(real, cfg, tmp_path / "pre", tmp_path / "B", from_scratch=True, log=None)
    assert d["label_support"]["fog"]["calibration"] == "kept_pretrained"
    assert b["label_support"]["fog"]["calibration"] == "default_unfitted"
    assert b["untrained_labels"] == ["fog"] and d["untrained_labels"] == []
    assert b["category_support"]["status"] == "not_applicable" and b["insufficient_calibration_data"] is False
    # B never learned fog: the exported expert abstains on fog for every input, at every operating point.
    from src.inference.predict import PhysicsExpert

    test = real.split("test_iid")
    expert_b = PhysicsExpert.load(tmp_path / "B")
    for op in ("balanced", "low_false_alarm"):
        pb = expert_b.predict(test.X, op)
        assert np.all(pb["decisions"][:, 1] == -1)
        assert all("fog_near_threshold" in r for r, ok in zip(pb["abstain_reasons"], pb["valid_input"]) if ok)
    expert_b.config["abstain"]["margins"]["balanced"]["fog"] = 0.0          # it is the untrained margin that abstains
    pb = expert_b.predict(test.X, "balanced")
    assert pb["valid_input"].any() and np.all(pb["decisions"][pb["valid_input"], 1] != -1)
    pre, cd = read_yaml(tmp_path / "pre" / "model_config.yaml"), read_yaml(tmp_path / "D" / "model_config.yaml")
    assert cd["temperatures"]["fog"] == pre["temperatures"]["fog"]
    assert cd["thresholds"]["balanced"]["fog"] == pre["thresholds"]["balanced"]["fog"]
    assert cd["temperatures"]["smoke"] != pre["temperatures"]["smoke"] or \
        cd["thresholds"]["balanced"]["smoke"] != pre["thresholds"]["balanced"]["smoke"]
    assert cd["fit"]["known_labels"]["train"] == {"smoke": int(np.sum(real.splits == "train")), "fog": 0}
    assert cd["fit"]["training"]["loss"] == "bce_with_logits_masked_unknown"

    res = evaluate_real(real, cfg, {"B": tmp_path / "B", "C": tmp_path / "pre", "D": tmp_path / "D"}, n_boot=20)
    assert res["status"] == "partial" and res["evaluated_labels"] == ["smoke"]
    assert res["label_status"]["smoke"]["status"] == "evaluated"
    assert res["label_status"]["fog"]["status"] == "no_labels" and "fog" in res["label_status"]["fog"]["reason"]
    assert res["counts"]["test_iid"]["known_labels"]["fog"] == 0
    for exp in ("B", "C", "D"):
        ov = res[exp]["test_iid"]["balanced"]["overall"]
        assert ov["per_label"]["smoke"]["auroc"] is not None and ov["per_label"]["smoke"]["n"] == len(real.split("test_iid"))
        assert ov["per_label"]["fog"]["status"] == "no_labels"
        assert all(ov["per_label"]["fog"][k] is None for k in ("precision", "recall", "f1", "auroc", "ece"))
        assert ov["false_smoke_on_fog_only"]["rate"] is None and ov["confusion_4class"]["n"] == 0
        assert ov["ci95"]["fog.auroc"]["lo"] is None and compact(ov)["fog"]["status"] == "no_labels"
    doc = build_evaluation({"model": "C"}, None, None, None, real=res)
    assert doc["ablations"]["B"] is res["B"] and doc["ablations"]["real_label_status"]["fog"]["status"] == "no_labels"
    merged = update_real_section({"synthetic": {"kept": True}, "real": {"status": "pending"},
                                  "ablations": {"A": {"a": 1}, "B": None, "C": {"c": 1}, "D": None,
                                                "reasons": {"B": "x", "D": "x"}}}, res)
    assert merged["synthetic"] == {"kept": True} and merged["real"]["status"] == "partial"
    assert merged["ablations"]["A"] == {"a": 1} and merged["ablations"]["D"] is res["D"]
    assert "B" not in merged["ablations"]["reasons"]
    write_json(tmp_path / "eval.json", merged)                            # finite JSON only

    # The Markdown report renders the real section; fog (no_labels) is n/a there, never 0.
    write_json(tmp_path / "eval_real.json", update_real_section(None, res))
    assert _script("summarize_results").main(["--evaluation", str(tmp_path / "eval_real.json"),
                                              "--out", str(tmp_path / "results.md")]) == 0
    md = (tmp_path / "results.md").read_text()
    assert "**Real-world status: partial.**" in md and "## Real footage" in md and "fog: no_labels" in md
    rows = [ln for ln in md.splitlines() if ln.startswith(("| C: ", "| B: ", "| D: "))]
    assert len(rows) == 6
    for ln in rows:
        # model | op | smoke AUROC | fog AUROC | smoke F1 | fog F1 | false smoke on fog_only | ECE | abstain
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        assert cells[2] != "n/a" and cells[3] == cells[5] == cells[6] == "n/a"
        assert cells[7].endswith(" / n/a") and cells[8].endswith(" / n/a")

    # The script: Stage B on its own keeps the synthetic sections of an existing evaluation.json.
    import shutil

    art = tmp_path / "art"
    shutil.copytree(tmp_path / "pre", art)
    write_json(art / "evaluation.json", {"synthetic": {"kept": True}, "real": {"status": "pending"},
                                         "ablations": {"A": None, "B": None, "C": None, "D": None, "reasons": {}}})
    real.save(tmp_path / "real.npz")
    scripts = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, scripts)
    try:
        ev = importlib.import_module("evaluate_physics")
        assert ev.main(["--artifacts", str(art), "--reports", str(tmp_path / "rep"), "--real", str(tmp_path / "real.npz"),
                        "--real-out", str(tmp_path / "runs"), "--n-boot", "10", "--set", "training.seeds=[0]",
                        "--set", "training.max_epochs=2", "--set", "finetune.max_epochs=2"]) == 0
    finally:
        sys.path.remove(scripts)
    out = read_json(tmp_path / "rep" / "evaluation.json")
    assert out["synthetic"] == {"kept": True} and out["real"]["status"] == "partial"
    assert out["real"]["D"]["test_iid"]["balanced"]["overall"]["per_label"]["fog"]["auroc"] is None
    assert out["ablations"]["B"] is not None and out["ablations"]["D"] is not None


# ============================================================================= real-result bookkeeping, scripts
def _script(name):
    """Import scripts/<name>.py as a fresh module.

    A copy cached by another test may have bound names that test patched while
    it was imported (tests/test_image_stats.py imports extract_features with
    src.data.dataset.load_sequence monkeypatched), so it is never reused.
    """
    import importlib
    import sys

    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.modules.pop(name, None)
    sys.path.insert(0, scripts)
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(scripts)


def test_update_real_section_never_keeps_stale_real_results():
    from src.training.evaluate import BD_REASON, update_real_section

    old = {"synthetic": {"kept": True}, "real": {"status": "complete"},
           "ablations": {"A": {"a": 1}, "B": {"old": "B"}, "C": {"c": 1}, "D": {"old": "D"}, "reasons": {},
                         "real_label_status": {"fog": {"status": "evaluated"}}, "extra_baselines": {"p": 1}}}
    snapshot = copy.deepcopy(old)
    ab = update_real_section(old, {"status": "pending", "reason": "no footage"})["ablations"]
    assert ab["B"] is None and ab["D"] is None and ab["reasons"] == {"B": BD_REASON, "D": BD_REASON}
    assert "real_label_status" not in ab
    assert (ab["A"], ab["C"], ab["extra_baselines"]) == ({"a": 1}, {"c": 1}, {"p": 1})
    assert old == snapshot                                                   # the input is not modified
    new = {"status": "partial", "B": {"new": "B"}, "D": None, "label_status": {"smoke": {"status": "evaluated"}}}
    doc = update_real_section(old, new)
    ab = doc["ablations"]
    assert ab["B"] == {"new": "B"} and ab["D"] is None and ab["reasons"] == {"D": BD_REASON}
    assert ab["real_label_status"] == {"smoke": {"status": "evaluated"}} and doc["real"]["status"] == "partial"
    assert doc["synthetic"] == {"kept": True}


def test_curves_and_roc_plot_skip_a_label_without_known_targets(tmp_path):
    from src.training.evaluate import curves, plot_roc_pr

    rng = np.random.default_rng(0)
    y = np.stack([(rng.random(60) < 0.5).astype(float), np.full(60, np.nan)], axis=1)
    probs = rng.random((60, 2))
    c = curves(probs, y)
    assert c["fog"] == {"status": "no_labels", "roc": None, "pr": None}
    assert "status" not in c["smoke"] and c["smoke"]["roc"]["fpr"][0] == 0.0 and c["smoke"]["roc"]["tpr"][-1] == 1.0
    json.dumps(c, allow_nan=False)
    assert plot_roc_pr(probs, y, tmp_path / "roc.png", "smoke only").exists()
    assert plot_roc_pr(probs, np.full((60, 2), np.nan), tmp_path / "none.png", "nothing known").exists()


def test_real_feature_config_sources_and_timing_conflicts(tmp_path, capsys):
    from src.data.pipeline import feature_timing
    from src.utils.io import write_yaml

    ef = _script("extract_features")
    art = tmp_path / "art"
    art.mkdir()
    write_yaml(art / "model_config.yaml", {"preprocessing": {"feature_config": {
        "expected_interval_s": 60.0, "flow_lag_s": 0.0, "max_offset_s": 10.0, "work_size": 96, "unknown_key": 1}}})
    base = ["--real-manifest", "m.csv", "--out", "o.npz"]
    fcfg, rec = ef.real_feature_config(ef.parse_args(base + ["--artifacts", str(art)]))
    assert (fcfg.expected_interval_s, fcfg.flow_lag_s, fcfg.max_offset_s, fcfg.work_size) == (60.0, 0.0, 10.0, 96)
    assert fcfg.flow_mode == "consecutive" and rec["source"] == f"artifacts:{art}" and rec["forced"] is False
    # an explicit value equal to the model's is fine (none and 0 both mean consecutive flow) ...
    _, rec = ef.real_feature_config(ef.parse_args(base + ["--artifacts", str(art), "--flow-lag-s", "none",
                                                          "--max-offset-s", "10"]))
    assert rec["forced"] is False
    # ... a contradicting one is refused unless forced
    with pytest.raises(SystemExit, match="contradicts the exported model"):
        ef.real_feature_config(ef.parse_args(base + ["--artifacts", str(art), "--interval-s", "5"]))
    fcfg, rec = ef.real_feature_config(ef.parse_args(base + ["--artifacts", str(art), "--interval-s", "5",
                                                             "--force-timing"]))
    assert fcfg.expected_interval_s == 5.0 and rec["forced"] is True
    assert rec["overridden"] == {"expected_interval_s": 5.0}
    # the exported model defines the timing: a profile on top of it is an error
    far = Path(__file__).resolve().parents[1] / "configs" / "simulation_far.yaml"
    with pytest.raises(SystemExit):
        ef.parse_args(base + ["--artifacts", str(art), "--profile", str(far)])
    assert "exclude each other" in capsys.readouterr().err
    fcfg, rec = ef.real_feature_config(ef.parse_args(base + ["--profile", str(far)]))
    want = feature_timing(read_yaml(far))
    assert (fcfg.expected_interval_s, fcfg.flow_lag_s, fcfg.max_offset_s) == \
        (want["expected_interval_s"], want["flow_lag_s"], want["max_offset_s"])
    assert rec["source"] == f"profile:{far}"


def _minute_folder(path, n=14, seed=0):
    """One frame per minute (+- 3 s jitter): a dark plume rising from a fixed base, timestamps.csv sidecar."""
    import cv2

    rng = np.random.default_rng(seed)
    path.mkdir(parents=True)
    t = 60.0 * np.arange(n) + rng.uniform(-3, 3, n)
    t[0] = 0.0
    yy, xx = np.mgrid[0:72, 0:96]
    rows = ["filename,timestamp_s"]
    for k in range(n):
        img = 150.0 + 20.0 * np.sin(xx / 7.0) * np.cos(yy / 5.0) + rng.normal(0, 2, (72, 96))
        plume = (np.abs(xx - 48) < 6 + k) & (yy > 60 - 3 * k)
        img[plume] -= 60.0
        name = f"f{k:03d}.jpg"
        cv2.imwrite(str(path / name), np.clip(img, 0, 255).astype(np.uint8))
        rows.append(f"{name},{t[k]:.3f}")
    (path / "timestamps.csv").write_text("\n".join(rows) + "\n")
    return t


def test_extract_real_consecutive_windows_and_synthetic_stored_config(tmp_path):
    from dataclasses import asdict

    from src.data.features import FeatureConfig, extract_features_with_meta
    from src.data.pipeline import feature_timing

    ef = _script("extract_features")
    far = Path(__file__).resolve().parents[1] / "configs" / "simulation_far.yaml"
    timing = feature_timing(read_yaml(far))
    t = _minute_folder(tmp_path / "seq")
    manifest = tmp_path / "real.csv"
    manifest.write_text("video_path,camera_id,video_id,smoke,fog,split,start_s,max_windows,window_id,camera_mount\n"
                        f"{tmp_path / 'seq'},stgo-s-mobo-c,seq,1,,test_iid,0.000,1,seq-post0,stgo-s\n")
    assert ef.main(["--real-manifest", str(manifest), "--out", str(tmp_path / "far.npz"), "--profile", str(far)]) == 0
    fs = FeatureSet.load(tmp_path / "far.npz")
    assert len(fs) == 1 and fs.y[0, 0] == 1.0 and np.isnan(fs.y[0, 1]) and fs.source == "real"
    m = fs.meta[0]
    assert m["camera_mount"] == "stgo-s" and m["flow_mode"] == "consecutive" and m["window_id"] == "seq-post0"
    assert m["feature_config"]["expected_interval_s"] == timing["expected_interval_s"]
    assert m["feature_config"]["max_offset_s"] == timing["max_offset_s"]
    assert not m["feature_config"]["flow_lag_s"]                              # no lag frames
    np.testing.assert_allclose(m["obs_times"], t[:12], atol=1e-3)            # 12 consecutive observations
    # The near-field timing (lag pairs 1 s apart) refuses one-minute imagery: nothing is extracted.
    assert ef.main(["--real-manifest", str(manifest), "--out", str(tmp_path / "near.npz")]) == 1
    assert "flow_lag_s" in json.dumps(read_json(tmp_path / "near.skipped.json"))

    # Synthetic re-extraction: each row with its stored feature_config and roi_scale, not near-field defaults.
    rng = np.random.default_rng(1)
    T, H, W = 12, 48, 64
    frames = rng.integers(60, 200, (T, H, W), dtype=np.uint8)
    masks = np.zeros((T, H, W), dtype=bool)
    masks[:, 20:40, 24:40] = True
    frames[masks] = (frames[masks] * 0.5).astype(np.uint8)
    rendered = SimpleNamespace(frames=frames, lag_frames=None, masks=masks, timestamps=np.arange(T) * 60.0,
                               lag_timestamps=None, mask_source="degraded", visibility={}, render_params={})
    sim = SimpleNamespace(sim_id="s", category="smoke_only", regime="ridge_cloud", labels=(1, 0), seed=1,
                          params={"split": "test_iid"}, valid=True, validity={}, dx=50.0)
    data = tmp_path / "syn"
    seq_path = save_sequence(data / "seqs" / "s-v0", rendered, sim, "s-v0")
    stored = {"expected_interval_s": 60.0, "flow_lag_s": 0.0, "max_offset_s": 10.0, "work_size": 64}
    FeatureSet(X=np.zeros((1, 12, 14), np.float32), y=[[1.0, 0.0]], seq_ids=["s-v0"], sim_ids=["s"], groups=["s"],
               splits=["test_iid"], categories=["smoke_only"], regimes=["ridge_cloud"], tags=[""],
               mask_sources=["degraded"], meta=[{"seq_path": str(seq_path.relative_to(data)), "roi_scale": 1.7,
                                                 "feature_config": stored}]).save(data / "features.npz")
    assert ef.main(["--data", str(data), "--out", str(tmp_path / "re.npz")]) == 0
    want, meta = extract_features_with_meta(frames, np.arange(T) * 60.0, masks, None, None,
                                            FeatureConfig(**stored, roi_scale=1.7))
    re_fs = FeatureSet.load(tmp_path / "re.npz")
    np.testing.assert_array_equal(re_fs.X[0], want)
    assert meta["flow_mode"] == "consecutive" == re_fs.meta[0]["flow_mode"]
    assert re_fs.meta[0]["feature_config"] == {k: v for k, v in asdict(FeatureConfig(**stored)).items()
                                               if k != "roi_scale"} and re_fs.meta[0]["roi_scale"] == 1.7
    # timing overrides: the rows record the timing they were re-extracted with
    assert ef.main(["--data", str(data), "--out", str(tmp_path / "re30.npz"), "--interval-s", "30"]) == 0
    m30 = FeatureSet.load(tmp_path / "re30.npz").meta[0]
    assert m30["feature_config"]["expected_interval_s"] == 30.0 and m30["feature_config"]["work_size"] == 64
    assert m30["feature_meta"]["timing"] == extract_features_with_meta(
        frames, np.arange(T) * 60.0, masks, None, None,
        FeatureConfig(**{**stored, "expected_interval_s": 30.0}, roi_scale=1.7))[1]["timing"]
    near, _ = extract_features_with_meta(frames, np.arange(T) * 60.0, masks, None, None, FeatureConfig())
    assert not np.array_equal(near, want)                                    # the defaults would differ
