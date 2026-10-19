"""Training, calibration, metrics, baselines, leakage checks and evaluation.json (src.training, src.data.dataset)."""

import copy
import json
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
from src.utils.io import read_json, read_yaml
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
    assert T == 1.0 and info["at_bound"] == "upper" and info["reason"] == "optimum_at_upper_bound"
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
