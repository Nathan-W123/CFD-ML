"""Model, normaliser, artifacts and the deployment API (src.models, src.data.normalization, src.inference)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.feature_spec import FEATURE_GATES, FEATURE_INDEX, N_FEATURES, gate_masks
from src.data.normalization import FeatureNormalizer
from src.inference.predict import PhysicsExpert
from src.models.physics_gru import (ModelConfig, PhysicsEncoder, PhysicsGRU, load_encoder, load_model,
                                    save_model_artifacts)

REPO = Path(__file__).resolve().parents[1]
NORMALISED = [i for i, g in enumerate(FEATURE_GATES) if g is not None]


def _features(n=64, seed=0, invalid_frac=0.2):
    """Raw-looking features: random values, some closed gates (exactly 0), flags 0/1."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12, N_FEATURES)) * 0.05 + 0.02
    X[..., FEATURE_INDEX["region_area"]] = rng.uniform(0.05, 0.5, size=(n, 12))
    X[..., 12] = (rng.random((n, 12)) > invalid_frac).astype(float)
    X[..., 13] = X[..., 12] * (rng.random((n, 12)) > invalid_frac)
    return np.where(gate_masks(X), X, 0.0).astype(np.float32)


# ----------------------------------------------------------------------------- architecture
def test_shapes_and_architecture():
    torch.manual_seed(0)
    model = PhysicsGRU(ModelConfig())
    x = torch.randn(5, 12, 14)
    assert model.encode(x).shape == (5, 16)
    assert model(x).shape == (5, 2)
    out = model.predict(x)
    assert out["logits"].shape == (5, 2)
    assert torch.all((out["probs"] >= 0) & (out["probs"] <= 1))
    assert model.encode(torch.randn(12, 14)).shape == (1, 16)          # single sequence
    gru = model.encoder.gru
    assert (gru.input_size, gru.hidden_size, gru.num_layers, gru.bidirectional) == (14, 32, 1, False)
    assert model.encoder.dropout.p == pytest.approx(0.1)
    assert (model.encoder.proj.in_features, model.encoder.proj.out_features) == (32, 16)
    assert isinstance(model.encoder.norm, torch.nn.LayerNorm) and isinstance(model.encoder.act, torch.nn.GELU)
    assert (model.head.in_features, model.head.out_features) == (16, 2)
    with pytest.raises(ValueError):
        model(torch.randn(2, 12, 13))


def test_encode_does_not_use_the_head():
    torch.manual_seed(0)
    model = PhysicsGRU().eval()
    x = torch.randn(3, 12, 14)
    ref = model.encode(x).detach().clone()
    with torch.no_grad():
        model.head.weight.fill_(float("nan"))
    emb = model.encode(x)
    assert torch.isfinite(emb).all() and torch.equal(emb, ref)
    assert not any("head" in k for k in model.encoder.state_dict())


def test_predict_is_eval_mode_and_restores_training_flag():
    torch.manual_seed(0)
    model = PhysicsGRU().train()
    x = torch.randn(4, 12, 14)
    a, b = model.predict(x)["logits"], model.predict(x)["logits"]
    assert torch.equal(a, b)                      # dropout off inside predict
    assert model.training


# ----------------------------------------------------------------------------- normaliser
def test_normalizer_statistics_are_gated():
    X = _features(200, seed=1)
    norm = FeatureNormalizer().fit(X)
    g = gate_masks(X)
    for i in NORMALISED:
        vals = X[..., i][g[..., i]]
        assert norm.mean[i] == pytest.approx(vals.mean(), rel=1e-6, abs=1e-9)
        assert norm.std_raw[i] == pytest.approx(vals.std(), rel=1e-6, abs=1e-9)
    # zeros at closed gates would drag the mean of region_area down if they were counted
    area = FEATURE_INDEX["region_area"]
    assert norm.mean[area] > X[..., area].mean()
    assert norm.count[area] == int(g[..., area].sum())


def test_normalizer_transform_closed_gates_clip_and_flags():
    X = _features(100, seed=2)
    norm = FeatureNormalizer(clip=3.0).fit(X)
    Y = X.copy()
    Y[0, 5, FEATURE_INDEX["region_area"]] = 1e3 if Y[0, 5, 12] > 0.5 else 0.0
    Z = norm.transform(Y)
    assert Z.dtype == np.float32 and Z.shape == Y.shape
    g = gate_masks(Y)
    assert np.all(Z[~g] == 0.0)
    assert np.all(np.abs(Z[..., NORMALISED]) <= 3.0)
    np.testing.assert_array_equal(Z[..., 12:], Y[..., 12:])          # flags untouched
    single = norm.transform(Y[0])
    np.testing.assert_array_equal(single, Z[0])
    # clip_fraction: share of open, non-flag entries at the clip
    big = Y.copy()
    open_ = g & np.array([gt is not None for gt in FEATURE_GATES])
    big[1][open_[1]] = 1e6
    cf = norm.clip_fraction(big)
    assert cf.shape == (len(big),) and cf[1] == pytest.approx(1.0)
    assert np.all(cf[2:] < 0.2)


def test_normalizer_std_floor_and_json_roundtrip(tmp_path):
    X = _features(50, seed=3)
    X[..., FEATURE_INDEX["boundary_roughness"]] = np.where(X[..., 12] > 0.5, 0.3, 0.0)   # constant when open
    norm = FeatureNormalizer().fit(X)
    i = FEATURE_INDEX["boundary_roughness"]
    assert norm.std_raw[i] == pytest.approx(0.0, abs=1e-7)
    assert norm.std[i] == pytest.approx(norm.std_floor[i]) and norm.std[i] > 0
    doc = norm.to_dict()
    assert doc["floored"][i] and len(doc["std_floor"]) == N_FEATURES
    path = norm.save(tmp_path / "norm.json")
    back = FeatureNormalizer.load(path)
    np.testing.assert_array_equal(back.transform(X), norm.transform(X))
    np.testing.assert_array_equal(back.std_floor, norm.std_floor)


def test_normalizer_uses_only_the_data_it_is_fitted_on():
    train, other = _features(80, seed=4), _features(80, seed=5) * 3.0
    a = FeatureNormalizer().fit(train)
    b = FeatureNormalizer().fit(train)
    a.transform(other)                       # transforming never refits
    np.testing.assert_array_equal(a.mean, b.mean)
    c = FeatureNormalizer().fit(np.concatenate([train, other]))
    assert not np.allclose(c.mean[NORMALISED], a.mean[NORMALISED])


# ----------------------------------------------------------------------------- artifacts
def _artifacts(tmp_path, seed=0):
    torch.manual_seed(seed)
    model = PhysicsGRU().eval()
    X = _features(32, seed=seed)
    norm = FeatureNormalizer().fit(X)
    extra = {"temperatures": {"smoke": 1.5, "fog": 0.8},
             "thresholds": {"balanced": {"smoke": {"logit": 0.1, "prob": 0.525}, "fog": {"logit": -0.2, "prob": 0.45}},
                            "low_false_alarm": {"smoke": {"logit": 1.0, "prob": 0.73},
                                                "fog": {"logit": -0.2, "prob": 0.45}}},
             "abstain": {"margins": {"balanced": {"smoke": 0.3, "fog": 0.25},
                                     "low_false_alarm": {"smoke": 0.3, "fog": 0.25}},
                         "input_validity": {"min_mask_frames": 3, "min_flow_frames": 2, "max_clip_fraction": 0.2}},
             "calibration": {"fit_source": "synthetic_val_calib"},
             "preprocessing": {"feature_config": {"expected_interval_s": 5.0, "flow_lag_s": 1.0}}}
    paths = save_model_artifacts(model, tmp_path / "art", norm, extra)
    return model, norm, X, paths


def test_artifacts_roundtrip_identical_outputs(tmp_path):
    model, norm, X, paths = _artifacts(tmp_path)
    assert {p.name for p in paths.values()} == {"physics_encoder.pt", "physics_classifier.pt",
                                                "feature_normalizer.json", "model_config.yaml"}
    loaded, lnorm, cfg = load_model(tmp_path / "art")
    Z = torch.from_numpy(norm.transform(X))
    assert torch.equal(loaded(torch.from_numpy(lnorm.transform(X))), model(Z))
    enc, enorm, _ = load_encoder(tmp_path / "art")
    assert isinstance(enc, PhysicsEncoder)
    assert torch.equal(enc(Z), model.encode(Z))
    for key in ("model", "feature_names", "preprocessing", "temperatures", "thresholds", "abstain", "calibration"):
        assert key in cfg
    assert cfg["preprocessing"]["normalizer_fit_split"] == "train"
    assert cfg["preprocessing"]["feature_config"]["flow_lag_s"] == 1.0
    # checkpoints are plain state dicts loadable with weights_only=True
    sd = torch.load(paths["encoder"], weights_only=True)
    assert set(sd) == set(model.encoder.state_dict())


def test_physics_expert_predict_decisions_and_abstention(tmp_path):
    model, norm, X, _ = _artifacts(tmp_path)
    expert = PhysicsExpert.load(tmp_path / "art")
    emb = expert.encode(X)
    assert emb.shape == (len(X), 16) and emb.dtype == np.float32
    assert expert.encode(X[0]).shape == (1, 16)
    out = expert.predict(X)
    assert out["calibration_source"] == "synthetic_val_calib"
    probs, dec, z = out["probs"], out["decisions"], out["calibrated_logits"]
    assert probs.shape == (len(X), 2) and np.all((probs >= 0) & (probs <= 1))
    raw = model(torch.from_numpy(norm.transform(X))).detach().numpy()
    np.testing.assert_allclose(out["logits"], raw, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(z, raw / np.array([1.5, 0.8]), rtol=1e-5, atol=1e-6)
    thr = np.array([0.1, -0.2])
    np.testing.assert_array_equal(out["hard_decisions"], (z >= thr).astype(int))
    assert set(np.unique(dec)) <= {-1, 0, 1}
    for i in range(len(X)):
        valid = out["valid_input"][i]
        for j, m in enumerate((0.3, 0.25)):
            if not valid:
                assert dec[i, j] == -1
            elif abs(z[i, j] - thr[j]) < m:
                assert dec[i, j] == -1
            else:
                assert dec[i, j] == out["hard_decisions"][i, j]
    # input validity: too few valid masks / flows
    bad = X[:2].copy()
    bad[0, :, 12] = 0.0
    bad[0, :, 13] = 0.0
    bad[0] = np.where(gate_masks(bad[0]), bad[0], 0.0)
    bad[1, :, 12] = 1.0
    bad[1, :, 13] = 0.0
    bad[1, 0, 13] = 1.0
    out = expert.predict(bad)
    assert np.all(out["decisions"] == -1)
    assert "few_valid_mask_frames" in out["abstain_reasons"][0]
    assert "few_valid_flow_frames" in out["abstain_reasons"][1]
    # timing from the extractor's meta, and the consecutive-flow refusal
    ok = X[:1].copy()
    ok[0, :, 12:] = 1.0
    res = expert.predict(ok, meta={"flow_mode": "consecutive", "timing": {"interval_compatible": False}})
    assert {"consecutive_flow", "incompatible_frame_interval"} <= set(res["abstain_reasons"][0])
    res = expert.predict(ok, operating_point="low_false_alarm")
    assert res["thresholds"]["smoke"] == pytest.approx(1 / (1 + np.exp(-1.0)))
    with pytest.raises(KeyError):
        expert.predict(ok, operating_point="nope")


def test_inference_does_not_import_training_code(tmp_path):
    _artifacts(tmp_path)
    np.save(tmp_path / "x.npy", _features(3, seed=7))
    code = (
        "import sys, numpy as np\n"
        "from src.inference.predict import PhysicsExpert\n"
        f"e = PhysicsExpert.load({str(tmp_path / 'art')!r})\n"
        f"x = np.load({str(tmp_path / 'x.npy')!r})\n"
        "assert e.encode(x).shape == (3, 16)\n"
        "out = e.predict(x)\n"
        "bad = [m for m in sys.modules if m == 'src.training' or m.startswith('src.training.') "
        "or m.startswith('matplotlib')]\n"
        "assert not bad, bad\n"
        "print('ok')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0 and "ok" in res.stdout, res.stderr
    cli = subprocess.run([sys.executable, "-m", "src.inference.predict", "--artifacts", str(tmp_path / "art"),
                          "--features", str(tmp_path / "x.npy"), "--embedding"],
                         cwd=REPO, capture_output=True, text=True, timeout=120)
    assert cli.returncode == 0, cli.stderr
    import json

    doc = json.loads(cli.stdout)
    assert np.asarray(doc["probs"]).shape == (3, 2) and np.asarray(doc["embedding"]).shape == (3, 16)
