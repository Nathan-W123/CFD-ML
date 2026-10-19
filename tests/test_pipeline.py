"""Dataset-generation helpers (split counts, caching fingerprint, configuration overrides) and the
far-field profile end to end (simulate -> render without lag frames -> consecutive-flow features)."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.data.features import FeatureConfig, extract_features
from src.data.pipeline import (cached_simulation, cfd_fingerprint, context_summary, feature_timing,
                               render_and_extract, split_counts)
from src.cfd.scenarios import sample_scenario
from src.utils.config import apply_overrides, load_simulation_config
from src.utils.runtime import child_rng, derive_seed

REPO = Path(__file__).resolve().parents[1]
CFG = load_simulation_config()
FAR_CONFIG = REPO / "configs" / "simulation_far.yaml"
# Shorter source-free spin-up (60 s precursor, 60-90 s pre-roll instead of 240 s and 2-15 min): the
# same profile and frame timing, a third of the CPU. The slot below is accepted on its first attempt.
FAR_FAST = ["time.precursor_s=60", "scenarios.common.spinup_s=[60, 90]"]
FAR_SLOT = ("smoke_only", "test_iid", 0)


def test_split_counts_are_integers_summing_to_n():
    fr = {"train": 0.6, "val_select": 0.1, "val_calib": 0.15, "test_iid": 0.15}
    for n in (7, 50, 250, 251):
        counts = split_counts(n, fr)
        assert sum(counts.values()) == n
        assert all(isinstance(v, int) for v in counts.values())
    assert split_counts(50, fr) == {"train": 30, "val_select": 5, "val_calib": 8, "test_iid": 7} or \
        split_counts(50, fr) == {"train": 30, "val_select": 5, "val_calib": 7, "test_iid": 8}


def test_fingerprint_ignores_render_settings_but_not_physics():
    base = cfd_fingerprint(CFG)
    render = copy.deepcopy(CFG)
    render["render"]["zoom"] = [0.5, 0.9]
    assert cfd_fingerprint(render) == base
    physics = copy.deepcopy(CFG)
    physics["lbm"]["sponge_rate_s"] = 0.5
    assert cfd_fingerprint(physics) != base


def test_overrides_reject_misspelt_keys():
    with pytest.raises(KeyError):
        apply_overrides(CFG, ["dataset.n_per_categroy=250"])
    out = apply_overrides(CFG, ["dataset.n_per_category=250", "+dataset.new_key=1"])
    assert out["dataset"]["n_per_category"] == 250 and out["dataset"]["new_key"] == 1


def test_seed_streams_do_not_collide_on_trailing_zeros():
    assert child_rng(5, 1).random() != child_rng(5, 1, 0).random()
    assert derive_seed(5, 1) != derive_seed(5, 1, 0)


def test_context_summary_is_flat_and_class_independent_in_keys():
    keys = None
    for c in ("smoke_only", "fog_only", "neither", "both"):
        s = context_summary(sample_scenario(c, "train", 0, CFG))
        assert all(isinstance(v, (int, float, str, bool)) for v in s.values())
        keys = keys or set(s)
        assert set(s) == keys


# ----------------------------------------------------------------------------- profiles
def test_feature_timing_follows_the_profile():
    near = feature_timing(CFG)
    assert near == {"expected_interval_s": 5.0, "flow_lag_s": 1.0, "max_offset_s": 0.5}
    d = FeatureConfig()                    # the near-field extractor is exactly the default one
    assert (d.expected_interval_s, d.flow_lag_s, d.max_offset_s) == (5.0, 1.0, 0.5)
    far = feature_timing(load_simulation_config(FAR_CONFIG))
    assert far == {"expected_interval_s": 60.0, "flow_lag_s": 0.0, "max_offset_s": 10.0}
    assert FeatureConfig(**far).flow_mode == "consecutive"
    assert feature_timing({"time": {"frame_interval_s": 30.0, "flow_lag_s": None, "max_offset_s": None}}) == \
        {"expected_interval_s": 30.0, "flow_lag_s": 0.0, "max_offset_s": 0.5}
    assert feature_timing({}) == near


@pytest.fixture(scope="module")
def far_slot(tmp_path_factory):
    """One far-field simulation, cached the way generate_dataset caches it (so the CLI test reuses it)."""
    cfg = load_simulation_config(FAR_CONFIG, FAR_FAST)
    out = tmp_path_factory.mktemp("far")
    category, split, index = FAR_SLOT
    sim, rejected, _ = cached_simulation(out, category, split, index, cfg)
    assert sim is not None, rejected
    return cfg, out, sim


def test_far_profile_renders_without_lag_frames_and_extracts_consecutive_flow(far_slot):
    cfg, _, sim = far_slot
    assert not sim.has_lag and np.allclose(np.diff(sim.times), 60.0)
    feats, info, rendered, _ = render_and_extract(sim, cfg, 0)
    assert feats is not None and feats.shape == (12, 14) and np.isfinite(feats).all()
    assert rendered.lag_frames is None and rendered.lag_timestamps is None
    fmeta = info["feature_meta"]
    assert info["flow_mode"] == fmeta["flow_mode"] == "consecutive"
    assert fmeta["timing"]["compatible"] and fmeta["timing"]["reasons"] == []
    fc = info["feature_config"]
    assert (fc["expected_interval_s"], fc["flow_lag_s"], fc["max_offset_s"]) == (60.0, 0.0, 10.0)
    assert feats[0, 13] == 0.0                      # no observation before the first: no flow pair
    # The stored configuration rebuilds the same extractor (what the exported model does at inference).
    X = extract_features(rendered.frames, rendered.timestamps, rendered.masks, None, None, FeatureConfig(**fc))
    np.testing.assert_array_equal(X, feats)


def test_generate_dataset_far_profile_end_to_end(far_slot):
    """CLI: --config simulation_far.yaml, one test_iid slot, robustness suites and example figures."""
    from src.data.dataset import FeatureSet, load_sequence

    cfg, out, sim = far_slot
    cmd = [sys.executable, "scripts/generate_dataset.py", "--config", str(FAR_CONFIG), "--out", str(out),
           "--splits", "test_iid", "--limit", "1", "--suites", "--examples", str(out / "figures")]
    for item in FAR_FAST:
        cmd += ["--set", item]
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-4000:]
    fs = FeatureSet.load(out / "features.npz")
    assert len(fs) >= 1 and fs.X.shape[1:] == (12, 14) and np.isfinite(fs.X).all()
    assert set(fs.sim_ids) == {sim.sim_id}          # the cached simulation was reused, not re-run
    for m in fs.meta:
        assert m["flow_mode"] == m["feature_meta"]["flow_mode"] == "consecutive"
        assert m["feature_meta"]["timing"]["compatible"]
        fc = m["feature_config"]
        assert (fc["expected_interval_s"], fc["flow_lag_s"], fc["max_offset_s"]) == (60.0, 0.0, 10.0)
        seq = load_sequence(out / m["seq_path"])
        assert seq["lag_frames"] is None and seq["lag_timestamps"] is None
        assert seq["frames"].shape[0] == 12 and np.allclose(np.diff(seq["timestamps"]), 60.0)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["profile"]["flow_modes"] == {"consecutive": len(fs)}
    assert summary["profile"]["expected_interval_s"] == 60.0
    rob = FeatureSet.load(out / "features_robustness.npz")
    assert {m["flow_mode"] for m in rob.meta} == {"consecutive"}
    assert {str(s) for s in rob.splits} == {f"robust:{name}" for name in cfg["robustness"]}
    assert (out / "figures" / f"examples_{FAR_SLOT[0]}.png").is_file()
    # What an exported model trained on these rows records: consecutive flow only, 60 s timing.
    from src.training.train import _preprocessing_record

    rec = _preprocessing_record(fs, {})
    assert rec["training_flow_modes"] == {"consecutive": len(fs)}
    assert FeatureConfig(**rec["feature_config"]).flow_mode == "consecutive"
    assert rec["feature_config"]["expected_interval_s"] == 60.0 and rec["feature_config"]["max_offset_s"] == 10.0
