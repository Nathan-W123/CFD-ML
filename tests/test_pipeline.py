"""Dataset-generation helpers: split counts, caching fingerprint, configuration overrides."""

import copy

import pytest

from src.data.pipeline import cfd_fingerprint, context_summary, split_counts
from src.cfd.scenarios import sample_scenario
from src.utils.config import apply_overrides, load_simulation_config
from src.utils.runtime import child_rng, derive_seed

CFG = load_simulation_config()


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
