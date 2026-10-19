"""Labels, scenario categories and regimes shared by simulation, data and training.

Two independent binary targets (smoke present, fog present) give four
scenario categories. Each category is generated from several physical
regimes; the regime is metadata for stratified evaluation, never a model
input.
"""

from __future__ import annotations

from typing import Dict, Tuple

LABEL_NAMES: Tuple[str, str] = ("smoke", "fog")

CATEGORIES: Tuple[str, ...] = ("smoke_only", "fog_only", "neither", "both")

# Regime strings: a single regime name, or "smoke_regime+fog_regime" for "both".
# category -> (smoke target, fog target)
CATEGORY_LABELS: Dict[str, Tuple[int, int]] = {
    "smoke_only": (1, 0),
    "fog_only": (0, 1),
    "neither": (0, 0),
    "both": (1, 1),
}

SMOKE_REGIMES: Tuple[str, ...] = (
    "hot_plume",        # strong heat source, rises
    "smoldering",       # weak heat, hugs the ground and drifts
    "crosswind",        # strong wind bends the plume over
    "pulsing",          # intermittent emission
    "advected_smoke",   # cooled smoke entering from outside the scene, no heat
)

FOG_REGIMES: Tuple[str, ...] = (
    "radiation_fog",      # near-ground fog under an inversion, light wind
    "advected_fog_bank",  # fog bank carried in through the upwind boundary
    "terrain_fog",        # valley fog over uneven terrain with cold-air drainage
    "fog_patch",          # localised patch drifting and evaporating at its edges
    "fog_dissipating",    # established fog eroded by dry air or warming, or shifted by wind
    "steam_fog",          # warm moist strip under cold air: rising condensate wisps (hard negative)
    "ridge_cloud",        # orographic cap / low cloud: humid flow lifted over a ridge condenses
)

NEITHER_REGIMES: Tuple[str, ...] = (
    "haze_drift",           # broad, faint aerosol layer drifting through
    "dust_burst",           # short unheated ground emission that settles (vehicle / wind dust)
    "illumination_change",  # moving cloud shadow / exposure change, no aerosol
    "camera_artifact",      # shake, zoom jitter, flicker with spurious detector regions
    "distant_cloud",        # a bright cloud drifting slowly in the sky band (classic false positive)
    "rain_shaft",           # precipitation curtain falling from cloud base, drifting with the wind
)

# Regimes a profile must enable explicitly with a positive weight in
# scenarios.regime_weights (the near-field profile leaves them at 0).
OPT_IN_REGIMES: Tuple[str, ...] = ("ridge_cloud", "rain_shaft")

CATEGORY_INDEX: Dict[str, int] = {name: i for i, name in enumerate(CATEGORIES)}


def category_from_labels(smoke: int, fog: int) -> str:
    """The category with the given pair of binary targets."""
    for name, pair in CATEGORY_LABELS.items():
        if pair == (int(smoke), int(fog)):
            return name
    raise ValueError(f"no category for labels smoke={smoke}, fog={fog}")
