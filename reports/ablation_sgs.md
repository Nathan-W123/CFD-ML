# Ablation: subgrid texture motion (near field, synthetic only)

Engine v2 displaces the rendered aerosol texture with the subgrid (Smagorinsky)
velocity scale of the CFD (`render.texture.sgs_gain`). To test whether it
explains the lower v2 smoke AUROC, the v2 near-field dataset was re-rendered
from the SAME simulations with `render.texture.sgs_gain = [0, 0]` (every
other render draw identical), and the model retrained and evaluated the same
way (3 seeds, mean ± sd; balanced-point AUROC; false smoke on fog_only at the
low-false-alarm point).

| model / data | test_iid smoke AUROC | test_iid fog AUROC | test_ood smoke AUROC | test_ood fog AUROC | test_iid false smoke (LFA) | test_ood false smoke (LFA) |
|---|---|---|---|---|---|---|
| engine v1 (v1 data) | 0.638 ± 0.007 | 0.699 ± 0.013 | 0.637 ± 0.010 | 0.653 ± 0.010 | 4.5 ± 0.8 % | 11.0 ± 1.7 % |
| engine v2, subgrid texture motion on (v2 data) | 0.590 ± 0.020 | 0.724 ± 0.020 | 0.624 ± 0.016 | 0.694 ± 0.005 | 4.1 ± 2.7 % | 14.3 ± 3.2 % |
| engine v2, subgrid texture motion off (same v2 simulations and renders otherwise) | 0.595 ± 0.013 | 0.726 ± 0.008 | 0.630 ± 0.007 | 0.682 ± 0.011 | 4.5 ± 0.8 % | 14.0 ± 2.6 % |

The subgrid texture motion neither helps nor hurts (all differences are
within the seed spread). The lower v2 smoke AUROC therefore comes from the
other v2 changes: more frequent shadows, artefacts and distant clouds in
every class, more look-alike scenes in "neither", and the time-correlated
camera shake. The v2 test sets are harder than the v1 ones, so v1 and v2
numbers are not directly comparable; v2 fog AUROC is higher despite that.
The CFD-oracle diagnostic shows the simulated smoke is as separable as in v1
(oracle 0.81 vs 0.78), so the loss is in what the camera features can
measure, not in the simulation.
