# Results (synthetic evaluation)

> Synthetic evaluation only: these numbers measure how well the motion signal of simulated, rendered sequences is learned and how robust it is to simulated nuisances. They are not real-world accuracy and do not show that the model reduces fog-related false alarms on real cameras.

**Real-world status: pending.** No labelled real surveillance footage with frame timing compatible with the model (its exported observation interval and flow lag, e.g. 12 observations ~5 s apart each with a frame ~1 s earlier for the near-field profile) was available; real-world validation is pending.

## Data

| split | sequences | simulations | per category (sequences) |
|---|---|---|---|
| test_iid | 296 | 148 | both 74, fog_only 74, neither 74, smoke_only 74 |
| test_ood | 398 | 199 | both 98, fog_only 100, neither 100, smoke_only 100 |
| train | 1800 | 600 | both 450, fog_only 450, neither 450, smoke_only 450 |
| val_calib | 453 | 151 | both 114, fog_only 114, neither 114, smoke_only 111 |
| val_select | 300 | 100 | both 75, fog_only 75, neither 75, smoke_only 75 |

Generation: simulations accepted 1198, simulation attempts rejected 28, render retries 130, sequences 3247, slots without valid simulation 0.

## Data QA (qa_dataset.py, train split, 1800 sequences)

| category | n | mask valid steps | flow valid steps |
|---|---|---|---|
| both | 450 | 87.0 % | 47.9 % |
| fog_only | 450 | 82.8 % | 47.7 % |
| neither | 450 | 62.3 % | 37.1 % |
| smoke_only | 450 | 79.3 % | 56.1 % |

Optical flow vs CFD velocity (158 sequences): horizontal correlation 0.832, sign agreement where the CFD moves 91.8 %; vertical correlation 0.409, sign agreement where the CFD moves 78.4 %.

Validity flags / parameters with folded univariate AUROC above 0.6:
* validity flag mask_valid predicts smoke (AUC 0.600)
* validity flag mask_valid predicts fog (AUC 0.639)

## Headline: experiment C (CFD-pretrained GRU) against baselines

AUROC is threshold-free; F1, false smoke on fog_only and abstention use the operating point (thresholds fitted on val_calib only). False smoke on fog_only applies the threshold to every fog_only clip, abstentions included (the rate on the clips C decides follows the table). Temperature scaling and abstention apply to C only (its temperatures and abstention margins are also fitted on val_calib), so only C's ECE is after temperature scaling; A and the pooled logistic baseline report raw sigmoid probabilities and never abstain. Exported model = the seed with the best val_select loss.

| model | split | operating point | smoke AUROC | fog AUROC | smoke F1 | fog F1 | false smoke on fog_only | ECE smoke / fog | abstain (any) |
|---|---|---|---|---|---|---|---|---|---|
| C: GRU (exported seed) | test_iid | balanced | 0.595 | 0.747 | 0.595 | 0.720 | 77.0 % | 0.083 / 0.068 | 75.0 % |
| C: GRU (exported seed) | test_iid | low_false_alarm | 0.595 | 0.747 | 0.186 | 0.720 | 4.1 % | 0.083 / 0.068 | 62.2 % |
| C: GRU (exported seed) | test_ood | balanced | 0.641 | 0.692 | 0.655 | 0.635 | 70.0 % | 0.034 / 0.076 | 71.4 % |
| C: GRU (exported seed) | test_ood | low_false_alarm | 0.641 | 0.692 | 0.380 | 0.635 | 18.0 % | 0.034 / 0.076 | 69.3 % |
| A: motion-cue heuristic | test_iid | balanced | 0.574 | 0.500 | 0.613 | 0.612 | 63.5 % | 0.099 / 0.095 | 0.0 % |
| A: motion-cue heuristic | test_iid | low_false_alarm | 0.574 | 0.500 | 0.000 | 0.612 | 1.4 % | 0.099 / 0.095 | 0.0 % |
| A: motion-cue heuristic | test_ood | balanced | 0.545 | 0.508 | 0.556 | 0.600 | 54.0 % | 0.093 / 0.058 | 0.0 % |
| A: motion-cue heuristic | test_ood | low_false_alarm | 0.545 | 0.508 | 0.000 | 0.600 | 1.0 % | 0.093 / 0.058 | 0.0 % |
| pooled logistic (14 x mean/std) | test_iid | balanced | 0.597 | 0.732 | 0.624 | 0.653 | 70.3 % | 0.081 / 0.048 | 0.0 % |
| pooled logistic (14 x mean/std) | test_iid | low_false_alarm | 0.597 | 0.732 | 0.087 | 0.653 | 4.1 % | 0.081 / 0.048 | 0.0 % |
| pooled logistic (14 x mean/std) | test_ood | balanced | 0.621 | 0.691 | 0.634 | 0.616 | 70.0 % | 0.071 / 0.073 | 0.0 % |
| pooled logistic (14 x mean/std) | test_ood | low_false_alarm | 0.621 | 0.691 | 0.326 | 0.616 | 14.0 % | 0.071 / 0.073 | 0.0 % |

C temperatures: fog T = 1.141, smoke T = 1.567. C false smoke on the fog_only clips it does not abstain on: test_iid balanced 76.2 % (16/21); test_iid low_false_alarm 0.0 % (0/37); test_ood balanced 92.0 % (46/50); test_ood low_false_alarm 12.2 % (5/41).

Mean ± sd over the 3 training seeds (balanced point):

| split | smoke AUROC | fog AUROC | false smoke on fog_only | smoke ECE | fog ECE |
|---|---|---|---|---|---|
| test_iid | 0.590 ± 0.020 | 0.724 ± 0.020 | 64.0 ± 17.1 % | 0.078 ± 0.012 | 0.057 ± 0.022 |
| test_ood | 0.624 ± 0.016 | 0.694 ± 0.005 | 66.0 ± 5.3 % | 0.062 ± 0.025 | 0.065 ± 0.010 |

95 % grouped-bootstrap intervals (resampling simulations), exported model:

| split | operating point | smoke AUROC | fog AUROC | false smoke on fog_only |
|---|---|---|---|---|
| test_iid | balanced | 0.595 [0.53, 0.66] | 0.747 [0.68, 0.81] | 77.0 % [67, 86] |
| test_iid | low_false_alarm | 0.595 [0.53, 0.66] | 0.747 [0.68, 0.81] | 4.1 % [0, 9] |
| test_ood | balanced | 0.641 [0.58, 0.71] | 0.692 [0.63, 0.75] | 70.0 % [59, 80] |
| test_ood | low_false_alarm | 0.641 [0.58, 0.71] | 0.692 [0.63, 0.75] | 18.0 % [10, 26] |

### C on test_iid by category (balanced point)

| category | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| both | 74 | 0.784 | 0.757 | n/a | n/a | n/a | 78.4 % |
| fog_only | 74 | n/a | 0.770 | n/a | n/a | 77.0 % | 82.4 % |
| neither | 74 | n/a | n/a | n/a | n/a | n/a | 75.7 % |
| smoke_only | 74 | 0.595 | n/a | n/a | n/a | n/a | 63.5 % |

### C on test_iid by regime (balanced point)

| regime | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| advected_fog_bank | 20 | n/a | 0.900 | n/a | n/a | 90.0 % | 85.0 % |
| advected_smoke | 12 | 0.750 | n/a | n/a | n/a | n/a | 91.7 % |
| advected_smoke+advected_fog_bank | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 100.0 % |
| advected_smoke+fog_dissipating | 6 | 0.500 | 1.000 | n/a | n/a | n/a | 100.0 % |
| advected_smoke+fog_patch | 2 | 0.500 | 0.500 | n/a | n/a | n/a | 100.0 % |
| advected_smoke+steam_fog | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 100.0 % |
| camera_artifact | 10 | n/a | n/a | n/a | n/a | n/a | 90.0 % |
| crosswind | 6 | 1.000 | n/a | n/a | n/a | n/a | 66.7 % |
| crosswind+advected_fog_bank | 6 | 0.667 | 1.000 | n/a | n/a | n/a | 100.0 % |
| crosswind+fog_dissipating | 6 | 0.667 | 1.000 | n/a | n/a | n/a | 100.0 % |
| distant_cloud | 6 | n/a | n/a | n/a | n/a | n/a | 83.3 % |
| dust_burst | 18 | n/a | n/a | n/a | n/a | n/a | 44.4 % |
| fog_dissipating | 16 | n/a | 0.875 | n/a | n/a | 87.5 % | 75.0 % |
| fog_patch | 20 | n/a | 0.650 | n/a | n/a | 65.0 % | 80.0 % |
| haze_drift | 26 | n/a | n/a | n/a | n/a | n/a | 96.2 % |
| hot_plume | 18 | 0.611 | n/a | n/a | n/a | n/a | 66.7 % |
| hot_plume+fog_dissipating | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 50.0 % |
| hot_plume+fog_patch | 4 | 0.750 | 0.500 | n/a | n/a | n/a | 50.0 % |
| hot_plume+radiation_fog | 2 | 1.000 | 0.000 | n/a | n/a | n/a | 0.0 % |
| hot_plume+steam_fog | 4 | 0.750 | 0.250 | n/a | n/a | n/a | 75.0 % |
| illumination_change | 14 | n/a | n/a | n/a | n/a | n/a | 64.3 % |
| pulsing | 26 | 0.346 | n/a | n/a | n/a | n/a | 46.2 % |
| pulsing+advected_fog_bank | 6 | 0.833 | 1.000 | n/a | n/a | n/a | 83.3 % |
| pulsing+fog_dissipating | 6 | 0.833 | 0.833 | n/a | n/a | n/a | 83.3 % |
| pulsing+fog_patch | 4 | 0.750 | 0.250 | n/a | n/a | n/a | 75.0 % |
| pulsing+steam_fog | 2 | 0.500 | 0.500 | n/a | n/a | n/a | 100.0 % |
| radiation_fog | 8 | n/a | 0.875 | n/a | n/a | 62.5 % | 100.0 % |
| smoldering | 12 | 0.750 | n/a | n/a | n/a | n/a | 66.7 % |
| smoldering+advected_fog_bank | 10 | 1.000 | 0.900 | n/a | n/a | n/a | 80.0 % |
| smoldering+fog_dissipating | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 0.0 % |
| smoldering+fog_patch | 8 | 0.750 | 0.500 | n/a | n/a | n/a | 62.5 % |
| steam_fog | 8 | n/a | 0.500 | n/a | n/a | 75.0 % | 75.0 % |
| terrain_fog | 2 | n/a | 0.500 | n/a | n/a | 50.0 % | 100.0 % |

### C on test_iid by mask_source (balanced point)

| mask_source | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| bbox | 62 | 0.657 | 0.897 | 0.608 | 0.844 | 78.6 % | 71.0 % |
| degraded | 140 | 0.710 | 0.721 | 0.580 | 0.733 | 80.6 % | 72.9 % |
| detector | 94 | 0.682 | 0.745 | 0.617 | 0.715 | 70.8 % | 80.9 % |

### C on test_iid by wind_bin (balanced point)

| wind_bin | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| 0.0-1.5 | 64 | 0.618 | 0.575 | 0.558 | 0.595 | 62.5 % | 71.9 % |
| 1.5-3.0 | 74 | 0.632 | 0.750 | 0.633 | 0.727 | 77.8 % | 81.1 % |
| 3.0-4.5 | 64 | 0.692 | 0.844 | 0.613 | 0.810 | 85.0 % | 78.1 % |
| 4.5-6.0 | 58 | 0.733 | 0.893 | 0.517 | 0.831 | 92.9 % | 69.0 % |
| 6.0-7.5 | 36 | 0.850 | 0.850 | 0.681 | 0.884 | 50.0 % | 72.2 % |

### C on test_ood by category (balanced point)

| category | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| both | 98 | 0.827 | 0.633 | n/a | n/a | n/a | 77.6 % |
| fog_only | 100 | n/a | 0.630 | n/a | n/a | 70.0 % | 72.0 % |
| neither | 100 | n/a | n/a | n/a | n/a | n/a | 73.0 % |
| smoke_only | 100 | 0.720 | n/a | n/a | n/a | n/a | 63.0 % |

### C on test_ood by holdout_tag (balanced point)

| holdout_tag | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| elevated_source | 200 | 0.760 | 0.600 | 0.628 | 0.705 | 74.0 % | 65.0 % |
| fast_wind | 198 | 0.786 | 0.663 | 0.660 | 0.682 | 66.0 % | 77.8 % |

### C on test_ood by regime (balanced point)

| regime | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| advected_fog_bank | 30 | n/a | 0.767 | n/a | n/a | 76.7 % | 80.0 % |
| advected_smoke | 18 | 0.722 | n/a | n/a | n/a | n/a | 83.3 % |
| advected_smoke+advected_fog_bank | 2 | 0.500 | 0.000 | n/a | n/a | n/a | 50.0 % |
| advected_smoke+fog_patch | 2 | 0.500 | 1.000 | n/a | n/a | n/a | 100.0 % |
| advected_smoke+steam_fog | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 100.0 % |
| camera_artifact | 20 | n/a | n/a | n/a | n/a | n/a | 75.0 % |
| crosswind | 18 | 0.833 | n/a | n/a | n/a | n/a | 38.9 % |
| crosswind+advected_fog_bank | 10 | 1.000 | 0.700 | n/a | n/a | n/a | 80.0 % |
| crosswind+fog_patch | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 50.0 % |
| distant_cloud | 10 | n/a | n/a | n/a | n/a | n/a | 80.0 % |
| dust_burst | 24 | n/a | n/a | n/a | n/a | n/a | 66.7 % |
| fog_dissipating | 32 | n/a | 0.688 | n/a | n/a | 62.5 % | 84.4 % |
| fog_patch | 20 | n/a | 0.450 | n/a | n/a | 65.0 % | 60.0 % |
| haze_drift | 36 | n/a | n/a | n/a | n/a | n/a | 75.0 % |
| hot_plume | 28 | 0.821 | n/a | n/a | n/a | n/a | 50.0 % |
| hot_plume+advected_fog_bank | 16 | 0.688 | 0.500 | n/a | n/a | n/a | 75.0 % |
| hot_plume+fog_dissipating | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 100.0 % |
| hot_plume+fog_patch | 6 | 0.667 | 0.833 | n/a | n/a | n/a | 100.0 % |
| hot_plume+radiation_fog | 2 | 0.000 | 1.000 | n/a | n/a | n/a | 100.0 % |
| illumination_change | 10 | n/a | n/a | n/a | n/a | n/a | 70.0 % |
| pulsing | 18 | 0.444 | n/a | n/a | n/a | n/a | 72.2 % |
| pulsing+advected_fog_bank | 10 | 0.900 | 0.600 | n/a | n/a | n/a | 80.0 % |
| pulsing+fog_dissipating | 4 | 1.000 | 0.750 | n/a | n/a | n/a | 50.0 % |
| pulsing+fog_patch | 4 | 1.000 | 0.250 | n/a | n/a | n/a | 50.0 % |
| pulsing+radiation_fog | 4 | 1.000 | 0.750 | n/a | n/a | n/a | 25.0 % |
| radiation_fog | 12 | n/a | 0.667 | n/a | n/a | 91.7 % | 58.3 % |
| smoldering | 18 | 0.722 | n/a | n/a | n/a | n/a | 77.8 % |
| smoldering+advected_fog_bank | 8 | 0.875 | 0.500 | n/a | n/a | n/a | 87.5 % |
| smoldering+fog_dissipating | 10 | 0.900 | 0.600 | n/a | n/a | n/a | 90.0 % |
| smoldering+fog_patch | 10 | 0.900 | 0.700 | n/a | n/a | n/a | 80.0 % |
| smoldering+radiation_fog | 2 | 0.000 | 1.000 | n/a | n/a | n/a | 100.0 % |
| smoldering+steam_fog | 2 | 1.000 | 0.000 | n/a | n/a | n/a | 50.0 % |
| steam_fog | 6 | n/a | 0.167 | n/a | n/a | 50.0 % | 33.3 % |

### C on test_ood by mask_source (balanced point)

| mask_source | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| bbox | 84 | 0.707 | 0.718 | 0.680 | 0.782 | 44.4 % | 70.2 % |
| degraded | 189 | 0.809 | 0.694 | 0.594 | 0.694 | 88.0 % | 71.4 % |
| detector | 125 | 0.765 | 0.475 | 0.683 | 0.618 | 56.2 % | 72.0 % |

### C on test_ood by wind_bin (balanced point)

| wind_bin | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| 0.0-1.5 | 58 | 0.556 | 0.583 | 0.479 | 0.667 | 68.8 % | 70.7 % |
| 1.5-3.0 | 50 | 0.750 | 0.789 | 0.460 | 0.553 | 85.0 % | 64.0 % |
| 3.0-4.5 | 42 | 0.846 | 0.562 | 0.624 | 0.710 | 75.0 % | 59.5 % |
| 4.5-6.0 | 32 | 0.792 | 0.286 | 0.807 | 0.659 | 0.0 % | 78.1 % |
| 6.0-7.5 | 80 | 0.800 | 0.571 | 0.738 | 0.726 | 77.8 % | 67.5 % |
| 7.5-9.0 | 136 | 0.786 | 0.667 | 0.633 | 0.629 | 61.1 % | 78.7 % |

## Robustness (re-renders of the test_iid simulations, balanced point)

| suite | n | smoke AUROC | fog AUROC | false smoke on fog_only | abstain | Δ smoke AUROC | Δ fog AUROC |
|---|---|---|---|---|---|---|---|
| bbox_masks | 148 | 0.648 | 0.709 | 54.1 % | 75.0 % | -0.012 | -0.003 |
| detector_masks | 147 | 0.702 | 0.698 | 67.6 % | 76.9 % | 0.042 | -0.014 |
| nominal | 147 | 0.660 | 0.712 | 75.7 % | 74.8 % | n/a | n/a |
| occlusion_heavy | 147 | 0.607 | 0.736 | 70.3 % | 70.1 % | -0.053 | 0.025 |
| render_shift | 147 | 0.648 | 0.609 | 48.6 % | 100.0 % | -0.011 | -0.103 |
| sgs_off | 148 | 0.641 | 0.650 | 64.9 % | 85.1 % | -0.019 | -0.061 |
| shake_heavy | 148 | 0.520 | 0.753 | 75.7 % | 85.1 % | -0.140 | 0.042 |
| zoom_out | 144 | 0.684 | 0.722 | 67.6 % | 76.4 % | 0.024 | 0.011 |

## Leave-one-regime-out (seed 0, evaluated on test_iid)

For a held-out regime positive for a label: recall and AUROC of its sequences against the split's negatives. Negative for a label: false-positive rate and AUROC of the split's positives against it.

| held-out regime | n | smoke | fog |
|---|---|---|---|
| advected_fog_bank | 44 | accuracy 0.409 | recall 0.909, AUROC 0.772 |
| advected_smoke | 24 | recall 0.167, AUROC 0.516 | accuracy 0.500 |
| camera_artifact | 10 | FPR 0.400, AUROC 0.668 | FPR 0.400, AUROC 0.724 |
| crosswind | 18 | recall 1.000, AUROC 0.602 | accuracy 0.778 |
| distant_cloud | 6 | FPR 0.667, AUROC 0.479 | FPR 0.167, AUROC 0.761 |
| dust_burst | 18 | FPR 0.444, AUROC 0.561 | FPR 0.056, AUROC 0.838 |
| fog_dissipating | 38 | accuracy 0.500 | recall 0.868, AUROC 0.788 |
| fog_patch | 38 | accuracy 0.553 | recall 0.342, AUROC 0.550 |
| haze_drift | 26 | FPR 0.731, AUROC 0.471 | FPR 0.923, AUROC 0.386 |
| hot_plume | 30 | recall 0.400, AUROC 0.587 | accuracy 0.433 |
| illumination_change | 14 | FPR 0.071, AUROC 0.799 | FPR 0.500, AUROC 0.798 |
| pulsing | 44 | recall 0.636, AUROC 0.521 | accuracy 0.841 |
| radiation_fog | 10 | accuracy 0.500 | recall 0.700, AUROC 0.673 |
| smoldering | 32 | recall 0.844, AUROC 0.611 | accuracy 0.594 |
| steam_fog | 16 | accuracy 0.438 | recall 0.375, AUROC 0.491 |
| terrain_fog | 2 | FPR 0.000, AUROC 0.895 | recall 0.500, AUROC 0.466 |

## Shortcut audit

Overall flag: **True**.
* flags-only probe above threshold: fog@pooled_held_out=0.6564
* Flags-only probe (validity flags alone), pooled held-out AUROC: fog 0.656, smoke 0.551
* Highest context / render parameter AUROCs for fog (folded, threshold 0.6): wind_speed_ms 0.523, diffusivity_m2s 0.520, lapse_dtheta_dz 0.520, render_view_x_frac 0.520
* Highest context / render parameter AUROCs for smoke (folded, threshold 0.6): wind_sign 0.529, hint_fp_boost 0.526, render_fp_boost 0.526, nu_m2s 0.524

