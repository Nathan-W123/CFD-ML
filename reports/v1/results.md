# Results (synthetic evaluation)

> Synthetic evaluation only: these numbers measure how well the motion signal of simulated, rendered sequences is learned and how robust it is to simulated nuisances. They are not real-world accuracy and do not show that the model reduces fog-related false alarms on real cameras.

**Real-world status: pending.** No labelled real surveillance footage with compatible frame timing (12 observations ~5 s apart, each with a frame ~1 s earlier) was available; real-world validation is pending.

## Data

| split | sequences | simulations | per category (sequences) |
|---|---|---|---|
| test_iid | 295 | 148 | both 74, fog_only 74, neither 74, smoke_only 73 |
| test_ood | 398 | 199 | both 98, fog_only 100, neither 100, smoke_only 100 |
| train | 1800 | 600 | both 450, fog_only 450, neither 450, smoke_only 450 |
| val_calib | 453 | 151 | both 114, fog_only 114, neither 114, smoke_only 111 |
| val_select | 300 | 100 | both 75, fog_only 75, neither 75, smoke_only 75 |

Generation: simulations accepted 1198, simulation attempts rejected 29, render retries 130, sequences 3246, slots without valid simulation 0.

## Data QA (qa_dataset.py, train split, 1800 sequences)

| category | n | mask valid steps | flow valid steps |
|---|---|---|---|
| both | 450 | 87.6 % | 49.9 % |
| fog_only | 450 | 82.9 % | 48.1 % |
| neither | 450 | 67.4 % | 35.5 % |
| smoke_only | 450 | 78.8 % | 57.0 % |

Optical flow vs CFD velocity (163 sequences): horizontal correlation 0.848, sign agreement where the CFD moves 94.0 %; vertical correlation 0.548, sign agreement where the CFD moves 76.9 %.

Validity flags / parameters with folded univariate AUROC above 0.6:
* validity flag mask_valid predicts fog (AUC 0.613)

## Headline: experiment C (CFD-pretrained GRU) against baselines

AUROC is threshold-free; F1, false smoke on fog_only and abstention use the operating point (thresholds fitted on val_calib only). False smoke on fog_only applies the threshold to every fog_only clip, abstentions included (the rate on the clips C decides follows the table). Temperature scaling and abstention apply to C only (its temperatures and abstention margins are also fitted on val_calib), so only C's ECE is after temperature scaling; A and the pooled logistic baseline report raw sigmoid probabilities and never abstain. Exported model = the seed with the best val_select loss.

| model | split | operating point | smoke AUROC | fog AUROC | smoke F1 | fog F1 | false smoke on fog_only | ECE smoke / fog | abstain (any) |
|---|---|---|---|---|---|---|---|---|---|
| C: GRU (exported seed) | test_iid | balanced | 0.642 | 0.687 | 0.567 | 0.660 | 45.9 % | 0.067 / 0.073 | 72.2 % |
| C: GRU (exported seed) | test_iid | low_false_alarm | 0.642 | 0.687 | 0.204 | 0.660 | 4.1 % | 0.067 / 0.073 | 74.9 % |
| C: GRU (exported seed) | test_ood | balanced | 0.638 | 0.646 | 0.615 | 0.632 | 52.0 % | 0.054 / 0.071 | 70.6 % |
| C: GRU (exported seed) | test_ood | low_false_alarm | 0.638 | 0.646 | 0.331 | 0.632 | 12.0 % | 0.054 / 0.071 | 78.1 % |
| A: motion-cue heuristic | test_iid | balanced | 0.560 | 0.487 | 0.502 | 0.639 | 44.6 % | 0.054 / 0.060 | 0.0 % |
| A: motion-cue heuristic | test_iid | low_false_alarm | 0.560 | 0.487 | 0.039 | 0.639 | 1.4 % | 0.054 / 0.060 | 0.0 % |
| A: motion-cue heuristic | test_ood | balanced | 0.491 | 0.500 | 0.480 | 0.619 | 46.0 % | 0.059 / 0.100 | 0.0 % |
| A: motion-cue heuristic | test_ood | low_false_alarm | 0.491 | 0.500 | 0.020 | 0.619 | 1.0 % | 0.059 / 0.100 | 0.0 % |
| pooled logistic (14 x mean/std) | test_iid | balanced | 0.622 | 0.731 | 0.585 | 0.686 | 67.6 % | 0.096 / 0.058 | 0.0 % |
| pooled logistic (14 x mean/std) | test_iid | low_false_alarm | 0.622 | 0.731 | 0.170 | 0.686 | 4.1 % | 0.096 / 0.058 | 0.0 % |
| pooled logistic (14 x mean/std) | test_ood | balanced | 0.614 | 0.637 | 0.621 | 0.608 | 65.0 % | 0.073 / 0.079 | 0.0 % |
| pooled logistic (14 x mean/std) | test_ood | low_false_alarm | 0.614 | 0.637 | 0.349 | 0.608 | 18.0 % | 0.073 / 0.079 | 0.0 % |

C temperatures: fog T = 1.516, smoke T = 1.520. C false smoke on the fog_only clips it does not abstain on: test_iid balanced 75.9 % (22/29); test_iid low_false_alarm 0.0 % (0/24); test_ood balanced 77.3 % (34/44); test_ood low_false_alarm 0.0 % (0/31).

Mean ± sd over the 3 training seeds (balanced point):

| split | smoke AUROC | fog AUROC | false smoke on fog_only | smoke ECE | fog ECE |
|---|---|---|---|---|---|
| test_iid | 0.638 ± 0.007 | 0.699 ± 0.013 | 39.2 ± 5.9 % | 0.063 ± 0.007 | 0.068 ± 0.005 |
| test_ood | 0.637 ± 0.010 | 0.653 ± 0.010 | 50.7 ± 3.2 % | 0.048 ± 0.012 | 0.068 ± 0.004 |

95 % grouped-bootstrap intervals (resampling simulations), exported model:

| split | operating point | smoke AUROC | fog AUROC | false smoke on fog_only |
|---|---|---|---|---|
| test_iid | balanced | 0.642 [0.57, 0.71] | 0.687 [0.62, 0.76] | 45.9 % [36, 57] |
| test_iid | low_false_alarm | 0.642 [0.57, 0.71] | 0.687 [0.62, 0.76] | 4.1 % [0, 9] |
| test_ood | balanced | 0.638 [0.58, 0.70] | 0.646 [0.57, 0.70] | 52.0 % [41, 64] |
| test_ood | low_false_alarm | 0.638 [0.58, 0.70] | 0.646 [0.57, 0.70] | 12.0 % [5, 20] |

### C on test_iid by category (balanced point)

| category | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| both | 74 | 0.541 | 0.716 | n/a | n/a | n/a | 74.3 % |
| fog_only | 74 | n/a | 0.730 | n/a | n/a | 45.9 % | 70.3 % |
| neither | 74 | n/a | n/a | n/a | n/a | n/a | 78.4 % |
| smoke_only | 73 | 0.548 | n/a | n/a | n/a | n/a | 65.8 % |

### C on test_iid by regime (balanced point)

| regime | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| advected_fog_bank | 20 | n/a | 0.700 | n/a | n/a | 45.0 % | 80.0 % |
| advected_smoke | 12 | 0.333 | n/a | n/a | n/a | n/a | 100.0 % |
| advected_smoke+advected_fog_bank | 2 | 0.000 | 1.000 | n/a | n/a | n/a | 100.0 % |
| advected_smoke+fog_dissipating | 6 | 0.333 | 1.000 | n/a | n/a | n/a | 83.3 % |
| advected_smoke+fog_patch | 2 | 0.500 | 0.500 | n/a | n/a | n/a | 50.0 % |
| advected_smoke+steam_fog | 2 | 0.500 | 0.500 | n/a | n/a | n/a | 100.0 % |
| camera_artifact | 2 | n/a | n/a | n/a | n/a | n/a | 100.0 % |
| crosswind | 6 | 0.500 | n/a | n/a | n/a | n/a | 83.3 % |
| crosswind+advected_fog_bank | 6 | 0.167 | 0.500 | n/a | n/a | n/a | 83.3 % |
| crosswind+fog_dissipating | 6 | 0.167 | 0.833 | n/a | n/a | n/a | 83.3 % |
| distant_cloud | 2 | n/a | n/a | n/a | n/a | n/a | 100.0 % |
| dust_burst | 30 | n/a | n/a | n/a | n/a | n/a | 56.7 % |
| fog_dissipating | 16 | n/a | 0.938 | n/a | n/a | 43.8 % | 81.2 % |
| fog_patch | 20 | n/a | 0.600 | n/a | n/a | 45.0 % | 65.0 % |
| haze_drift | 38 | n/a | n/a | n/a | n/a | n/a | 94.7 % |
| hot_plume | 18 | 0.667 | n/a | n/a | n/a | n/a | 55.6 % |
| hot_plume+fog_dissipating | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 100.0 % |
| hot_plume+fog_patch | 4 | 1.000 | 0.500 | n/a | n/a | n/a | 25.0 % |
| hot_plume+radiation_fog | 2 | 0.500 | 0.500 | n/a | n/a | n/a | 50.0 % |
| hot_plume+steam_fog | 4 | 0.500 | 0.250 | n/a | n/a | n/a | 75.0 % |
| illumination_change | 2 | n/a | n/a | n/a | n/a | n/a | 50.0 % |
| pulsing | 25 | 0.520 | n/a | n/a | n/a | n/a | 48.0 % |
| pulsing+advected_fog_bank | 6 | 0.333 | 1.000 | n/a | n/a | n/a | 66.7 % |
| pulsing+fog_dissipating | 6 | 0.833 | 0.833 | n/a | n/a | n/a | 83.3 % |
| pulsing+fog_patch | 4 | 0.500 | 0.250 | n/a | n/a | n/a | 75.0 % |
| pulsing+steam_fog | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 50.0 % |
| radiation_fog | 8 | n/a | 1.000 | n/a | n/a | 37.5 % | 100.0 % |
| smoldering | 12 | 0.667 | n/a | n/a | n/a | n/a | 75.0 % |
| smoldering+advected_fog_bank | 10 | 0.700 | 0.800 | n/a | n/a | n/a | 80.0 % |
| smoldering+fog_dissipating | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 0.0 % |
| smoldering+fog_patch | 8 | 0.625 | 0.625 | n/a | n/a | n/a | 87.5 % |
| steam_fog | 8 | n/a | 0.500 | n/a | n/a | 75.0 % | 12.5 % |
| terrain_fog | 2 | n/a | 0.500 | n/a | n/a | 0.0 % | 50.0 % |

### C on test_iid by mask_source (balanced point)

| mask_source | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| bbox | 62 | 0.629 | 0.690 | 0.711 | 0.770 | 21.4 % | 66.1 % |
| degraded | 140 | 0.522 | 0.779 | 0.571 | 0.685 | 61.1 % | 72.1 % |
| detector | 93 | 0.512 | 0.667 | 0.680 | 0.657 | 37.5 % | 76.3 % |

### C on test_iid by wind_bin (balanced point)

| wind_bin | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| 0.0-1.5 | 64 | 0.529 | 0.600 | 0.695 | 0.599 | 37.5 % | 67.2 % |
| 1.5-3.0 | 74 | 0.605 | 0.679 | 0.707 | 0.622 | 44.4 % | 74.3 % |
| 3.0-4.5 | 64 | 0.692 | 0.844 | 0.669 | 0.705 | 45.0 % | 76.6 % |
| 4.5-6.0 | 57 | 0.414 | 0.714 | 0.521 | 0.798 | 57.1 % | 71.9 % |
| 6.0-7.5 | 36 | 0.450 | 0.850 | 0.566 | 0.825 | 50.0 % | 69.4 % |

### C on test_ood by category (balanced point)

| category | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| both | 98 | 0.602 | 0.663 | n/a | n/a | n/a | 68.4 % |
| fog_only | 100 | n/a | 0.640 | n/a | n/a | 52.0 % | 70.0 % |
| neither | 100 | n/a | n/a | n/a | n/a | n/a | 83.0 % |
| smoke_only | 100 | 0.650 | n/a | n/a | n/a | n/a | 61.0 % |

### C on test_ood by holdout_tag (balanced point)

| holdout_tag | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| elevated_source | 200 | 0.650 | 0.640 | 0.628 | 0.666 | 60.0 % | 67.5 % |
| fast_wind | 198 | 0.602 | 0.663 | 0.653 | 0.629 | 44.0 % | 73.7 % |

### C on test_ood by regime (balanced point)

| regime | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| advected_fog_bank | 30 | n/a | 0.767 | n/a | n/a | 60.0 % | 66.7 % |
| advected_smoke | 18 | 0.611 | n/a | n/a | n/a | n/a | 77.8 % |
| advected_smoke+advected_fog_bank | 2 | 0.000 | 0.000 | n/a | n/a | n/a | 100.0 % |
| advected_smoke+fog_patch | 2 | 0.500 | 0.500 | n/a | n/a | n/a | 100.0 % |
| advected_smoke+steam_fog | 2 | 0.500 | 1.000 | n/a | n/a | n/a | 100.0 % |
| crosswind | 18 | 0.556 | n/a | n/a | n/a | n/a | 55.6 % |
| crosswind+advected_fog_bank | 10 | 0.800 | 0.700 | n/a | n/a | n/a | 60.0 % |
| crosswind+fog_patch | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 50.0 % |
| dust_burst | 42 | n/a | n/a | n/a | n/a | n/a | 83.3 % |
| fog_dissipating | 32 | n/a | 0.719 | n/a | n/a | 37.5 % | 81.2 % |
| fog_patch | 20 | n/a | 0.400 | n/a | n/a | 55.0 % | 55.0 % |
| haze_drift | 56 | n/a | n/a | n/a | n/a | n/a | 83.9 % |
| hot_plume | 28 | 0.857 | n/a | n/a | n/a | n/a | 53.6 % |
| hot_plume+advected_fog_bank | 16 | 0.688 | 0.625 | n/a | n/a | n/a | 68.8 % |
| hot_plume+fog_dissipating | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 50.0 % |
| hot_plume+fog_patch | 6 | 0.333 | 0.833 | n/a | n/a | n/a | 66.7 % |
| hot_plume+radiation_fog | 2 | 0.000 | 1.000 | n/a | n/a | n/a | 100.0 % |
| illumination_change | 2 | n/a | n/a | n/a | n/a | n/a | 50.0 % |
| pulsing | 18 | 0.444 | n/a | n/a | n/a | n/a | 61.1 % |
| pulsing+advected_fog_bank | 10 | 0.700 | 0.600 | n/a | n/a | n/a | 70.0 % |
| pulsing+fog_dissipating | 4 | 0.500 | 0.750 | n/a | n/a | n/a | 100.0 % |
| pulsing+fog_patch | 4 | 0.750 | 0.500 | n/a | n/a | n/a | 50.0 % |
| pulsing+radiation_fog | 4 | 1.000 | 0.750 | n/a | n/a | n/a | 50.0 % |
| radiation_fog | 12 | n/a | 0.750 | n/a | n/a | 58.3 % | 75.0 % |
| smoldering | 18 | 0.667 | n/a | n/a | n/a | n/a | 61.1 % |
| smoldering+advected_fog_bank | 8 | 0.375 | 0.375 | n/a | n/a | n/a | 75.0 % |
| smoldering+fog_dissipating | 10 | 0.600 | 0.800 | n/a | n/a | n/a | 90.0 % |
| smoldering+fog_patch | 10 | 0.500 | 0.700 | n/a | n/a | n/a | 40.0 % |
| smoldering+radiation_fog | 2 | 0.000 | 0.500 | n/a | n/a | n/a | 100.0 % |
| smoldering+steam_fog | 2 | 1.000 | 0.500 | n/a | n/a | n/a | 0.0 % |
| steam_fog | 6 | n/a | 0.167 | n/a | n/a | 66.7 % | 66.7 % |

### C on test_ood by mask_source (balanced point)

| mask_source | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| bbox | 84 | 0.610 | 0.641 | 0.673 | 0.660 | 33.3 % | 73.8 % |
| degraded | 189 | 0.685 | 0.724 | 0.609 | 0.664 | 68.0 % | 67.2 % |
| detector | 125 | 0.559 | 0.541 | 0.674 | 0.600 | 37.5 % | 73.6 % |

### C on test_ood by wind_bin (balanced point)

| wind_bin | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| 0.0-1.5 | 58 | 0.500 | 0.667 | 0.458 | 0.586 | 62.5 % | 67.2 % |
| 1.5-3.0 | 50 | 0.500 | 0.842 | 0.462 | 0.658 | 60.0 % | 66.0 % |
| 3.0-4.5 | 42 | 0.808 | 0.375 | 0.712 | 0.647 | 62.5 % | 69.0 % |
| 4.5-6.0 | 32 | 0.667 | 0.500 | 0.823 | 0.770 | 0.0 % | 71.9 % |
| 6.0-7.5 | 80 | 0.675 | 0.643 | 0.723 | 0.705 | 55.6 % | 75.0 % |
| 7.5-9.0 | 136 | 0.586 | 0.641 | 0.628 | 0.554 | 41.7 % | 71.3 % |

## Robustness (re-renders of the test_iid simulations, balanced point)

| suite | n | smoke AUROC | fog AUROC | false smoke on fog_only | abstain | Δ smoke AUROC | Δ fog AUROC |
|---|---|---|---|---|---|---|---|
| bbox_masks | 147 | 0.613 | 0.662 | 32.4 % | 75.5 % | -0.054 | -0.026 |
| detector_masks | 147 | 0.712 | 0.680 | 45.9 % | 72.1 % | 0.045 | -0.008 |
| nominal | 147 | 0.668 | 0.688 | 40.5 % | 72.8 % | n/a | n/a |
| occlusion_heavy | 145 | 0.562 | 0.742 | 51.4 % | 67.6 % | -0.105 | 0.054 |
| render_shift | 147 | 0.572 | 0.625 | 21.6 % | 100.0 % | -0.096 | -0.063 |
| shake_heavy | 148 | 0.565 | 0.666 | 43.2 % | 81.8 % | -0.103 | -0.022 |
| zoom_out | 143 | 0.628 | 0.676 | 54.1 % | 70.6 % | -0.040 | -0.012 |

## Leave-one-regime-out (seed 0, evaluated on test_iid)

For a held-out regime positive for a label: recall and AUROC of its sequences against the split's negatives. Negative for a label: false-positive rate and AUROC of the split's positives against it.

| held-out regime | n | smoke | fog |
|---|---|---|---|
| advected_fog_bank | 44 | accuracy 0.455 | recall 0.614, AUROC 0.610 |
| advected_smoke | 24 | recall 0.417, AUROC 0.523 | accuracy 0.625 |
| camera_artifact | 2 | FPR 0.000, AUROC 0.964 | FPR 0.000, AUROC 0.811 |
| crosswind | 18 | recall 0.278, AUROC 0.521 | accuracy 0.556 |
| distant_cloud | 2 | FPR 0.500, AUROC 0.565 | FPR 1.000, AUROC 0.608 |
| dust_burst | 30 | FPR 0.600, AUROC 0.461 | FPR 0.233, AUROC 0.700 |
| fog_dissipating | 38 | accuracy 0.474 | recall 0.605, AUROC 0.744 |
| fog_patch | 38 | accuracy 0.474 | recall 0.395, AUROC 0.493 |
| haze_drift | 38 | FPR 0.711, AUROC 0.537 | FPR 0.816, AUROC 0.398 |
| hot_plume | 30 | recall 0.700, AUROC 0.694 | accuracy 0.500 |
| illumination_change | 2 | FPR 0.000, AUROC 0.711 | FPR 0.500, AUROC 0.645 |
| pulsing | 43 | recall 0.326, AUROC 0.582 | accuracy 0.581 |
| radiation_fog | 10 | accuracy 0.900 | recall 0.800, AUROC 0.721 |
| smoldering | 32 | recall 0.594, AUROC 0.616 | accuracy 0.656 |
| steam_fog | 16 | accuracy 0.438 | recall 0.438, AUROC 0.478 |
| terrain_fog | 2 | FPR 0.000, AUROC 0.912 | recall 0.500, AUROC 0.408 |

## Shortcut audit

Overall flag: **True**.
* flags-only probe above threshold: fog@pooled_held_out=0.6029
* Flags-only probe (validity flags alone), pooled held-out AUROC: fog 0.603, smoke 0.562
* Highest context / render parameter AUROCs for fog (folded, threshold 0.6): wind_speed_ms 0.523, diffusivity_m2s 0.520, lapse_dtheta_dz 0.520, render_view_x_frac 0.519
* Highest context / render parameter AUROCs for smoke (folded, threshold 0.6): wind_sign 0.529, hint_fp_boost 0.526, render_fp_boost 0.526, nu_m2s 0.524

## Synthetic transfer proxy (NOT real-world evidence)

Separately generated synthetic simulations (different base seed) rendered with detector / box masks and shifted rendering; re-split by simulation. Not real footage.

| model | smoke AUROC | fog AUROC | false smoke on fog_only (balanced) |
|---|---|---|---|
| C_zero_shot | 0.566 [0.46, 0.67] | 0.704 [0.62, 0.78] | 30.3 % |
| B_scratch | 0.596 [0.48, 0.71] | 0.609 [0.51, 0.70] | 24.2 % |
| D_finetuned | 0.586 [0.47, 0.69] | 0.674 [0.59, 0.75] | 36.4 % |

## CFD-oracle diagnostic (oracle features are never model inputs)

CFD-oracle diagnostic. Oracle features are computed from the CFD fields (exact velocities, region from the optical depth at the nominal line of sight) and are never model inputs; they bound what a perfect measurement of the same motion representation could achieve. Synthetic data only.

Pooled logistic regression on per-feature time mean and std, test AUROC [95 % CI]:

| features | label | oracle test_iid | camera test_iid | oracle test_ood | camera test_ood |
|---|---|---|---|---|---|
| all | smoke | 0.783 [0.71, 0.85] | 0.600 [0.53, 0.67] | 0.745 [0.67, 0.81] | 0.590 [0.53, 0.65] |
| all | fog | 0.751 [0.67, 0.83] | 0.728 [0.67, 0.79] | 0.663 [0.59, 0.73] | 0.636 [0.57, 0.69] |
| region | smoke | 0.715 [0.63, 0.80] | 0.615 [0.55, 0.68] | 0.734 [0.66, 0.80] | 0.526 [0.46, 0.59] |
| region | fog | 0.727 [0.65, 0.81] | 0.733 [0.68, 0.79] | 0.768 [0.70, 0.84] | 0.680 [0.62, 0.74] |
| velocity | smoke | 0.744 [0.66, 0.82] | 0.552 [0.48, 0.62] | 0.707 [0.63, 0.77] | 0.603 [0.54, 0.67] |
| velocity | fog | 0.628 [0.54, 0.72] | 0.625 [0.55, 0.69] | 0.488 [0.41, 0.57] | 0.567 [0.51, 0.63] |

## CPU benchmark (one core)

| Stage | Single-core time |
|---|---|
| CFD simulation (accepted, incl. precursor) | 5.6 s/sim (637 sims/h) |
| Render 12 + 12 lag frames | 142 ms |
| Feature extraction (12 x 14) | 58 ms |
| GRU training epoch (1800 sequences) | 0.17 s |
| Inference, one sequence (features -> probabilities) | 0.92 ms |
| Inference, batch of 256 | 3.0 ms |

