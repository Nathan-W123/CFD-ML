# Results (synthetic evaluation; real footage in its own section)

> Synthetic evaluation only: these numbers measure how well the motion signal of simulated, rendered sequences is learned and how robust it is to simulated nuisances. They are not real-world accuracy and do not show that the model reduces fog-related false alarms on real cameras.

**Real-world status: partial.** Evaluated labels: smoke; fog n/a (no_labels).

## Data

| split | sequences | simulations | per category (sequences) |
|---|---|---|---|
| test_iid | 296 | 148 | both 74, fog_only 74, neither 74, smoke_only 74 |
| test_ood | 400 | 200 | both 100, fog_only 100, neither 100, smoke_only 100 |
| train | 1799 | 600 | both 450, fog_only 450, neither 450, smoke_only 449 |
| val_calib | 456 | 152 | both 114, fog_only 114, neither 114, smoke_only 114 |
| val_select | 297 | 99 | both 75, fog_only 75, neither 75, smoke_only 72 |

Generation: simulations accepted 1199, simulation attempts rejected 62, render retries 107, sequences 3248, slots without valid simulation 0.

## Data QA (qa_dataset.py, train split, 1799 sequences)

| category | n | mask valid steps | flow valid steps |
|---|---|---|---|
| both | 450 | 89.6 % | 38.4 % |
| fog_only | 450 | 85.6 % | 38.8 % |
| neither | 450 | 74.3 % | 30.2 % |
| smoke_only | 449 | 81.5 % | 38.7 % |

Optical flow vs CFD velocity (149 sequences): horizontal correlation 0.205, sign agreement where the CFD moves 63.4 %; vertical correlation 0.562, sign agreement where the CFD moves 63.8 %.

Validity flags / parameters with folded univariate AUROC above 0.6: none.

## Headline: experiment C (CFD-pretrained GRU) against baselines

AUROC is threshold-free; F1, false smoke on fog_only and abstention use the operating point (thresholds fitted on val_calib only). False smoke on fog_only applies the threshold to every fog_only clip, abstentions included (the rate on the clips C decides follows the table). Temperature scaling and abstention apply to C only (its temperatures and abstention margins are also fitted on val_calib), so only C's ECE is after temperature scaling; A and the pooled logistic baseline report raw sigmoid probabilities and never abstain. Exported model = the seed with the best val_select loss.

| model | split | operating point | smoke AUROC | fog AUROC | smoke F1 | fog F1 | false smoke on fog_only | ECE smoke / fog | abstain (any) |
|---|---|---|---|---|---|---|---|---|---|
| C: GRU (exported seed) | test_iid | balanced | 0.549 | 0.667 | 0.555 | 0.697 | 67.6 % | 0.090 / 0.082 | 78.7 % |
| C: GRU (exported seed) | test_iid | low_false_alarm | 0.549 | 0.667 | 0.201 | 0.697 | 13.5 % | 0.090 / 0.082 | 80.4 % |
| C: GRU (exported seed) | test_ood | balanced | 0.589 | 0.699 | 0.576 | 0.688 | 61.0 % | 0.041 / 0.084 | 80.8 % |
| C: GRU (exported seed) | test_ood | low_false_alarm | 0.589 | 0.699 | 0.166 | 0.688 | 9.0 % | 0.041 / 0.084 | 75.2 % |
| A: motion-cue heuristic | test_iid | balanced | 0.580 | 0.607 | 0.640 | 0.674 | 85.1 % | 0.067 / 0.105 | 0.0 % |
| A: motion-cue heuristic | test_iid | low_false_alarm | 0.580 | 0.607 | 0.075 | 0.674 | 5.4 % | 0.067 / 0.105 | 0.0 % |
| A: motion-cue heuristic | test_ood | balanced | 0.537 | 0.563 | 0.627 | 0.632 | 71.0 % | 0.093 / 0.074 | 0.0 % |
| A: motion-cue heuristic | test_ood | low_false_alarm | 0.537 | 0.563 | 0.047 | 0.632 | 6.0 % | 0.093 / 0.074 | 0.0 % |
| pooled logistic (14 x mean/std) | test_iid | balanced | 0.571 | 0.673 | 0.600 | 0.633 | 66.2 % | 0.079 / 0.076 | 0.0 % |
| pooled logistic (14 x mean/std) | test_iid | low_false_alarm | 0.571 | 0.673 | 0.172 | 0.633 | 9.5 % | 0.079 / 0.076 | 0.0 % |
| pooled logistic (14 x mean/std) | test_ood | balanced | 0.563 | 0.656 | 0.551 | 0.616 | 55.0 % | 0.082 / 0.078 | 0.0 % |
| pooled logistic (14 x mean/std) | test_ood | low_false_alarm | 0.563 | 0.656 | 0.147 | 0.616 | 8.0 % | 0.082 / 0.078 | 0.0 % |

C temperatures: fog T = 1.076, smoke T = 1.228. C false smoke on the fog_only clips it does not abstain on: test_iid balanced 70.4 % (19/27); test_iid low_false_alarm 0.0 % (0/25); test_ood balanced 68.4 % (13/19); test_ood low_false_alarm 0.0 % (0/34).

Mean ± sd over the 3 training seeds (balanced point):

| split | smoke AUROC | fog AUROC | false smoke on fog_only | smoke ECE | fog ECE |
|---|---|---|---|---|---|
| test_iid | 0.559 ± 0.008 | 0.687 ± 0.018 | 65.3 ± 2.1 % | 0.082 ± 0.008 | 0.079 ± 0.006 |
| test_ood | 0.589 ± 0.012 | 0.692 ± 0.013 | 58.7 ± 7.8 % | 0.042 ± 0.006 | 0.077 ± 0.006 |

95 % grouped-bootstrap intervals (resampling simulations), exported model:

| split | operating point | smoke AUROC | fog AUROC | false smoke on fog_only |
|---|---|---|---|---|
| test_iid | balanced | 0.549 [0.47, 0.62] | 0.667 [0.60, 0.74] | 67.6 % [55, 80] |
| test_iid | low_false_alarm | 0.549 [0.47, 0.62] | 0.667 [0.60, 0.74] | 13.5 % [7, 20] |
| test_ood | balanced | 0.589 [0.53, 0.65] | 0.699 [0.64, 0.75] | 61.0 % [52, 70] |
| test_ood | low_false_alarm | 0.589 [0.53, 0.65] | 0.699 [0.64, 0.75] | 9.0 % [4, 16] |

### C on test_iid by category (balanced point)

| category | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| both | 74 | 0.662 | 0.905 | n/a | n/a | n/a | 74.3 % |
| fog_only | 74 | n/a | 0.851 | n/a | n/a | 67.6 % | 71.6 % |
| neither | 74 | n/a | n/a | n/a | n/a | n/a | 87.8 % |
| smoke_only | 74 | 0.500 | n/a | n/a | n/a | n/a | 81.1 % |

### C on test_iid by regime (balanced point)

| regime | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| advected_fog_bank | 14 | n/a | 0.929 | n/a | n/a | 78.6 % | 71.4 % |
| advected_smoke | 18 | 0.667 | n/a | n/a | n/a | n/a | 94.4 % |
| advected_smoke+advected_fog_bank | 4 | 0.750 | 0.750 | n/a | n/a | n/a | 100.0 % |
| advected_smoke+fog_dissipating | 4 | 0.250 | 0.500 | n/a | n/a | n/a | 100.0 % |
| advected_smoke+ridge_cloud | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 100.0 % |
| camera_artifact | 8 | n/a | n/a | n/a | n/a | n/a | 87.5 % |
| crosswind | 10 | 0.500 | n/a | n/a | n/a | n/a | 70.0 % |
| crosswind+advected_fog_bank | 4 | 0.750 | 1.000 | n/a | n/a | n/a | 75.0 % |
| distant_cloud | 8 | n/a | n/a | n/a | n/a | n/a | 100.0 % |
| dust_burst | 6 | n/a | n/a | n/a | n/a | n/a | 83.3 % |
| fog_dissipating | 14 | n/a | 0.786 | n/a | n/a | 50.0 % | 78.6 % |
| fog_patch | 16 | n/a | 0.625 | n/a | n/a | 56.2 % | 68.8 % |
| haze_drift | 30 | n/a | n/a | n/a | n/a | n/a | 93.3 % |
| hot_plume | 14 | 0.643 | n/a | n/a | n/a | n/a | 71.4 % |
| hot_plume+advected_fog_bank | 4 | 0.750 | 1.000 | n/a | n/a | n/a | 50.0 % |
| hot_plume+fog_dissipating | 4 | 1.000 | 1.000 | n/a | n/a | n/a | 75.0 % |
| hot_plume+fog_patch | 4 | 0.750 | 1.000 | n/a | n/a | n/a | 75.0 % |
| hot_plume+radiation_fog | 2 | 0.500 | 0.500 | n/a | n/a | n/a | 100.0 % |
| hot_plume+ridge_cloud | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 100.0 % |
| illumination_change | 10 | n/a | n/a | n/a | n/a | n/a | 90.0 % |
| pulsing | 22 | 0.318 | n/a | n/a | n/a | n/a | 86.4 % |
| pulsing+advected_fog_bank | 6 | 0.500 | 1.000 | n/a | n/a | n/a | 100.0 % |
| pulsing+fog_dissipating | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 0.0 % |
| pulsing+radiation_fog | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 50.0 % |
| pulsing+terrain_fog | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 50.0 % |
| radiation_fog | 14 | n/a | 1.000 | n/a | n/a | 78.6 % | 57.1 % |
| rain_shaft | 12 | n/a | n/a | n/a | n/a | n/a | 66.7 % |
| ridge_cloud | 6 | n/a | 1.000 | n/a | n/a | 83.3 % | 66.7 % |
| smoldering | 10 | 0.400 | n/a | n/a | n/a | n/a | 70.0 % |
| smoldering+advected_fog_bank | 4 | 0.500 | 0.750 | n/a | n/a | n/a | 100.0 % |
| smoldering+fog_dissipating | 2 | 0.500 | 1.000 | n/a | n/a | n/a | 100.0 % |
| smoldering+fog_patch | 8 | 0.500 | 1.000 | n/a | n/a | n/a | 62.5 % |
| smoldering+radiation_fog | 6 | 0.500 | 0.833 | n/a | n/a | n/a | 66.7 % |
| smoldering+ridge_cloud | 10 | 0.700 | 0.900 | n/a | n/a | n/a | 60.0 % |
| smoldering+steam_fog | 2 | 0.500 | 1.000 | n/a | n/a | n/a | 50.0 % |
| terrain_fog | 10 | n/a | 0.900 | n/a | n/a | 70.0 % | 90.0 % |

### C on test_iid by mask_source (balanced point)

| mask_source | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| bbox | 60 | 0.559 | 0.957 | 0.576 | 0.640 | 77.8 % | 88.3 % |
| degraded | 153 | 0.617 | 0.880 | 0.574 | 0.696 | 69.4 % | 78.4 % |
| detector | 83 | 0.515 | 0.840 | 0.492 | 0.653 | 62.1 % | 72.3 % |

### C on test_iid by wind_bin (balanced point)

| wind_bin | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| 0.0-1.5 | 46 | 0.615 | 0.844 | 0.429 | 0.632 | 56.2 % | 71.7 % |
| 1.5-3.0 | 44 | 0.600 | 1.000 | 0.523 | 0.750 | 100.0 % | 75.0 % |
| 10.5-12.0 | 30 | 0.400 | 0.700 | 0.615 | 0.570 | 0.0 % | 80.0 % |
| 3.0-4.5 | 40 | 0.500 | 0.950 | 0.473 | 0.630 | 66.7 % | 80.0 % |
| 4.5-6.0 | 36 | 0.643 | 0.958 | 0.481 | 0.872 | 81.2 % | 83.3 % |
| 6.0-7.5 | 38 | 0.550 | 0.778 | 0.672 | 0.700 | 50.0 % | 73.7 % |
| 7.5-9.0 | 30 | 0.500 | 0.833 | 0.473 | 0.625 | 87.5 % | 83.3 % |
| 9.0-10.5 | 32 | 0.812 | 0.900 | 0.707 | 0.468 | 37.5 % | 87.5 % |

### C on test_ood by category (balanced point)

| category | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| both | 100 | 0.700 | 0.890 | n/a | n/a | n/a | 78.0 % |
| fog_only | 100 | n/a | 0.810 | n/a | n/a | 61.0 % | 86.0 % |
| neither | 100 | n/a | n/a | n/a | n/a | n/a | 83.0 % |
| smoke_only | 100 | 0.500 | n/a | n/a | n/a | n/a | 76.0 % |

### C on test_ood by holdout_tag (balanced point)

| holdout_tag | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| elevated_source | 200 | 0.610 | 0.900 | 0.612 | 0.698 | 58.0 % | 82.5 % |
| fast_wind | 200 | 0.590 | 0.800 | 0.564 | 0.702 | 64.0 % | 79.0 % |

### C on test_ood by regime (balanced point)

| regime | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| advected_fog_bank | 26 | n/a | 0.923 | n/a | n/a | 65.4 % | 84.6 % |
| advected_smoke | 8 | 0.500 | n/a | n/a | n/a | n/a | 100.0 % |
| advected_smoke+advected_fog_bank | 6 | 0.667 | 0.833 | n/a | n/a | n/a | 66.7 % |
| advected_smoke+fog_dissipating | 2 | 0.500 | 1.000 | n/a | n/a | n/a | 100.0 % |
| advected_smoke+fog_patch | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 100.0 % |
| advected_smoke+ridge_cloud | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 100.0 % |
| advected_smoke+steam_fog | 2 | 0.500 | 0.500 | n/a | n/a | n/a | 100.0 % |
| camera_artifact | 16 | n/a | n/a | n/a | n/a | n/a | 87.5 % |
| crosswind | 20 | 0.450 | n/a | n/a | n/a | n/a | 75.0 % |
| crosswind+advected_fog_bank | 6 | 0.833 | 1.000 | n/a | n/a | n/a | 66.7 % |
| crosswind+ridge_cloud | 10 | 0.600 | 0.700 | n/a | n/a | n/a | 100.0 % |
| crosswind+steam_fog | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 100.0 % |
| distant_cloud | 18 | n/a | n/a | n/a | n/a | n/a | 88.9 % |
| dust_burst | 10 | n/a | n/a | n/a | n/a | n/a | 60.0 % |
| fog_dissipating | 22 | n/a | 0.909 | n/a | n/a | 72.7 % | 68.2 % |
| fog_patch | 12 | n/a | 0.833 | n/a | n/a | 58.3 % | 91.7 % |
| haze_drift | 20 | n/a | n/a | n/a | n/a | n/a | 90.0 % |
| hot_plume | 34 | 0.618 | n/a | n/a | n/a | n/a | 64.7 % |
| hot_plume+advected_fog_bank | 8 | 0.750 | 1.000 | n/a | n/a | n/a | 50.0 % |
| hot_plume+fog_dissipating | 4 | 1.000 | 1.000 | n/a | n/a | n/a | 75.0 % |
| hot_plume+fog_patch | 10 | 0.600 | 0.700 | n/a | n/a | n/a | 90.0 % |
| hot_plume+radiation_fog | 2 | 0.500 | 1.000 | n/a | n/a | n/a | 100.0 % |
| illumination_change | 14 | n/a | n/a | n/a | n/a | n/a | 78.6 % |
| pulsing | 16 | 0.312 | n/a | n/a | n/a | n/a | 87.5 % |
| pulsing+advected_fog_bank | 8 | 0.500 | 1.000 | n/a | n/a | n/a | 75.0 % |
| pulsing+fog_dissipating | 2 | 0.500 | 1.000 | n/a | n/a | n/a | 100.0 % |
| pulsing+radiation_fog | 2 | 0.500 | 1.000 | n/a | n/a | n/a | 50.0 % |
| pulsing+ridge_cloud | 8 | 0.750 | 1.000 | n/a | n/a | n/a | 75.0 % |
| radiation_fog | 8 | n/a | 0.875 | n/a | n/a | 75.0 % | 87.5 % |
| rain_shaft | 22 | n/a | n/a | n/a | n/a | n/a | 81.8 % |
| ridge_cloud | 26 | n/a | 0.615 | n/a | n/a | 42.3 % | 100.0 % |
| smoldering | 22 | 0.500 | n/a | n/a | n/a | n/a | 77.3 % |
| smoldering+advected_fog_bank | 8 | 0.875 | 0.875 | n/a | n/a | n/a | 75.0 % |
| smoldering+fog_patch | 2 | 0.500 | 0.500 | n/a | n/a | n/a | 50.0 % |
| smoldering+radiation_fog | 2 | 1.000 | 1.000 | n/a | n/a | n/a | 100.0 % |
| smoldering+ridge_cloud | 12 | 0.667 | 0.917 | n/a | n/a | n/a | 66.7 % |
| steam_fog | 6 | n/a | 0.667 | n/a | n/a | 66.7 % | 83.3 % |

### C on test_ood by mask_source (balanced point)

| mask_source | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| bbox | 85 | 0.432 | 0.938 | 0.430 | 0.758 | 73.9 % | 88.2 % |
| degraded | 195 | 0.731 | 0.914 | 0.646 | 0.782 | 74.5 % | 76.9 % |
| detector | 120 | 0.524 | 0.678 | 0.577 | 0.588 | 30.0 % | 81.7 % |

### C on test_ood by wind_bin (balanced point)

| wind_bin | n | smoke recall | fog recall | smoke AUROC | fog AUROC | false smoke on fog_only | abstain |
|---|---|---|---|---|---|---|---|
| 0.0-1.5 | 24 | 0.667 | 0.917 | 0.493 | 0.556 | 62.5 % | 70.8 % |
| 1.5-3.0 | 32 | 0.800 | 0.958 | 0.664 | 0.891 | 56.2 % | 84.4 % |
| 10.5-12.0 | 30 | 0.500 | 0.667 | 0.638 | 0.549 | 50.0 % | 76.7 % |
| 12.0-13.5 | 98 | 0.580 | 0.780 | 0.676 | 0.666 | 40.9 % | 80.6 % |
| 13.5-15.0 | 102 | 0.600 | 0.820 | 0.464 | 0.750 | 82.1 % | 77.5 % |
| 3.0-4.5 | 22 | 0.600 | 1.000 | 0.525 | 0.732 | 83.3 % | 77.3 % |
| 4.5-6.0 | 18 | 0.500 | 1.000 | 0.681 | 0.713 | 50.0 % | 88.9 % |
| 6.0-7.5 | 32 | 0.667 | 0.857 | 0.804 | 0.746 | 50.0 % | 90.6 % |
| 7.5-9.0 | 12 | 0.375 | 0.800 | 0.531 | 0.500 | 0.0 % | 83.3 % |
| 9.0-10.5 | 30 | 0.650 | 0.833 | 0.555 | 0.741 | 66.7 % | 86.7 % |

## Real footage (Stage B, test_iid split, camera-disjoint)

Status: **partial**; evaluated labels: smoke.
* fog: no_labels (no known fog targets on the real test_iid split (fog not annotated: NaN = unknown); fog metrics are null, not 0); its metrics are n/a, not 0.
* test_iid: 171 windows, 46 videos, 35 cameras; known labels: fog 0, smoke 171.

| model | operating point | smoke AUROC | fog AUROC | smoke F1 | fog F1 | false smoke on fog_only | ECE smoke / fog | abstain smoke / fog |
|---|---|---|---|---|---|---|---|---|
| C: CFD-pretrained, zero-shot | balanced | 0.596 [0.52, 0.67] | n/a | 0.601 | n/a | n/a | 0.095 / n/a | 71.3 % / n/a |
| C: CFD-pretrained, zero-shot | low_false_alarm | 0.596 [0.52, 0.67] | n/a | 0.206 | n/a | n/a | 0.095 / n/a | 74.9 % / n/a |
| B: real data, from scratch | balanced | 0.569 [0.49, 0.66] | n/a | 0.513 | n/a | n/a | 0.074 / n/a | 60.8 % / n/a |
| B: real data, from scratch | low_false_alarm | 0.569 [0.49, 0.66] | n/a | 0.513 | n/a | n/a | 0.074 / n/a | 60.8 % / n/a |
| D: CFD-pretrained, fine-tuned | balanced | 0.580 [0.50, 0.66] | n/a | 0.613 | n/a | n/a | 0.092 / n/a | 63.7 % / n/a |
| D: CFD-pretrained, fine-tuned | low_false_alarm | 0.580 [0.50, 0.66] | n/a | 0.613 | n/a | n/a | 0.092 / n/a | 63.7 % / n/a |

Over the training seeds (balanced point):

* C: CFD-pretrained, zero-shot: smoke AUROC 0.609 ± 0.022 (mean ± sd over 3 seeds)
* B: real data, from scratch: smoke AUROC 0.579 ± 0.020 (mean ± sd over 3 seeds)
* D: CFD-pretrained, fine-tuned: smoke AUROC 0.582 ± 0.001 (mean ± sd over 3 seeds)

Thresholds:

* B: real data, from scratch, smoke: low_false_alarm threshold not fitted as such: no fog_only sequences with both labels known: the threshold equals the balanced one.
* D: CFD-pretrained, fine-tuned, smoke: low_false_alarm threshold not fitted as such: no fog_only sequences with both labels known: the threshold equals the balanced one.

## Robustness (re-renders of the test_iid simulations, balanced point)

| suite | n | smoke AUROC | fog AUROC | false smoke on fog_only | abstain | Δ smoke AUROC | Δ fog AUROC |
|---|---|---|---|---|---|---|---|
| bbox_masks | 148 | 0.550 | 0.668 | 67.6 % | 80.4 % | -0.107 | 0.071 |
| detector_masks | 148 | 0.623 | 0.746 | 51.4 % | 79.1 % | -0.034 | 0.149 |
| nominal | 148 | 0.657 | 0.598 | 54.1 % | 77.7 % | n/a | n/a |
| occlusion_heavy | 147 | 0.623 | 0.572 | 54.1 % | 66.7 % | -0.034 | -0.026 |
| render_shift | 143 | 0.662 | 0.590 | 14.7 % | 100.0 % | 0.005 | -0.007 |
| sgs_off | 148 | 0.585 | 0.668 | 56.8 % | 68.9 % | -0.072 | 0.071 |
| shake_heavy | 148 | 0.623 | 0.630 | 37.8 % | 82.4 % | -0.034 | 0.032 |
| zoom_out | 147 | 0.582 | 0.698 | 56.8 % | 83.0 % | -0.075 | 0.101 |

## Leave-one-regime-out (seed 0, evaluated on test_iid)

For a held-out regime positive for a label: recall and AUROC of its sequences against the split's negatives. Negative for a label: false-positive rate and AUROC of the split's positives against it.

| held-out regime | n | smoke | fog |
|---|---|---|---|
| advected_fog_bank | 36 | accuracy 0.528 | recall 0.694, AUROC 0.746 |
| advected_smoke | 28 | recall 0.464, AUROC 0.375 | accuracy 0.571 |
| camera_artifact | 8 | FPR 0.375, AUROC 0.845 | FPR 0.000, AUROC 0.916 |
| crosswind | 14 | recall 0.500, AUROC 0.583 | accuracy 0.643 |
| distant_cloud | 8 | FPR 0.750, AUROC 0.626 | FPR 0.250, AUROC 0.808 |
| dust_burst | 6 | FPR 0.667, AUROC 0.639 | FPR 0.500, AUROC 0.727 |
| fog_dissipating | 26 | accuracy 0.462 | recall 0.808, AUROC 0.592 |
| fog_patch | 28 | accuracy 0.500 | recall 0.679, AUROC 0.646 |
| haze_drift | 30 | FPR 0.533, AUROC 0.490 | FPR 0.533, AUROC 0.659 |
| hot_plume | 30 | recall 0.733, AUROC 0.564 | accuracy 0.633 |
| illumination_change | 10 | FPR 0.400, AUROC 0.626 | FPR 0.200, AUROC 0.764 |
| pulsing | 34 | recall 0.500, AUROC 0.493 | accuracy 0.735 |
| radiation_fog | 24 | accuracy 0.375 | recall 0.708, AUROC 0.791 |
| rain_shaft | 12 | FPR 0.500, AUROC 0.579 | FPR 0.333, AUROC 0.686 |
| ridge_cloud | 20 | accuracy 0.700 | recall 0.700, AUROC 0.779 |
| smoldering | 42 | recall 0.571, AUROC 0.553 | accuracy 0.619 |
| steam_fog | 2 | recall 1.000, AUROC 0.868 | recall 1.000, AUROC 0.838 |
| terrain_fog | 12 | accuracy 0.417 | recall 0.500, AUROC 0.560 |

## Shortcut audit

Overall flag: **True**.
* flags-only probe above threshold: fog@pooled_held_out=0.6058
* Flags-only probe (validity flags alone), pooled held-out AUROC: fog 0.606, smoke 0.583
* Highest context / render parameter AUROCs for fog (folded, threshold 0.6): hint_distant_cloud 0.537, theta_ground_k 0.534, render_exposure_flicker 0.533, render_camera_shake_px 0.529
* Highest context / render parameter AUROCs for smoke (folded, threshold 0.6): terrain_kind=flat 0.523, wind_speed_ms 0.521, background_haze 0.521, gust_fraction 0.519

## CFD-oracle diagnostic (oracle features are never model inputs)

CFD-oracle diagnostic. Oracle features are computed from the CFD fields (exact velocities, region from the optical depth at the nominal line of sight) and are never model inputs; they bound what a perfect measurement of the same motion representation could achieve. Synthetic data only.

Pooled logistic regression on per-feature time mean and std, test AUROC [95 % CI]:

| features | label | oracle test_iid | camera test_iid | oracle test_ood | camera test_ood |
|---|---|---|---|---|---|
| all | smoke | 0.746 [0.66, 0.82] | 0.572 [0.50, 0.65] | 0.771 [0.71, 0.83] | 0.554 [0.50, 0.60] |
| all | fog | 0.806 [0.72, 0.87] | 0.636 [0.56, 0.71] | 0.724 [0.65, 0.79] | 0.660 [0.60, 0.72] |
| region | smoke | 0.749 [0.67, 0.82] | 0.541 [0.47, 0.62] | 0.765 [0.70, 0.83] | 0.555 [0.50, 0.61] |
| region | fog | 0.789 [0.71, 0.86] | 0.645 [0.57, 0.71] | 0.748 [0.68, 0.81] | 0.688 [0.63, 0.75] |
| velocity | smoke | 0.658 [0.56, 0.75] | 0.577 [0.51, 0.64] | 0.728 [0.65, 0.79] | 0.536 [0.48, 0.59] |
| velocity | fog | 0.623 [0.53, 0.71] | 0.495 [0.43, 0.56] | 0.613 [0.54, 0.69] | 0.503 [0.44, 0.56] |

## CPU benchmark (one core)

| Stage | Single-core time |
|---|---|
| CFD simulation (accepted, incl. precursor) | 8.6 s/sim (417 sims/h) |
| Render 12 frames (no lag frames) | 119 ms |
| Feature extraction (12 x 14) | 75 ms |
| GRU training epoch (1799 sequences) | 0.15 s |
| Inference, one sequence (features -> probabilities) | 0.80 ms |
| Inference, batch of 256 | 3.9 ms |

