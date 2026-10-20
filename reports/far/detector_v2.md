# Real-footage detector (far field): smoke and fog

> Linear heads on frozen CLIP appearance features (and, if selected, the motion features; for smoke also a tile head on 4 x 6 fine tiles and, if given, a zoom head on full-resolution crops and the prompt-contrast change within the window and against a reference frame 14 min earlier, all stacked with the window head), trained on real labelled windows; model selection and thresholds on the dev cameras only; intervals resample cameras. Imagery courtesy of HPWREN (https://www.hpwren.ucsd.edu/).

Goals: fog AUROC >= 0.90; smoke AUROC >= 0.85; at the deploy point at most 5 % false smoke on fog windows while detecting at least 50 % of the smoke windows.

| head | selected feature groups | C | CV AUROC pooled | CV AUROC within sets | dev windows |
|---|---|---|---|---|---|
| fog | full, sims | 0.01 | 0.977 | 0.945 | 1303 |
| smoke | full, tiles, sims, fine | 0.003 | 0.888 | 0.804 | 2051 |
| smoke: tile head | fine tiles, best tile | 0.0003 | 0.867 | 0.769 | 2051 |
| smoke: zoom head | full-resolution zoom crops, best crop | 0.0001 | 0.877 | 0.806 | 2051 |
| smoke: v1 stack | head + tile head | | 0.896 | 0.809 | 2051 |
| smoke: stacked (deployed) | head+tile+zoom+chg_max+chg_rz+fog_chg+fogd_chg (10); head+tile+zoom+chg_max+chg_rz+ref_max+ref_rz+fog_chg+fogd_chg+fog_ref+fogd_ref (2041) | | 0.910 | 0.838 | 2051 |

Thresholds: smoke deploy 0.269, smoke strict 0.530, smoke balanced 0.282, fog balanced 0.243; smoke alarms withheld when P(fog) >= 0.95.

Rules: deploy_false_smoke 0.025, deploy_recall 0.8, strict_false_alarm 0.05, fog_veto 0.95, stack_fog_weight 3.0, select_on within.

## v2 against the v1 stack (window + tile heads), same operating rules

| | dev v1 | dev v2 | test v1 | test v2 |
|---|---|---|---|---|
| smoke AUROC within sets | 0.809 | 0.838 | 0.761 | 0.771 |
| smoke AUROC, figlib | 0.760 | 0.794 | 0.713 | 0.726 |
| smoke AUROC, hpwren_smoke | 0.901 | 0.926 | 0.844 | 0.871 |
| smoke AUROC, hpwren_smoke2 | 0.892 | 0.909 | 0.928 | 0.919 |
| fog AUROC | 0.977 | 0.977 | 0.963 | 0.963 |
| deploy: smoke recall | 80.1% | 80.1% | 81.7% | 76.3% |
| deploy: alarms on fog windows | 2.1% | 1.4% | 3.7% | 3.7% |
| deploy: alarms on all smoke-free windows | 20.5% | 15.3% | 26.9% | 20.3% |
| deploy: alarms on smoke-free windows, figlib | 55.1% | 41.5% | 65.1% | 49.4% |
| deploy: alarms on smoke-free windows, hpwren_smoke | 13.4% | 5.8% | 29.0% | 22.6% |
| deploy: alarms on smoke-free windows, hpwren_smoke2 | 18.9% | 14.1% | 21.5% | 15.2% |
| strict: smoke recall | 59.7% | 65.4% | 53.4% | 61.8% |
| strict: alarms on all smoke-free windows | 4.9% | 4.9% | 5.9% | 7.9% |

Test windows by stack: head+tile+zoom+chg_max+chg_rz+ref_max+ref_rz+fog_chg+fogd_chg+fog_ref+fogd_ref: 437; head+tile+zoom+chg_max+chg_rz+fog_chg+fogd_chg: 3.

## Dev cameras, out-of-fold (cross-validated)

2064 windows, 143 camera mounts; smoke labelled 2051 (653 smoke), fog labelled 1303 (300 fog), 291 fog windows without smoke.

| metric | value | goal | met |
|---|---|---|---|
| fog AUROC | 0.977 [0.96, 0.99] | >= 0.90 | True |
| fog AUROC within sets | 0.945 [0.92, 0.97] | | |
| smoke AUROC | 0.910 [0.89, 0.93] | >= 0.85 | True |
| smoke AUROC within sets | 0.838 [0.81, 0.87] | | |
| false smoke on fog windows (deploy) | 1.4% [0, 3] | <= 5 % | True |
| smoke recall (deploy) | 80.1% [76, 85] | >= 50 % | |
| false alarms on all no-smoke windows (deploy) | 15.3% [12, 19] | | |
| strict point: false alarms on all no-smoke windows | 4.9% [3, 7] | | |
| strict point: false smoke on fog windows | 0.3% [0, 1] | | |
| strict point: smoke recall | 65.4% [60, 71] | | |

| set | smoke AUROC | fog AUROC | false smoke on fog | false alarms, no-smoke | smoke recall | strict: false alarms, no-smoke | strict: smoke recall |
|---|---|---|---|---|---|---|---|
| figlib | 0.794 | n/a | n/a | 41.5% | 79.7% | 16.7% | 62.5% |
| hpwren_fog | n/a | 0.935 | 0.0% | 0.5% | n/a | 0.0% | n/a |
| hpwren_fog2 | n/a | 0.949 | 2.2% | 4.1% | n/a | 1.1% | n/a |
| hpwren_smoke | 0.926 | 0.978 | 0.0% | 5.8% | 81.0% | 0.6% | 71.9% |
| hpwren_smoke2 | 0.909 | n/a | n/a | 14.1% | 80.2% | 1.7% | 66.7% |

## Held-out test cameras

440 windows, 35 camera mounts; smoke labelled 436 (131 smoke), fog labelled 261 (56 fog), 54 fog windows without smoke.

| metric | value | goal | met |
|---|---|---|---|
| fog AUROC | 0.963 [0.91, 0.99] | >= 0.90 | True |
| fog AUROC within sets | 0.932 [0.85, 0.99] | | |
| smoke AUROC | 0.881 [0.82, 0.93] | >= 0.85 | True |
| smoke AUROC within sets | 0.771 [0.67, 0.86] | | |
| false smoke on fog windows (deploy) | 3.7% [0, 10] | <= 5 % | True |
| smoke recall (deploy) | 76.3% [66, 86] | >= 50 % | |
| false alarms on all no-smoke windows (deploy) | 20.3% [14, 28] | | |
| strict point: false alarms on all no-smoke windows | 7.9% [4, 12] | | |
| strict point: false smoke on fog windows | 1.9% [0, 6] | | |
| strict point: smoke recall | 61.8% [51, 74] | | |

| set | smoke AUROC | fog AUROC | false smoke on fog | false alarms, no-smoke | smoke recall | strict: false alarms, no-smoke | strict: smoke recall |
|---|---|---|---|---|---|---|---|
| figlib | 0.726 | n/a | n/a | 49.4% | 73.9% | 19.3% | 55.7% |
| hpwren_fog | n/a | 0.878 | 4.8% | 2.4% | n/a | 0.0% | n/a |
| hpwren_fog2 | n/a | 0.941 | 3.2% | 1.4% | n/a | 1.4% | n/a |
| hpwren_smoke | 0.871 | n/a | n/a | 22.6% | 80.0% | 6.5% | 70.0% |
| hpwren_smoke2 | 0.919 | 1.000 | 0.0% | 15.2% | 82.6% | 6.3% | 78.3% |
