# Real-footage detector (far field): smoke and fog

> Linear heads on frozen CLIP appearance features (and, if selected, the motion features; for smoke also a tile head on 4 x 6 fine tiles and, if given, a zoom head on full-resolution crops and the prompt-contrast change within the window and against a reference frame 14 min earlier, all stacked with the window head), trained on real labelled windows; model selection and thresholds on the dev cameras only; intervals resample cameras. Imagery courtesy of HPWREN (https://www.hpwren.ucsd.edu/).

Goals: fog AUROC >= 0.90; smoke AUROC >= 0.85; at the deploy point at most 5 % false smoke on fog windows while detecting at least 50 % of the smoke windows.

| head | selected feature groups | C | CV AUROC pooled | CV AUROC within sets | dev windows |
|---|---|---|---|---|---|
| fog | full, sims | 0.01 | 0.965 | 0.930 | 1745 |
| smoke | full, tiles, sims, fine | 0.003 | 0.896 | 0.798 | 2535 |
| smoke: tile head | fine tiles, best tile (box labels) | 0.003 | 0.886 | 0.859 | 2535 |
| smoke: zoom head | full-resolution zoom crops, best crop (box labels) | 0.003 | 0.849 | 0.838 | 2535 |
| smoke: tileref head | fine tiles against the reference frame, best tile (box labels) | 0.003 | 0.906 | 0.881 | 2525 |
| smoke: v1 stack | head + tile head | | 0.920 | 0.853 | 2535 |
| smoke: stacked (deployed) | head+tile+zoom+chg_max+chg_rz+fog_chg+fogd_chg (10); head+tile+zoom+chg_max+chg_rz+ref_max+ref_rz+fog_chg+fogd_chg+fog_ref+fogd_ref+tileref (2525) | | 0.929 | 0.870 | 2535 |

Thresholds: smoke deploy 0.234, smoke strict 0.354, smoke balanced 0.312, fog balanced 0.133; smoke alarms withheld when P(fog) >= 0.95.

Rules: deploy_false_smoke 0.025, deploy_recall 0.8, strict_false_alarm 0.05, fog_veto 0.95, stack_fog_weight 3.0, select_on within, boxes ['configs/real/smoke_dev_boxes.csv'], box_min_confidence low, box_pos_frac 0.25, box_balance False, ref_tile_head True.

## Deployed stack against the v1 stack (window + tile heads), same operating rules

| | dev, v1 stack | dev, deployed | test, v1 stack | test, deployed |
|---|---|---|---|---|
| smoke AUROC within sets | 0.853 | 0.870 | 0.805 | 0.796 |
| smoke AUROC, figlib | 0.807 | 0.832 | 0.763 | 0.756 |
| smoke AUROC, hpwren_bg | 0.803 | 0.874 | n/a | n/a |
| smoke AUROC, hpwren_smoke | 0.942 | 0.937 | 0.910 | 0.911 |
| smoke AUROC, hpwren_smoke2 | 0.930 | 0.936 | 0.935 | 0.916 |
| fog AUROC | 0.965 | 0.965 | 0.968 | 0.968 |
| deploy: smoke recall | 76.6% | 80.1% | 71.0% | 74.8% |
| deploy: alarms on fog windows | 2.5% | 1.2% | 1.9% | 1.9% |
| deploy: alarms on all smoke-free windows | 7.9% | 9.5% | 11.8% | 15.1% |
| deploy: alarms on smoke-free windows, figlib | 30.5% | 33.6% | 31.3% | 36.1% |
| deploy: alarms on smoke-free windows, hpwren_bg | 4.8% | 6.4% | n/a | n/a |
| deploy: alarms on smoke-free windows, hpwren_fog | 0.9% | 0.0% | 0.0% | 0.0% |
| deploy: alarms on smoke-free windows, hpwren_fog2 | 1.6% | 1.9% | 1.4% | 2.8% |
| deploy: alarms on smoke-free windows, hpwren_smoke | 0.0% | 2.9% | 6.5% | 12.9% |
| deploy: alarms on smoke-free windows, hpwren_smoke2 | 3.1% | 5.8% | 8.9% | 12.7% |
| strict: smoke recall | 70.6% | 74.3% | 65.6% | 67.9% |
| strict: alarms on all smoke-free windows | 5.0% | 5.0% | 5.9% | 9.2% |

Test windows by stack: head+tile+zoom+chg_max+chg_rz+ref_max+ref_rz+fog_chg+fogd_chg+fog_ref+fogd_ref+tileref: 437; head+tile+zoom+chg_max+chg_rz+fog_chg+fogd_chg: 3.

## Dev cameras, out-of-fold (cross-validated)

2548 windows, 143 camera mounts; smoke labelled 2535 (654 smoke), fog labelled 1745 (333 fog), 324 fog windows without smoke.

| metric | value | goal | met |
|---|---|---|---|
| fog AUROC | 0.965 [0.95, 0.98] | >= 0.90 | True |
| fog AUROC within sets | 0.930 [0.91, 0.95] | | |
| smoke AUROC | 0.929 [0.91, 0.95] | >= 0.85 | True |
| smoke AUROC within sets | 0.870 [0.84, 0.90] | | |
| false smoke on fog windows (deploy) | 1.2% [0, 3] | <= 5 % | True |
| smoke recall (deploy) | 80.1% [75, 85] | >= 50 % | |
| false alarms on all no-smoke windows (deploy) | 9.5% [7, 12] | | |
| strict point: false alarms on all no-smoke windows | 5.0% [4, 7] | | |
| strict point: false smoke on fog windows | 0.9% [0, 2] | | |
| strict point: smoke recall | 74.3% [69, 79] | | |

| set | smoke AUROC | fog AUROC | false smoke on fog | false alarms, no-smoke | smoke recall | strict: false alarms, no-smoke | strict: smoke recall |
|---|---|---|---|---|---|---|---|
| figlib | 0.832 | n/a | n/a | 33.6% | 79.2% | 19.5% | 72.4% |
| hpwren_bg | 0.874 | 0.910 | 6.1% | 6.4% | 0.0% | 2.9% | 0.0% |
| hpwren_fog | n/a | 0.922 | 0.0% | 0.0% | n/a | 0.0% | n/a |
| hpwren_fog2 | n/a | 0.940 | 1.1% | 1.9% | n/a | 1.1% | n/a |
| hpwren_smoke | 0.937 | 0.952 | 0.0% | 2.9% | 80.2% | 0.0% | 77.7% |
| hpwren_smoke2 | 0.936 | n/a | n/a | 5.8% | 82.5% | 2.4% | 76.3% |

## Held-out test cameras

440 windows, 35 camera mounts; smoke labelled 436 (131 smoke), fog labelled 261 (56 fog), 54 fog windows without smoke.

| metric | value | goal | met |
|---|---|---|---|
| fog AUROC | 0.968 [0.91, 0.99] | >= 0.90 | True |
| fog AUROC within sets | 0.935 [0.85, 0.99] | | |
| smoke AUROC | 0.886 [0.83, 0.94] | >= 0.85 | True |
| smoke AUROC within sets | 0.796 [0.71, 0.88] | | |
| false smoke on fog windows (deploy) | 1.9% [0, 6] | <= 5 % | True |
| smoke recall (deploy) | 74.8% [64, 85] | >= 50 % | |
| false alarms on all no-smoke windows (deploy) | 15.1% [10, 21] | | |
| strict point: false alarms on all no-smoke windows | 9.2% [6, 14] | | |
| strict point: false smoke on fog windows | 1.9% [0, 6] | | |
| strict point: smoke recall | 67.9% [56, 80] | | |

| set | smoke AUROC | fog AUROC | false smoke on fog | false alarms, no-smoke | smoke recall | strict: false alarms, no-smoke | strict: smoke recall |
|---|---|---|---|---|---|---|---|
| figlib | 0.756 | n/a | n/a | 36.1% | 70.5% | 24.1% | 61.4% |
| hpwren_fog | n/a | 0.900 | 0.0% | 0.0% | n/a | 0.0% | n/a |
| hpwren_fog2 | n/a | 0.938 | 3.2% | 2.8% | n/a | 1.4% | n/a |
| hpwren_smoke | 0.911 | n/a | n/a | 12.9% | 80.0% | 6.5% | 80.0% |
| hpwren_smoke2 | 0.916 | 1.000 | 0.0% | 12.7% | 87.0% | 6.3% | 82.6% |
