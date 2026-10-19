# Real-footage detector (far field): smoke and fog

> Linear heads on frozen CLIP appearance features (and, if selected, the motion features; for smoke also a tile head on 4 x 6 fine tiles, stacked with the window head), trained on real labelled windows; model selection and thresholds on the dev cameras only; intervals resample cameras. Imagery courtesy of HPWREN (https://www.hpwren.ucsd.edu/).

Goals: fog AUROC >= 0.90; smoke AUROC >= 0.85; at the deploy point at most 5 % false smoke on fog windows while detecting at least 50 % of the smoke windows.

| head | selected feature groups | C | CV AUROC pooled | CV AUROC within sets | dev windows |
|---|---|---|---|---|---|
| smoke | full, tiles, sims, fine | 0.003 | 0.888 | 0.804 | 2051 |
| smoke: tile head | fine tiles, best tile | 0.0003 | 0.867 | 0.769 | 2051 |
| smoke: stacked (deployed) | head + tile head | | 0.896 | 0.809 | 2051 |
| fog | full, sims | 0.01 | 0.977 | 0.945 | 1303 |

Thresholds: smoke deploy 0.278, smoke strict 0.568, smoke balanced 0.298, fog balanced 0.243.

## Dev cameras, out-of-fold (cross-validated)

2064 windows, 143 camera mounts; smoke labelled 2051 (653 smoke), fog labelled 1303 (300 fog), 291 fog windows without smoke.

| metric | value | goal | met |
|---|---|---|---|
| fog AUROC | 0.977 [0.96, 0.99] | >= 0.90 | True |
| smoke AUROC | 0.896 [0.87, 0.92] | >= 0.85 | True |
| smoke AUROC within sets | 0.809 [0.77, 0.84] | | |
| false smoke on fog windows (deploy) | 2.4% [1, 4] | <= 5 % | True |
| smoke recall (deploy) | 79.9% [76, 85] | >= 50 % | |
| false alarms on all no-smoke windows (deploy) | 20.5% [16, 25] | | |
| strict point: false alarms on all no-smoke windows | 4.9% [3, 7] | | |
| strict point: false smoke on fog windows | 0.3% [0, 1] | | |
| strict point: smoke recall | 59.9% [54, 66] | | |

| set | smoke AUROC | fog AUROC | false smoke on fog | false alarms, no-smoke | smoke recall | strict: false alarms, no-smoke | strict: smoke recall |
|---|---|---|---|---|---|---|---|
| figlib | 0.760 | n/a | n/a | 54.8% | 80.3% | 16.7% | 59.7% |
| hpwren_fog | n/a | 0.935 | 2.8% | 1.9% | n/a | 0.5% | n/a |
| hpwren_fog2 | n/a | 0.949 | 2.2% | 3.0% | n/a | 0.5% | n/a |
| hpwren_smoke | 0.901 | 0.978 | 0.0% | 12.8% | 78.5% | 1.2% | 58.7% |
| hpwren_smoke2 | 0.892 | n/a | n/a | 18.9% | 80.2% | 1.7% | 61.0% |

## Held-out test cameras

440 windows, 35 camera mounts; smoke labelled 436 (131 smoke), fog labelled 261 (56 fog), 54 fog windows without smoke.

| metric | value | goal | met |
|---|---|---|---|
| fog AUROC | 0.963 [0.91, 0.99] | >= 0.90 | True |
| smoke AUROC | 0.880 [0.82, 0.93] | >= 0.85 | True |
| smoke AUROC within sets | 0.761 [0.66, 0.85] | | |
| false smoke on fog windows (deploy) | 3.7% [0, 10] | <= 5 % | True |
| smoke recall (deploy) | 82.4% [73, 91] | >= 50 % | |
| false alarms on all no-smoke windows (deploy) | 26.6% [19, 35] | | |
| strict point: false alarms on all no-smoke windows | 5.9% [3, 11] | | |
| strict point: false smoke on fog windows | 0.0% [0, 0] | | |
| strict point: smoke recall | 53.4% [42, 67] | | |

| set | smoke AUROC | fog AUROC | false smoke on fog | false alarms, no-smoke | smoke recall | strict: false alarms, no-smoke | strict: smoke recall |
|---|---|---|---|---|---|---|---|
| figlib | 0.713 | n/a | n/a | 65.1% | 83.0% | 20.5% | 53.4% |
| hpwren_fog | n/a | 0.878 | 4.8% | 2.4% | n/a | 0.0% | n/a |
| hpwren_fog2 | n/a | 0.941 | 3.2% | 1.4% | n/a | 0.0% | n/a |
| hpwren_smoke | 0.844 | n/a | n/a | 29.0% | 75.0% | 0.0% | 50.0% |
| hpwren_smoke2 | 0.928 | 1.000 | 0.0% | 20.3% | 87.0% | 1.3% | 56.5% |
