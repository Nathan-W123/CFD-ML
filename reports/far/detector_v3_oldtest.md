# Detectors scored side by side

Sets: figlib, hpwren_fog, hpwren_fog2, hpwren_smoke, hpwren_smoke2. Each detector uses its own stored thresholds (deploy, strict, fog veto); intervals resample cameras (95 %).

| | v1 | v2 | v3 |
|---|---|---|---|
| smoke AUROC | 0.880 [0.82, 0.93] | 0.881 [0.82, 0.93] | 0.886 [0.83, 0.94] |
| smoke AUROC within sets | 0.761 [0.66, 0.85] | 0.771 [0.67, 0.86] | 0.796 [0.71, 0.88] |
| fog AUROC | 0.963 [0.91, 0.99] | 0.963 [0.91, 0.99] | 0.968 [0.91, 0.99] |
| deploy: smoke recall | 82.4% [73, 91] | 76.3% [66, 86] | 74.8% [64, 85] |
| deploy: alarms on smoke-free windows | 26.6% [19, 35] | 20.3% [14, 28] | 15.1% [10, 21] |
| deploy: alarms on fog windows | 3.7% [0, 10] | 3.7% [0, 10] | 1.9% [0, 6] |
| strict: smoke recall | 53.4% [42, 67] | 61.8% [51, 74] | 67.9% [56, 80] |
| strict: alarms on smoke-free windows | 5.9% [3, 11] | 7.9% [4, 12] | 9.2% [6, 14] |
| figlib: smoke AUROC | 0.713 | 0.726 | 0.756 |
| figlib: deploy: alarms on smoke-free | 65.1% | 49.4% | 36.1% |
| hpwren_fog: deploy: alarms on smoke-free | 2.4% | 2.4% | 0.0% |
| hpwren_fog: deploy: alarms on fog | 4.8% | 4.8% | 0.0% |
| hpwren_fog2: deploy: alarms on smoke-free | 1.4% | 1.4% | 2.8% |
| hpwren_fog2: deploy: alarms on fog | 3.2% | 3.2% | 3.2% |
| hpwren_smoke: smoke AUROC | 0.844 | 0.871 | 0.911 |
| hpwren_smoke: deploy: alarms on smoke-free | 29.0% | 22.6% | 12.9% |
| hpwren_smoke2: smoke AUROC | 0.928 | 0.919 | 0.916 |
| hpwren_smoke2: deploy: alarms on smoke-free | 20.3% | 15.2% | 12.7% |
| hpwren_smoke2: deploy: alarms on fog | 0.0% | 0.0% | 0.0% |

440 windows, 35 camera mounts; smoke labelled 436 (131 smoke), fog labelled 261 (56 fog).

* v1 at 80% recall on these windows (descriptive): alarms on smoke-free 24.3%; figlib 57.8%, hpwren_fog 2.4%, hpwren_fog2 1.4%, hpwren_smoke 29.0%, hpwren_smoke2 19.0%
* v2 at 80% recall on these windows (descriptive): alarms on smoke-free 27.2%; figlib 61.4%, hpwren_fog 2.4%, hpwren_fog2 1.4%, hpwren_smoke 32.3%, hpwren_smoke2 25.3%
* v3 at 80% recall on these windows (descriptive): alarms on smoke-free 27.5%; figlib 60.2%, hpwren_fog 0.0%, hpwren_fog2 7.0%, hpwren_smoke 25.8%, hpwren_smoke2 26.6%
* v1, smoke caught at a matched false-alarm rate (descriptive): 4.9% alarms: 51.1%; 9.8% alarms: 62.6%; 14.8% alarms: 72.5%
* v2, smoke caught at a matched false-alarm rate (descriptive): 4.9% alarms: 57.3%; 9.8% alarms: 64.1%; 14.8% alarms: 69.5%
* v3, smoke caught at a matched false-alarm rate (descriptive): 4.9% alarms: 63.4%; 9.8% alarms: 67.9%; 14.8% alarms: 74.8%

## Paired differences v2-v1 (same windows, same resampled cameras)

| | difference | 95 % |
|---|---|---|
| smoke_auroc_within_sets | +0.010 | [-0.025, +0.051] |
| smoke_auroc/figlib | +0.013 | [-0.032, +0.063] |
| smoke_auroc/hpwren_smoke | +0.027 | [-0.028, +0.083] |
| smoke_auroc/hpwren_smoke2 | -0.009 | [-0.038, +0.024] |
| false_alarm_no_smoke/deploy | -6.2 pts | [-10.5 pts, -3.0 pts] |
| false_alarm_no_smoke/deploy/figlib | -15.7 pts | [-26.5 pts, -7.3 pts] |
| false_alarm_no_smoke/deploy/hpwren_fog | +0.0 pts | [+0.0 pts, +0.0 pts] |
| false_alarm_no_smoke/deploy/hpwren_fog2 | +0.0 pts | [+0.0 pts, +0.0 pts] |
| false_alarm_no_smoke/deploy/hpwren_smoke | -6.5 pts | [-16.7 pts, +0.0 pts] |
| false_alarm_no_smoke/deploy/hpwren_smoke2 | -5.1 pts | [-17.1 pts, +2.4 pts] |
| smoke_recall/deploy | -6.1 pts | [-11.1 pts, -1.6 pts] |
| false_alarm_no_smoke/strict | +2.0 pts | [-2.2 pts, +5.3 pts] |
| false_alarm_no_smoke/strict/figlib | -1.2 pts | [-10.8 pts, +7.0 pts] |
| false_alarm_no_smoke/strict/hpwren_fog | +0.0 pts | [+0.0 pts, +0.0 pts] |
| false_alarm_no_smoke/strict/hpwren_fog2 | +1.4 pts | [+0.0 pts, +4.2 pts] |
| false_alarm_no_smoke/strict/hpwren_smoke | +6.5 pts | [+0.0 pts, +16.2 pts] |
| false_alarm_no_smoke/strict/hpwren_smoke2 | +5.1 pts | [+0.0 pts, +10.1 pts] |
| smoke_recall/strict | +8.4 pts | [+1.4 pts, +16.2 pts] |

## Paired differences v3-v1 (same windows, same resampled cameras)

| | difference | 95 % |
|---|---|---|
| smoke_auroc_within_sets | +0.035 | [-0.000, +0.074] |
| smoke_auroc/figlib | +0.043 | [+0.000, +0.090] |
| smoke_auroc/hpwren_smoke | +0.068 | [+0.002, +0.144] |
| smoke_auroc/hpwren_smoke2 | -0.012 | [-0.055, +0.020] |
| false_alarm_no_smoke/deploy | -11.5 pts | [-18.9 pts, -5.5 pts] |
| false_alarm_no_smoke/deploy/figlib | -28.9 pts | [-40.6 pts, -17.4 pts] |
| false_alarm_no_smoke/deploy/hpwren_fog | -2.4 pts | [-9.1 pts, +0.0 pts] |
| false_alarm_no_smoke/deploy/hpwren_fog2 | +1.4 pts | [+0.0 pts, +4.5 pts] |
| false_alarm_no_smoke/deploy/hpwren_smoke | -16.1 pts | [-43.8 pts, -2.9 pts] |
| false_alarm_no_smoke/deploy/hpwren_smoke2 | -7.6 pts | [-23.2 pts, +3.1 pts] |
| smoke_recall/deploy | -7.6 pts | [-15.6 pts, -0.8 pts] |
| false_alarm_no_smoke/strict | +3.3 pts | [-0.4 pts, +6.5 pts] |
| false_alarm_no_smoke/strict/figlib | +3.6 pts | [-6.5 pts, +12.3 pts] |
| false_alarm_no_smoke/strict/hpwren_fog | +0.0 pts | [+0.0 pts, +0.0 pts] |
| false_alarm_no_smoke/strict/hpwren_fog2 | +1.4 pts | [+0.0 pts, +4.2 pts] |
| false_alarm_no_smoke/strict/hpwren_smoke | +6.5 pts | [+0.0 pts, +16.2 pts] |
| false_alarm_no_smoke/strict/hpwren_smoke2 | +5.1 pts | [+0.0 pts, +11.1 pts] |
| smoke_recall/strict | +14.5 pts | [+6.0 pts, +24.5 pts] |
