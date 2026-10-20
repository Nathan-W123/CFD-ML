# Detectors scored side by side

Sets: figlib_test2, hpwren_bg_test. Each detector uses its own stored thresholds (deploy, strict, fog veto); intervals resample cameras (95 %).

| | v1 | v2 | v3 |
|---|---|---|---|
| smoke AUROC | 0.839 [0.79, 0.90] | 0.879 [0.84, 0.93] | 0.880 [0.85, 0.92] |
| smoke AUROC within sets | 0.739 [0.69, 0.81] | 0.799 [0.75, 0.86] | 0.813 [0.78, 0.86] |
| fog AUROC | 0.867 [0.77, 0.94] | 0.867 [0.77, 0.94] | 0.864 [0.76, 0.94] |
| deploy: smoke recall | 76.4% [66, 87] | 76.4% [69, 86] | 72.6% [64, 82] |
| deploy: alarms on smoke-free windows | 29.6% [20, 39] | 24.3% [16, 31] | 14.0% [10, 18] |
| deploy: alarms on fog windows | 4.2% [0, 17] | 4.2% [0, 17] | 0.0% [0, 0] |
| strict: smoke recall | 53.5% [44, 69] | 59.9% [50, 74] | 66.9% [60, 79] |
| strict: alarms on smoke-free windows | 5.9% [3, 9] | 7.0% [4, 10] | 8.6% [5, 13] |
| figlib_test2: smoke AUROC | 0.739 | 0.799 | 0.813 |
| figlib_test2: deploy: alarms on smoke-free | 52.3% | 41.7% | 26.5% |
| hpwren_bg_test: deploy: alarms on smoke-free | 14.1% | 12.3% | 5.5% |
| hpwren_bg_test: deploy: alarms on fog | 4.2% | 4.2% | 0.0% |

528 windows, 29 camera mounts; smoke labelled 528 (157 smoke), fog labelled 202 (24 fog).

* v1 at 80% recall on these windows (descriptive): alarms on smoke-free 33.4%; figlib_test2 57.0%, hpwren_bg_test 17.3%
* v2 at 80% recall on these windows (descriptive): alarms on smoke-free 26.1%; figlib_test2 45.7%, hpwren_bg_test 12.7%
* v3 at 80% recall on these windows (descriptive): alarms on smoke-free 26.1%; figlib_test2 42.4%, hpwren_bg_test 15.0%
* v1, smoke caught at a matched false-alarm rate (descriptive): 4.9% alarms: 52.9%; 10.0% alarms: 56.1%; 14.8% alarms: 61.1%
* v2, smoke caught at a matched false-alarm rate (descriptive): 4.9% alarms: 57.3%; 10.0% alarms: 63.7%; 14.8% alarms: 71.3%
* v3, smoke caught at a matched false-alarm rate (descriptive): 4.9% alarms: 62.4%; 10.0% alarms: 69.4%; 14.8% alarms: 74.5%

## Paired differences v2-v1 (same windows, same resampled cameras)

| | difference | 95 % |
|---|---|---|
| smoke_auroc_within_sets | +0.061 | [+0.043, +0.075] |
| smoke_auroc/figlib_test2 | +0.061 | [+0.043, +0.075] |
| false_alarm_no_smoke/deploy | -5.4 pts | [-10.3 pts, -0.7 pts] |
| false_alarm_no_smoke/deploy/figlib_test2 | -10.6 pts | [-17.9 pts, -3.2 pts] |
| false_alarm_no_smoke/deploy/hpwren_bg_test | -1.8 pts | [-7.3 pts, +2.6 pts] |
| smoke_recall/deploy | +0.0 pts | [-4.6 pts, +4.4 pts] |
| false_alarm_no_smoke/strict | +1.1 pts | [-1.3 pts, +3.2 pts] |
| false_alarm_no_smoke/strict/figlib_test2 | +3.3 pts | [+0.0 pts, +7.0 pts] |
| false_alarm_no_smoke/strict/hpwren_bg_test | -0.5 pts | [-2.9 pts, +1.7 pts] |
| smoke_recall/strict | +6.4 pts | [+1.9 pts, +10.5 pts] |

## Paired differences v3-v1 (same windows, same resampled cameras)

| | difference | 95 % |
|---|---|---|
| smoke_auroc_within_sets | +0.074 | [+0.041, +0.107] |
| smoke_auroc/figlib_test2 | +0.074 | [+0.041, +0.107] |
| false_alarm_no_smoke/deploy | -15.6 pts | [-21.6 pts, -8.8 pts] |
| false_alarm_no_smoke/deploy/figlib_test2 | -25.8 pts | [-35.3 pts, -17.3 pts] |
| false_alarm_no_smoke/deploy/hpwren_bg_test | -8.6 pts | [-14.9 pts, -2.6 pts] |
| smoke_recall/deploy | -3.8 pts | [-9.8 pts, +2.4 pts] |
| false_alarm_no_smoke/strict | +2.7 pts | [-0.8 pts, +5.9 pts] |
| false_alarm_no_smoke/strict/figlib_test2 | +6.6 pts | [+0.0 pts, +12.1 pts] |
| false_alarm_no_smoke/strict/hpwren_bg_test | +0.0 pts | [-2.8 pts, +2.7 pts] |
| smoke_recall/strict | +13.4 pts | [+7.4 pts, +20.7 pts] |
