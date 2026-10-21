# Detectors scored side by side

Sets: figlib_test2, hpwren_bg_test. Each detector uses its own stored thresholds (deploy, strict, fog veto); intervals resample cameras (95 %).

| | v3 |
|---|---|
| smoke AUROC | 0.878 [0.84, 0.92] |
| smoke AUROC within sets | 0.811 [0.78, 0.86] |
| fog AUROC | 0.864 [0.76, 0.94] |
| deploy: smoke recall | 73.9% [66, 83] |
| deploy: alarms on smoke-free windows | 16.2% [11, 22] |
| deploy: alarms on fog windows | 0.0% [0, 0] |
| strict: smoke recall | 65.6% [57, 76] |
| strict: alarms on smoke-free windows | 8.1% [4, 12] |
| figlib_test2: smoke AUROC | 0.811 |
| figlib_test2: deploy: alarms on smoke-free | 27.8% |
| hpwren_bg_test: deploy: alarms on smoke-free | 8.2% |
| hpwren_bg_test: deploy: alarms on fog | 0.0% |

528 windows, 29 camera mounts; smoke labelled 528 (157 smoke), fog labelled 202 (24 fog).

* v3 at 80% recall on these windows (descriptive): alarms on smoke-free 24.8%; figlib_test2 41.1%, hpwren_bg_test 13.6%
* v3, smoke caught at a matched false-alarm rate (descriptive): 4.9% alarms: 58.6%; 10.0% alarms: 68.2%
