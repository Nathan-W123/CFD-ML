# Real fog test: HPWREN fog set (far-field model, zero-shot)

> The exported far-field model and its synthetic thresholds, unchanged. Labels are visual (blind contact sheets, independent votes); METAR fog reports only proposed the windows. Imagery courtesy of HPWREN (https://www.hpwren.ucsd.edu/).

## Labels

* 357 windows labelled; fog {'1': 152, '0': 144, 'unknown': 61}; smoke {'0': 341, 'unknown': 16}; quality {'ok': 305, 'glare': 47, 'obstructed': 5}.
* Fog types (fog = 1): {'enveloping': 38, 'below': 100, 'bank': 4, 'patches': 1, 'low_cloud_on_terrain': 9}; extent: {'large': 45, 'medium': 27, 'small': 80}.
* Votes 1 and 2 agree on fog for 99% of the sheets both decided (276); Cohen's kappa 0.99.
* METAR candidate kind -> visual fog label: {'clear->0': 123, 'clear->1': 20, 'clear->unknown': 35, 'fog->0': 21, 'fog->1': 132, 'fog->unknown': 26}.

## all (262 windows: 131 fog, 131 clear; 88 cameras)

* **Fog AUROC** (P(fog), fog against clear): 0.352 [0.28, 0.43]; over the seeds 0.386 ± 0.030.
* P(smoke), fog against clear: AUROC 0.382 [0.31, 0.46] (0.5: fog raises the smoke score no more than a clear morning).
* Median P(fog): fog 0.475, clear 0.530; median P(smoke): fog 0.501, clear 0.531.
* Valid input (enough masks and flow): fog 38%, clear 56%.

| operating point | false smoke on fog (all / decided) | false smoke on clear | fog found (decided) | fog called on clear (decided) | fog windows: smoke yes / no / abstain | clear windows: fog yes / no / abstain |
|---|---|---|---|---|---|---|
| balanced | 15% / 66% | 18% | 92% | 100% | 19 / 10 / 102 | 56 / 0 / 75 |
| low_false_alarm | 0% / 0% | 0% | 92% | 100% | 0 / 22 / 109 | 56 / 0 / 75 |

Abstention reasons (balanced): fog {'few_valid_mask_frames': 20, 'few_valid_flow_frames': 81, 'smoke_near_threshold': 21, 'fog_near_threshold': 14}; clear {'smoke_near_threshold': 42, 'few_valid_flow_frames': 57, 'few_valid_mask_frames': 1, 'fog_near_threshold': 18}.

| fog type | n | valid input | median P(fog) | AUROC against clear | fog yes (balanced) | smoke yes (balanced) |
|---|---|---|---|---|---|---|
| bank | 4 | 50% | 0.380 | 0.210 | 1 | 1 |
| below | 81 | 43% | 0.513 | 0.454 | 25 | 14 |
| enveloping | 36 | 17% | 0.318 | 0.073 | 1 | 1 |
| low_cloud_on_terrain | 9 | 67% | 0.555 | 0.653 | 6 | 3 |
| patches | 1 | 100% | 0.368 | 0.046 | 0 | 0 |

## test_iid (42 windows: 21 fog, 21 clear; 16 cameras)

* **Fog AUROC** (P(fog), fog against clear): 0.433 [0.24, 0.62]; over the seeds 0.437 ± 0.017.
* P(smoke), fog against clear: AUROC 0.390 [0.21, 0.55] (0.5: fog raises the smoke score no more than a clear morning).
* Median P(fog): fog 0.489, clear 0.514; median P(smoke): fog 0.514, clear 0.525.
* Valid input (enough masks and flow): fog 48%, clear 43%.

| operating point | false smoke on fog (all / decided) | false smoke on clear | fog found (decided) | fog called on clear (decided) | fog windows: smoke yes / no / abstain | clear windows: fog yes / no / abstain |
|---|---|---|---|---|---|---|
| balanced | 24% / 83% | 14% | 90% | 100% | 5 / 1 / 15 | 8 / 0 / 13 |
| low_false_alarm | 0% / 0% | 0% | 90% | 100% | 0 / 3 / 18 | 8 / 0 / 13 |

Abstention reasons (balanced): fog {'few_valid_flow_frames': 11, 'few_valid_mask_frames': 3, 'smoke_near_threshold': 4}; clear {'few_valid_flow_frames': 12, 'smoke_near_threshold': 5, 'fog_near_threshold': 1}.

| fog type | n | valid input | median P(fog) | AUROC against clear | fog yes (balanced) | smoke yes (balanced) |
|---|---|---|---|---|---|---|
| below | 7 | 57% | 0.536 | 0.585 | 4 | 3 |
| enveloping | 8 | 12% | 0.331 | 0.030 | 0 | 0 |
| low_cloud_on_terrain | 6 | 83% | 0.576 | 0.794 | 5 | 2 |

## Smoke against fog across datasets (confounded, indicative only)

FIgLib smoke windows (fires 2016-2024, any hour) against HPWREN windows (2026 mornings): different years, cameras' state and time of day.

* all: P(smoke) AUROC FIgLib smoke vs HPWREN fog 0.553 (443 vs 131); vs HPWREN clear 0.457 (131).
* test_iid: P(smoke) AUROC FIgLib smoke vs HPWREN fog 0.557 (88 vs 21); vs HPWREN clear 0.464 (21).
