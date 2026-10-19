# Real fog test: HPWREN fog set (far-field model, zero-shot)

> The exported far-field model and its synthetic thresholds, unchanged. Labels are visual (blind contact sheets, independent votes); METAR fog reports only proposed the windows. Imagery courtesy of HPWREN (https://www.hpwren.ucsd.edu/).

## Labels

* 357 windows labelled; fog {'1': 152, '0': 144, 'unknown': 61}; smoke {'0': 341, 'unknown': 16}; quality {'ok': 305, 'glare': 47, 'obstructed': 5}.
* Fog types (fog = 1): {'enveloping': 38, 'below': 100, 'bank': 4, 'patches': 1, 'low_cloud_on_terrain': 9}; extent: {'large': 45, 'medium': 27, 'small': 80}.
* Votes 1 and 2 agree on fog for 99% of the sheets both decided (276); Cohen's kappa 0.99.
* METAR candidate kind -> visual fog label: {'clear->0': 123, 'clear->1': 20, 'clear->unknown': 35, 'fog->0': 21, 'fog->1': 132, 'fog->unknown': 26}.

## all (253 windows: 127 fog, 126 clear; 88 cameras)

* **Fog AUROC** (P(fog), fog against clear): 0.267 [0.20, 0.34]; over the seeds 0.271 ± 0.041.
* P(smoke), fog against clear: AUROC 0.228 [0.17, 0.30] (0.5: fog raises the smoke score no more than a clear morning).
* Median P(fog): fog 0.637, clear 0.651; median P(smoke): fog 0.621, clear 0.632.
* Valid input (enough masks and flow): fog 63%, clear 66%.

| operating point | false smoke on fog (all / decided) | false smoke on clear | fog found (decided) | fog called on clear (decided) | fog windows: smoke yes / no / abstain | clear windows: fog yes / no / abstain |
|---|---|---|---|---|---|---|
| balanced | 61% / 99% | 66% | 100% | 100% | 78 / 1 / 48 | 83 / 0 / 43 |
| low_false_alarm | 0% / 0% | 1% | 100% | 100% | 0 / 2 / 125 | 83 / 0 / 43 |

Abstention reasons (balanced): fog {'few_valid_flow_frames': 47, 'fog_near_threshold': 1, 'smoke_near_threshold': 1}; clear {'few_valid_flow_frames': 43}.

| fog type | n | valid input | median P(fog) | AUROC against clear | fog yes (balanced) | smoke yes (balanced) |
|---|---|---|---|---|---|---|
| bank | 4 | 100% | 0.650 | 0.498 | 4 | 4 |
| below | 79 | 84% | 0.645 | 0.350 | 66 | 65 |
| enveloping | 36 | 6% | 0.552 | 0.020 | 2 | 2 |
| low_cloud_on_terrain | 7 | 100% | 0.652 | 0.510 | 7 | 7 |
| patches | 1 | 100% | 0.357 | 0.000 | 0 | 0 |

## test_iid (37 windows: 18 fog, 19 clear; 16 cameras)

* **Fog AUROC** (P(fog), fog against clear): 0.170 [0.03, 0.30]; over the seeds 0.146 ± 0.029.
* P(smoke), fog against clear: AUROC 0.082 [0.00, 0.25] (0.5: fog raises the smoke score no more than a clear morning).
* Median P(fog): fog 0.614, clear 0.653; median P(smoke): fog 0.592, clear 0.636.
* Valid input (enough masks and flow): fog 56%, clear 63%.

| operating point | false smoke on fog (all / decided) | false smoke on clear | fog found (decided) | fog called on clear (decided) | fog windows: smoke yes / no / abstain | clear windows: fog yes / no / abstain |
|---|---|---|---|---|---|---|
| balanced | 56% / 100% | 63% | 100% | 100% | 10 / 0 / 8 | 12 / 0 / 7 |
| low_false_alarm | 0% / n/a | 0% | 100% | 100% | 0 / 0 / 18 | 12 / 0 / 7 |

Abstention reasons (balanced): fog {'few_valid_flow_frames': 8}; clear {'few_valid_flow_frames': 7}.

| fog type | n | valid input | median P(fog) | AUROC against clear | fog yes (balanced) | smoke yes (balanced) |
|---|---|---|---|---|---|---|
| below | 6 | 100% | 0.630 | 0.184 | 6 | 6 |
| enveloping | 8 | 0% | 0.552 | 0.000 | 0 | 0 |
| low_cloud_on_terrain | 4 | 100% | 0.649 | 0.487 | 4 | 4 |

## Smoke against fog across datasets (confounded, indicative only)

FIgLib smoke windows (fires 2016-2024, any hour) against HPWREN windows (2026 mornings): different years, cameras' state and time of day.

* all: P(smoke) AUROC FIgLib smoke vs HPWREN fog 0.256 (443 vs 127); vs HPWREN clear 0.060 (126).
* test_iid: P(smoke) AUROC FIgLib smoke vs HPWREN fog 0.359 (88 vs 18); vs HPWREN clear 0.056 (19).
