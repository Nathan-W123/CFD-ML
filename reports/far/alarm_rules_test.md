# Alarm rules over per-minute window scores

Rules: R0_deploy: single T=0.234; R0_strict: single T=0.354; chosen: single T=0.259 with change >= -2.46. Intervals resample camera mounts (95 %).

| | R0_deploy | R0_strict | chosen |
|---|---|---|---|
| fires_within_5 | 20.3% [12.8%, 27.4%] | 27.8% [18.9%, 34.8%] | 22.8% [14.5%, 29.5%] |
| fires_within_10 | 24.1% [16.4%, 30.6%] | 34.2% [26.1%, 40.6%] | 26.6% [19.7%, 33.3%] |
| fires_within_15 | 44.3% [35.2%, 55.8%] | 48.1% [40.5%, 55.6%] | 43.0% [33.3%, 52.1%] |
| fires_within_20 | 74.7% [68.5%, 81.8%] | 68.4% [61.8%, 76.6%] | 74.7% [68.8%, 80.8%] |
| fires_within_30 | 92.4% [87.7%, 96.3%] | 88.6% [84.9%, 92.9%] | 91.1% [85.4%, 96.0%] |
| fires_firing_within_15 | 89.9% [83.0%, 95.8%] | 81.0% [72.5%, 89.6%] | 86.1% [79.6%, 92.5%] |
| fires_firing_within_30 | 96.2% [90.4%, 100.0%] | 89.9% [85.3%, 95.1%] | 94.9% [89.7%, 98.9%] |
| fires_detected | 98.7% [95.0%, 100.0%] | 93.7% [89.7%, 97.9%] | 96.2% [92.9%, 100.0%] |
| median_delay_detected | 15.98 [15.00, 16.97] | 15.00 [15.00, 16.02] | 15.97 [15.00, 16.00] |
| fire_pre_events_per_hour | 2.83 [2.36, 3.35] | 2.07 [1.50, 2.75] | 2.45 [1.96, 3.01] |
| ordinary_events_per_hour | 0.81 [0.48, 1.18] | 0.45 [0.20, 0.75] | 0.48 [0.23, 0.76] |
| events_per_hour | 1.43 [1.01, 1.85] | 0.94 [0.61, 1.33] | 1.08 [0.70, 1.52] |
| fire_pre_window_share | 28.5% [19.8%, 38.9%] | 16.9% [10.0%, 25.2%] | 19.1% [13.4%, 24.9%] |
| ordinary_window_share | 8.1% [2.6%, 16.1%] | 4.4% [1.1%, 9.3%] | 4.9% [1.3%, 10.5%] |

79 fire sequences; smoke-free footage: 18.4 camera-hours before the first visible plume, 42.0 camera-hours on ordinary days.

Paired, chosen - R0_deploy (same resampled mounts):

| | difference | 95 % |
|---|---|---|
| fires_within_5 | 2.5% | [0.0%, 5.1%] |
| fires_within_10 | 2.5% | [0.0%, 6.7%] |
| fires_within_15 | -1.3% | [-9.8%, 3.8%] |
| fires_within_20 | 0.0% | [-6.9%, 6.4%] |
| fires_within_30 | -1.3% | [-3.9%, 0.0%] |
| fires_firing_within_15 | -3.8% | [-6.0%, 0.0%] |
| fires_firing_within_30 | -1.3% | [-3.3%, 0.0%] |
| fires_detected | -2.5% | [-4.7%, 0.0%] |
| median_delay_detected | -0.01 | [-0.98, 0.98] |
| fire_pre_events_per_hour | -0.38 | [-0.71, -0.12] |
| ordinary_events_per_hour | -0.33 | [-0.50, -0.18] |
| events_per_hour | -0.35 | [-0.49, -0.22] |
| fire_pre_window_share | -9.3% | [-15.2%, -5.4%] |
| ordinary_window_share | -3.2% | [-5.9%, -1.0%] |

With a 10-minute hold-off instead of 30 (sensitivity):

| | R0_deploy | R0_strict | chosen |
|---|---|---|---|
| fires_within_5 | 48.1% | 53.2% | 44.3% |
| fires_within_10 | 81.0% | 77.2% | 77.2% |
| fires_within_15 | 88.6% | 81.0% | 84.8% |
| fires_within_20 | 96.2% | 84.8% | 92.4% |
| fires_within_30 | 96.2% | 89.9% | 94.9% |
| fires_firing_within_15 | 89.9% | 81.0% | 86.1% |
| fires_firing_within_30 | 96.2% | 89.9% | 94.9% |
| fires_detected | 98.7% | 94.9% | 97.5% |
| median_delay_detected | 5.02 | 5.00 | 6.00 |
| fire_pre_events_per_hour | 4.14 | 2.67 | 3.43 |
| ordinary_events_per_hour | 1.19 | 0.71 | 0.83 |
| events_per_hour | 2.09 | 1.31 | 1.62 |
| fire_pre_window_share | 28.5% | 16.9% | 19.1% |
| ordinary_window_share | 8.1% | 4.4% | 4.9% |

Targets: fire_pre_events not met, ordinary_events met, fires_within_30 met, fires_within_15 met, all_met not met.
