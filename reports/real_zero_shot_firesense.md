# Zero-shot check on real footage: firesense

> Sanity check, not an accuracy estimate: the exported synthetic model and its synthetic thresholds, no fine-tuning or recalibration on real data (Stage B not run).

FIRESENSE smoke videos (CERTH / Bilkent, Zenodo record 836749, CC BY 4.0): 13 smoke and 9 non-smoke clips. Only clips of at least 56 s fit the 12 x 5 s window with 1 s lag frames: 4 smoke videos (a building fire, a handheld smoke-grenade clip, an indoor warehouse with detector outlines burned into the video, a courtyard CCTV bucket fire) and 1 non-smoke video (bushes and a fence, fixed camera). No clip contains fog, so fog is only tested for false alarms. Masks come from the built-in fallback detector (none are provided). Scenes are close-range smoke, not long-range wildfire camera views.

9 windows from 5 videos (5 cameras); masks: fallback_detector 9.

**balanced**: abstained (either label) on 6 / 9 windows (few_valid_flow_frames 3, smoke_near_threshold 3, fog_near_threshold 2).
* smoke: positives 7 (decided 2: 2 detected, 0 missed); negatives 2 (decided 1: 0 false alarms, 1 correct)
* fog: positives 0 (decided 0: 0 detected, 0 missed); negatives 9 (decided 4: 2 false alarms, 2 correct)

**low_false_alarm**: abstained (either label) on 7 / 9 windows (few_valid_flow_frames 3, fog_near_threshold 2, smoke_near_threshold 2).
* smoke: positives 7 (decided 3: 0 detected, 3 missed); negatives 2 (decided 1: 0 false alarms, 1 correct)
* fog: positives 0 (decided 0: 0 detected, 0 missed); negatives 9 (decided 4: 2 false alarms, 2 correct)

Window-level AUROC of the balanced-point probabilities (tiny n, windows of one video are correlated): smoke 1.00 (7 pos / 2 neg windows, 4 / 1 cameras); fog n/a (0 pos / 9 neg windows, 0 / 5 cameras)

| window | labels smoke/fog | valid mask / flow frames | stabilisation | P(smoke) | P(fog) | balanced smoke/fog | low-false-alarm smoke/fog | abstain reasons (balanced) |
|---|---|---|---|---|---|---|---|---|
| testneg04-w0 | 0/0 | 8 / 3 | applied | 0.41 | 0.37 | no/no | no/no | - |
| testneg04-w1 | 0/0 | 6 / 0 | applied | 0.36 | 0.33 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| testpos02-w0 | 1/0 | 12 / 9 | applied | 0.54 | 0.54 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| testpos04-w0 | 1/0 | 12 / 0 | unreliable_background | 0.49 | 0.62 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| testpos04-w1 | 1/0 | 12 / 0 | unreliable_background | 0.45 | 0.55 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| testpos05-w0 | 1/0 | 12 / 12 | applied | 0.47 | 0.44 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| testpos05-w1 | 1/0 | 12 / 12 | applied | 0.63 | 0.70 | yes/yes | abstain/yes | - |
| testpos07-w0 | 1/0 | 12 / 12 | applied | 0.52 | 0.70 | abstain/yes | no/yes | smoke_near_threshold |
| testpos07-w1 | 1/0 | 12 / 12 | applied | 0.72 | 0.26 | yes/no | abstain/no | - |
