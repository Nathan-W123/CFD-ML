# Zero-shot check on real footage: firesense

> Sanity check, not an accuracy estimate: the exported synthetic model and its synthetic thresholds, no fine-tuning or recalibration on real data (Stage B not run).

FIRESENSE smoke videos (CERTH / Bilkent, Zenodo record 836749, CC BY 4.0). Clips of at least 56 s fit the 12 x 5 s window with 1 s lag frames; testpos04 (edited footage) and testpos05 (wrong container clock) are excluded, testpos03 crashes OpenCV. Used: a building fire (testpos02), a courtyard CCTV bucket fire (testpos07, two windows) and one smoke-free fixed view of bushes and a fence (testneg04, two windows). No fog in any clip; masks from the built-in fallback detector; close-range smoke, not long-range wildfire views.

5 windows from 3 videos (3 cameras); masks: fallback_detector 5.

**balanced**: abstained (either label) on 3 / 5 windows (smoke_near_threshold 1, few_valid_flow_frames 1, fog_near_threshold 1).
* smoke: positives 3 (decided 3: 3 detected, 0 missed); negatives 2 (decided 0: 0 false alarms, 0 correct)
* fog: positives 0 (decided 0: 0 detected, 0 missed); negatives 5 (decided 3: 1 false alarms, 2 correct)

**low_false_alarm**: abstained (either label) on 3 / 5 windows (few_valid_flow_frames 1, smoke_near_threshold 2, fog_near_threshold 1).
* smoke: positives 3 (decided 1: 1 detected, 0 missed); negatives 2 (decided 1: 0 false alarms, 1 correct)
* fog: positives 0 (decided 0: 0 detected, 0 missed); negatives 5 (decided 3: 1 false alarms, 2 correct)

Window-level AUROC of the balanced-point probabilities (tiny n, windows of one video are correlated): smoke 1.00 (3 pos / 2 neg windows, 2 / 1 cameras); fog n/a (0 pos / 5 neg windows, 0 / 3 cameras)

| window | labels smoke/fog | valid mask / flow frames | stabilisation | P(smoke) | P(fog) | balanced smoke/fog | low-false-alarm smoke/fog | abstain reasons (balanced) |
|---|---|---|---|---|---|---|---|---|
| testneg04-w0 | 0/0 | 8 / 3 | applied | 0.42 | 0.33 | abstain/no | no/no | smoke_near_threshold |
| testneg04-w1 | 0/0 | 6 / 0 | applied | 0.32 | 0.30 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| testpos02-w0 | 1/0 | 12 / 9 | applied | 0.58 | 0.56 | yes/abstain | abstain/abstain | fog_near_threshold |
| testpos07-w0 | 1/0 | 12 / 12 | applied | 0.63 | 0.67 | yes/yes | abstain/yes | - |
| testpos07-w1 | 1/0 | 12 / 12 | applied | 0.69 | 0.24 | yes/no | yes/no | - |
