# Zero-shot check on real footage: figlib (all splits)

> Sanity check, not an accuracy estimate: the exported synthetic model and its synthetic thresholds, no fine-tuning or recalibration on real data (Stage B is reported separately).

HPWREN FIgLib wildfire camera sequences (imagery courtesy of HPWREN, https://www.hpwren.ucsd.edu/), one frame per minute: 12-frame windows entirely before the first visible plume (smoke 0) and starting at least 60 s after it (smoke 1); fog is not annotated (unknown). Splits are camera-disjoint by camera mount; the far-field render ranges were set from image statistics of the train split only (no labels). Masks from the built-in fallback detector; the plumes are early, small and often distant.

880 windows from 229 videos (137 cameras); masks: fallback_detector 880.

**balanced**: abstained (either label) on 690 / 880 windows (fog_near_threshold 137, smoke_near_threshold 255, few_valid_flow_frames 382, few_valid_mask_frames 203).
* smoke: positives 443 (decided 136: 106 detected, 30 missed); negatives 437 (decided 107: 74 false alarms, 33 correct)
* fog: positives 0 (decided 0: 0 detected, 0 missed); negatives 0 (decided 0: 0 false alarms, 0 correct); label unknown on 880

**low_false_alarm**: abstained (either label) on 755 / 880 windows (smoke_near_threshold 252, fog_near_threshold 137, few_valid_flow_frames 382, few_valid_mask_frames 203).
* smoke: positives 443 (decided 129: 3 detected, 126 missed); negatives 437 (decided 117: 0 false alarms, 117 correct)
* fog: positives 0 (decided 0: 0 detected, 0 missed); negatives 0 (decided 0: 0 false alarms, 0 correct); label unknown on 880

Window-level AUROC of the balanced-point probabilities (tiny n, windows of one video are correlated): smoke 0.58 (443 pos / 437 neg windows, 135 / 132 cameras); fog n/a (0 pos / 0 neg windows, 0 / 0 cameras, 880 unknown)

| window | labels smoke/fog | valid mask / flow frames | stabilisation | P(smoke) | P(fog) | balanced smoke/fog | low-false-alarm smoke/fog | abstain reasons (balanced) |
|---|---|---|---|---|---|---|---|---|
| 20220302-Jimfire-0921-stgo-s-mobo-c-pre0-w0 | 0/unknown | 11 / 4 | applied | 0.58 | 0.49 | yes/abstain | abstain/abstain | fog_near_threshold |
| 20220302-Jimfire-0921-stgo-s-mobo-c-pre1-w0 | 0/unknown | 12 / 2 | applied | 0.55 | 0.45 | abstain/abstain | abstain/abstain | smoke_near_threshold, fog_near_threshold |
| 20220302-Jimfire-0921-stgo-s-mobo-c-post0-w0 | 1/unknown | 10 / 2 | applied | 0.43 | 0.37 | no/abstain | no/abstain | fog_near_threshold |
| 20220302-Jimfire-0921-stgo-s-mobo-c-post1-w0 | 1/unknown | 10 / 2 | applied | 0.47 | 0.41 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20180614_FIRE_hp-s-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.60 | 0.65 | yes/yes | abstain/yes | - |
| 20180614_FIRE_hp-s-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.62 | 0.62 | yes/yes | abstain/yes | - |
| 20180614_FIRE_hp-s-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.51 | 0.55 | abstain/yes | no/yes | smoke_near_threshold |
| 20180614_FIRE_hp-s-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.58 | 0.60 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20200806_SpringsFire_sm-e-mobo-m-post0-w0 | 1/unknown | 11 / 0 | applied | 0.52 | 0.44 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20200806_SpringsFire_sm-e-mobo-m-post1-w0 | 1/unknown | 10 / 0 | applied | 0.62 | 0.55 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20241113_Fire_dwpgm-w-mobo-c-pre0-w0 | 0/unknown | 8 / 0 | applied | 0.45 | 0.39 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20241113_Fire_dwpgm-w-mobo-c-pre1-w0 | 0/unknown | 10 / 0 | applied | 0.53 | 0.48 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20241113_Fire_dwpgm-w-mobo-c-post0-w0 | 1/unknown | 12 / 3 | applied | 0.65 | 0.46 | yes/abstain | abstain/abstain | fog_near_threshold |
| 20241113_Fire_dwpgm-w-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.60 | 0.65 | yes/yes | abstain/yes | - |
| 20191006_FIRE_lp-e-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180727_FIRE_wc-n-mobo-c-pre0-w0 | 0/unknown | 9 / 2 | applied | 0.61 | 0.59 | yes/yes | abstain/yes | - |
| 20180727_FIRE_wc-n-mobo-c-pre1-w0 | 0/unknown | 6 / 0 | applied | 0.50 | 0.45 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180727_FIRE_wc-n-mobo-c-post0-w0 | 1/unknown | 6 / 0 | applied | 0.48 | 0.40 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180727_FIRE_wc-n-mobo-c-post1-w0 | 1/unknown | 10 / 1 | applied | 0.63 | 0.59 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170519_FIRE_rm-w-mobo-c-pre0-w0 | 0/unknown | 12 / 7 | applied | 0.54 | 0.58 | abstain/yes | no/yes | smoke_near_threshold |
| 20170519_FIRE_rm-w-mobo-c-pre1-w0 | 0/unknown | 10 / 3 | applied | 0.53 | 0.50 | abstain/yes | no/yes | smoke_near_threshold |
| 20170519_FIRE_rm-w-mobo-c-post0-w0 | 1/unknown | 12 / 9 | applied | 0.57 | 0.60 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20170519_FIRE_rm-w-mobo-c-post1-w0 | 1/unknown | 9 / 1 | applied | 0.60 | 0.59 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20210115_Bonitafire_tp-w-mobo-c-pre0-w0 | 0/unknown | 3 / 2 | applied | 0.53 | 0.45 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20210115_Bonitafire_tp-w-mobo-c-pre1-w0 | 0/unknown | 4 / 2 | applied | 0.57 | 0.49 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20210115_Bonitafire_tp-w-mobo-c-post0-w0 | 1/unknown | 2 / 1 | applied | 0.48 | 0.45 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200306_FIRE_mlo-n-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.59 | 0.66 | yes/yes | abstain/yes | - |
| 20200306_FIRE_mlo-n-mobo-c-pre1-w0 | 0/unknown | 12 / 10 | applied | 0.55 | 0.65 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20200306_FIRE_mlo-n-mobo-c-post0-w0 | 1/unknown | 12 / 10 | applied | 0.61 | 0.65 | yes/yes | abstain/yes | - |
| 20200306_FIRE_mlo-n-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.61 | 0.65 | yes/yes | abstain/yes | - |
| 20240825_TenajaFire_buff-n-mobo-c-pre0-w0 | 0/unknown | 2 / 1 | applied | 0.42 | 0.38 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20240825_TenajaFire_buff-n-mobo-c-pre1-w0 | 0/unknown | 3 / 2 | applied | 0.40 | 0.33 | no/no | no/no | - |
| 20240825_TenajaFire_buff-n-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20240825_TenajaFire_buff-n-mobo-c-post1-w0 | 1/unknown | 10 / 8 | applied | 0.67 | 0.57 | yes/yes | abstain/yes | - |
| 20230806.161759-PassFire-ws-n-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20230806.161759-PassFire-ws-n-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20230806.161759-PassFire-ws-n-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.64 | 0.63 | yes/yes | abstain/yes | - |
| 20230806.161759-PassFire-ws-n-mobo-c-post1-w0 | 1/unknown | 12 / 6 | applied | 0.65 | 0.54 | yes/yes | abstain/yes | - |
| 20170807_FIRE_bh-n-mobo-c-pre0-w0 | 0/unknown | 3 / 2 | applied | 0.45 | 0.40 | no/abstain | no/abstain | fog_near_threshold |
| 20170807_FIRE_bh-n-mobo-c-pre1-w0 | 0/unknown | 7 / 4 | applied | 0.45 | 0.42 | no/abstain | no/abstain | fog_near_threshold |
| 20170807_FIRE_bh-n-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170807_FIRE_bh-n-mobo-c-post1-w0 | 1/unknown | 3 / 2 | applied | 0.56 | 0.49 | abstain/abstain | abstain/abstain | smoke_near_threshold, fog_near_threshold |
| 20170711_FIRE_bm-s-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170711_FIRE_bm-s-mobo-c-pre1-w0 | 0/unknown | 4 / 3 | applied | 0.61 | 0.57 | yes/yes | abstain/yes | - |
| 20170711_FIRE_bm-s-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180727_FIRE_mg-w-mobo-c-pre0-w0 | 0/unknown | 11 / 5 | applied | 0.48 | 0.50 | abstain/yes | no/yes | smoke_near_threshold |
| 20180727_FIRE_mg-w-mobo-c-pre1-w0 | 0/unknown | 10 / 6 | applied | 0.53 | 0.59 | abstain/yes | no/yes | smoke_near_threshold |
| 20180727_FIRE_mg-w-mobo-c-post0-w0 | 1/unknown | 12 / 5 | applied | 0.51 | 0.56 | abstain/yes | no/yes | smoke_near_threshold |
| 20180727_FIRE_mg-w-mobo-c-post1-w0 | 1/unknown | 12 / 3 | applied | 0.53 | 0.56 | abstain/yes | no/yes | smoke_near_threshold |
| 20241209_FranklinFire_69bravo-w-mobo-c-pre0-w0 | 0/unknown | 12 / 2 | applied | 0.45 | 0.38 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20241209_FranklinFire_69bravo-w-mobo-c-pre1-w0 | 0/unknown | 10 / 0 | applied | 0.50 | 0.43 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20241209_FranklinFire_69bravo-w-mobo-c-post0-w0 | 1/unknown | 10 / 4 | applied | 0.48 | 0.46 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20241209_FranklinFire_69bravo-w-mobo-c-post1-w0 | 1/unknown | 12 / 7 | applied | 0.53 | 0.50 | abstain/yes | no/yes | smoke_near_threshold |
| 20180725_Cranston_sp-e-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180725_Cranston_sp-e-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180725_Cranston_sp-e-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180725_Cranston_sp-e-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170722_FIRE_mg-n-iqeye-pre0-w0 | 0/unknown | 2 / 1 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170722_FIRE_mg-n-iqeye-pre1-w0 | 0/unknown | 10 / 0 | applied | 0.57 | 0.50 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170722_FIRE_mg-n-iqeye-post0-w0 | 1/unknown | 2 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170722_FIRE_mg-n-iqeye-post1-w0 | 1/unknown | 12 / 11 | applied | 0.60 | 0.65 | yes/yes | abstain/yes | - |
| 20231027.095514-prescribed-bi-s-mobo-c-pre0-w0 | 0/unknown | 8 / 3 | applied | 0.58 | 0.51 | yes/yes | abstain/yes | - |
| 20231027.095514-prescribed-bi-s-mobo-c-pre1-w0 | 0/unknown | 10 / 3 | applied | 0.46 | 0.40 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20231027.095514-prescribed-bi-s-mobo-c-post0-w0 | 1/unknown | 9 / 4 | applied | 0.46 | 0.41 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20231027.095514-prescribed-bi-s-mobo-c-post1-w0 | 1/unknown | 9 / 6 | applied | 0.50 | 0.42 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20161113_FIRE_bl-n-mobo-c-pre0-w0 | 0/unknown | 8 / 4 | applied | 0.47 | 0.46 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20161113_FIRE_bl-n-mobo-c-pre1-w0 | 0/unknown | 5 / 0 | applied | 0.45 | 0.37 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20161113_FIRE_bl-n-mobo-c-post0-w0 | 1/unknown | 8 / 5 | applied | 0.48 | 0.47 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20161113_FIRE_bl-n-mobo-c-post1-w0 | 1/unknown | 8 / 4 | applied | 0.54 | 0.55 | abstain/yes | no/yes | smoke_near_threshold |
| 20170713_FIRE_smer-tcs8-mobo-c-pre0-w0 | 0/unknown | 11 / 3 | applied | 0.49 | 0.44 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20170713_FIRE_smer-tcs8-mobo-c-pre1-w0 | 0/unknown | 12 / 4 | applied | 0.42 | 0.38 | no/abstain | no/abstain | fog_near_threshold |
| 20170713_FIRE_smer-tcs8-mobo-c-post0-w0 | 1/unknown | 12 / 3 | applied | 0.52 | 0.42 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20170713_FIRE_smer-tcs8-mobo-c-post1-w0 | 1/unknown | 10 / 1 | applied | 0.44 | 0.36 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20160619_FIRE_om-e-mobo-c-pre0-w0 | 0/unknown | 5 / 1 | applied | 0.43 | 0.35 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20160619_FIRE_om-e-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20160619_FIRE_om-e-mobo-c-post0-w0 | 1/unknown | 5 / 4 | applied | 0.40 | 0.31 | no/no | no/no | - |
| 20160619_FIRE_om-e-mobo-c-post1-w0 | 1/unknown | 3 / 2 | applied | 0.42 | 0.36 | no/abstain | no/abstain | fog_near_threshold |
| 20251102_ScissorsFire_tp-s-mobo-m-pre0-w0 | 0/unknown | 12 / 0 | insufficient_textured_background | 0.51 | 0.57 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20251102_ScissorsFire_tp-s-mobo-m-pre1-w0 | 0/unknown | 12 / 0 | insufficient_textured_background | 0.52 | 0.56 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20251102_ScissorsFire_tp-s-mobo-m-post0-w0 | 1/unknown | 12 / 0 | insufficient_textured_background | 0.54 | 0.58 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20251102_ScissorsFire_tp-s-mobo-m-post1-w0 | 1/unknown | 12 / 11 | applied | 0.54 | 0.58 | abstain/yes | no/yes | smoke_near_threshold |
| 20231114.093653-prescribed-mlo-w-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20231114.093653-prescribed-mlo-w-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20231114.093653-prescribed-mlo-w-mobo-c-post0-w0 | 1/unknown | 7 / 2 | applied | 0.54 | 0.52 | abstain/yes | no/yes | smoke_near_threshold |
| 20231114.093653-prescribed-mlo-w-mobo-c-post1-w0 | 1/unknown | 12 / 4 | applied | 0.68 | 0.56 | yes/yes | yes/yes | - |
| 20240909_BridgeFire_starr-n-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20240909_BridgeFire_starr-n-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20240909_BridgeFire_starr-n-mobo-c-post0-w0 | 1/unknown | 2 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20240909_BridgeFire_starr-n-mobo-c-post1-w0 | 1/unknown | 6 / 3 | applied | 0.62 | 0.42 | yes/abstain | abstain/abstain | fog_near_threshold |
| 20180606_FIRE_ml-s-mobo-c-pre0-w0 | 0/unknown | 12 / 0 | applied | 0.63 | 0.60 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180606_FIRE_ml-s-mobo-c-pre1-w0 | 0/unknown | 12 / 0 | applied | 0.57 | 0.52 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180606_FIRE_ml-s-mobo-c-post0-w0 | 1/unknown | 12 / 8 | applied | 0.64 | 0.63 | yes/yes | abstain/yes | - |
| 20180606_FIRE_ml-s-mobo-c-post1-w0 | 1/unknown | 12 / 2 | applied | 0.60 | 0.62 | yes/yes | abstain/yes | - |
| 20190716_FIRE_mg-n-mobo-c-pre0-w0 | 0/unknown | 9 / 0 | applied | 0.56 | 0.47 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20190716_FIRE_mg-n-mobo-c-pre1-w0 | 0/unknown | 9 / 7 | applied | 0.61 | 0.62 | yes/yes | abstain/yes | - |
| 20190716_FIRE_mg-n-mobo-c-post0-w0 | 1/unknown | 6 / 0 | applied | 0.46 | 0.38 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170613_FIRE_bh-w-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170613_FIRE_bh-w-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170613_FIRE_bh-w-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170613_FIRE_bh-w-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20240701_Kitchenfire_ws-w-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20240701_Kitchenfire_ws-w-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20240701_Kitchenfire_ws-w-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20240701_Kitchenfire_ws-w-mobo-c-post1-w0 | 1/unknown | 3 / 2 | applied | 0.64 | 0.48 | yes/abstain | abstain/abstain | fog_near_threshold |
| 20171207_Lilac_rm-s-mobo-pre0-w0 | 0/unknown | 12 / 7 | applied | 0.63 | 0.60 | yes/yes | abstain/yes | - |
| 20171207_Lilac_rm-s-mobo-pre1-w0 | 0/unknown | 12 / 8 | applied | 0.57 | 0.61 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20171207_Lilac_rm-s-mobo-post0-w0 | 1/unknown | 12 / 8 | applied | 0.42 | 0.41 | no/abstain | no/abstain | fog_near_threshold |
| 20171207_Lilac_rm-s-mobo-post1-w0 | 1/unknown | 12 / 8 | applied | 0.53 | 0.57 | abstain/yes | no/yes | smoke_near_threshold |
| 20200812_LakeFire_dwpgm-n-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.53 | 0.62 | abstain/yes | no/yes | smoke_near_threshold |
| 20200812_LakeFire_dwpgm-n-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.51 | 0.54 | abstain/yes | no/yes | smoke_near_threshold |
| 20200812_LakeFire_dwpgm-n-mobo-c-post0-w0 | 1/unknown | 12 / 0 | applied | 0.52 | 0.49 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20200812_LakeFire_dwpgm-n-mobo-c-post1-w0 | 1/unknown | 12 / 0 | applied | 0.58 | 0.49 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20200813_Ranch2Fire_wilson-e-mobo-c-pre0-w0 | 0/unknown | 12 / 7 | applied | 0.52 | 0.62 | abstain/yes | no/yes | smoke_near_threshold |
| 20200813_Ranch2Fire_wilson-e-mobo-c-post0-w0 | 1/unknown | 12 / 9 | applied | 0.51 | 0.61 | abstain/yes | no/yes | smoke_near_threshold |
| 20200813_Ranch2Fire_wilson-e-mobo-c-post1-w0 | 1/unknown | 12 / 10 | applied | 0.53 | 0.61 | abstain/yes | no/yes | smoke_near_threshold |
| 20251102_ScissorsFire_mp-n-mobo-m-pre0-w0 | 0/unknown | 12 / 7 | applied | 0.56 | 0.61 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20251102_ScissorsFire_mp-n-mobo-m-pre1-w0 | 0/unknown | 12 / 9 | applied | 0.60 | 0.63 | yes/yes | abstain/yes | - |
| 20251102_ScissorsFire_mp-n-mobo-m-post0-w0 | 1/unknown | 12 / 8 | applied | 0.51 | 0.61 | abstain/yes | no/yes | smoke_near_threshold |
| 20251102_ScissorsFire_mp-n-mobo-m-post1-w0 | 1/unknown | 12 / 4 | applied | 0.63 | 0.65 | yes/yes | abstain/yes | - |
| 20190813_Topanga_69bravo-n-mobo-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.57 | 0.52 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20190813_Topanga_69bravo-n-mobo-pre1-w0 | 0/unknown | 12 / 10 | applied | 0.51 | 0.49 | abstain/yes | no/yes | smoke_near_threshold |
| 20190813_Topanga_69bravo-n-mobo-post0-w0 | 1/unknown | 10 / 2 | applied | 0.42 | 0.38 | no/abstain | no/abstain | fog_near_threshold |
| 20190813_Topanga_69bravo-n-mobo-post1-w0 | 1/unknown | 12 / 3 | applied | 0.47 | 0.44 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20260622_BernardoFire_wc-w-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20260622_BernardoFire_wc-w-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20260622_BernardoFire_wc-w-mobo-c-post0-w0 | 1/unknown | 3 / 2 | applied | 0.48 | 0.46 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20260622_BernardoFire_wc-w-mobo-c-post1-w0 | 1/unknown | 4 / 0 | applied | 0.59 | 0.50 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20171021_FIRE_pi-e-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20171021_FIRE_pi-e-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20171021_FIRE_pi-e-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20171021_FIRE_pi-e-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20251102_ScissorsFire_vo-e-mobo-m-pre0-w0 | 0/unknown | 12 / 9 | applied | 0.49 | 0.58 | abstain/yes | no/yes | smoke_near_threshold |
| 20251102_ScissorsFire_vo-e-mobo-m-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.50 | 0.60 | abstain/yes | no/yes | smoke_near_threshold |
| 20251102_ScissorsFire_vo-e-mobo-m-post0-w0 | 1/unknown | 12 / 9 | applied | 0.55 | 0.65 | abstain/yes | no/yes | smoke_near_threshold |
| 20251102_ScissorsFire_vo-e-mobo-m-post1-w0 | 1/unknown | 12 / 10 | applied | 0.57 | 0.65 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20160604_FIRE_rm-n-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20160604_FIRE_rm-n-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20160604_FIRE_rm-n-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.63 | 0.65 | yes/yes | abstain/yes | - |
| 20160604_FIRE_rm-n-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.61 | 0.63 | yes/yes | abstain/yes | - |
| 20220405-fire-in-Fallbrook-rm-s-mobo-m-pre0-w0 | 0/unknown | 12 / 7 | applied | 0.40 | 0.36 | no/abstain | no/abstain | fog_near_threshold |
| 20220405-fire-in-Fallbrook-rm-s-mobo-m-pre1-w0 | 0/unknown | 12 / 4 | applied | 0.47 | 0.39 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20220405-fire-in-Fallbrook-rm-s-mobo-m-post0-w0 | 1/unknown | 12 / 11 | applied | 0.65 | 0.54 | yes/yes | abstain/yes | - |
| 20220405-fire-in-Fallbrook-rm-s-mobo-m-post1-w0 | 1/unknown | 12 / 7 | applied | 0.55 | 0.57 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20170625_BBM_bm-n-mobo-pre0-w0 | 0/unknown | 5 / 1 | applied | 0.40 | 0.31 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170625_BBM_bm-n-mobo-pre1-w0 | 0/unknown | 3 / 2 | applied | 0.40 | 0.32 | no/no | no/no | - |
| 20170625_BBM_bm-n-mobo-post0-w0 | 1/unknown | 4 / 2 | applied | 0.58 | 0.50 | yes/yes | abstain/yes | - |
| 20170625_BBM_bm-n-mobo-post1-w0 | 1/unknown | 9 / 2 | applied | 0.46 | 0.39 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20220210-EmeraldFire-signal-s-mobo-c-pre0-w0 | 0/unknown | 5 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20220210-EmeraldFire-signal-s-mobo-c-pre1-w0 | 0/unknown | 7 / 0 | insufficient_textured_background | 0.41 | 0.32 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20220210-EmeraldFire-signal-s-mobo-c-post0-w0 | 1/unknown | 9 / 5 | applied | 0.54 | 0.55 | abstain/yes | no/yes | smoke_near_threshold |
| 20220210-EmeraldFire-signal-s-mobo-c-post1-w0 | 1/unknown | 12 / 9 | applied | 0.47 | 0.44 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20220210-EmeraldFire-stgo-w-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20220210-EmeraldFire-stgo-w-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20220210-EmeraldFire-stgo-w-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20220210-EmeraldFire-stgo-w-mobo-c-post1-w0 | 1/unknown | 4 / 3 | applied | 0.40 | 0.33 | no/no | no/no | - |
| 20171010_FIRE_hp-w-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20171010_FIRE_hp-w-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20171010_FIRE_hp-w-mobo-c-post0-w0 | 1/unknown | 4 / 2 | applied | 0.47 | 0.37 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20171010_FIRE_hp-w-mobo-c-post1-w0 | 1/unknown | 10 / 6 | applied | 0.58 | 0.39 | yes/abstain | abstain/abstain | fog_near_threshold |
| 20241113_Fire_dwpgm-s-mobo-c-pre0-w0 | 0/unknown | 8 / 6 | applied | 0.46 | 0.41 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20241113_Fire_dwpgm-s-mobo-c-pre1-w0 | 0/unknown | 8 / 6 | applied | 0.49 | 0.43 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20241113_Fire_dwpgm-s-mobo-c-post0-w0 | 1/unknown | 12 / 6 | applied | 0.60 | 0.59 | yes/yes | abstain/yes | - |
| 20241113_Fire_dwpgm-s-mobo-c-post1-w0 | 1/unknown | 12 / 7 | applied | 0.65 | 0.63 | yes/yes | abstain/yes | - |
| 20251102_ScissorsFire_vo-e-mobo-c-pre0-w0 | 0/unknown | 12 / 4 | applied | 0.50 | 0.59 | abstain/yes | no/yes | smoke_near_threshold |
| 20251102_ScissorsFire_vo-e-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.54 | 0.65 | abstain/yes | no/yes | smoke_near_threshold |
| 20251102_ScissorsFire_vo-e-mobo-c-post0-w0 | 1/unknown | 12 / 10 | applied | 0.56 | 0.65 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20251102_ScissorsFire_vo-e-mobo-c-post1-w0 | 1/unknown | 12 / 8 | applied | 0.57 | 0.64 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20250120_TedWilliamsFire_wc-w-mobo-m-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.64 | 0.51 | yes/yes | abstain/yes | - |
| 20250120_TedWilliamsFire_wc-w-mobo-m-pre1-w0 | 0/unknown | 12 / 3 | applied | 0.54 | 0.58 | abstain/yes | no/yes | smoke_near_threshold |
| 20250120_TedWilliamsFire_wc-w-mobo-m-post0-w0 | 1/unknown | 12 / 8 | applied | 0.56 | 0.59 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20250120_TedWilliamsFire_wc-w-mobo-m-post1-w0 | 1/unknown | 12 / 5 | applied | 0.44 | 0.39 | no/abstain | no/abstain | fog_near_threshold |
| 20251102_ScissorsFire_mp-n-mobo-c-pre0-w0 | 0/unknown | 12 / 5 | applied | 0.60 | 0.63 | yes/yes | abstain/yes | - |
| 20251102_ScissorsFire_mp-n-mobo-c-pre1-w0 | 0/unknown | 12 / 6 | applied | 0.59 | 0.65 | yes/yes | abstain/yes | - |
| 20251102_ScissorsFire_mp-n-mobo-c-post0-w0 | 1/unknown | 12 / 4 | applied | 0.60 | 0.63 | yes/yes | abstain/yes | - |
| 20251102_ScissorsFire_mp-n-mobo-c-post1-w0 | 1/unknown | 12 / 5 | applied | 0.61 | 0.63 | yes/yes | abstain/yes | - |
| 20220405-fire-in-Fallbrook-rm-s-mobo-c-pre0-w0 | 0/unknown | 12 / 4 | applied | 0.42 | 0.40 | no/abstain | no/abstain | fog_near_threshold |
| 20220405-fire-in-Fallbrook-rm-s-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.58 | 0.59 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20220405-fire-in-Fallbrook-rm-s-mobo-c-post0-w0 | 1/unknown | 12 / 2 | applied | 0.59 | 0.53 | yes/yes | abstain/yes | - |
| 20220405-fire-in-Fallbrook-rm-s-mobo-c-post1-w0 | 1/unknown | 12 / 4 | applied | 0.54 | 0.49 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20170826_FIRE_tp-s-mobo-c-pre0-w0 | 0/unknown | 12 / 0 | applied | 0.56 | 0.57 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170826_FIRE_tp-s-mobo-c-pre1-w0 | 0/unknown | 2 / 1 | applied | 0.47 | 0.37 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170826_FIRE_tp-s-mobo-c-post0-w0 | 1/unknown | 10 / 0 | applied | 0.62 | 0.60 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170826_FIRE_tp-s-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.64 | 0.64 | yes/yes | abstain/yes | - |
| 20200727_Border11Fire_om-e-mobo-m-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200727_Border11Fire_om-e-mobo-m-pre1-w0 | 0/unknown | 2 / 0 | applied | 0.55 | 0.48 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200727_Border11Fire_om-e-mobo-m-post0-w0 | 1/unknown | 4 / 3 | applied | 0.40 | 0.33 | no/no | no/no | - |
| 20200727_Border11Fire_om-e-mobo-m-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20160711_FIRE_ml-n-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20160711_FIRE_ml-n-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20160711_FIRE_ml-n-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20160711_FIRE_ml-n-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170711_FIRE_sdsc-e-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170711_FIRE_sdsc-e-mobo-c-pre1-w0 | 0/unknown | 7 / 0 | applied | 0.50 | 0.42 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170711_FIRE_sdsc-e-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170711_FIRE_sdsc-e-mobo-c-post1-w0 | 1/unknown | 4 / 3 | applied | 0.40 | 0.32 | no/no | no/no | - |
| 20190610_Pauma_bh-w-mobo-m-pre0-w0 | 0/unknown | 12 / 9 | applied | 0.60 | 0.63 | yes/yes | abstain/yes | - |
| 20190610_Pauma_bh-w-mobo-m-pre1-w0 | 0/unknown | 12 / 7 | applied | 0.61 | 0.59 | yes/yes | abstain/yes | - |
| 20190610_Pauma_bh-w-mobo-m-post0-w0 | 1/unknown | 12 / 10 | applied | 0.61 | 0.60 | yes/yes | abstain/yes | - |
| 20190610_Pauma_bh-w-mobo-m-post1-w0 | 1/unknown | 12 / 11 | applied | 0.58 | 0.57 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20161113_FIRE_bm-w-mobo-c-pre0-w0 | 0/unknown | 2 / 1 | applied | 0.49 | 0.45 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20161113_FIRE_bm-w-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20161113_FIRE_bm-w-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20161113_FIRE_bm-w-mobo-c-post1-w0 | 1/unknown | 2 / 0 | applied | 0.39 | 0.31 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180806_FIRE_mg-s-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180806_FIRE_mg-s-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180806_FIRE_mg-s-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180806_FIRE_mg-s-mobo-c-post1-w0 | 1/unknown | 2 / 1 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20250804_CoolFire_bi-w-mobo-c-post0-w0 | 1/unknown | 3 / 0 | applied | 0.40 | 0.31 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20250804_CoolFire_bi-w-mobo-c-post1-w0 | 1/unknown | 3 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20200601_WILDLAND-DRILLS_mlo-s-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.62 | 0.62 | yes/yes | abstain/yes | - |
| 20200601_WILDLAND-DRILLS_mlo-s-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.61 | 0.63 | yes/yes | abstain/yes | - |
| 20200601_WILDLAND-DRILLS_mlo-s-mobo-c-post0-w0 | 1/unknown | 12 / 10 | applied | 0.60 | 0.61 | yes/yes | abstain/yes | - |
| 20200601_WILDLAND-DRILLS_mlo-s-mobo-c-post1-w0 | 1/unknown | 12 / 7 | applied | 0.63 | 0.59 | yes/yes | abstain/yes | - |
| 20190922_FIRE_ml-w-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.57 | 0.61 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20190922_FIRE_ml-w-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.57 | 0.60 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20190922_FIRE_ml-w-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.62 | 0.63 | yes/yes | abstain/yes | - |
| 20190922_FIRE_ml-w-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.63 | 0.63 | yes/yes | abstain/yes | - |
| 20261015_HendersonFire_buff-s-mobo-c-pre0-w0 | 0/unknown | 5 / 3 | applied | 0.62 | 0.57 | yes/yes | abstain/yes | - |
| 20261015_HendersonFire_buff-s-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20261015_HendersonFire_buff-s-mobo-c-post0-w0 | 1/unknown | 10 / 4 | applied | 0.57 | 0.56 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20261015_HendersonFire_buff-s-mobo-c-post1-w0 | 1/unknown | 12 / 1 | applied | 0.63 | 0.63 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180727_FIRE_bh-s-mobo-c-pre0-w0 | 0/unknown | 12 / 8 | applied | 0.50 | 0.52 | abstain/yes | no/yes | smoke_near_threshold |
| 20180727_FIRE_bh-s-mobo-c-pre1-w0 | 0/unknown | 10 / 3 | applied | 0.54 | 0.58 | abstain/yes | no/yes | smoke_near_threshold |
| 20180727_FIRE_bh-s-mobo-c-post0-w0 | 1/unknown | 12 / 8 | applied | 0.62 | 0.58 | yes/yes | abstain/yes | - |
| 20180727_FIRE_bh-s-mobo-c-post1-w0 | 1/unknown | 12 / 9 | applied | 0.55 | 0.61 | abstain/yes | no/yes | smoke_near_threshold |
| 20240907_FIRE_ws-s-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.56 | 0.62 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20240907_FIRE_ws-s-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.55 | 0.60 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20240907_FIRE_ws-s-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.59 | 0.64 | yes/yes | abstain/yes | - |
| 20240907_FIRE_ws-s-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.60 | 0.64 | yes/yes | abstain/yes | - |
| 20170711_FIRE_bl-s-mobo-c-pre0-w0 | 0/unknown | 12 / 0 | applied | 0.61 | 0.58 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170711_FIRE_bl-s-mobo-c-pre1-w0 | 0/unknown | 6 / 0 | applied | 0.59 | 0.50 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170711_FIRE_bl-s-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.63 | 0.62 | yes/yes | abstain/yes | - |
| 20170711_FIRE_bl-s-mobo-c-post1-w0 | 1/unknown | 12 / 0 | applied | 0.60 | 0.56 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20231108.221213-AmagoFire-bi-s-mobo-m-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20231108.221213-AmagoFire-bi-s-mobo-m-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20231108.221213-AmagoFire-bi-s-mobo-m-post0-w0 | 1/unknown | 6 / 4 | applied | 0.52 | 0.36 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20231108.221213-AmagoFire-bi-s-mobo-m-post1-w0 | 1/unknown | 4 / 2 | applied | 0.59 | 0.40 | yes/abstain | abstain/abstain | fog_near_threshold |
| 20260319_CrosleyFire_mpo-w-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20260319_CrosleyFire_mpo-w-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20260319_CrosleyFire_mpo-w-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20260319_CrosleyFire_mpo-w-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180706_FIRE_sm-e-mobo-c-pre0-w0 | 0/unknown | 11 / 1 | applied | 0.61 | 0.55 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180706_FIRE_sm-e-mobo-c-pre1-w0 | 0/unknown | 12 / 1 | applied | 0.55 | 0.56 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180706_FIRE_sm-e-mobo-c-post0-w0 | 1/unknown | 12 / 3 | applied | 0.61 | 0.62 | yes/yes | abstain/yes | - |
| 20180706_FIRE_sm-e-mobo-c-post1-w0 | 1/unknown | 12 / 7 | applied | 0.58 | 0.63 | yes/yes | abstain/yes | - |
| 20170613_FIRE_hp-n-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170613_FIRE_hp-n-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170613_FIRE_hp-n-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170613_FIRE_hp-n-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20250823_Clubfire_stgo-s-mobo-m-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.59 | 0.66 | yes/yes | abstain/yes | - |
| 20250823_Clubfire_stgo-s-mobo-m-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.59 | 0.60 | yes/yes | abstain/yes | - |
| 20250823_Clubfire_stgo-s-mobo-m-post0-w0 | 1/unknown | 12 / 11 | applied | 0.61 | 0.66 | yes/yes | abstain/yes | - |
| 20240721_EagleFire_stgo-n-mobo-c-pre0-w0 | 0/unknown | 5 / 0 | applied | 0.50 | 0.40 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20240721_EagleFire_stgo-n-mobo-c-pre1-w0 | 0/unknown | 7 / 4 | applied | 0.43 | 0.39 | no/abstain | no/abstain | fog_near_threshold |
| 20240721_EagleFire_stgo-n-mobo-c-post0-w0 | 1/unknown | 12 / 7 | applied | 0.57 | 0.60 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20240721_EagleFire_stgo-n-mobo-c-post1-w0 | 1/unknown | 12 / 0 | applied | 0.58 | 0.52 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20210302_FIRE_lp-e-mobo-m-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20210302_FIRE_lp-e-mobo-m-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20210302_FIRE_lp-e-mobo-m-post0-w0 | 1/unknown | 12 / 9 | applied | 0.42 | 0.39 | no/abstain | no/abstain | fog_near_threshold |
| 20210302_FIRE_lp-e-mobo-m-post1-w0 | 1/unknown | 8 / 6 | applied | 0.48 | 0.46 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20190924_FIRE_wc-s-mobo-c-pre0-w0 | 0/unknown | 12 / 7 | applied | 0.55 | 0.54 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20190924_FIRE_wc-s-mobo-c-pre1-w0 | 0/unknown | 8 / 3 | applied | 0.46 | 0.41 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20220622-HighlandFire-wc-n-mobo-m-pre0-w0 | 0/unknown | 12 / 0 | insufficient_textured_background | 0.52 | 0.60 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20220622-HighlandFire-wc-n-mobo-m-pre1-w0 | 0/unknown | 12 / 9 | applied | 0.51 | 0.53 | abstain/yes | no/yes | smoke_near_threshold |
| 20220622-HighlandFire-wc-n-mobo-m-post0-w0 | 1/unknown | 12 / 0 | insufficient_textured_background | 0.51 | 0.59 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20220622-HighlandFire-wc-n-mobo-m-post1-w0 | 1/unknown | 12 / 0 | insufficient_textured_background | 0.51 | 0.58 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20190924_FIRE_hp-s-mobo-c-pre0-w0 | 0/unknown | 12 / 8 | applied | 0.57 | 0.59 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20190924_FIRE_hp-s-mobo-c-pre1-w0 | 0/unknown | 12 / 2 | applied | 0.53 | 0.48 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20190924_FIRE_hp-s-mobo-c-post0-w0 | 1/unknown | 12 / 6 | applied | 0.62 | 0.60 | yes/yes | abstain/yes | - |
| 20190924_FIRE_hp-s-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.62 | 0.65 | yes/yes | abstain/yes | - |
| 20200822_BrattonFire_lp-e-mobo-c-pre0-w0 | 0/unknown | 12 / 2 | applied | 0.59 | 0.64 | yes/yes | abstain/yes | - |
| 20200822_BrattonFire_lp-e-mobo-c-pre1-w0 | 0/unknown | 12 / 0 | applied | 0.62 | 0.60 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20200822_BrattonFire_lp-e-mobo-c-post0-w0 | 1/unknown | 12 / 5 | applied | 0.53 | 0.56 | abstain/yes | no/yes | smoke_near_threshold |
| 20200822_BrattonFire_lp-e-mobo-c-post1-w0 | 1/unknown | 12 / 6 | applied | 0.53 | 0.61 | abstain/yes | no/yes | smoke_near_threshold |
| 20180603_FIRE_rm-w-mobo-c-pre0-w0 | 0/unknown | 2 / 1 | applied | 0.47 | 0.43 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180603_FIRE_rm-w-mobo-c-pre1-w0 | 0/unknown | 2 / 1 | applied | 0.45 | 0.40 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180603_FIRE_rm-w-mobo-c-post0-w0 | 1/unknown | 2 / 1 | applied | 0.43 | 0.38 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180603_FIRE_rm-w-mobo-c-post1-w0 | 1/unknown | 7 / 5 | applied | 0.59 | 0.57 | yes/yes | abstain/yes | - |
| 20210209_FIRE_tp-w-mobo-c-pre0-w0 | 0/unknown | 12 / 10 | applied | 0.61 | 0.55 | yes/yes | abstain/yes | - |
| 20210209_FIRE_tp-w-mobo-c-pre1-w0 | 0/unknown | 12 / 8 | applied | 0.61 | 0.63 | yes/yes | abstain/yes | - |
| 20210209_FIRE_tp-w-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.61 | 0.57 | yes/yes | abstain/yes | - |
| 20210209_FIRE_tp-w-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.66 | 0.62 | yes/yes | abstain/yes | - |
| 20240201.064352-FIRE-mlo-n-mobo-c-pre0-w0 | 0/unknown | 12 / 8 | applied | 0.53 | 0.60 | abstain/yes | no/yes | smoke_near_threshold |
| 20240201.064352-FIRE-mlo-n-mobo-c-pre1-w0 | 0/unknown | 12 / 8 | applied | 0.59 | 0.66 | yes/yes | abstain/yes | - |
| 20240201.064352-FIRE-mlo-n-mobo-c-post0-w0 | 1/unknown | 12 / 2 | applied | 0.54 | 0.62 | abstain/yes | no/yes | smoke_near_threshold |
| 20240201.064352-FIRE-mlo-n-mobo-c-post1-w0 | 1/unknown | 12 / 4 | applied | 0.52 | 0.62 | abstain/yes | no/yes | smoke_near_threshold |
| 20241007_Posta3Fire_ws-n-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.61 | 0.65 | yes/yes | abstain/yes | - |
| 20241007_Posta3Fire_ws-n-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.60 | 0.65 | yes/yes | abstain/yes | - |
| 20241007_Posta3Fire_ws-n-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.61 | 0.66 | yes/yes | abstain/yes | - |
| 20241007_Posta3Fire_ws-n-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.53 | 0.60 | abstain/yes | no/yes | smoke_near_threshold |
| 20171207_FIRE_bh-n-mobo-c-pre0-w0 | 0/unknown | 12 / 3 | applied | 0.62 | 0.64 | yes/yes | abstain/yes | - |
| 20171207_FIRE_bh-n-mobo-c-pre1-w0 | 0/unknown | 12 / 6 | applied | 0.62 | 0.64 | yes/yes | abstain/yes | - |
| 20171207_FIRE_bh-n-mobo-c-post0-w0 | 1/unknown | 12 / 8 | applied | 0.44 | 0.38 | no/abstain | no/abstain | fog_near_threshold |
| 20171207_FIRE_bh-n-mobo-c-post1-w0 | 1/unknown | 12 / 4 | applied | 0.61 | 0.54 | yes/yes | abstain/yes | - |
| 20200521_VEGMGMT_bm-s-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200521_VEGMGMT_bm-s-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200521_VEGMGMT_bm-s-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200521_VEGMGMT_bm-s-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180809_FIRE_mg-w-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.55 | 0.64 | abstain/yes | no/yes | smoke_near_threshold |
| 20180809_FIRE_mg-w-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.57 | 0.63 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20180809_FIRE_mg-w-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.58 | 0.66 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20180809_FIRE_mg-w-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.56 | 0.64 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20180827_Holyflareup_sp-e-mobo-c-pre0-w0 | 0/unknown | 10 / 6 | applied | 0.46 | 0.45 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20180827_Holyflareup_sp-e-mobo-c-pre1-w0 | 0/unknown | 10 / 7 | applied | 0.44 | 0.42 | no/abstain | no/abstain | fog_near_threshold |
| 20180827_Holyflareup_sp-e-mobo-c-post0-w0 | 1/unknown | 10 / 4 | applied | 0.56 | 0.50 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20180827_Holyflareup_sp-e-mobo-c-post1-w0 | 1/unknown | 11 / 1 | applied | 0.53 | 0.47 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20231108.221213-AmagoFire-bi-s-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20231108.221213-AmagoFire-bi-s-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20231108.221213-AmagoFire-bi-s-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20231108.221213-AmagoFire-bi-s-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180813_FIRE_bl-n-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180813_FIRE_bl-n-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180813_FIRE_bl-n-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180813_FIRE_bl-n-mobo-c-post1-w0 | 1/unknown | 2 / 1 | applied | 0.43 | 0.40 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20171026_FIRE_smer-tcs8-mobo-c-pre0-w0 | 0/unknown | 11 / 2 | applied | 0.44 | 0.36 | no/abstain | no/abstain | fog_near_threshold |
| 20171026_FIRE_smer-tcs8-mobo-c-pre1-w0 | 0/unknown | 11 / 3 | applied | 0.43 | 0.37 | no/abstain | no/abstain | fog_near_threshold |
| 20171026_FIRE_smer-tcs8-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20171026_FIRE_smer-tcs8-mobo-c-post1-w0 | 1/unknown | 3 / 0 | applied | 0.42 | 0.34 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170627_FIRE_om-e-mobo-c-pre0-w0 | 0/unknown | 10 / 6 | applied | 0.46 | 0.46 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20170627_FIRE_om-e-mobo-c-pre1-w0 | 0/unknown | 8 / 6 | applied | 0.55 | 0.51 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20170627_FIRE_om-e-mobo-c-post0-w0 | 1/unknown | 10 / 8 | applied | 0.55 | 0.51 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20170627_FIRE_om-e-mobo-c-post1-w0 | 1/unknown | 8 / 6 | applied | 0.40 | 0.36 | no/abstain | no/abstain | fog_near_threshold |
| 20200206_FIRE_ml-s-mobo-c-pre0-w0 | 0/unknown | 10 / 8 | applied | 0.49 | 0.48 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20200206_FIRE_ml-s-mobo-c-pre1-w0 | 0/unknown | 10 / 7 | applied | 0.55 | 0.55 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20200206_FIRE_ml-s-mobo-c-post0-w0 | 1/unknown | 12 / 6 | applied | 0.64 | 0.63 | yes/yes | abstain/yes | - |
| 20200206_FIRE_ml-s-mobo-c-post1-w0 | 1/unknown | 12 / 5 | applied | 0.62 | 0.64 | yes/yes | abstain/yes | - |
| 20241108_ResortFire_mg-n-mobo-c-pre0-w0 | 0/unknown | 9 / 1 | applied | 0.55 | 0.52 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20241108_ResortFire_mg-n-mobo-c-pre1-w0 | 0/unknown | 10 / 0 | applied | 0.58 | 0.52 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20241108_ResortFire_mg-n-mobo-c-post0-w0 | 1/unknown | 9 / 3 | applied | 0.61 | 0.58 | yes/yes | abstain/yes | - |
| 20241108_ResortFire_mg-n-mobo-c-post1-w0 | 1/unknown | 9 / 2 | applied | 0.49 | 0.47 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20171207_FIRE_bh-w-mobo-c-pre0-w0 | 0/unknown | 12 / 2 | applied | 0.47 | 0.38 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20171207_FIRE_bh-w-mobo-c-pre1-w0 | 0/unknown | 12 / 9 | applied | 0.42 | 0.34 | no/no | no/no | - |
| 20171207_FIRE_bh-w-mobo-c-post0-w0 | 1/unknown | 12 / 0 | applied | 0.43 | 0.37 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20171207_FIRE_bh-w-mobo-c-post1-w0 | 1/unknown | 12 / 0 | applied | 0.42 | 0.35 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20240907_ChurchFire_ws-w-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.54 | 0.60 | abstain/yes | no/yes | smoke_near_threshold |
| 20240907_ChurchFire_ws-w-mobo-c-pre1-w0 | 0/unknown | 12 / 9 | applied | 0.60 | 0.64 | yes/yes | abstain/yes | - |
| 20240907_ChurchFire_ws-w-mobo-c-post0-w0 | 1/unknown | 12 / 9 | applied | 0.56 | 0.61 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20240907_ChurchFire_ws-w-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.52 | 0.61 | abstain/yes | no/yes | smoke_near_threshold |
| 20200906-BobcatFire-wilson-e-mobo-c-pre0-w0 | 0/unknown | 12 / 10 | applied | 0.63 | 0.64 | yes/yes | abstain/yes | - |
| 20200906-BobcatFire-wilson-e-mobo-c-pre1-w0 | 0/unknown | 12 / 7 | applied | 0.61 | 0.64 | yes/yes | abstain/yes | - |
| 20200906-BobcatFire-wilson-e-mobo-c-post0-w0 | 1/unknown | 12 / 10 | applied | 0.59 | 0.63 | yes/yes | abstain/yes | - |
| 20200906-BobcatFire-wilson-e-mobo-c-post1-w0 | 1/unknown | 12 / 9 | applied | 0.56 | 0.64 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20180606_FIRE_pi-e-mobo-c-pre0-w0 | 0/unknown | 12 / 1 | applied | 0.60 | 0.60 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180606_FIRE_pi-e-mobo-c-pre1-w0 | 0/unknown | 12 / 0 | applied | 0.64 | 0.65 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180606_FIRE_pi-e-mobo-c-post0-w0 | 1/unknown | 12 / 9 | applied | 0.62 | 0.66 | yes/yes | abstain/yes | - |
| 20180606_FIRE_pi-e-mobo-c-post1-w0 | 1/unknown | 12 / 0 | applied | 0.63 | 0.65 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20171026_FIRE_rm-n-mobo-c-pre0-w0 | 0/unknown | 10 / 8 | applied | 0.47 | 0.44 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20171026_FIRE_rm-n-mobo-c-pre1-w0 | 0/unknown | 10 / 0 | applied | 0.44 | 0.36 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20171026_FIRE_rm-n-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.49 | 0.49 | abstain/yes | no/yes | smoke_near_threshold |
| 20171026_FIRE_rm-n-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.51 | 0.53 | abstain/yes | no/yes | smoke_near_threshold |
| 20231003.210106-Fallbrook_RV-rm-s-mobo-m-pre0-w0 | 0/unknown | 10 / 1 | applied | 0.42 | 0.34 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20231003.210106-Fallbrook_RV-rm-s-mobo-m-pre1-w0 | 0/unknown | 12 / 1 | applied | 0.40 | 0.34 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20231003.210106-Fallbrook_RV-rm-s-mobo-m-post0-w0 | 1/unknown | 12 / 0 | applied | 0.44 | 0.39 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20231003.210106-Fallbrook_RV-rm-s-mobo-m-post1-w0 | 1/unknown | 12 / 2 | applied | 0.43 | 0.36 | no/abstain | no/abstain | fog_near_threshold |
| 20241006_SaddlebackFire_stgo-w-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.48 | 0.48 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20241006_SaddlebackFire_stgo-w-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.54 | 0.56 | abstain/yes | no/yes | smoke_near_threshold |
| 20241006_SaddlebackFire_stgo-w-mobo-c-post0-w0 | 1/unknown | 12 / 3 | applied | 0.50 | 0.51 | abstain/yes | no/yes | smoke_near_threshold |
| 20241006_SaddlebackFire_stgo-w-mobo-c-post1-w0 | 1/unknown | 12 / 2 | applied | 0.41 | 0.37 | no/abstain | no/abstain | fog_near_threshold |
| 20200202_FIRE_hp-w-mobo-c-pre0-w0 | 0/unknown | 3 / 2 | applied | 0.40 | 0.32 | no/no | no/no | - |
| 20200202_FIRE_hp-w-mobo-c-pre1-w0 | 0/unknown | 10 / 0 | applied | 0.46 | 0.35 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20200202_FIRE_hp-w-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200202_FIRE_hp-w-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20250101_Fire_dwpgm-s-mobo-c-pre0-w0 | 0/unknown | 12 / 0 | insufficient_textured_background | 0.59 | 0.63 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20250101_Fire_dwpgm-s-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.56 | 0.66 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20250101_Fire_dwpgm-s-mobo-c-post0-w0 | 1/unknown | 12 / 0 | insufficient_textured_background | 0.61 | 0.64 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20250101_Fire_dwpgm-s-mobo-c-post1-w0 | 1/unknown | 12 / 0 | insufficient_textured_background | 0.63 | 0.64 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20251019_RanchFire_wc-w-mobo-m-pre0-w0 | 0/unknown | 10 / 5 | applied | 0.58 | 0.54 | yes/yes | abstain/yes | - |
| 20251019_RanchFire_wc-w-mobo-m-pre1-w0 | 0/unknown | 8 / 6 | applied | 0.58 | 0.54 | yes/yes | abstain/yes | - |
| 20251019_RanchFire_wc-w-mobo-m-post0-w0 | 1/unknown | 12 / 8 | applied | 0.48 | 0.47 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20251019_RanchFire_wc-w-mobo-m-post1-w0 | 1/unknown | 10 / 6 | applied | 0.62 | 0.58 | yes/yes | abstain/yes | - |
| 20231003.210207-Fallbrook_RV-rm-s-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.54 | 0.56 | abstain/yes | no/yes | smoke_near_threshold |
| 20231003.210207-Fallbrook_RV-rm-s-mobo-c-pre1-w0 | 0/unknown | 8 / 0 | applied | 0.60 | 0.51 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20231003.210207-Fallbrook_RV-rm-s-mobo-c-post0-w0 | 1/unknown | 10 / 0 | applied | 0.42 | 0.36 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20231003.210207-Fallbrook_RV-rm-s-mobo-c-post1-w0 | 1/unknown | 10 / 0 | applied | 0.52 | 0.46 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20210204_FIRE_tp-s-mobo-c-pre0-w0 | 0/unknown | 5 / 3 | applied | 0.44 | 0.36 | no/abstain | no/abstain | fog_near_threshold |
| 20210204_FIRE_tp-s-mobo-c-pre1-w0 | 0/unknown | 9 / 7 | applied | 0.43 | 0.36 | no/abstain | no/abstain | fog_near_threshold |
| 20210204_FIRE_tp-s-mobo-c-post0-w0 | 1/unknown | 4 / 0 | applied | 0.48 | 0.39 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20210204_FIRE_tp-s-mobo-c-post1-w0 | 1/unknown | 2 / 1 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20250501_FelipeFire_ml-n-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.51 | 0.55 | abstain/yes | no/yes | smoke_near_threshold |
| 20250501_FelipeFire_ml-n-mobo-c-pre1-w0 | 0/unknown | 9 / 2 | applied | 0.41 | 0.38 | no/abstain | no/abstain | fog_near_threshold |
| 20250501_FelipeFire_ml-n-mobo-c-post0-w0 | 1/unknown | 12 / 10 | applied | 0.47 | 0.51 | abstain/yes | no/yes | smoke_near_threshold |
| 20250501_FelipeFire_ml-n-mobo-c-post1-w0 | 1/unknown | 9 / 5 | applied | 0.46 | 0.44 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20171016_FIRE_sdsc-e-mobo-c-pre0-w0 | 0/unknown | 11 / 6 | applied | 0.61 | 0.59 | yes/yes | abstain/yes | - |
| 20171016_FIRE_sdsc-e-mobo-c-pre1-w0 | 0/unknown | 11 / 3 | applied | 0.48 | 0.43 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20171016_FIRE_sdsc-e-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20171016_FIRE_sdsc-e-mobo-c-post1-w0 | 1/unknown | 6 / 4 | applied | 0.40 | 0.31 | no/no | no/no | - |
| 20200705_FIRE_bm-w-mobo-c-pre0-w0 | 0/unknown | 5 / 3 | applied | 0.55 | 0.53 | abstain/yes | no/yes | smoke_near_threshold |
| 20200705_FIRE_bm-w-mobo-c-pre1-w0 | 0/unknown | 10 / 0 | applied | 0.40 | 0.31 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20200705_FIRE_bm-w-mobo-c-post0-w0 | 1/unknown | 8 / 0 | applied | 0.40 | 0.31 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20200705_FIRE_bm-w-mobo-c-post1-w0 | 1/unknown | 10 / 0 | applied | 0.42 | 0.35 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20230726.153347-FIRE-mg-s-mobo-c-pre0-w0 | 0/unknown | 2 / 0 | applied | 0.43 | 0.35 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20230726.153347-FIRE-mg-s-mobo-c-pre1-w0 | 0/unknown | 6 / 0 | applied | 0.43 | 0.35 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20230726.153347-FIRE-mg-s-mobo-c-post0-w0 | 1/unknown | 3 / 2 | applied | 0.40 | 0.32 | no/no | no/no | - |
| 20230726.153347-FIRE-mg-s-mobo-c-post1-w0 | 1/unknown | 7 / 0 | applied | 0.44 | 0.38 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20200829_inside-Mexico_mlo-s-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200829_inside-Mexico_mlo-s-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200829_inside-Mexico_mlo-s-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200829_inside-Mexico_mlo-s-mobo-c-post1-w0 | 1/unknown | 5 / 3 | applied | 0.63 | 0.47 | yes/abstain | abstain/abstain | fog_near_threshold |
| 20190924_FIRE_ml-w-mobo-c-pre0-w0 | 0/unknown | 11 / 8 | applied | 0.55 | 0.52 | abstain/yes | no/yes | smoke_near_threshold |
| 20190924_FIRE_ml-w-mobo-c-pre1-w0 | 0/unknown | 8 / 0 | applied | 0.45 | 0.37 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20190924_FIRE_ml-w-mobo-c-post0-w0 | 1/unknown | 11 / 2 | applied | 0.47 | 0.45 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20190924_FIRE_ml-w-mobo-c-post1-w0 | 1/unknown | 12 / 9 | applied | 0.60 | 0.49 | yes/abstain | abstain/abstain | fog_near_threshold |
| 20170722_FIRE_bm-n-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170722_FIRE_bm-n-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170722_FIRE_bm-n-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170722_FIRE_bm-n-mobo-c-post1-w0 | 1/unknown | 2 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180809_FIRE_bh-s-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.59 | 0.60 | yes/yes | abstain/yes | - |
| 20180809_FIRE_bh-s-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.56 | 0.61 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20180809_FIRE_bh-s-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.56 | 0.60 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20180809_FIRE_bh-s-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.62 | 0.60 | yes/yes | abstain/yes | - |
| 20180603_FIRE_bl-s-mobo-c-pre0-w0 | 0/unknown | 8 / 0 | applied | 0.56 | 0.47 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180603_FIRE_bl-s-mobo-c-pre1-w0 | 0/unknown | 10 / 0 | applied | 0.58 | 0.50 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180603_FIRE_bl-s-mobo-c-post0-w0 | 1/unknown | 8 / 0 | applied | 0.58 | 0.50 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180603_FIRE_bl-s-mobo-c-post1-w0 | 1/unknown | 8 / 0 | applied | 0.58 | 0.49 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20200806_SpringsFire_sm-e-mobo-c-post0-w0 | 1/unknown | 10 / 0 | applied | 0.58 | 0.51 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20200806_SpringsFire_sm-e-mobo-c-post1-w0 | 1/unknown | 10 / 0 | applied | 0.58 | 0.50 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20171010_FIRE_hp-n-mobo-c-pre0-w0 | 0/unknown | 7 / 5 | applied | 0.46 | 0.40 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20171010_FIRE_hp-n-mobo-c-pre1-w0 | 0/unknown | 4 / 2 | applied | 0.56 | 0.48 | abstain/abstain | abstain/abstain | smoke_near_threshold, fog_near_threshold |
| 20171010_FIRE_hp-n-mobo-c-post0-w0 | 1/unknown | 3 / 2 | applied | 0.48 | 0.45 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20171010_FIRE_hp-n-mobo-c-post1-w0 | 1/unknown | 8 / 6 | applied | 0.55 | 0.42 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20180602_FIRE_smer-tcs9-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180602_FIRE_smer-tcs9-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180602_FIRE_smer-tcs9-mobo-c-post0-w0 | 1/unknown | 8 / 2 | applied | 0.57 | 0.48 | abstain/abstain | abstain/abstain | smoke_near_threshold, fog_near_threshold |
| 20180602_FIRE_smer-tcs9-mobo-c-post1-w0 | 1/unknown | 12 / 6 | applied | 0.64 | 0.57 | yes/yes | abstain/yes | - |
| 20250823_Clubfire_stgo-s-mobo-c-pre0-w0 | 0/unknown | 12 / 10 | applied | 0.59 | 0.66 | yes/yes | abstain/yes | - |
| 20250823_Clubfire_stgo-s-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.58 | 0.60 | yes/yes | abstain/yes | - |
| 20250823_Clubfire_stgo-s-mobo-c-post0-w0 | 1/unknown | 12 / 9 | applied | 0.62 | 0.63 | yes/yes | abstain/yes | - |
| 20191005_FIRE_hp-s-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20191005_FIRE_hp-s-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20191005_FIRE_hp-s-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20191005_FIRE_hp-s-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200911_FIRE_lp-e-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200911_FIRE_lp-e-mobo-c-pre1-w0 | 0/unknown | 3 / 2 | applied | 0.45 | 0.39 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20200911_FIRE_lp-e-mobo-c-post0-w0 | 1/unknown | 3 / 2 | applied | 0.52 | 0.48 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20200911_FIRE_lp-e-mobo-c-post1-w0 | 1/unknown | 2 / 1 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20191005_FIRE_wc-n-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20191005_FIRE_wc-n-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20191005_FIRE_wc-n-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20191005_FIRE_wc-n-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20190529_94Fire_om-n-mobo-c-pre0-w0 | 0/unknown | 9 / 0 | applied | 0.55 | 0.52 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20190529_94Fire_om-n-mobo-c-pre1-w0 | 0/unknown | 8 / 0 | applied | 0.57 | 0.52 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20190529_94Fire_om-n-mobo-c-post0-w0 | 1/unknown | 9 / 3 | applied | 0.59 | 0.56 | yes/yes | abstain/yes | - |
| 20190529_94Fire_om-n-mobo-c-post1-w0 | 1/unknown | 12 / 0 | applied | 0.56 | 0.48 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180723_FIRE_tp-e-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.49 | 0.59 | abstain/yes | no/yes | smoke_near_threshold |
| 20180723_FIRE_tp-e-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.51 | 0.59 | abstain/yes | no/yes | smoke_near_threshold |
| 20180723_FIRE_tp-e-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.49 | 0.58 | abstain/yes | no/yes | smoke_near_threshold |
| 20180723_FIRE_tp-e-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.57 | 0.60 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20250123_GilmanFire_tdlln-s-mobo-c-pre0-w0 | 0/unknown | 12 / 5 | applied | 0.57 | 0.53 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20250123_GilmanFire_tdlln-s-mobo-c-pre1-w0 | 0/unknown | 12 / 7 | applied | 0.43 | 0.37 | no/abstain | no/abstain | fog_near_threshold |
| 20250123_GilmanFire_tdlln-s-mobo-c-post0-w0 | 1/unknown | 12 / 5 | applied | 0.45 | 0.44 | no/abstain | no/abstain | fog_near_threshold |
| 20250123_GilmanFire_tdlln-s-mobo-c-post1-w0 | 1/unknown | 12 / 5 | applied | 0.56 | 0.60 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20160718_FIRE_mw-e-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20160718_FIRE_mw-e-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20160718_FIRE_mw-e-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20160718_FIRE_mw-e-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.60 | 0.63 | yes/yes | abstain/yes | - |
| 20180504_FIRE_smer-tcs10-mobo-c-pre0-w0 | 0/unknown | 4 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180504_FIRE_smer-tcs10-mobo-c-pre1-w0 | 0/unknown | 6 / 0 | applied | 0.43 | 0.36 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180504_FIRE_smer-tcs10-mobo-c-post0-w0 | 1/unknown | 12 / 9 | applied | 0.69 | 0.55 | yes/yes | yes/yes | - |
| 20180504_FIRE_smer-tcs10-mobo-c-post1-w0 | 1/unknown | 12 / 4 | applied | 0.61 | 0.61 | yes/yes | abstain/yes | - |
| 20200306_FIRE_pi-n-mobo-c-pre0-w0 | 0/unknown | 12 / 6 | applied | 0.54 | 0.61 | abstain/yes | no/yes | smoke_near_threshold |
| 20200306_FIRE_pi-n-mobo-c-pre1-w0 | 0/unknown | 12 / 6 | applied | 0.49 | 0.52 | abstain/yes | no/yes | smoke_near_threshold |
| 20200306_FIRE_pi-n-mobo-c-post0-w0 | 1/unknown | 12 / 4 | applied | 0.65 | 0.62 | yes/yes | abstain/yes | - |
| 20200306_FIRE_pi-n-mobo-c-post1-w0 | 1/unknown | 12 / 1 | applied | 0.51 | 0.54 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180614_Hope_wc-e-mobo-c-pre0-w0 | 0/unknown | 12 / 1 | applied | 0.64 | 0.62 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180614_Hope_wc-e-mobo-c-pre1-w0 | 0/unknown | 12 / 0 | applied | 0.64 | 0.60 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180614_Hope_wc-e-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.61 | 0.63 | yes/yes | abstain/yes | - |
| 20180614_Hope_wc-e-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.59 | 0.64 | yes/yes | abstain/yes | - |
| 20190728_Dehesa_lp-n-mobo-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.58 | 0.64 | yes/yes | abstain/yes | - |
| 20190728_Dehesa_lp-n-mobo-pre1-w0 | 0/unknown | 7 / 0 | applied | 0.59 | 0.51 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20190728_Dehesa_lp-n-mobo-post0-w0 | 1/unknown | 12 / 11 | applied | 0.54 | 0.64 | abstain/yes | no/yes | smoke_near_threshold |
| 20190728_Dehesa_lp-n-mobo-post1-w0 | 1/unknown | 12 / 11 | applied | 0.56 | 0.64 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20180706_West_lp-n-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.59 | 0.62 | yes/yes | abstain/yes | - |
| 20180706_West_lp-n-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.60 | 0.58 | yes/yes | abstain/yes | - |
| 20180706_West_lp-n-mobo-c-post0-w0 | 1/unknown | 12 / 7 | applied | 0.50 | 0.57 | abstain/yes | no/yes | smoke_near_threshold |
| 20180706_West_lp-n-mobo-c-post1-w0 | 1/unknown | 12 / 0 | unreliable_background | 0.50 | 0.58 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180719_Skyline_sp-n-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.59 | 0.65 | yes/yes | abstain/yes | - |
| 20180719_Skyline_sp-n-mobo-c-pre1-w0 | 0/unknown | 12 / 3 | applied | 0.52 | 0.58 | abstain/yes | no/yes | smoke_near_threshold |
| 20180719_Skyline_sp-n-mobo-c-post0-w0 | 1/unknown | 12 / 6 | applied | 0.65 | 0.64 | yes/yes | abstain/yes | - |
| 20180719_Skyline_sp-n-mobo-c-post1-w0 | 1/unknown | 12 / 2 | applied | 0.64 | 0.64 | yes/yes | abstain/yes | - |
| 20240701_McCainfire_ws-e-mobo-c-pre0-w0 | 0/unknown | 6 / 4 | applied | 0.51 | 0.50 | abstain/yes | no/yes | smoke_near_threshold |
| 20240701_McCainfire_ws-e-mobo-c-pre1-w0 | 0/unknown | 6 / 4 | applied | 0.48 | 0.45 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20240701_McCainfire_ws-e-mobo-c-post0-w0 | 1/unknown | 5 / 0 | applied | 0.51 | 0.42 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20240701_McCainfire_ws-e-mobo-c-post1-w0 | 1/unknown | 2 / 0 | applied | 0.50 | 0.41 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200601_WILDLAND-DRILLS_mlo-e-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.60 | 0.64 | yes/yes | abstain/yes | - |
| 20200601_WILDLAND-DRILLS_mlo-e-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.60 | 0.65 | yes/yes | abstain/yes | - |
| 20200601_WILDLAND-DRILLS_mlo-e-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.55 | 0.65 | abstain/yes | no/yes | smoke_near_threshold |
| 20200601_WILDLAND-DRILLS_mlo-e-mobo-c-post1-w0 | 1/unknown | 12 / 6 | applied | 0.55 | 0.62 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20190803_Sage_om-n-mobo-pre0-w0 | 0/unknown | 12 / 9 | applied | 0.53 | 0.59 | abstain/yes | no/yes | smoke_near_threshold |
| 20190803_Sage_om-n-mobo-pre1-w0 | 0/unknown | 12 / 10 | applied | 0.60 | 0.62 | yes/yes | abstain/yes | - |
| 20170609_FIRE_sm-n-mobo-c-pre0-w0 | 0/unknown | 9 / 0 | applied | 0.60 | 0.54 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170609_FIRE_sm-n-mobo-c-pre1-w0 | 0/unknown | 10 / 0 | applied | 0.62 | 0.56 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170609_FIRE_sm-n-mobo-c-post0-w0 | 1/unknown | 9 / 0 | applied | 0.61 | 0.55 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170609_FIRE_sm-n-mobo-c-post1-w0 | 1/unknown | 7 / 0 | applied | 0.56 | 0.47 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170722_FIRE_hp-e-mobo-c-pre0-w0 | 0/unknown | 12 / 0 | applied | 0.48 | 0.41 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170722_FIRE_hp-e-mobo-c-pre1-w0 | 0/unknown | 9 / 0 | applied | 0.52 | 0.46 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170722_FIRE_hp-e-mobo-c-post0-w0 | 1/unknown | 12 / 0 | applied | 0.46 | 0.42 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170722_FIRE_hp-e-mobo-c-post1-w0 | 1/unknown | 12 / 0 | applied | 0.52 | 0.46 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180726_FIRE_so-n-mobo-c-pre0-w0 | 0/unknown | 7 / 3 | applied | 0.46 | 0.40 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20180726_FIRE_so-n-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180726_FIRE_so-n-mobo-c-post0-w0 | 1/unknown | 5 / 3 | applied | 0.53 | 0.48 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20180726_FIRE_so-n-mobo-c-post1-w0 | 1/unknown | 10 / 2 | applied | 0.52 | 0.55 | abstain/yes | no/yes | smoke_near_threshold |
| 20200806_SpringsFire_om-n-mobo-m-post0-w0 | 1/unknown | 6 / 2 | applied | 0.47 | 0.43 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20200806_SpringsFire_om-n-mobo-m-post1-w0 | 1/unknown | 5 / 1 | applied | 0.50 | 0.41 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170627_FIRE_lp-n-iqeye-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170627_FIRE_lp-n-iqeye-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170627_FIRE_lp-n-iqeye-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170627_FIRE_lp-n-iqeye-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20190728_FIRE_om-n-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.60 | 0.65 | yes/yes | abstain/yes | - |
| 20190728_FIRE_om-n-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.59 | 0.63 | yes/yes | abstain/yes | - |
| 20190728_FIRE_om-n-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.61 | 0.66 | yes/yes | abstain/yes | - |
| 20190728_FIRE_om-n-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.61 | 0.65 | yes/yes | abstain/yes | - |
| 20240927_86Fire_tp-e-mobo-c-pre0-w0 | 0/unknown | 12 / 0 | applied | 0.62 | 0.60 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20240927_86Fire_tp-e-mobo-c-pre1-w0 | 0/unknown | 10 / 0 | applied | 0.62 | 0.60 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20240927_86Fire_tp-e-mobo-c-post0-w0 | 1/unknown | 12 / 0 | applied | 0.62 | 0.58 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20240927_86Fire_tp-e-mobo-c-post1-w0 | 1/unknown | 10 / 0 | applied | 0.62 | 0.56 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20160722_FIRE_mw-e-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20160722_FIRE_mw-e-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20160722_FIRE_mw-e-mobo-c-post0-w0 | 1/unknown | 3 / 2 | applied | 0.57 | 0.54 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20160722_FIRE_mw-e-mobo-c-post1-w0 | 1/unknown | 12 / 7 | applied | 0.51 | 0.50 | abstain/yes | no/yes | smoke_near_threshold |
| 20220214-PrescribedFire-pi-n-mobo-c-pre0-w0 | 0/unknown | 10 / 0 | applied | 0.61 | 0.56 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20220214-PrescribedFire-pi-n-mobo-c-pre1-w0 | 0/unknown | 11 / 0 | applied | 0.61 | 0.56 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20220214-PrescribedFire-pi-n-mobo-c-post0-w0 | 1/unknown | 12 / 0 | applied | 0.61 | 0.58 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20220214-PrescribedFire-pi-n-mobo-c-post1-w0 | 1/unknown | 12 / 0 | applied | 0.64 | 0.60 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180706_FIRE_wc-e-mobo-c-pre0-w0 | 0/unknown | 12 / 0 | applied | 0.63 | 0.60 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180706_FIRE_wc-e-mobo-c-pre1-w0 | 0/unknown | 12 / 0 | applied | 0.61 | 0.57 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180706_FIRE_wc-e-mobo-c-post0-w0 | 1/unknown | 12 / 5 | applied | 0.59 | 0.60 | yes/yes | abstain/yes | - |
| 20180706_FIRE_wc-e-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.60 | 0.59 | yes/yes | abstain/yes | - |
| 20190717_FIRE_lp-n-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20190717_FIRE_lp-n-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20190717_FIRE_lp-n-mobo-c-post0-w0 | 1/unknown | 10 / 8 | applied | 0.62 | 0.52 | yes/yes | abstain/yes | - |
| 20190717_FIRE_lp-n-mobo-c-post1-w0 | 1/unknown | 9 / 3 | applied | 0.51 | 0.44 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20190728_FIRE_sp-n-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.56 | 0.64 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20190728_FIRE_sp-n-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.59 | 0.65 | yes/yes | abstain/yes | - |
| 20190728_FIRE_sp-n-mobo-c-post0-w0 | 1/unknown | 12 / 9 | applied | 0.61 | 0.65 | yes/yes | abstain/yes | - |
| 20190728_FIRE_sp-n-mobo-c-post1-w0 | 1/unknown | 12 / 8 | applied | 0.53 | 0.62 | abstain/yes | no/yes | smoke_near_threshold |
| 20170711_FIRE_sm-n-mobo-c-pre0-w0 | 0/unknown | 10 / 2 | applied | 0.61 | 0.61 | yes/yes | abstain/yes | - |
| 20170711_FIRE_sm-n-mobo-c-pre1-w0 | 0/unknown | 10 / 7 | applied | 0.51 | 0.42 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20170711_FIRE_sm-n-mobo-c-post0-w0 | 1/unknown | 10 / 0 | applied | 0.61 | 0.57 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170711_FIRE_sm-n-mobo-c-post1-w0 | 1/unknown | 12 / 0 | applied | 0.63 | 0.58 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20210209_FIRE_hp-e-mobo-c-pre0-w0 | 0/unknown | 12 / 7 | applied | 0.57 | 0.65 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20210209_FIRE_hp-e-mobo-c-pre1-w0 | 0/unknown | 12 / 10 | applied | 0.58 | 0.66 | yes/yes | abstain/yes | - |
| 20210209_FIRE_hp-e-mobo-c-post0-w0 | 1/unknown | 12 / 9 | applied | 0.56 | 0.65 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20210209_FIRE_hp-e-mobo-c-post1-w0 | 1/unknown | 12 / 8 | applied | 0.58 | 0.66 | yes/yes | abstain/yes | - |
| 20191006_FIRE_om-n-mobo-c-pre0-w0 | 0/unknown | 2 / 0 | applied | 0.49 | 0.41 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20191006_FIRE_om-n-mobo-c-pre1-w0 | 0/unknown | 6 / 4 | applied | 0.51 | 0.47 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20191006_FIRE_om-n-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20191006_FIRE_om-n-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180725_FIRE_smer-tcs10-mobo-c-pre0-w0 | 0/unknown | 4 / 2 | applied | 0.53 | 0.48 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20180725_FIRE_smer-tcs10-mobo-c-pre1-w0 | 0/unknown | 5 / 2 | applied | 0.43 | 0.38 | no/abstain | no/abstain | fog_near_threshold |
| 20180725_FIRE_smer-tcs10-mobo-c-post0-w0 | 1/unknown | 3 / 0 | applied | 0.48 | 0.40 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180725_FIRE_smer-tcs10-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.64 | 0.62 | yes/yes | abstain/yes | - |
| 20180720_Cinnamon_wc-e-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180720_Cinnamon_wc-e-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180720_Cinnamon_wc-e-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180720_Cinnamon_wc-e-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20190913_FIRE_lp-n-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20190913_FIRE_lp-n-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20190913_FIRE_lp-n-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20190913_FIRE_lp-n-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200813_SkylineFire_sp-n-mobo-c-pre0-w0 | 0/unknown | 11 / 8 | applied | 0.44 | 0.41 | no/abstain | no/abstain | fog_near_threshold |
| 20200813_SkylineFire_sp-n-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.53 | 0.61 | abstain/yes | no/yes | smoke_near_threshold |
| 20200813_SkylineFire_sp-n-mobo-c-post1-w0 | 1/unknown | 12 / 8 | applied | 0.52 | 0.58 | abstain/yes | no/yes | smoke_near_threshold |
| 20180603_FIRE_sm-n-mobo-c-pre0-w0 | 0/unknown | 11 / 8 | applied | 0.51 | 0.48 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20180603_FIRE_sm-n-mobo-c-pre1-w0 | 0/unknown | 11 / 4 | applied | 0.60 | 0.58 | yes/yes | abstain/yes | - |
| 20180603_FIRE_sm-n-mobo-c-post0-w0 | 1/unknown | 12 / 5 | applied | 0.44 | 0.45 | no/abstain | no/abstain | fog_near_threshold |
| 20180603_FIRE_sm-n-mobo-c-post1-w0 | 1/unknown | 3 / 0 | applied | 0.54 | 0.46 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20230727.144836-BonnyFire-hp-e-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.60 | 0.64 | yes/yes | abstain/yes | - |
| 20230727.144836-BonnyFire-hp-e-mobo-c-pre1-w0 | 0/unknown | 12 / 8 | applied | 0.61 | 0.63 | yes/yes | abstain/yes | - |
| 20230727.144836-BonnyFire-hp-e-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.56 | 0.65 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20230727.144836-BonnyFire-hp-e-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.58 | 0.64 | yes/yes | abstain/yes | - |
| 20200521_FIRE_om-n-mobo-c-pre0-w0 | 0/unknown | 8 / 6 | applied | 0.55 | 0.56 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20200521_FIRE_om-n-mobo-c-pre1-w0 | 0/unknown | 8 / 6 | applied | 0.49 | 0.50 | abstain/yes | no/yes | smoke_near_threshold |
| 20200521_FIRE_om-n-mobo-c-post0-w0 | 1/unknown | 8 / 4 | applied | 0.58 | 0.58 | yes/yes | abstain/yes | - |
| 20200521_FIRE_om-n-mobo-c-post1-w0 | 1/unknown | 5 / 2 | applied | 0.52 | 0.47 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20190712_RockHouse_wc-e-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20190712_RockHouse_wc-e-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20190712_RockHouse_wc-e-mobo-c-post0-w0 | 1/unknown | 3 / 2 | applied | 0.53 | 0.45 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20190712_RockHouse_wc-e-mobo-c-post1-w0 | 1/unknown | 4 / 3 | applied | 0.40 | 0.32 | no/no | no/no | - |
| 20260629_JunctionFire_mg-e-mobo-c-pre0-w0 | 0/unknown | 12 / 3 | applied | 0.44 | 0.37 | no/abstain | no/abstain | fog_near_threshold |
| 20260629_JunctionFire_mg-e-mobo-c-pre1-w0 | 0/unknown | 12 / 1 | applied | 0.48 | 0.43 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20260629_JunctionFire_mg-e-mobo-c-post0-w0 | 1/unknown | 12 / 2 | applied | 0.44 | 0.40 | no/abstain | no/abstain | fog_near_threshold |
| 20260629_JunctionFire_mg-e-mobo-c-post1-w0 | 1/unknown | 12 / 3 | applied | 0.47 | 0.45 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20250122_BernardoFire_ch-s-mobo-c-pre0-w0 | 0/unknown | 12 / 5 | applied | 0.58 | 0.60 | yes/yes | abstain/yes | - |
| 20250122_BernardoFire_ch-s-mobo-c-pre1-w0 | 0/unknown | 12 / 2 | applied | 0.57 | 0.54 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20250122_BernardoFire_ch-s-mobo-c-post0-w0 | 1/unknown | 12 / 6 | applied | 0.57 | 0.62 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20250122_BernardoFire_ch-s-mobo-c-post1-w0 | 1/unknown | 12 / 8 | applied | 0.56 | 0.63 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20220210-EmeraldFire-marconi-w-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20220210-EmeraldFire-marconi-w-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20220210-EmeraldFire-marconi-w-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20220210-EmeraldFire-marconi-w-mobo-c-post1-w0 | 1/unknown | 3 / 2 | applied | 0.40 | 0.32 | no/no | no/no | - |
| 20171010_FIRE_rm-e-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.46 | 0.41 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20171010_FIRE_rm-e-mobo-c-pre1-w0 | 0/unknown | 10 / 8 | applied | 0.46 | 0.42 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20171010_FIRE_rm-e-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.48 | 0.47 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20171010_FIRE_rm-e-mobo-c-post1-w0 | 1/unknown | 12 / 6 | applied | 0.43 | 0.35 | no/no | no/no | - |
| 20170708_Whittier_syp-n-mobo-m-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170708_Whittier_syp-n-mobo-m-pre1-w0 | 0/unknown | 6 / 0 | applied | 0.52 | 0.43 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170708_Whittier_syp-n-mobo-m-post0-w0 | 1/unknown | 7 / 4 | applied | 0.45 | 0.40 | no/abstain | no/abstain | fog_near_threshold |
| 20170708_Whittier_syp-n-mobo-m-post1-w0 | 1/unknown | 12 / 7 | applied | 0.42 | 0.38 | no/abstain | no/abstain | fog_near_threshold |
| 20230809.111733-BunnieFire-cp-w-mobo-c-pre0-w0 | 0/unknown | 11 / 0 | applied | 0.52 | 0.44 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20230809.111733-BunnieFire-cp-w-mobo-c-pre1-w0 | 0/unknown | 2 / 0 | applied | 0.49 | 0.39 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20230809.111733-BunnieFire-cp-w-mobo-c-post0-w0 | 1/unknown | 9 / 0 | applied | 0.55 | 0.47 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20230809.111733-BunnieFire-cp-w-mobo-c-post1-w0 | 1/unknown | 4 / 0 | applied | 0.41 | 0.33 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180806_FIRE_vo-w-mobo-c-pre0-w0 | 0/unknown | 5 / 0 | applied | 0.44 | 0.35 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180806_FIRE_vo-w-mobo-c-pre1-w0 | 0/unknown | 7 / 0 | applied | 0.45 | 0.36 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180806_FIRE_vo-w-mobo-c-post0-w0 | 1/unknown | 4 / 0 | applied | 0.45 | 0.36 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180806_FIRE_vo-w-mobo-c-post1-w0 | 1/unknown | 5 / 0 | applied | 0.48 | 0.38 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20220302-Jimfire-0921-stgo-e-mobo-c-pre0-w0 | 0/unknown | 10 / 3 | applied | 0.53 | 0.45 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20220302-Jimfire-0921-stgo-e-mobo-c-pre1-w0 | 0/unknown | 4 / 2 | applied | 0.40 | 0.31 | no/no | no/no | - |
| 20220302-Jimfire-0921-stgo-e-mobo-c-post0-w0 | 1/unknown | 10 / 8 | applied | 0.57 | 0.55 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20220302-Jimfire-0921-stgo-e-mobo-c-post1-w0 | 1/unknown | 10 / 8 | applied | 0.55 | 0.54 | abstain/yes | no/yes | smoke_near_threshold |
| 20170708_Whittier_syp-n-mobo-c-pre0-w0 | 0/unknown | 12 / 0 | applied | 0.47 | 0.44 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170708_Whittier_syp-n-mobo-c-pre1-w0 | 0/unknown | 5 / 0 | applied | 0.42 | 0.34 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170708_Whittier_syp-n-mobo-c-post0-w0 | 1/unknown | 11 / 3 | applied | 0.45 | 0.41 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20170708_Whittier_syp-n-mobo-c-post1-w0 | 1/unknown | 12 / 9 | applied | 0.47 | 0.49 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20191005_FIRE_vo-n-mobo-c-pre0-w0 | 0/unknown | 7 / 0 | applied | 0.45 | 0.37 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20191005_FIRE_vo-n-mobo-c-pre1-w0 | 0/unknown | 8 / 6 | applied | 0.56 | 0.57 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20191005_FIRE_vo-n-mobo-c-post0-w0 | 1/unknown | 9 / 0 | applied | 0.53 | 0.48 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180522_FIRE_rm-e-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.56 | 0.65 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20180522_FIRE_rm-e-mobo-c-pre1-w0 | 0/unknown | 12 / 9 | applied | 0.53 | 0.63 | abstain/yes | no/yes | smoke_near_threshold |
| 20180522_FIRE_rm-e-mobo-c-post0-w0 | 1/unknown | 12 / 9 | applied | 0.53 | 0.62 | abstain/yes | no/yes | smoke_near_threshold |
| 20180522_FIRE_rm-e-mobo-c-post1-w0 | 1/unknown | 12 / 9 | applied | 0.57 | 0.65 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20240909_BridgeFire_marconi-n-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20240909_BridgeFire_marconi-n-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20240909_BridgeFire_marconi-n-mobo-c-post0-w0 | 1/unknown | 12 / 7 | applied | 0.67 | 0.58 | yes/yes | yes/yes | - |
| 20240909_BridgeFire_marconi-n-mobo-c-post1-w0 | 1/unknown | 12 / 0 | applied | 0.61 | 0.57 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20230809.111832-BunnieFire-cp-w-mobo-c-pre0-w0 | 0/unknown | 12 / 0 | applied | 0.51 | 0.45 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20230809.111832-BunnieFire-cp-w-mobo-c-pre1-w0 | 0/unknown | 3 / 0 | applied | 0.49 | 0.41 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20230809.111832-BunnieFire-cp-w-mobo-c-post0-w0 | 1/unknown | 9 / 0 | applied | 0.55 | 0.47 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20230809.111832-BunnieFire-cp-w-mobo-c-post1-w0 | 1/unknown | 4 / 0 | applied | 0.41 | 0.33 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20231026.103052-prescribed-vo-w-mobo-c-pre0-w0 | 0/unknown | 9 / 7 | applied | 0.38 | 0.33 | no/no | no/no | - |
| 20231026.103052-prescribed-vo-w-mobo-c-pre1-w0 | 0/unknown | 10 / 0 | applied | 0.42 | 0.35 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20231026.103052-prescribed-vo-w-mobo-c-post0-w0 | 1/unknown | 11 / 0 | applied | 0.56 | 0.52 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20231026.103052-prescribed-vo-w-mobo-c-post1-w0 | 1/unknown | 12 / 0 | applied | 0.50 | 0.47 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20220302-Jimfire-1101-stgo-e-mobo-c-pre0-w0 | 0/unknown | 12 / 8 | applied | 0.60 | 0.61 | yes/yes | abstain/yes | - |
| 20220302-Jimfire-1101-stgo-e-mobo-c-pre1-w0 | 0/unknown | 12 / 8 | applied | 0.63 | 0.64 | yes/yes | abstain/yes | - |
| 20220302-Jimfire-1101-stgo-e-mobo-c-post0-w0 | 1/unknown | 12 / 9 | applied | 0.57 | 0.56 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20220302-Jimfire-1101-stgo-e-mobo-c-post1-w0 | 1/unknown | 12 / 10 | applied | 0.61 | 0.62 | yes/yes | abstain/yes | - |
| 20260629_JunctionFire_vo-n-mobo-c-pre0-w0 | 0/unknown | 12 / 2 | applied | 0.42 | 0.37 | no/abstain | no/abstain | fog_near_threshold |
| 20260629_JunctionFire_vo-n-mobo-c-pre1-w0 | 0/unknown | 12 / 5 | applied | 0.52 | 0.45 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20260629_JunctionFire_vo-n-mobo-c-post0-w0 | 1/unknown | 12 / 5 | applied | 0.47 | 0.50 | abstain/yes | no/yes | smoke_near_threshold |
| 20260629_JunctionFire_vo-n-mobo-c-post1-w0 | 1/unknown | 12 / 2 | applied | 0.49 | 0.52 | abstain/yes | no/yes | smoke_near_threshold |
| 20240905_ValleyFire_cp-s-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.57 | 0.63 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20240905_ValleyFire_cp-s-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.50 | 0.55 | abstain/yes | no/yes | smoke_near_threshold |
| 20240905_ValleyFire_cp-s-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.59 | 0.57 | yes/yes | abstain/yes | - |
| 20240905_ValleyFire_cp-s-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.60 | 0.65 | yes/yes | abstain/yes | - |
| 20180919_FIRE_rm-e-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.45 | 0.39 | no/abstain | no/abstain | fog_near_threshold |
| 20180919_FIRE_rm-e-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.44 | 0.39 | no/abstain | no/abstain | fog_near_threshold |
| 20180919_FIRE_rm-e-mobo-c-post0-w0 | 1/unknown | 12 / 5 | applied | 0.43 | 0.36 | no/abstain | no/abstain | fog_near_threshold |
| 20180919_FIRE_rm-e-mobo-c-post1-w0 | 1/unknown | 12 / 5 | applied | 0.43 | 0.37 | no/abstain | no/abstain | fog_near_threshold |
| 20231026.103933-prescribed-cp-w-mobo-c-pre0-w0 | 0/unknown | 11 / 0 | applied | 0.53 | 0.46 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20231026.103933-prescribed-cp-w-mobo-c-pre1-w0 | 0/unknown | 11 / 0 | applied | 0.54 | 0.50 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20231026.103933-prescribed-cp-w-mobo-c-post0-w0 | 1/unknown | 11 / 0 | applied | 0.51 | 0.45 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20231026.103933-prescribed-cp-w-mobo-c-post1-w0 | 1/unknown | 6 / 0 | applied | 0.47 | 0.39 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20231027.095451-prescribed-vo-w-mobo-c-pre0-w0 | 0/unknown | 11 / 0 | applied | 0.43 | 0.35 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20231027.095451-prescribed-vo-w-mobo-c-pre1-w0 | 0/unknown | 12 / 1 | applied | 0.53 | 0.46 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20231027.095451-prescribed-vo-w-mobo-c-post0-w0 | 1/unknown | 6 / 1 | applied | 0.42 | 0.35 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20231027.095451-prescribed-vo-w-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20220905-FairviewFire-stgo-e-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.55 | 0.61 | abstain/yes | no/yes | smoke_near_threshold |
| 20220905-FairviewFire-stgo-e-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.51 | 0.58 | abstain/yes | no/yes | smoke_near_threshold |
| 20220905-FairviewFire-stgo-e-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.54 | 0.56 | abstain/yes | no/yes | smoke_near_threshold |
| 20220905-FairviewFire-stgo-e-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.55 | 0.57 | abstain/yes | no/yes | smoke_near_threshold |
| 20200226_FIRE_rm-e-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200226_FIRE_rm-e-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200226_FIRE_rm-e-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200226_FIRE_rm-e-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20250617_MonteFire_cp-w-mobo-c-pre0-w0 | 0/unknown | 9 / 7 | applied | 0.61 | 0.54 | yes/yes | abstain/yes | - |
| 20250617_MonteFire_cp-w-mobo-c-pre1-w0 | 0/unknown | 10 / 9 | applied | 0.39 | 0.34 | no/no | no/no | - |
| 20250617_MonteFire_cp-w-mobo-c-post0-w0 | 1/unknown | 4 / 2 | applied | 0.48 | 0.45 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20250617_MonteFire_cp-w-mobo-c-post1-w0 | 1/unknown | 2 / 1 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20260629_JunctionFire_vo-w-mobo-c-pre0-w0 | 0/unknown | 12 / 4 | applied | 0.54 | 0.62 | abstain/yes | no/yes | smoke_near_threshold |
| 20260629_JunctionFire_vo-w-mobo-c-pre1-w0 | 0/unknown | 12 / 5 | applied | 0.60 | 0.64 | yes/yes | abstain/yes | - |
| 20260629_JunctionFire_vo-w-mobo-c-post0-w0 | 1/unknown | 12 / 1 | applied | 0.53 | 0.60 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20260629_JunctionFire_vo-w-mobo-c-post1-w0 | 1/unknown | 12 / 0 | applied | 0.53 | 0.60 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20240909_BridgeFire_stgo-e-mobo-c-pre0-w0 | 0/unknown | 9 / 5 | applied | 0.60 | 0.47 | yes/abstain | abstain/abstain | fog_near_threshold |
| 20240909_BridgeFire_stgo-e-mobo-c-pre1-w0 | 0/unknown | 9 / 3 | applied | 0.55 | 0.48 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20240909_BridgeFire_stgo-e-mobo-c-post0-w0 | 1/unknown | 11 / 6 | applied | 0.49 | 0.51 | abstain/yes | no/yes | smoke_near_threshold |
| 20240909_BridgeFire_stgo-e-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.59 | 0.63 | yes/yes | abstain/yes | - |
| 20200615_Rainbow_rm-e-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200615_Rainbow_rm-e-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200615_Rainbow_rm-e-mobo-c-post0-w0 | 1/unknown | 12 / 10 | applied | 0.60 | 0.65 | yes/yes | abstain/yes | - |
| 20200615_Rainbow_rm-e-mobo-c-post1-w0 | 1/unknown | 12 / 7 | applied | 0.46 | 0.46 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20260722_CreelmanFire_cp-w-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20260722_CreelmanFire_cp-w-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20260722_CreelmanFire_cp-w-mobo-c-post0-w0 | 1/unknown | 2 / 1 | applied | 0.61 | 0.46 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20260722_CreelmanFire_cp-w-mobo-c-post1-w0 | 1/unknown | 10 / 7 | applied | 0.66 | 0.60 | yes/yes | abstain/yes | - |
| 20240706_StewartFire_rm-e-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20240706_StewartFire_rm-e-mobo-c-pre1-w0 | 0/unknown | 2 / 1 | applied | 0.41 | 0.33 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20240706_StewartFire_rm-e-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20240706_StewartFire_rm-e-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20240908_RiceFire_rm-e-mobo-c-pre0-w0 | 0/unknown | 8 / 3 | applied | 0.56 | 0.56 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20240908_RiceFire_rm-e-mobo-c-pre1-w0 | 0/unknown | 12 / 4 | applied | 0.66 | 0.54 | yes/yes | abstain/yes | - |
| 20240908_RiceFire_rm-e-mobo-c-post0-w0 | 1/unknown | 11 / 3 | applied | 0.56 | 0.55 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20240908_RiceFire_rm-e-mobo-c-post1-w0 | 1/unknown | 7 / 0 | applied | 0.50 | 0.42 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20241202_mulch_fire_rm-e-mobo-c-pre0-w0 | 0/unknown | 8 / 4 | applied | 0.49 | 0.50 | abstain/yes | no/yes | smoke_near_threshold |
| 20241202_mulch_fire_rm-e-mobo-c-pre1-w0 | 0/unknown | 9 / 8 | applied | 0.53 | 0.42 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20241202_mulch_fire_rm-e-mobo-c-post0-w0 | 1/unknown | 10 / 5 | applied | 0.48 | 0.44 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20241202_mulch_fire_rm-e-mobo-c-post1-w0 | 1/unknown | 10 / 4 | applied | 0.55 | 0.53 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20260408_RainbowFir_rm-e-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20260408_RainbowFir_rm-e-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20260408_RainbowFir_rm-e-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20260408_RainbowFir_rm-e-mobo-c-post1-w0 | 1/unknown | 12 / 0 | applied | 0.58 | 0.53 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20260502_MissionFire_rm-e-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.53 | 0.62 | abstain/yes | no/yes | smoke_near_threshold |
| 20260502_MissionFire_rm-e-mobo-c-pre1-w0 | 0/unknown | 12 / 9 | applied | 0.55 | 0.62 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20260502_MissionFire_rm-e-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.54 | 0.65 | abstain/yes | no/yes | smoke_near_threshold |
| 20260502_MissionFire_rm-e-mobo-c-post1-w0 | 1/unknown | 12 / 9 | applied | 0.54 | 0.62 | abstain/yes | no/yes | smoke_near_threshold |
| 20260722_RainbowFire_rm-e-mobo-c-pre0-w0 | 0/unknown | 4 / 3 | applied | 0.43 | 0.37 | no/abstain | no/abstain | fog_near_threshold |
| 20260722_RainbowFire_rm-e-mobo-c-pre1-w0 | 0/unknown | 8 / 2 | applied | 0.51 | 0.48 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20260722_RainbowFire_rm-e-mobo-c-post0-w0 | 1/unknown | 10 / 6 | applied | 0.55 | 0.49 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20260722_RainbowFire_rm-e-mobo-c-post1-w0 | 1/unknown | 10 / 1 | applied | 0.45 | 0.39 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180602_Alison_sp-w-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180602_Alison_sp-w-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180602_Alison_sp-w-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180602_Alison_sp-w-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20190813_FIRE_69bravo-e-mobo-c-pre0-w0 | 0/unknown | 7 / 4 | applied | 0.59 | 0.52 | yes/yes | abstain/yes | - |
| 20190813_FIRE_69bravo-e-mobo-c-pre1-w0 | 0/unknown | 8 / 4 | applied | 0.62 | 0.60 | yes/yes | abstain/yes | - |
| 20190813_FIRE_69bravo-e-mobo-c-post0-w0 | 1/unknown | 11 / 3 | applied | 0.55 | 0.54 | abstain/yes | no/yes | smoke_near_threshold |
| 20190813_FIRE_69bravo-e-mobo-c-post1-w0 | 1/unknown | 11 / 0 | applied | 0.58 | 0.50 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170520_FIRE_lp-s-iqeye-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170520_FIRE_lp-s-iqeye-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170520_FIRE_lp-s-iqeye-post0-w0 | 1/unknown | 12 / 7 | applied | 0.54 | 0.59 | abstain/yes | no/yes | smoke_near_threshold |
| 20170520_FIRE_lp-s-iqeye-post1-w0 | 1/unknown | 12 / 7 | applied | 0.62 | 0.60 | yes/yes | abstain/yes | - |
| 20191030_CopperCanyon_om-s-mobo-m-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20191030_CopperCanyon_om-s-mobo-m-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20191030_CopperCanyon_om-s-mobo-m-post0-w0 | 1/unknown | 12 / 10 | applied | 0.52 | 0.54 | abstain/yes | no/yes | smoke_near_threshold |
| 20191030_CopperCanyon_om-s-mobo-m-post1-w0 | 1/unknown | 11 / 7 | applied | 0.44 | 0.40 | no/abstain | no/abstain | fog_near_threshold |
| 20220831-Border32fire-pi-s-mobo-m-post0-w0 | 1/unknown | 12 / 11 | applied | 0.54 | 0.60 | abstain/yes | no/yes | smoke_near_threshold |
| 20220831-Border32fire-pi-s-mobo-m-post1-w0 | 1/unknown | 12 / 2 | applied | 0.60 | 0.59 | yes/yes | abstain/yes | - |
| 20190803_OtaySouth_om-s-mobo-pre0-w0 | 0/unknown | 8 / 1 | applied | 0.50 | 0.42 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20190803_OtaySouth_om-s-mobo-pre1-w0 | 0/unknown | 12 / 0 | applied | 0.41 | 0.33 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20190803_OtaySouth_om-s-mobo-post0-w0 | 1/unknown | 11 / 9 | applied | 0.64 | 0.54 | yes/yes | abstain/yes | - |
| 20190803_OtaySouth_om-s-mobo-post1-w0 | 1/unknown | 2 / 1 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200806_SpringsFire_lp-w-mobo-c-post0-w0 | 1/unknown | 10 / 5 | applied | 0.55 | 0.53 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20200806_SpringsFire_lp-w-mobo-c-post1-w0 | 1/unknown | 12 / 3 | applied | 0.54 | 0.49 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20170520_FIRE_pi-w-mobo-c-pre0-w0 | 0/unknown | 11 / 8 | applied | 0.46 | 0.44 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20170520_FIRE_pi-w-mobo-c-pre1-w0 | 0/unknown | 11 / 2 | applied | 0.55 | 0.48 | abstain/abstain | abstain/abstain | smoke_near_threshold, fog_near_threshold |
| 20170520_FIRE_pi-w-mobo-c-post0-w0 | 1/unknown | 9 / 7 | applied | 0.58 | 0.54 | yes/yes | abstain/yes | - |
| 20170520_FIRE_pi-w-mobo-c-post1-w0 | 1/unknown | 10 / 8 | applied | 0.53 | 0.54 | abstain/yes | no/yes | smoke_near_threshold |
| 20260909_GettyFire_wilson-w-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.59 | 0.66 | yes/yes | abstain/yes | - |
| 20260909_GettyFire_wilson-w-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.59 | 0.66 | yes/yes | abstain/yes | - |
| 20260909_GettyFire_wilson-w-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.58 | 0.65 | yes/yes | abstain/yes | - |
| 20260909_GettyFire_wilson-w-mobo-c-post1-w0 | 1/unknown | 12 / 7 | applied | 0.59 | 0.65 | yes/yes | abstain/yes | - |
| 20200727_Border11Fire_lp-s-mobo-m-pre0-w0 | 0/unknown | 10 / 6 | applied | 0.56 | 0.47 | abstain/abstain | abstain/abstain | smoke_near_threshold, fog_near_threshold |
| 20200727_Border11Fire_lp-s-mobo-m-pre1-w0 | 0/unknown | 11 / 1 | applied | 0.57 | 0.57 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20200727_Border11Fire_lp-s-mobo-m-post0-w0 | 1/unknown | 8 / 1 | applied | 0.46 | 0.41 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20200727_Border11Fire_lp-s-mobo-m-post1-w0 | 1/unknown | 5 / 0 | applied | 0.42 | 0.36 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170722_FIRE_so-s-mobo-c-pre0-w0 | 0/unknown | 9 / 7 | applied | 0.64 | 0.61 | yes/yes | abstain/yes | - |
| 20170722_FIRE_so-s-mobo-c-pre1-w0 | 0/unknown | 5 / 0 | applied | 0.43 | 0.35 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170722_FIRE_so-s-mobo-c-post0-w0 | 1/unknown | 11 / 7 | applied | 0.60 | 0.61 | yes/yes | abstain/yes | - |
| 20170722_FIRE_so-s-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.62 | 0.63 | yes/yes | abstain/yes | - |
| 20191007_FIRE_sm-s-mobo-c-pre0-w0 | 0/unknown | 11 / 0 | applied | 0.44 | 0.37 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20191007_FIRE_sm-s-mobo-c-pre1-w0 | 0/unknown | 12 / 10 | applied | 0.44 | 0.40 | no/abstain | no/abstain | fog_near_threshold |
| 20191007_FIRE_sm-s-mobo-c-post0-w0 | 1/unknown | 9 / 0 | applied | 0.44 | 0.35 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20191007_FIRE_sm-s-mobo-c-post1-w0 | 1/unknown | 10 / 0 | applied | 0.44 | 0.36 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20160604_FIRE_smer-tcs3-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20160604_FIRE_smer-tcs3-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20160604_FIRE_smer-tcs3-mobo-c-post0-w0 | 1/unknown | 12 / 7 | applied | 0.65 | 0.63 | yes/yes | abstain/yes | - |
| 20160604_FIRE_smer-tcs3-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.55 | 0.62 | abstain/yes | no/yes | smoke_near_threshold |
| 20170711_FIRE_bl-e-mobo-c-pre0-w0 | 0/unknown | 9 / 0 | applied | 0.46 | 0.40 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170711_FIRE_bl-e-mobo-c-pre1-w0 | 0/unknown | 7 / 0 | applied | 0.46 | 0.39 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170711_FIRE_bl-e-mobo-c-post0-w0 | 1/unknown | 10 / 0 | applied | 0.45 | 0.37 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20170711_FIRE_bl-e-mobo-c-post1-w0 | 1/unknown | 10 / 0 | applied | 0.61 | 0.57 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20240802_WarnersFire_sojr-s-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.52 | 0.53 | abstain/yes | no/yes | smoke_near_threshold |
| 20240802_WarnersFire_sojr-s-mobo-c-pre1-w0 | 0/unknown | 12 / 10 | applied | 0.50 | 0.58 | abstain/yes | no/yes | smoke_near_threshold |
| 20240802_WarnersFire_sojr-s-mobo-c-post0-w0 | 1/unknown | 12 / 8 | applied | 0.57 | 0.57 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20240802_WarnersFire_sojr-s-mobo-c-post1-w0 | 1/unknown | 12 / 9 | applied | 0.55 | 0.59 | abstain/yes | no/yes | smoke_near_threshold |
| 20170821_FIRE_lo-s-mobo-c-pre0-w0 | 0/unknown | 2 / 0 | applied | 0.41 | 0.35 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170821_FIRE_lo-s-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170821_FIRE_lo-s-mobo-c-post0-w0 | 1/unknown | 2 / 0 | applied | 0.57 | 0.49 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20170821_FIRE_lo-s-mobo-c-post1-w0 | 1/unknown | 3 / 2 | applied | 0.41 | 0.33 | no/no | no/no | - |
| 20180806_Holy_sp-s-mobo-m-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180806_Holy_sp-s-mobo-m-post0-w0 | 1/unknown | 12 / 11 | applied | 0.63 | 0.59 | yes/yes | abstain/yes | - |
| 20200521_FIRE_om-w-mobo-c-pre0-w0 | 0/unknown | 2 / 1 | applied | 0.57 | 0.43 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200521_FIRE_om-w-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200521_FIRE_om-w-mobo-c-post0-w0 | 1/unknown | 8 / 0 | applied | 0.59 | 0.51 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20200521_FIRE_om-w-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20220905-FairviewFire-smer-tcs3-mobo-m-pre0-w0 | 0/unknown | 12 / 3 | applied | 0.43 | 0.35 | no/abstain | no/abstain | fog_near_threshold |
| 20220905-FairviewFire-smer-tcs3-mobo-m-pre1-w0 | 0/unknown | 12 / 7 | applied | 0.52 | 0.56 | abstain/yes | no/yes | smoke_near_threshold |
| 20220905-FairviewFire-smer-tcs3-mobo-m-post0-w0 | 1/unknown | 12 / 3 | applied | 0.58 | 0.57 | yes/yes | abstain/yes | - |
| 20220905-FairviewFire-smer-tcs3-mobo-m-post1-w0 | 1/unknown | 12 / 5 | applied | 0.57 | 0.57 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20191005_FIRE_bm-e-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20191005_FIRE_bm-e-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20191005_FIRE_bm-e-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20191005_FIRE_bm-e-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180602_Alison_sp-s-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180602_Alison_sp-s-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180602_Alison_sp-s-mobo-c-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180602_Alison_sp-s-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20231110.112257-Border36Fire-om-w-mobo-m-pre0-w0 | 0/unknown | 12 / 7 | applied | 0.58 | 0.64 | yes/yes | abstain/yes | - |
| 20231110.112257-Border36Fire-om-w-mobo-m-pre1-w0 | 0/unknown | 12 / 7 | applied | 0.56 | 0.62 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20231110.112257-Border36Fire-om-w-mobo-m-post0-w0 | 1/unknown | 12 / 9 | applied | 0.60 | 0.60 | yes/yes | abstain/yes | - |
| 20231110.112257-Border36Fire-om-w-mobo-m-post1-w0 | 1/unknown | 11 / 2 | applied | 0.48 | 0.45 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20180718_FIRE_syp-w-mobo-c-pre0-w0 | 0/unknown | 11 / 0 | applied | 0.49 | 0.40 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180718_FIRE_syp-w-mobo-c-pre1-w0 | 0/unknown | 12 / 1 | applied | 0.58 | 0.60 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180718_FIRE_syp-w-mobo-c-post0-w0 | 1/unknown | 10 / 0 | applied | 0.54 | 0.46 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180718_FIRE_syp-w-mobo-c-post1-w0 | 1/unknown | 11 / 2 | applied | 0.60 | 0.62 | yes/yes | abstain/yes | - |
| 20190801_Caliente_om-w-mobo-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20190801_Caliente_om-w-mobo-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20190801_Caliente_om-w-mobo-post0-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20190801_Caliente_om-w-mobo-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20240625_DelMarFire_tdlln-n-mobo-c-pre0-w0 | 0/unknown | 11 / 2 | applied | 0.57 | 0.53 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20240625_DelMarFire_tdlln-n-mobo-c-pre1-w0 | 0/unknown | 9 / 0 | applied | 0.57 | 0.49 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20240625_DelMarFire_tdlln-n-mobo-c-post0-w0 | 1/unknown | 11 / 3 | applied | 0.41 | 0.39 | no/abstain | no/abstain | fog_near_threshold |
| 20240625_DelMarFire_tdlln-n-mobo-c-post1-w0 | 1/unknown | 9 / 1 | applied | 0.55 | 0.46 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20240903_NavajoFire_cp-n-mobo-c-pre0-w0 | 0/unknown | 3 / 2 | applied | 0.61 | 0.57 | yes/yes | abstain/yes | - |
| 20240903_NavajoFire_cp-n-mobo-c-pre1-w0 | 0/unknown | 7 / 0 | applied | 0.58 | 0.49 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20240903_NavajoFire_cp-n-mobo-c-post0-w0 | 1/unknown | 4 / 0 | applied | 0.46 | 0.38 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20240903_NavajoFire_cp-n-mobo-c-post1-w0 | 1/unknown | 7 / 0 | applied | 0.60 | 0.51 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20200813_Ranch2Fire_sjh-n-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.55 | 0.64 | abstain/yes | no/yes | smoke_near_threshold |
| 20200813_Ranch2Fire_sjh-n-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.57 | 0.65 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20200813_Ranch2Fire_sjh-n-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.56 | 0.64 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20200813_Ranch2Fire_sjh-n-mobo-c-post1-w0 | 1/unknown | 12 / 9 | applied | 0.53 | 0.61 | abstain/yes | no/yes | smoke_near_threshold |
| 20160619_FIRE_pi-s-mobo-c-pre0-w0 | 0/unknown | 12 / 9 | applied | 0.49 | 0.44 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20160619_FIRE_pi-s-mobo-c-pre1-w0 | 0/unknown | 6 / 4 | applied | 0.41 | 0.34 | no/no | no/no | - |
| 20160619_FIRE_pi-s-mobo-c-post0-w0 | 1/unknown | 8 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20160619_FIRE_pi-s-mobo-c-post1-w0 | 1/unknown | 6 / 4 | applied | 0.51 | 0.49 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20200806_SpringsFire_lp-w-mobo-m-post0-w0 | 1/unknown | 11 / 3 | applied | 0.59 | 0.58 | yes/yes | abstain/yes | - |
| 20250107_EatonFire_wilson-s-mobo-c-pre0-w0 | 0/unknown | 3 / 0 | applied | 0.41 | 0.34 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20250107_EatonFire_wilson-s-mobo-c-pre1-w0 | 0/unknown | 7 / 2 | applied | 0.55 | 0.47 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20250107_EatonFire_wilson-s-mobo-c-post0-w0 | 1/unknown | 12 / 7 | applied | 0.55 | 0.50 | abstain/yes | no/yes | smoke_near_threshold |
| 20250107_EatonFire_wilson-s-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.58 | 0.63 | yes/yes | abstain/yes | - |
| 20180603_FIRE_sm-w-mobo-c-pre0-w0 | 0/unknown | 7 / 3 | applied | 0.47 | 0.38 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20180603_FIRE_sm-w-mobo-c-pre1-w0 | 0/unknown | 2 / 1 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180603_FIRE_sm-w-mobo-c-post0-w0 | 1/unknown | 8 / 1 | applied | 0.46 | 0.39 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180603_FIRE_sm-w-mobo-c-post1-w0 | 1/unknown | 3 / 1 | applied | 0.59 | 0.41 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180726_FIRE_so-w-mobo-c-pre0-w0 | 0/unknown | 2 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180726_FIRE_so-w-mobo-c-pre1-w0 | 0/unknown | 4 / 2 | applied | 0.44 | 0.38 | no/abstain | no/abstain | fog_near_threshold |
| 20180726_FIRE_so-w-mobo-c-post0-w0 | 1/unknown | 2 / 1 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20180726_FIRE_so-w-mobo-c-post1-w0 | 1/unknown | 2 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20190924_FIRE_lo-w-mobo-c-pre0-w0 | 0/unknown | 6 / 4 | applied | 0.50 | 0.41 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20190924_FIRE_lo-w-mobo-c-pre1-w0 | 0/unknown | 6 / 4 | applied | 0.52 | 0.50 | abstain/yes | no/yes | smoke_near_threshold |
| 20190924_FIRE_lo-w-mobo-c-post0-w0 | 1/unknown | 7 / 3 | applied | 0.53 | 0.47 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20190924_FIRE_lo-w-mobo-c-post1-w0 | 1/unknown | 10 / 0 | applied | 0.58 | 0.52 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20201205_typical-range-fire_sclm-e-mobo-c-pre0-w0 | 0/unknown | 12 / 8 | applied | 0.53 | 0.55 | abstain/yes | no/yes | smoke_near_threshold |
| 20201205_typical-range-fire_sclm-e-mobo-c-pre1-w0 | 0/unknown | 11 / 3 | applied | 0.57 | 0.57 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20201205_typical-range-fire_sclm-e-mobo-c-post0-w0 | 1/unknown | 12 / 2 | applied | 0.50 | 0.55 | abstain/yes | no/yes | smoke_near_threshold |
| 20201205_typical-range-fire_sclm-e-mobo-c-post1-w0 | 1/unknown | 12 / 8 | applied | 0.54 | 0.61 | abstain/yes | no/yes | smoke_near_threshold |
| 20170520_FIRE_om-s-mobo-c-pre0-w0 | 0/unknown | 3 / 2 | applied | 0.60 | 0.54 | yes/yes | abstain/yes | - |
| 20170520_FIRE_om-s-mobo-c-post0-w0 | 1/unknown | 6 / 4 | applied | 0.62 | 0.56 | yes/yes | abstain/yes | - |
| 20170520_FIRE_om-s-mobo-c-post1-w0 | 1/unknown | 7 / 5 | applied | 0.62 | 0.57 | yes/yes | abstain/yes | - |
| 20201202_BondFire-nightime_sp-w-mobo-c-pre0-w0 | 0/unknown | 1 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20201202_BondFire-nightime_sp-w-mobo-c-pre1-w0 | 0/unknown | 2 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20201202_BondFire-nightime_sp-w-mobo-c-post0-w0 | 1/unknown | 6 / 4 | applied | 0.56 | 0.46 | abstain/abstain | abstain/abstain | smoke_near_threshold, fog_near_threshold |
| 20250101_Fire_69bravo-e-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.56 | 0.65 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20250101_Fire_69bravo-e-mobo-c-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.54 | 0.62 | abstain/yes | no/yes | smoke_near_threshold |
| 20250101_Fire_69bravo-e-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.53 | 0.60 | abstain/yes | no/yes | smoke_near_threshold |
| 20250101_Fire_69bravo-e-mobo-c-post1-w0 | 1/unknown | 12 / 11 | applied | 0.50 | 0.54 | abstain/yes | no/yes | smoke_near_threshold |
| 20200822_SloaneFire_lp-w-mobo-c-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.60 | 0.67 | yes/yes | abstain/yes | - |
| 20200822_SloaneFire_lp-w-mobo-c-pre1-w0 | 0/unknown | 12 / 8 | applied | 0.59 | 0.62 | yes/yes | abstain/yes | - |
| 20200822_SloaneFire_lp-w-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.59 | 0.65 | yes/yes | abstain/yes | - |
| 20200822_SloaneFire_lp-w-mobo-c-post1-w0 | 1/unknown | 12 / 5 | applied | 0.65 | 0.59 | yes/yes | abstain/yes | - |
| 20190717_FIRE_pi-w-mobo-c-pre0-w0 | 0/unknown | 4 / 3 | applied | 0.41 | 0.32 | no/no | no/no | - |
| 20190717_FIRE_pi-w-mobo-c-pre1-w0 | 0/unknown | 4 / 2 | applied | 0.52 | 0.38 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20190717_FIRE_pi-w-mobo-c-post0-w0 | 1/unknown | 2 / 1 | applied | 0.50 | 0.37 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20190717_FIRE_pi-w-mobo-c-post1-w0 | 1/unknown | 6 / 3 | applied | 0.50 | 0.50 | abstain/yes | no/yes | smoke_near_threshold |
| 20200828_BorderFire_sm-s-mobo-c-pre0-w0 | 0/unknown | 12 / 9 | applied | 0.62 | 0.55 | yes/yes | abstain/yes | - |
| 20200828_BorderFire_sm-s-mobo-c-pre1-w0 | 0/unknown | 12 / 3 | applied | 0.56 | 0.61 | abstain/yes | abstain/yes | smoke_near_threshold |
| 20200828_BorderFire_sm-s-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.65 | 0.59 | yes/yes | abstain/yes | - |
| 20200828_BorderFire_sm-s-mobo-c-post1-w0 | 1/unknown | 12 / 8 | applied | 0.49 | 0.55 | abstain/yes | no/yes | smoke_near_threshold |
| 20180727_FIRE_bl-e-mobo-c-pre0-w0 | 0/unknown | 10 / 2 | applied | 0.63 | 0.57 | yes/yes | abstain/yes | - |
| 20180727_FIRE_bl-e-mobo-c-pre1-w0 | 0/unknown | 7 / 0 | applied | 0.56 | 0.48 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180727_FIRE_bl-e-mobo-c-post0-w0 | 1/unknown | 12 / 1 | applied | 0.55 | 0.49 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180727_FIRE_bl-e-mobo-c-post1-w0 | 1/unknown | 8 / 6 | applied | 0.48 | 0.47 | abstain/abstain | no/abstain | smoke_near_threshold, fog_near_threshold |
| 20180606_FIRE_lo-s-mobo-c-pre0-w0 | 0/unknown | 12 / 0 | applied | 0.64 | 0.64 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180606_FIRE_lo-s-mobo-c-pre1-w0 | 0/unknown | 12 / 0 | applied | 0.63 | 0.60 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180606_FIRE_lo-s-mobo-c-post0-w0 | 1/unknown | 12 / 0 | applied | 0.57 | 0.56 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20180606_FIRE_lo-s-mobo-c-post1-w0 | 1/unknown | 12 / 0 | applied | 0.58 | 0.56 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20200618_FIRE_om-w-mobo-c-pre0-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200618_FIRE_om-w-mobo-c-pre1-w0 | 0/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20200618_FIRE_om-w-mobo-c-post0-w0 | 1/unknown | 8 / 0 | applied | 0.58 | 0.50 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20200618_FIRE_om-w-mobo-c-post1-w0 | 1/unknown | 0 / 0 | applied | 0.40 | 0.32 | abstain/abstain | abstain/abstain | few_valid_mask_frames, few_valid_flow_frames |
| 20260629_JunctionFire_bm-e-mobo-c-pre0-w0 | 0/unknown | 12 / 10 | applied | 0.42 | 0.36 | no/abstain | no/abstain | fog_near_threshold |
| 20260629_JunctionFire_bm-e-mobo-c-pre1-w0 | 0/unknown | 12 / 1 | applied | 0.59 | 0.54 | abstain/abstain | abstain/abstain | few_valid_flow_frames |
| 20260629_JunctionFire_bm-e-mobo-c-post0-w0 | 1/unknown | 12 / 6 | applied | 0.44 | 0.39 | no/abstain | no/abstain | fog_near_threshold |
| 20260629_JunctionFire_bm-e-mobo-c-post1-w0 | 1/unknown | 12 / 5 | applied | 0.60 | 0.58 | yes/yes | abstain/yes | - |
| 20180806_Holy_sp-s-mobo-c-pre0-w0 | 0/unknown | 3 / 2 | applied | 0.62 | 0.55 | yes/yes | abstain/yes | - |
| 20180806_Holy_sp-s-mobo-c-post0-w0 | 1/unknown | 12 / 11 | applied | 0.62 | 0.60 | yes/yes | abstain/yes | - |
| 20240806_Border68FireMayBeHardToFind_om-w-mobo-m-pre0-w0 | 0/unknown | 12 / 11 | applied | 0.60 | 0.64 | yes/yes | abstain/yes | - |
| 20240806_Border68FireMayBeHardToFind_om-w-mobo-m-pre1-w0 | 0/unknown | 12 / 11 | applied | 0.60 | 0.64 | yes/yes | abstain/yes | - |
| 20240806_Border68FireMayBeHardToFind_om-w-mobo-m-post0-w0 | 1/unknown | 12 / 11 | applied | 0.59 | 0.65 | yes/yes | abstain/yes | - |
| 20240806_Border68FireMayBeHardToFind_om-w-mobo-m-post1-w0 | 1/unknown | 12 / 11 | applied | 0.59 | 0.64 | yes/yes | abstain/yes | - |
