# Zero-shot check on real footage: figlib (test_iid cameras)

> Sanity check, not an accuracy estimate: the exported synthetic model and its synthetic thresholds, no fine-tuning or recalibration on real data (Stage B is reported separately).

HPWREN FIgLib wildfire camera sequences (imagery courtesy of HPWREN, https://www.hpwren.ucsd.edu/), one frame per minute: 12-frame windows entirely before the first visible plume (smoke 0) and starting at least 60 s after it (smoke 1); fog is not annotated (unknown). Splits are camera-disjoint by camera mount; the far-field render ranges were set from image statistics of the train split only (no labels). Masks from the built-in fallback detector; the plumes are early, small and often distant. Only the test_iid cameras (the split Stage B reports on).

171 windows from 46 videos (35 cameras); masks: fallback_detector 171.

**balanced**: abstained (either label) on 129 / 171 windows (few_valid_mask_frames 40, few_valid_flow_frames 76, smoke_near_threshold 46, fog_near_threshold 21).
* smoke: positives 88 (decided 28: 24 detected, 4 missed); negatives 83 (decided 21: 15 false alarms, 6 correct)
* fog: positives 0 (decided 0: 0 detected, 0 missed); negatives 0 (decided 0: 0 false alarms, 0 correct); label unknown on 171

**low_false_alarm**: abstained (either label) on 146 / 171 windows (few_valid_mask_frames 40, few_valid_flow_frames 76, smoke_near_threshold 52, fog_near_threshold 21).
* smoke: positives 88 (decided 24: 0 detected, 24 missed); negatives 83 (decided 19: 0 false alarms, 19 correct)
* fog: positives 0 (decided 0: 0 detected, 0 missed); negatives 0 (decided 0: 0 false alarms, 0 correct); label unknown on 171

Window-level AUROC of the balanced-point probabilities (tiny n, windows of one video are correlated): smoke 0.60 (88 pos / 83 neg windows, 35 / 33 cameras); fog n/a (0 pos / 0 neg windows, 0 / 0 cameras, 171 unknown)

| window | labels smoke/fog | valid mask / flow frames | stabilisation | P(smoke) | P(fog) | balanced smoke/fog | low-false-alarm smoke/fog | abstain reasons (balanced) |
|---|---|---|---|---|---|---|---|---|
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
