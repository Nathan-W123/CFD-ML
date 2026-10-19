# Image statistics, real vs synthetic: FIgLib train cameras (the split the far render ranges were set from)

> Diagnostic of the rendering, not a model input. Real clips are few, so a coverage fraction moves in steps of 1 / n; a statistic is flagged when more than 10% of the real clips fall outside the synthetic P1-P99 range.

* Real: `data/real/figlib/real.csv` rows of split train, one per video, 114 clips (0 skipped), native sizes (H x W) 683x1024, 768x1024; up to 12 frames 60 s apart, 114 clips with at least 3 distinct frames at that spacing for flicker and motion, 0 of them left out of motion because no frame pair registers (phase-correlation response below 0.15).
* Synthetic: `data/synthetic_far`, 200 of 2848 sequences in splits train, val_select, val_calib, test_iid (seed 0), native size 192 x 192, frames 60 s apart, 5 without a registered frame pair.
* Comparable basis: real frames resized to the synthetic pixel count (aspect kept), except blockiness, which exists only on the encoder's native 8x8 grid and is compared at native resolution.

## Comparison (comparable basis)

| statistic | unit | basis | real P5 / P25 / P50 / P75 / P95 | synthetic P5 / P25 / P50 / P75 / P95 | synthetic P1 - P99 | real below / above | verdict |
|---|---|---|---|---|---|---|---|
| noise_sigma | gray levels | resized | 0.575 / 0.8 / 0.986 / 1.16 / 1.71 (n 114) | 0.137 / 0.453 / 0.793 / 2.82 / 7.84 (n 200) | 0.0877 - 8.96 | 0% / 0% | covered |
| contrast | (P95 - P5) / 255 | resized | 0.27 / 0.631 / 0.712 / 0.804 / 0.887 (n 114) | 0.268 / 0.428 / 0.541 / 0.679 / 0.93 (n 200) | 0.18 - 0.992 | 3% / 0% | covered |
| brightness | mean / 255 | resized | 0.0895 / 0.396 / 0.418 / 0.444 / 0.491 (n 114) | 0.166 / 0.3 / 0.425 / 0.555 / 0.776 (n 200) | 0.103 - 0.82 | 6% / 1% | covered |
| sharpness | var(Laplacian) / (P95 - P5)^2 | resized | 0.00481 / 0.0125 / 0.0223 / 0.0349 / 0.141 (n 114) | 0.00415 / 0.00947 / 0.0172 / 0.034 / 0.0805 (n 200) | 0.00292 - 0.135 | 4% / 5% | covered |
| blockiness | boundary / interior gradient | native | 1.06 / 1.08 / 1.13 / 1.2 / 1.37 (n 114) | 0.977 / 1.06 / 1.4 / 1.67 / 2.06 (n 200) | 0.962 - 2.19 | 0% / 0% | covered |
| flicker | std / mean of frame means | resized | 0.00071 / 0.00202 / 0.00408 / 0.0143 / 0.0243 (n 114) | 0.00509 / 0.0127 / 0.0207 / 0.0531 / 0.134 (n 200) | 0.00236 - 0.166 | 29% / 0% | widen_low |
| motion_px | px per frame interval | resized | 0.00848 / 0.0174 / 0.0594 / 0.498 / 0.535 (n 114) | 0.0791 / 0.203 / 0.349 / 4.6 / 21.6 (n 195) | 0.0494 - 51.4 | 43% / 0% | widen_low |

## Real footage at native resolution (reference)

| statistic | real P5 / P25 / P50 / P75 / P95 | outside synthetic P1 - P99 (below / above) |
|---|---|---|
| noise_sigma | 0.342 / 0.472 / 0.586 / 0.796 / 1.41 (n 114) | 1% (1% / 0%) |
| contrast | 0.269 / 0.655 / 0.749 / 0.824 / 0.906 (n 114) | 3% (3% / 0%) |
| brightness | 0.0895 / 0.396 / 0.418 / 0.444 / 0.491 (n 114) | 7% (6% / 1%) |
| sharpness | 0.00297 / 0.00899 / 0.0191 / 0.0322 / 0.116 (n 114) | 11% (5% / 5%) |
| blockiness | 1.06 / 1.08 / 1.13 / 1.2 / 1.37 (n 114) | 0% (0% / 0%) |
| flicker | 0.000707 / 0.00202 / 0.00407 / 0.0143 / 0.0243 (n 114) | 29% (29% / 0%) |
| motion_px | 0.0114 / 0.0184 / 0.0326 / 0.0549 / 0.142 (n 113) | 71% (71% / 0%) |

## What to change in `render:`

* **noise_sigma**: Covered: 0% of 114 real clips fall outside the synthetic P1-P99 range [0.0877, 8.96]; no change needed.
* **contrast**: Covered: 3% of 114 real clips fall outside the synthetic P1-P99 range [0.18, 0.992]; no change needed.
* **brightness**: Covered: 7% of 114 real clips fall outside the synthetic P1-P99 range [0.103, 0.82]; no change needed.
* **sharpness**: Covered: 9% of 114 real clips fall outside the synthetic P1-P99 range [0.00292, 0.135]; no change needed.
* **blockiness**: Covered: 0% of 114 real clips fall outside the synthetic P1-P99 range [0.962, 2.19]; no change needed.
* **flicker**: 29% of 114 real clips fall outside the synthetic P1-P99 range [0.00236, 0.166] (29% below, 0% above). Real clips are steadier than the synthetic ones (real P5 0.001 vs synthetic P1 0.002). render.exposure_flicker already starts at 0, so the synthetic floor comes from the scenes themselves (aerosol growing or drifting through the view changes the frame mean); nothing to widen in the camera model.
* **motion_px**: 43% of 114 real clips fall outside the synthetic P1-P99 range [0.0494, 51.4] (43% below, 0% above). Real cameras are steadier (real P5 0.01 px vs synthetic P1 0.05 px per 60 s). render.camera_shake_px already starts at 0 (now [0, 0.3]) but is drawn uniformly, so few synthetic clips are as steady as a rigid mount: make near-zero shake more common or lengthen render.camera_shake_tau_s (now [1, 30] s) so that frames stay correlated; part of the synthetic floor can also be aerosol motion that the phase correlation follows.

## Do the statistics follow their render parameters? (synthetic, Spearman rank correlation)

| statistic ~ parameter | rho |
|---|---|
| noise_sigma~noise_sigma | 0.38 |
| contrast~contrast | 0.41 |
| contrast~global_haze | -0.19 |
| brightness~brightness_offset | 0.61 |
| sharpness~noise_sigma | 0.41 |
| blockiness~jpeg_strength | 0.89 |
| flicker~exposure_flicker | 0.73 |
| motion_px~camera_shake_px | 0.77 |

## Per real clip (comparable basis; flicker and motion only at the comparison spacing)

| clip | native H x W | frames | spacing (s) | noise_sigma | contrast | brightness | sharpness | blockiness | flicker | motion_px |
|---|---|---|---|---|---|---|---|---|---|---|
| 20220302-Jimfire-0921-stgo-s-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.828 | 0.882 | 0.395 | 0.0132 | 1.14 | 0.00815 | 0.0124 |
| 20180614_FIRE_hp-s-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.55 | 0.741 | 0.418 | 0.0223 | 1.11 | 0.0179 | 0.405 |
| 20200806_SpringsFire_sm-e-mobo-m | 683 x 1024 | 12 | 60 (interval) | 0.659 | 0.682 | 0.353 | 0.0317 | 1.16 | 0.0151 | 0.032 |
| 20241113_Fire_dwpgm-w-mobo-c | 683 x 1024 | 12 | 60 (interval) | 1.09 | 0.722 | 0.413 | 0.0429 | 1.08 | 0.00245 | 0.0131 |
| 20191006_FIRE_lp-e-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.874 | 0.692 | 0.43 | 0.0457 | 1.1 | 0.00149 | 0.0105 |
| 20180727_FIRE_wc-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.8 | 0.49 | 0.439 | 0.0351 | 1.15 | 0.00309 | 0.0121 |
| 20170519_FIRE_rm-w-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.2 | 0.733 | 0.465 | 0.0297 | 1.07 | 0.00275 | 0.493 |
| 20210115_Bonitafire_tp-w-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.0923 | 0.11 | 0.0202 | 0.203 | 1.4 | 0.00305 | 0.0411 |
| 20200306_FIRE_mlo-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 1.35 | 0.731 | 0.399 | 0.0218 | 1.08 | 0.0156 | 0.0553 |
| 20240825_TenajaFire_buff-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.834 | 0.624 | 0.433 | 0.02 | 1.12 | 0.00967 | 0.00691 |
| 20230806.161759-PassFire-ws-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.854 | 0.573 | 0.428 | 0.0315 | 1.13 | 0.00253 | 0.0183 |
| 20170807_FIRE_bh-n-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.3 | 0.694 | 0.482 | 0.0523 | 1.05 | 0.00161 | 0.505 |
| 20170711_FIRE_bm-s-mobo-c | 768 x 1024 | 12 | 60 (interval) | 0.89 | 0.69 | 0.435 | 0.0123 | 1.12 | 0.00454 | 0.501 |
| 20180727_FIRE_mg-w-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.781 | 0.775 | 0.459 | 0.00313 | 1.36 | 0.0159 | 0.0454 |
| 20241209_FranklinFire_69bravo-w-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.76 | 0.192 | 0.0995 | 0.185 | 1.38 | 0.0104 | 0.123 |
| 20180725_Cranston_sp-e-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.808 | 0.576 | 0.386 | 0.0248 | 1.15 | 0.0154 | 0.0061 |
| 20170722_FIRE_mg-n-iqeye | 768 x 1024 | 12 | 60 (interval) | 1.11 | 0.59 | 0.485 | 0.0214 | 1.22 | 0.00331 | 0.503 |
| 20231027.095514-prescribed-bi-s-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.761 | 0.871 | 0.438 | 0.0147 | 1.16 | 0.0167 | 0.0166 |
| 20161113_FIRE_bl-n-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.08 | 0.682 | 0.468 | 0.0296 | 1.07 | 0.00297 | 0.502 |
| 20170713_FIRE_smer-tcs8-mobo-c | 683 x 1024 | 12 | 60 (interval) | 1.29 | 0.812 | 0.532 | 0.0312 | 1.07 | 0.0186 | 0.00791 |
| 20160619_FIRE_om-e-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.07 | 0.841 | 0.451 | 0.00621 | 1.13 | 0.00401 | 0.512 |
| 20251102_ScissorsFire_tp-s-mobo-m | 683 x 1024 | 12 | 60 (interval) | 4.94 | 0.584 | 0.844 | 0.0499 | 1.02 | 0.0216 | 0.0495 |
| 20231114.093653-prescribed-mlo-w-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.854 | 0.792 | 0.403 | 0.0134 | 1.08 | 0.00161 | 0.0106 |
| 20240909_BridgeFire_starr-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.892 | 0.412 | 0.473 | 0.029 | 1.15 | 0.00312 | 0.0233 |
| 20180606_FIRE_ml-s-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.19 | 0.796 | 0.441 | 0.0196 | 1.07 | 0.000737 | 0.465 |
| 20190716_FIRE_mg-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 1.02 | 0.69 | 0.444 | 0.0191 | 1.08 | 0.00127 | 0.0147 |
| 20170613_FIRE_bh-w-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.4 | 0.663 | 0.423 | 0.0434 | 1.03 | 0.00215 | 0.5 |
| 20240701_Kitchenfire_ws-w-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.874 | 0.604 | 0.39 | 0.0386 | 1.09 | 0.00248 | 0.00946 |
| 20171207_Lilac_rm-s-mobo | 768 x 1024 | 12 | 60 (interval) | 1.06 | 0.918 | 0.42 | 0.0103 | 1.18 | 0.0123 | 0.513 |
| 20200812_LakeFire_dwpgm-n-mobo-c | 683 x 1024 | 12 | 64 (interval) | 1.25 | 0.733 | 0.424 | 0.0181 | 1.12 | 0.0043 | 0.0513 |
| 20200813_Ranch2Fire_wilson-e-mobo-c | 683 x 1024 | 12 | 60 (interval) | 1.71 | 0.667 | 0.408 | 0.0232 | 1.14 | 0.0219 | 0.0519 |
| 20251102_ScissorsFire_mp-n-mobo-m | 683 x 1024 | 12 | 60 (interval) | 1.02 | 0.647 | 0.363 | 0.0309 | 1.14 | 0.00481 | 0.0846 |
| 20190813_Topanga_69bravo-n-mobo | 683 x 1024 | 12 | 60 (interval) | 0.74 | 0.533 | 0.417 | 0.0377 | 1.14 | 0.000432 | 0.0182 |
| 20260622_BernardoFire_wc-w-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.835 | 0.649 | 0.4 | 0.0192 | 1.12 | 0.0182 | 0.0065 |
| 20171021_FIRE_pi-e-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.01 | 0.733 | 0.424 | 0.00926 | 1.16 | 0.00075 | 0.504 |
| 20251102_ScissorsFire_vo-e-mobo-m | 683 x 1024 | 12 | 60 (interval) | 1.71 | 0.694 | 0.419 | 0.0221 | 1.08 | 0.00411 | 0.0517 |
| 20160604_FIRE_rm-n-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.01 | 0.718 | 0.493 | 0.024 | 1.07 | 0.00185 | 0.499 |
| 20220405-fire-in-Fallbrook-rm-s-mobo-m | 683 x 1024 | 12 | 60 (interval) | 0.701 | 0.329 | 0.122 | 0.124 | 1.19 | 0.0209 | 0.0383 |
| 20170625_BBM_bm-n-mobo | 768 x 1024 | 12 | 60 (interval) | 2.12 | 0.686 | 0.456 | 0.0454 | 1.01 | 0.00364 | 0.498 |
| 20220210-EmeraldFire-signal-s-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.541 | 0.137 | 0.0492 | 0.532 | 1.41 | 0.0108 | 0.162 |
| 20220210-EmeraldFire-stgo-w-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.458 | 0.553 | 0.139 | 0.0251 | 1.33 | 0.00502 | 0.0176 |
| 20171010_FIRE_hp-w-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.11 | 0.882 | 0.432 | 0.012 | 1.12 | 0.0042 | 0.499 |
| 20241113_Fire_dwpgm-s-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.683 | 0.902 | 0.419 | 0.00288 | 1.31 | 0.00266 | 0.0129 |
| 20251102_ScissorsFire_vo-e-mobo-c | 683 x 1024 | 12 | 60 (interval) | 1.78 | 0.782 | 0.408 | 0.018 | 1.06 | 0.0285 | 0.0399 |
| 20250120_TedWilliamsFire_wc-w-mobo-m | 683 x 1024 | 12 | 60 (interval) | 1.06 | 0.835 | 0.421 | 0.0142 | 1.09 | 0.00805 | 0.0167 |
| 20251102_ScissorsFire_mp-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 1.1 | 0.673 | 0.361 | 0.0326 | 1.1 | 0.0241 | 0.106 |
| 20220405-fire-in-Fallbrook-rm-s-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.485 | 0.18 | 0.0508 | 0.246 | 1.25 | 0.0191 | 0.059 |
| 20170826_FIRE_tp-s-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.36 | 0.788 | 0.398 | 0.0419 | 1.04 | 0.00167 | 0.552 |
| 20200727_Border11Fire_om-e-mobo-m | 683 x 1024 | 12 | 60 (interval) | 0.575 | 0.629 | 0.366 | 0.00199 | 1.48 | 0.0472 | 0.0275 |
| 20160711_FIRE_ml-n-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.22 | 0.784 | 0.466 | 0.0202 | 1.06 | 0.00274 | 0.498 |
| 20170711_FIRE_sdsc-e-mobo-c | 768 x 1024 | 12 | 60 (interval) | 0.966 | 0.816 | 0.449 | 0.0476 | 1.06 | 0.00155 | 0.501 |
| 20190610_Pauma_bh-w-mobo-m | 768 x 1024 | 12 | 60 (interval) | 2.18 | 0.706 | 0.631 | 0.0259 | 1.29 | 0.0112 | 0.508 |
| 20161113_FIRE_bm-w-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.17 | 0.735 | 0.407 | 0.0205 | 1.06 | 0.0033 | 0.495 |
| 20180806_FIRE_mg-s-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.802 | 0.773 | 0.423 | 0.00703 | 1.2 | 0.00289 | 0.0174 |
| 20250804_CoolFire_bi-w-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.745 | 0.537 | 0.404 | 0.0318 | 1.1 | 0.000533 | 0.0138 |
| 20200601_WILDLAND-DRILLS_mlo-s-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.883 | 0.784 | 0.437 | 0.00672 | 1.27 | 0.0143 | 0.183 |
| 20190922_FIRE_ml-w-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.34 | 0.806 | 0.41 | 0.0261 | 1.12 | 0.0234 | 0.703 |
| 20261015_HendersonFire_buff-s-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.732 | 0.669 | 0.387 | 0.00757 | 1.23 | 0.00199 | 0.0111 |
| 20180727_FIRE_bh-s-mobo-c | 768 x 1024 | 12 | 60 (interval) | 0.99 | 0.784 | 0.55 | 0.00456 | 1.25 | 0.0162 | 0.749 |
| 20240907_FIRE_ws-s-mobo-c | 683 x 1024 | 12 | 59 (interval) | 1.71 | 0.851 | 0.41 | 0.019 | 1.11 | 0.0141 | 0.246 |

(54 more clips in the JSON report.)

## Notes

* Flicker and motion include scene changes (moving smoke, passing objects, pans, auto-exposure): they bound the camera's own flicker and shake from above, on both sides.
* Noise includes the finest scene texture; sharpness includes noise. Real values below the synthetic range usually mean in-camera denoising or codec smoothing.
* The synthetic frames carry the rendered nuisances of every class (strong artefacts included), so their upper percentiles reflect deliberately harsh clips.
