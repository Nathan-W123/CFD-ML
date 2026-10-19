# Image statistics, real vs synthetic: FIgLib held-out cameras (val + test splits; not used for the render calibration)

> Diagnostic of the rendering, not a model input. Real clips are few, so a coverage fraction moves in steps of 1 / n; a statistic is flagged when more than 10% of the real clips fall outside the synthetic P1-P99 range.

* Real: `data/real/figlib/real.csv` rows of split test_iid, val_calib, val_select, one per video, 115 clips (0 skipped), native sizes (H x W) 683x1024, 768x1024; up to 12 frames 60 s apart, 115 clips with at least 3 distinct frames at that spacing for flicker and motion, 0 of them left out of motion because no frame pair registers (phase-correlation response below 0.15).
* Synthetic: `data/synthetic_far`, 200 of 2848 sequences in splits train, val_select, val_calib, test_iid (seed 0), native size 192 x 192, frames 60 s apart, 5 without a registered frame pair.
* Comparable basis: real frames resized to the synthetic pixel count (aspect kept), except blockiness, which exists only on the encoder's native 8x8 grid and is compared at native resolution.

## Comparison (comparable basis)

| statistic | unit | basis | real P5 / P25 / P50 / P75 / P95 | synthetic P5 / P25 / P50 / P75 / P95 | synthetic P1 - P99 | real below / above | verdict |
|---|---|---|---|---|---|---|---|
| noise_sigma | gray levels | resized | 0.605 / 0.766 / 0.892 / 1.16 / 1.65 (n 115) | 0.137 / 0.453 / 0.793 / 2.82 / 7.84 (n 200) | 0.0877 - 8.96 | 0% / 0% | covered |
| contrast | (P95 - P5) / 255 | resized | 0.509 / 0.609 / 0.675 / 0.796 / 0.9 (n 115) | 0.268 / 0.428 / 0.541 / 0.679 / 0.93 (n 200) | 0.18 - 0.992 | 0% / 0% | covered |
| brightness | mean / 255 | resized | 0.324 / 0.4 / 0.42 / 0.441 / 0.48 (n 115) | 0.166 / 0.3 / 0.425 / 0.555 / 0.776 (n 200) | 0.103 - 0.82 | 1% / 0% | covered |
| sharpness | var(Laplacian) / (P95 - P5)^2 | resized | 0.00396 / 0.00969 / 0.0167 / 0.0304 / 0.0625 (n 115) | 0.00415 / 0.00947 / 0.0172 / 0.034 / 0.0805 (n 200) | 0.00292 - 0.135 | 1% / 0% | covered |
| blockiness | boundary / interior gradient | native | 1.04 / 1.1 / 1.14 / 1.21 / 1.35 (n 115) | 0.977 / 1.06 / 1.4 / 1.67 / 2.06 (n 200) | 0.962 - 2.19 | 0% / 0% | covered |
| flicker | std / mean of frame means | resized | 0.000918 / 0.0022 / 0.0043 / 0.0166 / 0.0307 (n 115) | 0.00509 / 0.0127 / 0.0207 / 0.0531 / 0.134 (n 200) | 0.00236 - 0.166 | 30% / 0% | widen_low |
| motion_px | px per frame interval | resized | 0.00787 / 0.0169 / 0.0965 / 0.496 / 0.513 (n 115) | 0.0791 / 0.203 / 0.349 / 4.6 / 21.6 (n 195) | 0.0494 - 51.4 | 45% / 0% | widen_low |

## Real footage at native resolution (reference)

| statistic | real P5 / P25 / P50 / P75 / P95 | outside synthetic P1 - P99 (below / above) |
|---|---|---|
| noise_sigma | 0.339 / 0.445 / 0.516 / 0.685 / 1.51 (n 115) | 0% (0% / 0%) |
| contrast | 0.513 / 0.624 / 0.69 / 0.809 / 0.919 (n 115) | 0% (0% / 0%) |
| brightness | 0.324 / 0.4 / 0.42 / 0.441 / 0.48 (n 115) | 1% (1% / 0%) |
| sharpness | 0.00248 / 0.00809 / 0.0148 / 0.0255 / 0.0532 (n 115) | 9% (9% / 0%) |
| blockiness | 1.04 / 1.1 / 1.14 / 1.21 / 1.35 (n 115) | 0% (0% / 0%) |
| flicker | 0.000919 / 0.0022 / 0.0043 / 0.0166 / 0.0307 (n 115) | 30% (30% / 0%) |
| motion_px | 0.0127 / 0.0201 / 0.0318 / 0.0584 / 0.169 (n 113) | 70% (70% / 0%) |

## What to change in `render:`

* **noise_sigma**: Covered: 0% of 115 real clips fall outside the synthetic P1-P99 range [0.0877, 8.96]; no change needed.
* **contrast**: Covered: 0% of 115 real clips fall outside the synthetic P1-P99 range [0.18, 0.992]; no change needed.
* **brightness**: Covered: 1% of 115 real clips fall outside the synthetic P1-P99 range [0.103, 0.82]; no change needed.
* **sharpness**: Covered: 1% of 115 real clips fall outside the synthetic P1-P99 range [0.00292, 0.135]; no change needed.
* **blockiness**: Covered: 0% of 115 real clips fall outside the synthetic P1-P99 range [0.962, 2.19]; no change needed.
* **flicker**: 30% of 115 real clips fall outside the synthetic P1-P99 range [0.00236, 0.166] (30% below, 0% above). Real clips are steadier than the synthetic ones (real P5 0.001 vs synthetic P1 0.002). render.exposure_flicker already starts at 0, so the synthetic floor comes from the scenes themselves (aerosol growing or drifting through the view changes the frame mean); nothing to widen in the camera model.
* **motion_px**: 45% of 115 real clips fall outside the synthetic P1-P99 range [0.0494, 51.4] (45% below, 0% above). Real cameras are steadier (real P5 0.01 px vs synthetic P1 0.05 px per 60 s). render.camera_shake_px already starts at 0 (now [0, 0.3]) but is drawn uniformly, so few synthetic clips are as steady as a rigid mount: make near-zero shake more common or lengthen render.camera_shake_tau_s (now [1, 30] s) so that frames stay correlated; part of the synthetic floor can also be aerosol motion that the phase correlation follows.

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
| 20190529_94Fire_om-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.778 | 0.624 | 0.432 | 0.0247 | 1.09 | 0.000968 | 0.0133 |
| 20180723_FIRE_tp-e-mobo-c | 768 x 1024 | 12 | 60 (interval) | 2.02 | 0.678 | 0.415 | 0.0636 | 1.07 | 0.0366 | 0.559 |
| 20250123_GilmanFire_tdlln-s-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.815 | 0.914 | 0.424 | 0.017 | 1.14 | 0.0131 | 0.0333 |
| 20160718_FIRE_mw-e-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.21 | 0.7 | 0.49 | 0.0667 | 1.04 | 0.000802 | 0.501 |
| 20180504_FIRE_smer-tcs10-mobo-c | 683 x 1024 | 12 | 60 (interval) | 1.32 | 0.616 | 0.461 | 0.0496 | 1.09 | 0.0168 | 0.0144 |
| 20200306_FIRE_pi-n-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.3 | 0.694 | 0.431 | 0.0517 | 1.04 | 0.000994 | 0.514 |
| 20180614_Hope_wc-e-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.858 | 0.627 | 0.399 | 0.0236 | 1.16 | 0.00854 | 0.0762 |
| 20190728_Dehesa_lp-n-mobo | 683 x 1024 | 12 | 60 (interval) | 0.899 | 0.71 | 0.429 | 0.0212 | 1.13 | 0.0179 | 0.0352 |
| 20180706_West_lp-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.96 | 0.727 | 0.466 | 0.0214 | 1.14 | 0.00837 | 0.328 |
| 20180719_Skyline_sp-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 1.14 | 0.612 | 0.399 | 0.0414 | 1.14 | 0.00818 | 0.0238 |
| 20240701_McCainfire_ws-e-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.854 | 0.518 | 0.435 | 0.0557 | 1.1 | 0.00184 | 0.0164 |
| 20200601_WILDLAND-DRILLS_mlo-e-mobo-c | 683 x 1024 | 12 | 60 (interval) | 2 | 0.806 | 0.414 | 0.012 | 1.13 | 0.0258 | 0.156 |
| 20190803_Sage_om-n-mobo | 683 x 1024 | 12 | 60 (interval) | 1.47 | 0.639 | 0.458 | 0.0167 | 1.12 | 0.0165 | 0.455 |
| 20170609_FIRE_sm-n-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.06 | 0.714 | 0.436 | 0.0305 | 1.1 | 0.000495 | 0.497 |
| 20170722_FIRE_hp-e-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.89 | 0.545 | 0.424 | 0.0576 | 1.03 | 0.00361 | 0.494 |
| 20180726_FIRE_so-n-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.6 | 0.863 | 0.46 | 0.0185 | 1.07 | 0.00067 | 0.497 |
| 20200806_SpringsFire_om-n-mobo-m | 683 x 1024 | 12 | 60 (interval) | 0.778 | 0.833 | 0.425 | 0.00805 | 1.15 | 0.0149 | 0.0179 |
| 20170627_FIRE_lp-n-iqeye | 768 x 1024 | 12 | 60 (interval) | 1.11 | 0.639 | 0.419 | 0.0383 | 1.12 | 0.00389 | 0.498 |
| 20190728_FIRE_om-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 1.13 | 0.827 | 0.445 | 0.0103 | 1.12 | 0.0137 | 0.314 |
| 20240927_86Fire_tp-e-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.759 | 0.708 | 0.328 | 0.0296 | 1.1 | 0.00237 | 0.0965 |
| 20160722_FIRE_mw-e-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.24 | 0.694 | 0.493 | 0.0763 | 1.04 | 0.0015 | 0.498 |
| 20220214-PrescribedFire-pi-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.892 | 0.757 | 0.405 | 0.0357 | 1.06 | 0.000979 | 0.00432 |
| 20180706_FIRE_wc-e-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.848 | 0.608 | 0.405 | 0.0212 | 1.21 | 0.00137 | 0.0435 |
| 20190717_FIRE_lp-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.81 | 0.596 | 0.444 | 0.0513 | 1.1 | 0.00156 | 0.00746 |
| 20190728_FIRE_sp-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 1.68 | 0.667 | 0.373 | 0.033 | 1.14 | 0.00783 | 0.112 |
| 20170711_FIRE_sm-n-mobo-c | 768 x 1024 | 12 | 60 (interval) | 0.878 | 0.549 | 0.426 | 0.0387 | 1.17 | 0.00112 | 0.508 |
| 20210209_FIRE_hp-e-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.52 | 0.906 | 0.457 | 0.012 | 1.13 | 0.0154 | 0.467 |
| 20191006_FIRE_om-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.838 | 0.624 | 0.41 | 0.0227 | 1.1 | 0.0179 | 0.00793 |
| 20180725_FIRE_smer-tcs10-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.968 | 0.875 | 0.507 | 0.00555 | 1.25 | 0.0749 | 0.0257 |
| 20180720_Cinnamon_wc-e-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.743 | 0.465 | 0.426 | 0.0534 | 1.15 | 0.0182 | 0.0109 |
| 20190913_FIRE_lp-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.808 | 0.639 | 0.431 | 0.0357 | 1.11 | 0.0174 | 0.00952 |
| 20200813_SkylineFire_sp-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 1.63 | 0.892 | 0.43 | 0.0229 | 1.15 | 0.0191 | 0.175 |
| 20180603_FIRE_sm-n-mobo-c | 768 x 1024 | 12 | 60 (interval) | 0.756 | 0.727 | 0.438 | 0.0148 | 1.26 | 0.00129 | 0.505 |
| 20230727.144836-BonnyFire-hp-e-mobo-c | 683 x 1024 | 12 | 60 (interval) | 1.45 | 0.596 | 0.398 | 0.0224 | 1.12 | 0.00634 | 0.116 |
| 20200521_FIRE_om-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.708 | 0.569 | 0.42 | 0.0263 | 1.1 | 0.000495 | 0.00977 |
| 20190712_RockHouse_wc-e-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.781 | 0.545 | 0.403 | 0.0401 | 1.16 | 0.00247 | 0.00774 |
| 20260629_JunctionFire_mg-e-mobo-c | 683 x 1024 | 12 | 60 (interval) | 1.04 | 0.671 | 0.418 | 0.0149 | 1.11 | 0.0019 | 0.0459 |
| 20250122_BernardoFire_ch-s-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.733 | 0.92 | 0.448 | 0.0038 | 1.25 | 0.0187 | 0.135 |
| 20220210-EmeraldFire-marconi-w-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.121 | 0.263 | 0.0571 | 0.0729 | 1.33 | 0.00101 | 0.0798 |
| 20171010_FIRE_rm-e-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.11 | 0.639 | 0.454 | 0.0272 | 1.11 | 0.00219 | 0.498 |
| 20170708_Whittier_syp-n-mobo-m | 683 x 1024 | 12 | 60 (interval) | 1.07 | 0.576 | 0.513 | 0.0195 | 1.17 | 0.12 | 0.0442 |
| 20230809.111733-BunnieFire-cp-w-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.736 | 0.612 | 0.379 | 0.00851 | 1.23 | 0.00237 | 0.19 |
| 20180806_FIRE_vo-w-mobo-c | 768 x 1024 | 12 | 60 (interval) | 3.07 | 0.8 | 0.428 | 0.0256 | 1.03 | 0.00159 | 0.513 |
| 20220302-Jimfire-0921-stgo-e-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.936 | 0.831 | 0.412 | 0.0165 | 1.14 | 0.00326 | 0.0112 |
| 20170708_Whittier_syp-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 1.09 | 0.582 | 0.452 | 0.0235 | 1.14 | 0.0071 | 0.0113 |
| 20191005_FIRE_vo-n-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.27 | 0.824 | 0.413 | 0.0167 | 1.06 | 0.000992 | 0.506 |
| 20180522_FIRE_rm-e-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.51 | 0.896 | 0.424 | 0.00668 | 1.19 | 0.0226 | 0.577 |
| 20240909_BridgeFire_marconi-n-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.827 | 0.496 | 0.463 | 0.0165 | 1.15 | 0.00362 | 0.0173 |
| 20230809.111832-BunnieFire-cp-w-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.736 | 0.61 | 0.379 | 0.00853 | 1.23 | 0.00243 | 0.191 |
| 20231026.103052-prescribed-vo-w-mobo-c | 683 x 1024 | 12 | 58 (interval) | 0.805 | 0.776 | 0.375 | 0.00916 | 1.13 | 0.00297 | 0.0211 |
| 20220302-Jimfire-1101-stgo-e-mobo-c | 683 x 1024 | 12 | 60 (interval) | 1.64 | 0.725 | 0.403 | 0.028 | 1.11 | 0.00295 | 0.0541 |
| 20260629_JunctionFire_vo-n-mobo-c | 683 x 1024 | 12 | 61 (interval) | 0.932 | 0.871 | 0.418 | 0.0117 | 1.08 | 0.0167 | 0.0287 |
| 20240905_ValleyFire_cp-s-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.789 | 0.675 | 0.394 | 0.014 | 1.16 | 0.00233 | 0.929 |
| 20180919_FIRE_rm-e-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.06 | 0.61 | 0.44 | 0.0248 | 1.14 | 0.00486 | 0.502 |
| 20231026.103933-prescribed-cp-w-mobo-c | 683 x 1024 | 12 | 59 (interval) | 0.711 | 0.675 | 0.386 | 0.00889 | 1.19 | 0.00227 | 0.159 |
| 20231027.095451-prescribed-vo-w-mobo-c | 683 x 1024 | 12 | 58 (interval) | 0.779 | 0.827 | 0.373 | 0.00924 | 1.12 | 0.0013 | 0.0153 |
| 20220905-FairviewFire-stgo-e-mobo-c | 683 x 1024 | 12 | 60 (interval) | 1.42 | 0.62 | 0.405 | 0.0466 | 1.1 | 0.008 | 0.139 |
| 20200226_FIRE_rm-e-mobo-c | 768 x 1024 | 12 | 60 (interval) | 1.12 | 0.778 | 0.414 | 0.0141 | 1.13 | 0.00222 | 0.499 |
| 20250617_MonteFire_cp-w-mobo-c | 683 x 1024 | 12 | 60 (interval) | 0.531 | 0.614 | 0.377 | 0.0088 | 1.23 | 0.0092 | 0.0194 |
| 20260629_JunctionFire_vo-w-mobo-c | 683 x 1024 | 12 | 61 (interval) | 0.739 | 0.796 | 0.365 | 0.00498 | 1.24 | 0.0164 | 0.169 |

(55 more clips in the JSON report.)

## Notes

* Flicker and motion include scene changes (moving smoke, passing objects, pans, auto-exposure): they bound the camera's own flicker and shake from above, on both sides.
* Noise includes the finest scene texture; sharpness includes noise. Real values below the synthetic range usually mean in-camera denoising or codec smoothing.
* The synthetic frames carry the rendered nuisances of every class (strong artefacts included), so their upper percentiles reflect deliberately harsh clips.
