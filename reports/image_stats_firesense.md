# Image statistics, real vs synthetic: firesense

> Diagnostic of the rendering, not a model input. Real clips are few, so a coverage fraction moves in steps of 1 / n; a statistic is flagged when more than 10% of the real clips fall outside the synthetic P1-P99 range.

* Real: `data/real/firesense/real.csv`, 19 clips (0 skipped), native sizes (H x W) 240x300, 240x320, 270x480, 288x352, 360x480, 360x640, 424x640, 480x640, 576x704; up to 12 frames 5 s apart, 19 clips with at least 3 distinct frames at that spacing for flicker and motion, 7 of them left out of motion because no frame pair registers (phase-correlation response below 0.15).
* Synthetic: `data/synthetic`, 200 of 2849 sequences in splits train, val_select, val_calib, test_iid (seed 0), native size 192 x 192, frames 5 s apart, 3 without a registered frame pair.
* Comparable basis: real frames resized to the synthetic pixel count (aspect kept), except blockiness, which exists only on the encoder's native 8x8 grid and is compared at native resolution.

## Comparison (comparable basis)

| statistic | unit | basis | real P5 / P25 / P50 / P75 / P95 | synthetic P5 / P25 / P50 / P75 / P95 | synthetic P1 - P99 | real below / above | verdict |
|---|---|---|---|---|---|---|---|
| noise_sigma | gray levels | resized | 0.302 / 0.592 / 1 / 1.51 / 6.11 (n 19) | 0.266 / 0.577 / 1.74 / 5.21 / 8.49 (n 200) | 0.139 - 9.09 | 0% / 0% | covered |
| contrast | (P95 - P5) / 255 | resized | 0.526 / 0.607 / 0.663 / 0.8 / 0.868 (n 19) | 0.17 / 0.264 / 0.384 / 0.502 / 0.709 (n 200) | 0.147 - 0.82 | 0% / 26% | widen_high |
| brightness | mean / 255 | resized | 0.435 / 0.459 / 0.491 / 0.521 / 0.559 (n 19) | 0.35 / 0.423 / 0.529 / 0.62 / 0.722 (n 200) | 0.239 - 0.756 | 0% / 0% | covered |
| sharpness | var(Laplacian) / (P95 - P5)^2 | resized | 0.00726 / 0.0156 / 0.0347 / 0.0876 / 0.151 (n 19) | 0.00603 / 0.0175 / 0.0385 / 0.0903 / 0.306 (n 200) | 0.00411 - 0.741 | 0% / 0% | covered |
| blockiness | boundary / interior gradient | native | 1.04 / 1.11 / 1.34 / 2.15 / 3 (n 19) | 0.989 / 1 / 1.12 / 1.58 / 2.06 (n 200) | 0.938 - 2.43 | 0% / 16% | widen_high |
| flicker | std / mean of frame means | resized | 0.00465 / 0.00988 / 0.0262 / 0.0656 / 0.11 (n 19) | 0.0061 / 0.0162 / 0.0274 / 0.0444 / 0.0917 (n 200) | 0.0038 - 0.133 | 0% / 0% | covered |
| motion_px | px per frame interval | resized | 0.143 / 0.395 / 0.521 / 0.689 / 1.12 (n 12) | 0.221 / 1.53 / 2.84 / 6.45 / 24 (n 197) | 0.0937 - 33.4 | 8% / 0% | covered |

## Real footage at native resolution (reference)

| statistic | real P5 / P25 / P50 / P75 / P95 | outside synthetic P1 - P99 (below / above) |
|---|---|---|
| noise_sigma | 0.105 / 0.297 / 0.519 / 0.841 / 7.14 (n 19) | 16% (16% / 0%) |
| contrast | 0.579 / 0.616 / 0.671 / 0.809 / 0.871 (n 19) | 26% (0% / 26%) |
| brightness | 0.435 / 0.459 / 0.491 / 0.521 / 0.559 (n 19) | 0% (0% / 0%) |
| sharpness | 0.00335 / 0.0152 / 0.0256 / 0.044 / 0.236 (n 19) | 11% (11% / 0%) |
| blockiness | 1.04 / 1.11 / 1.34 / 2.15 / 3 (n 19) | 16% (0% / 16%) |
| flicker | 0.00466 / 0.00988 / 0.0262 / 0.0656 / 0.11 (n 19) | 0% (0% / 0%) |
| motion_px | 0.0515 / 0.0952 / 0.188 / 0.774 / 1.72 (n 12) | 25% (25% / 0%) |

## What to change in `render:`

* **noise_sigma**: Covered: 0% of 19 real clips fall outside the synthetic P1-P99 range [0.139, 9.09]; no change needed.
* **contrast**: 26% of 19 real clips fall outside the synthetic P1-P99 range [0.147, 0.82] (0% below, 26% above). Real frames span a wider gray range (real P95 0.87 vs synthetic P99 0.82). Raise the upper end of render.contrast (now [0.6, 1.3]) and/or lower render.global_haze (now [0, 0.35]), which compresses the range; the scene's own range comes from render.background.sky_brightness / ground_brightness (now [0.45, 0.95] / [0.1, 0.5]).
* **brightness**: Covered: 0% of 19 real clips fall outside the synthetic P1-P99 range [0.239, 0.756]; no change needed.
* **sharpness**: Covered: 0% of 19 real clips fall outside the synthetic P1-P99 range [0.00411, 0.741]; no change needed.
* **blockiness**: 16% of 19 real clips fall outside the synthetic P1-P99 range [0.938, 2.43] (0% below, 16% above). Real frames are blockier (real P95 3.003 vs synthetic P99 2.431, native resolution). Lower the lower end of render.jpeg_quality (now [35, 95]) and/or raise render.jpeg_prob (now 0.6).
* **flicker**: Covered: 0% of 19 real clips fall outside the synthetic P1-P99 range [0.0038, 0.133]; no change needed.
* **motion_px**: Covered: 8% of 12 real clips fall outside the synthetic P1-P99 range [0.0937, 33.4]; no change needed.

## Do the statistics follow their render parameters? (synthetic, Spearman rank correlation)

| statistic ~ parameter | rho |
|---|---|
| noise_sigma~noise_sigma | 0.53 |
| contrast~contrast | 0.47 |
| contrast~global_haze | -0.38 |
| brightness~brightness_offset | 0.54 |
| sharpness~noise_sigma | 0.52 |
| blockiness~jpeg_strength | 0.91 |
| flicker~exposure_flicker | 0.87 |
| motion_px~camera_shake_px | 0.69 |

## Per real clip (comparable basis; flicker and motion only at the comparison spacing)

| clip | native H x W | frames | spacing (s) | noise_sigma | contrast | brightness | sharpness | blockiness | flicker | motion_px |
|---|---|---|---|---|---|---|---|---|---|---|
| testneg01 | 288 x 352 | 3 | 5 (interval) | 1.02 | 0.839 | 0.482 | 0.0347 | 1.13 | 0.00473 | 1.01 |
| testneg02 | 240 x 320 | 10 | 5 (interval) | 0.598 | 0.947 | 0.5 | 0.0177 | 1.21 | 0.0194 | 0.572 |
| testneg03 | 480 x 640 | 5 | 4.97 (interval) | 1.13 | 0.576 | 0.458 | 0.106 | 1.11 | 0.0104 | 0.508 |
| testneg04 | 576 x 704 | 12 | 5 (interval) | 2.24 | 0.702 | 0.526 | 0.069 | 1.4 | 0.00392 | 0.0354 |
| testneg05 | 288 x 352 | 3 | 5 (interval) | 6.46 | 0.608 | 0.561 | 0.142 | 1.04 | 0.00794 | 0.23 |
| testneg06 | 288 x 352 | 3 | 5 (interval) | 1.87 | 0.6 | 0.449 | 0.121 | 1.11 | 0.0313 | n/a |
| testneg07 | 288 x 352 | 7 | 5 (interval) | 6.07 | 0.498 | 0.491 | 0.232 | 1.03 | 0.00832 | 1.25 |
| testneg08 | 288 x 352 | 7 | 5 (interval) | 0.81 | 0.686 | 0.559 | 0.0598 | 1.06 | 0.0354 | 0.938 |
| testneg09 | 288 x 352 | 6 | 5 (interval) | 1.57 | 0.529 | 0.479 | 0.135 | 1.1 | 0.0274 | 0.418 |
| testpos01 | 360 x 640 | 4 | 5 (interval) | 1.24 | 0.608 | 0.465 | 0.0171 | 2.34 | 0.0839 | n/a |
| testpos02 | 424 x 640 | 12 | 5.01 (interval) | 1.45 | 0.694 | 0.543 | 0.057 | 1.18 | 0.0169 | 0.326 |
| testpos06 | 240 x 320 | 5 | 5 (interval) | 0.615 | 0.651 | 0.46 | 0.0279 | 1.74 | 0.0252 | 0.432 |
| testpos07 | 240 x 320 | 12 | 5 (interval) | 0.587 | 0.663 | 0.455 | 0.0259 | 1.66 | 0.00937 | 0.534 |
| testpos08 | 240 x 320 | 5 | 5 (interval) | 0.284 | 0.859 | 0.516 | 0.00873 | 2.59 | 0.0262 | 0.606 |
| testpos09 | 360 x 640 | 6 | 5 (interval) | 0.521 | 0.625 | 0.398 | 0.007 | 2.89 | 0.107 | n/a |
| testpos10 | 360 x 480 | 10 | 5.01 (interval) | 0.33 | 0.606 | 0.491 | 0.00729 | 3.99 | 0.0948 | n/a |
| testpos11 | 240 x 320 | 6 | 5 (interval) | 0.815 | 0.855 | 0.496 | 0.0142 | 1.34 | 0.108 | n/a |
| testpos12 | 270 x 480 | 3 | 4.99 (interval) | 1 | 0.855 | 0.532 | 0.0446 | 2 | 0.0472 | n/a |
| testpos13 | 240 x 300 | 4 | 5.01 (interval) | 0.304 | 0.761 | 0.439 | 0.0134 | 2.3 | 0.126 | n/a |

## Notes

* Flicker and motion include scene changes (moving smoke, passing objects, pans, auto-exposure): they bound the camera's own flicker and shake from above, on both sides.
* Noise includes the finest scene texture; sharpness includes noise. Real values below the synthetic range usually mean in-camera denoising or codec smoothing.
* The synthetic frames carry the rendered nuisances of every class (strong artefacts included), so their upper percentiles reflect deliberately harsh clips.
