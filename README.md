# CFD-ML: a CFD-pretrained physics expert for wildfire smoke vs fog

**A small temporal network (GRU) trained to tell wildfire smoke from fog by
how it moves, pretrained on a 2D lattice-Boltzmann simulator of smoke, fog
and look-alikes, and producing a 16-dimensional motion embedding for a
multimodal wildfire detector. Two camera profiles: near field (a 320 m
scene, frames 5 s apart with 1 s lag pairs, for surveillance video) and far
field (a 3.2 km scene, one frame per minute, for wildfire camera networks
such as HPWREN and AlertWildfire). Pretrained on synthetic data only. On
real one-frame-per-minute wildfire camera footage it separates smoke from
smoke-free windows only weakly (AUROC 0.58-0.61), and it does not recognise real
fog: real fog windows score lower on fog than clear mornings (AUROC 0.35).
A real-footage detector built next to it (frozen CLIP appearance features
over the whole frame and 2 x 3 and 4 x 6 tiles, logistic heads, trained on
real labelled windows) meets its stated goals on 35 held-out camera mounts
by point estimate: fog AUROC 0.96 and smoke AUROC 0.88 (both pooled over
five datasets), and 82 % of smoke windows caught at 3.7 % false smoke
alarms on fog. That operating point alarms on 27 % of all smoke-free
windows; a strict one catches 53 % at 5.9 %. Smoke within one dataset is
harder: 0.71 on FIgLib's early plumes, 0.84-0.93 on 2026 HPWREN fire days.
The 2026 labels come from AI labellers.**

The physics expert is one of four parts of a larger system (vision, physics,
weather, fusion). It sees only motion features measured from camera frames:
optical flow, region area and growth, centroid motion, boundary roughness and
how stable the region's base is. It never sees the simulated velocity,
pressure or concentration fields; the simulator only renders camera-like
frames, and real footage goes through the same feature extractor.

> **Status: the real-footage detector meets its stated targets on held-out
> cameras by point estimate, on pooled AUROCs
> ([`reports/far/detector.md`](reports/far/detector.md); caveats in the
> section "Real-footage detector" below); the CFD-pretrained physics expert
> alone does not transfer to real fog.** A second version (v2: full-resolution
> zoom, same-camera reference frames, fog-aware stacking) improved every number
> on dev but not on the test cameras, so v1 stays the default (section
> "Detector v2 attempt"). The far-field physics model was tested on
> real wildfire camera footage at its timing (one frame per minute):
>
> * **Smoke, HPWREN FIgLib** (880 windows, 137 cameras): smoke AUROC
>   0.61 ± 0.02 zero-shot on the 171 windows of the 35 held-out test cameras
>   of the FIgLib-only split (three seeds); the exported model scores 0.58
>   over all 880 windows. That is about what it reaches on its synthetic
>   test sets (0.56 ± 0.01 on test_iid, 0.59 ± 0.01 on test_ood).
> * **Fog, HPWREN archive** (262 labelled 2026 mornings, 88 cameras): built
>   for this project, because no public footage at this timing carries fog
>   labels. The model does not recognise real fog: fog AUROC 0.35
>   [0.28, 0.43], below chance. Fog does not raise its smoke score: smoke
>   alarms come on 15 % of fog windows and 18 % of clear ones (balanced
>   point), and on none of either at the low-false-alarm default.
>
> Training on the real footage from scratch learns fog better (AUROC
> 0.59 ± 0.15) than the pretrained model, fine-tuned or not. The near-field
> model has only a five-window sanity check on real video. The fog labels
> come from AI vision labellers, not people.

## Results (engine v2)

Full reports: near field [`reports/results.md`](reports/results.md) (from
[`reports/evaluation.json`](reports/evaluation.json)); far field
[`reports/far/results.md`](reports/far/results.md) (from
[`reports/far/evaluation.json`](reports/far/evaluation.json), including the
real-footage Stage B section); the real fog test in
[`reports/far/real_fog_hpwren.md`](reports/far/real_fog_hpwren.md). Data QA, CFD-oracle diagnostics, image statistics,
zero-shot reports and benchmarks sit next to them; figures in
`reports/figures/` and `reports/figures_far/`; the engine v1 reports are
archived in [`reports/v1/`](reports/v1). Single numbers are for the exported
model (the seed with the best validation loss) with 95 % bootstrap
intervals over simulations or cameras; "±" is mean ± sd over three
training seeds.

### Real-footage detector (far field)

[`reports/far/detector.md`](reports/far/detector.md) (from
[`reports/far/detector.json`](reports/far/detector.json)); model in
`artifacts_detector/`; design in `docs/design.md` section 13. One window is
12 frames one minute apart. Inputs: frozen CLIP ViT-B/32 embeddings of
frames 0, 6 and 11 (whole frame and a 2 x 3 grid) and of frames 0 and 11
on a 4 x 6 grid, so a small distant plume still fills a few CLIP patches.
The raw motion features (time-pooled; not the CFD-pretrained expert's
embedding, which was not tried) were offered to model selection and not
chosen (within-set AUROC 0.803 with them, 0.804 without).
Weather is not an input, because the windows were found through METAR fog
reports and CAL FIRE incident reports. Heads: an L2 logistic regression
per label; for smoke also a tile head (a separate logistic regression
applied to every 4 x 6 tile, the window scored by its best tile), stacked
with the window head. Everything was selected and calibrated on 2064 dev
windows from 143 camera mounts by cross-validation grouped by camera mount,
maximising the within-set AUROC; the 440 windows of the 35 held-out test
mounts were only scored. The one refitted model was scored three times:
the second and third runs only added report metrics (the strict point and
the within-set fog AUROC); model, thresholds and earlier numbers reproduced
exactly.

| data (windows used) | labels | smoke | fog |
|---|---|---|---|
| FIgLib, 229 fire sequences (880) | library ignition time: 2 windows before, 2 after the first visible plume | 443 | unknown |
| HPWREN 2026 fog mornings, two sets (262 + 448) | AI vision labels, METAR-proposed | 0 | 353 |
| HPWREN 2026 fire days, two sets (344 + 570) | AI vision labels, CAL FIRE-proposed, up to 70 km | 341 | 3 |

Held-out test cameras (95 % intervals resample camera mounts):

| metric | test | dev (cross-validated) | goal |
|---|---|---|---|
| fog AUROC, all sets pooled | **0.963** [0.91, 0.99] | 0.977 | >= 0.90 |
| fog AUROC within sets | 0.932 [0.85, 0.99] | 0.945 | |
| smoke AUROC, all sets pooled | **0.880** [0.82, 0.93] | 0.896 | >= 0.85 |
| smoke AUROC within sets (pairs from one set only) | 0.761 [0.66, 0.85] | 0.809 | |
| deploy point: false smoke alarms on fog windows | **3.7 %** (2 of 54) [0, 10] | 2.4 % (set to <= 2.5 %) | <= 5 % |
| deploy point: smoke windows caught | **82.4 %** [73, 91] | 79.9 % | >= 50 % |
| deploy point: alarms on all no-smoke windows | 26.6 % | 20.5 % | |
| strict point: alarms on all no-smoke windows | 5.9 % | 4.9 % (set to <= 5 %) | |
| strict point: false smoke on fog / smoke caught | 0 % / 53.4 % | 0.3 % / 59.9 % | |

| test set | smoke AUROC | fog AUROC | deploy: alarms on no-smoke | deploy: smoke caught | strict: alarms on no-smoke | strict: smoke caught |
|---|---|---|---|---|---|---|
| FIgLib (88 smoke, 83 not) | 0.713 | | 65 % | 83 % | 20 % | 53 % |
| HPWREN fog (21 fog, 21 clear) | | 0.878 | 2 % | | 0 % | |
| HPWREN fog 2 (33 fog, 41 clear) | | 0.941 | 1 % | | 0 % | |
| HPWREN fire days (20 smoke, 31 not) | 0.844 | | 29 % | 75 % | 0 % | 50 % |
| HPWREN fire days 2 (23 smoke, 79 not) | 0.928 | | 20 % | 87 % | 1 % | 57 % |

What this does and does not show:

* All three goals hold on the held-out cameras by point estimate; the 95 %
  intervals for smoke AUROC [0.82, 0.93] and false smoke on fog [0, 10 %]
  include values that miss them. Both pooled AUROCs are helped by easy
  negatives from other sets: the fog sets' windows are all smoke-free, and
  the fire-day windows are almost all fog-free. Within one dataset the smoke
  AUROC is 0.76 and the fog AUROC 0.93 (0.88 on the first fog set, below
  0.90; 0.94 on the second). The hardest part is FIgLib's early plumes
  (0.71; windows mostly within 24 minutes of the first visible plume, some
  to 38): the plume is often a few pixels in a frame stored 1024 px wide.
* The deploy point budgets false alarms on fog only, as the goal asks. It
  alarms on 27 % of all smoke-free test windows, 65 % of FIgLib's windows
  minutes before an ignition. The strict point (set on the dev cameras: at
  most 5 % alarms on all dev smoke-free windows) is the better one for
  unattended use: 5.9 % alarms overall, none on fog, 53 % of smoke windows
  caught; it still alarms on 20 % of FIgLib's pre-ignition windows. It was
  added after the first test run, without changing the model.
* 54 test fog windows: one more false alarm would be 5.6 %. The interval
  on that rate reaches 10 %.
* The 2026 HPWREN labels come from AI vision labellers, not from people:
  two blind votes per contact sheet, a third where either vote was unsure,
  the quality calls differed or they disagreed (14-25 % of sheets); Cohen's
  kappa 0.98-1.00 on the sheets both first votes decided. These are runs of
  one AI model, so this measures consistency, not accuracy; no human check
  yet.
* What helped smoke, from this run's dev cross-validation (`selection` in
  `detector.json`; within sets, FIgLib in brackets): window head without
  the 4 x 6 tiles 0.771 (0.725), with them 0.804 (0.756); the tile head
  alone 0.769 (0.728); the two stacked 0.809 (0.760). Earlier FIgLib-only
  probes (scratch scripts, not in the repo) found no gain from zoomed crops
  around changing regions, a reference frame from the same camera 12
  minutes earlier, change-only features or set-balanced weights.

### Detector v2 attempt: fewer false alarms, better tiny smoke (not confirmed on test)

Goal: fewer smoke alarms on clear scenes and on fog, a higher AUROC on
FIgLib's early plumes, fog and clear/large-smoke performance unchanged. The
rules were written down before v2 was scored on the test cameras
([`configs/real/detector_v2_eval_rules.json`](configs/real/detector_v2_eval_rules.json),
one dev-only revision recorded in it). What v2 adds, all chosen on dev
cameras: a zoom head on full-resolution crops of each window's top-4 tiles
(the stored frames are 2-3x downscaled; `scripts/zoom_windows.py` streams the
originals), the smoke-prompt contrast change within the window and against
the same camera 14 minutes before the window (`scripts/reference_frames.py`;
a fixed lag, so it cannot encode the label: before-fire windows at -24 and
-12 min score alike, AUROC 0.50), how fog-like the most-changed tiles are,
a stack fitted with fog windows weighted 3, a fog veto (no smoke alarm when
P(fog) >= 0.95) and a deploy threshold matched to v1's 80 % dev recall.

| v1 stack vs v2, same rules | dev v1 | dev v2 | test v1 | test v2 |
|---|---|---|---|---|
| smoke AUROC within sets | 0.809 | 0.838 | 0.761 | 0.771 |
| FIgLib (early plumes) | 0.760 | 0.794 | 0.713 | 0.726 |
| 2026 fire days (two sets) | 0.901 / 0.892 | 0.926 / 0.909 | 0.844 / 0.928 | 0.871 / 0.919 |
| fog AUROC | 0.977 | 0.977 | 0.963 | 0.963 |
| deploy: smoke caught | 80.1 % | 80.1 % | 81.7 % | 76.3 % |
| deploy: alarms on fog windows | 2.1 % | 1.4 % | 3.7 % | 3.7 % |
| deploy: alarms on all smoke-free windows | 20.5 % | 15.3 % | 26.9 % | 20.3 % |
| strict: smoke caught | 59.7 % | 65.4 % | 53.4 % | 61.8 % |
| strict: alarms on all smoke-free windows | 4.9 % | 4.9 % | 5.9 % | 7.9 % |

* On dev, v2 cut smoke-free alarms by a quarter and fog alarms by a third at
  the same catch rate, and raised every smoke AUROC. On the held-out test
  cameras that did not replicate: at the same test catch rate (80 %), v2
  alarms on 27.2 % of smoke-free windows (v1 26.6 %) and on 3.7 % of fog
  windows (v1 3.7 %); within-set AUROC +0.010 (95 % camera-bootstrap
  interval -0.025 to +0.051;
  [`reports/far/detector_v2_matched.json`](reports/far/detector_v2_matched.json)).
  The lower deploy-point alarm rate on test comes from catching less smoke
  (76 % vs 82 %), not from better ranking.
* Likely reasons: the dev numbers are optimistic (about ten stack variants
  were compared on dev, and the fog fix was found there), and the test set
  is small (35 camera mounts, 54 fog windows; a FIgLib difference below
  about 0.035 AUROC is within noise), so it can neither confirm the dev gain
  nor rule out a smaller real one.
* Because v2 is not shown to be better and needs the original frames and a
  reference frame per window, the default detector stays v1
  (`artifacts_detector/`, [`reports/far/detector.md`](reports/far/detector.md)).
  v2 is in `artifacts_detector_v2/` with
  [`reports/far/detector_v2.md`](reports/far/detector_v2.md).
* A blind AI audit of all 437 FIgLib pre-ignition windows found smoke in
  view in 8 (7 dev, 1 test; 41 too dark or unclear), so FIgLib's clear-scene
  alarms are almost all genuine false alarms
  ([`configs/real/figlib_audit_labels.csv`](configs/real/figlib_audit_labels.csv)).

### Real footage

**HPWREN FIgLib, far field, smoke only**
([`reports/far/stage_b_figlib_only.json`](reports/far/stage_b_figlib_only.json),
[`reports/far/real_zero_shot_figlib.md`](reports/far/real_zero_shot_figlib.md)).
The table below is the Stage B run on FIgLib alone. `evaluation.json` now
holds the combined FIgLib + HPWREN fog run (below).
Imagery courtesy of HPWREN (https://www.hpwren.ucsd.edu/). 229 fire
sequences from 137 cameras, one frame per minute. Each sequence gives up
to two 12-frame windows entirely before the first visible plume (smoke 0)
and two starting at least a minute after it (smoke 1): 880 windows, fog
unknown. Splits are camera-disjoint by camera mount; the test cameras give
171 windows (46 sequences, 35 cameras). Candidate masks come from the
built-in fallback detector, as there is no vision model yet.

| test cameras, 171 windows | smoke AUROC | 3 seeds |
|---|---|---|
| C: CFD-pretrained, zero-shot | 0.60 [0.52, 0.67] | 0.609 ± 0.022 |
| B: same network trained from scratch on the 439 FIgLib training windows | 0.57 [0.49, 0.66] | 0.579 ± 0.020 |
| D: C fine-tuned on those windows (lr 1e-4) | 0.58 [0.50, 0.66] | 0.582 ± 0.001 |

* **The signal is weak and uniform.** Zero-shot smoke AUROC is 0.57-0.60
  on every split (0.58 over all 880 windows). Pretraining on the simulator
  is no worse than learning from 439 real windows, and fine-tuning adds
  nothing; the intervals overlap, so B, C and D cannot be told apart.
* **The synthetic thresholds do not transfer.** Real windows score in a
  narrow band (P(smoke) 0.38-0.69, median 0.52) that straddles the
  synthetic thresholds (0.51 balanced, 0.61 low-false-alarm). Over all 880
  windows, at the balanced point the zero-shot model finds 106 of the 136
  smoke windows it decides on (78 %), but calls smoke on 74 of the 107
  smoke-free windows it decides on (69 %). At the low-false-alarm point it
  raises no false alarm (0 of 117) and finds 3 of 129. It abstains on
  78 % of the windows: 43 % have fewer than two valid flow frames and 23 %
  fewer than three valid masks.
* **No real low-false-alarm point from FIgLib alone.** B and D recalibrate
  on the real `val_calib` split, but without fog labels there are no
  fog_only clips to fit the low-false-alarm threshold on, so it equals the
  balanced one. This is recorded in their `model_config.yaml` and in the
  report. Adding the HPWREN fog windows (below) fixes this.
* **Why so little.** Single motion features carry little on FIgLib: the
  best univariate smoke AUROC is 0.57 (vorticity, flow spread, mask
  validity). Smoke at ignition is small and distant, and the smoke-free
  windows are mostly static landscapes, whereas every synthetic scene
  contains a moving aerosol or look-alike.
* **The stretch fix holds up on real data.** The model trained before
  the far-field stretch fix (below) does no better on FIgLib: 0.593 against
  0.609 on the test cameras (three-seed means).

**HPWREN fog, far field: the real fog test**
([`reports/far/real_fog_hpwren.md`](reports/far/real_fog_hpwren.md),
[`reports/far/real_fog_hpwren_boxes.md`](reports/far/real_fog_hpwren_boxes.md);
built with [`scripts/prepare_hpwren_fog.py`](scripts/prepare_hpwren_fog.py)).
No public footage at one frame per minute carries fog labels, so this set
was built from the HPWREN camera archive, which keeps every frame for about
three months.

* **Candidates.** Airport METARs proposed 400 morning windows from
  20 July to 18 October 2026: 200 fog mornings (an airport inside the
  camera's view reported fog, low visibility or a ceiling at or below
  1000 ft) and 200 clear mornings at the same cameras and times.
* **Labels.** The 357 windows that downloaded were labelled from the frames:
  blind contact sheets with the camera banner blanked, two independent votes
  per sheet and a third on 91 sheets. The first two votes agree on fog for
  99 % of the sheets both decided (Cohen's kappa 0.99). The labellers are
  AI vision runs of one model, so this measures consistency, not verified
  accuracy; a human spot check is still to do. The METARs are a rough
  proxy: 74 % of fog candidates show visible fog and 11 % of clear ones do.
* **Usable set.** 262 windows (131 fog, 131 clear, 88 cameras; 21 + 21 on
  the test cameras). The rest had sun glare, an archive placeholder, or an
  undecided label. No window shows smoke. The fog is mostly a fog or
  stratus sea below the camera (81 of 131), the camera inside fog (36), or
  low cloud on ridges (9).

| zero-shot, all usable windows | fog AUROC | smoke alarms on fog windows / on clear windows (balanced) | at the low-false-alarm default |
|---|---|---|---|
| built-in fallback masks (262 windows: 131 fog, 131 clear) | 0.35 [0.28, 0.43] (seeds 0.39 ± 0.03) | 19 / 131 (15 %), 24 / 131 (18 %) | 0 and 0 (22 and 40 decided) |
| boxes around the fog or look-alike (253 windows with boxes: 127 fog, 126 clear) | 0.27 [0.20, 0.34] (seeds 0.27 ± 0.04) | 78 / 127 (61 %), 83 / 126 (66 %) | 0 and 1 (2 and 1 decided; abstains on 125 of 127 and 125 of 126) |

* **The model does not recognise real fog.** It ranks fog windows below
  clear ones. On the windows it decides, its fog head says "fog" almost
  everywhere: 33 of 36 fog windows and 56 of 56 clear ones.
* **It abstains on the camera inside fog.** Only 17 % of those windows have
  valid input, and the median P(fog) there is 0.32.
* **Fog does not trigger smoke alarms more than clear mornings do.** The
  smoke score is, if anything, lower on fog (AUROC 0.38 of P(smoke) for fog
  against clear). At the default low-false-alarm point no window raises a
  smoke alarm with built-in masks (with boxes, 1 of 126 clear windows does).
* **Why built-in masks miss fog.** The fallback detector finds change against
  the window's median frame, so a fog sea lying still for 11 minutes is
  never a candidate. Valid input is 38 % on fog windows against 56 % on
  clear ones.
* **Boxes give the model the fog region, and it fails differently.** With a
  box around the fog (or, on clear mornings, around the most fog- or
  smoke-like region), every region scores about 0.64 for fog and 0.62 for
  smoke whatever it contains.
* **The physics runs backwards.** Region area points the right way (fog
  boxes are larger, AUROC 0.76), but boundary roughness (0.23) and anchor
  stability (0.14) run opposite to simulated fog (0.61 and 0.58). Real fog
  seas lie still; the simulated fog banks form, advect and dissipate. Part
  of this is how the boxes were drawn (four per window, interpolated), so
  the box test is not clean either.
* **Smoke against fog across datasets.** P(smoke) ranks FIgLib smoke windows
  above HPWREN fog windows with AUROC 0.55 (built-in masks). The two sets
  differ in year and time of day, so this is only indicative.

**Stage B with real smoke and real fog**
([`reports/far/results.md`](reports/far/results.md#real-footage-stage-b-test_iid-split-camera-disjoint)).
FIgLib (smoke labels) and the HPWREN fog set (fog labels) were merged,
camera-disjoint by mount, with built-in masks. B and D are trained,
calibrated and thresholded on the real splits; C (zero-shot) keeps its
synthetic calibration and thresholds, so its 2 / 21 is at the synthetic
low-false-alarm point. The test cameras give 213 windows from 41
cameras: 212 with a smoke label and 42 with a fog label (21 fog, 21 clear).

| test cameras | smoke AUROC | fog AUROC | false smoke on the 21 fog windows, low-false-alarm point |
|---|---|---|---|
| C: CFD-pretrained, zero-shot | 0.57 [0.51, 0.64] (seeds 0.570 ± 0.012) | 0.43 [0.25, 0.63] (0.437 ± 0.017) | 2 / 21 |
| B: same network, real data only | 0.61 [0.53, 0.69] (0.590 ± 0.020) | 0.76 [0.60, 0.90] (0.587 ± 0.152) | 0 / 21 |
| D: C fine-tuned on real data (lr 1e-4) | 0.55 [0.48, 0.63] (0.551 ± 0.002) | 0.44 [0.24, 0.66] (0.422 ± 0.020) | 1 / 21 |

* **The real low-false-alarm threshold now exists.** It is fitted on the
  real `val_calib` fog windows, but there are only 14 of them, and the test
  has 21: these rates are very uncertain.
* **Pretraining does not help on real fog.** Training from scratch on about
  600 real training windows learns fog better than the CFD-pretrained model, but
  unstably across seeds. Fine-tuning at the planned learning rate barely
  moves the pretrained model.
* **The models mostly abstain.** B and D, recalibrated on real data, abstain
  on 81-100 % of the test windows for smoke, at either operating point.

**FIRESENSE, near field**
([`reports/real_zero_shot_firesense.md`](reports/real_zero_shot_firesense.md)).
[FIRESENSE](https://zenodo.org/records/836749) (CC BY 4.0) is close-range
smoke video, not wildfire views. Only 5 windows from 3 videos are usable:
the other clips are shorter than one 12 x 5 s window with 1 s lag frames,
or are excluded as edited, mis-clocked or undecodable. That leaves 3 smoke
windows and 2 smoke-free ones, with no fog. All 3 smoke windows score above both smoke-free ones. The
balanced point detects the 3 smoke windows and decides on neither
smoke-free one; the low-false-alarm point decides on one window of each
and gets both right. One smoke window is called fog (P(fog) 0.67). This
shows that the real-footage path works; it measures nothing.

**Image statistics, far render against FIgLib**
([`reports/far/image_stats_figlib.md`](reports/far/image_stats_figlib.md)).
The far profile's imaging ranges were set from the FIgLib train cameras
only. On the held-out cameras, noise, contrast, brightness, sharpness and
JPEG blocking fall inside the synthetic range. Flicker and
frame-to-frame motion do not: 30 % and 45 % of the held-out clips are
steadier than 99 % of the synthetic ones. The camera model can be that
steady (synthetic clips with shake below 0.03 px move 0.08 px per minute,
against a real median of 0.06-0.10 px), but its uniform draws rarely are,
and part of the remaining motion is scene content. The near profile was
not tuned to FIRESENSE, its only real footage, because that footage is
also its check.

### Near field, synthetic

**Data.** 1200 simulation slots. 28 CFD attempts failed the label-support
check (fog or smoke not visible in enough frames) and were re-drawn; 2
slots kept no sequence because
no render variant passed the visibility check. That leaves 3247
sequences:

* train 1800 (600 simulations), val_select 300, val_calib 453;
* test_iid 296 (148 simulations);
* test_ood 398 (199 simulations; held-out recipes: wind 7-9 m/s, sources
  on 45-80 m hills);
* plus 1176 robustness re-renders of the test_iid simulations in 8 suites.

| | smoke AUROC | fog AUROC | false smoke on fog_only, balanced / low-false-alarm | ECE smoke / fog |
|---|---|---|---|---|
| C: CFD-pretrained GRU, test_iid | 0.60 [0.53, 0.66] | 0.75 [0.68, 0.81] | 77 % / 4 % [0, 9] | 0.08 / 0.07 |
| C, test_ood (held-out recipes) | 0.64 [0.58, 0.71] | 0.69 [0.63, 0.75] | 70 % / 18 % [10, 26] | 0.03 / 0.08 |
| A: motion-cue heuristic, test_iid | 0.57 | 0.50 | 64 % / 1 % | |
| pooled logistic (14 features x mean/std), test_iid | 0.60 | 0.73 | 70 % / 4 % | |

* **Over seeds:** smoke 0.590 ± 0.020 and fog 0.724 ± 0.020 on test_iid;
  smoke 0.624 ± 0.016 and fog 0.694 ± 0.005 on test_ood.
* **The low-false-alarm point (the default) holds its 5 % target in
  distribution.** It reaches 4.1 ± 2.7 % on test_iid but 14.3 ± 3.2 % on
  the held-out recipes. Counting only the fog_only clips the model decides
  on, the balanced point calls smoke on 76 % (16/21 test_iid) and 92 %
  (46/50 test_ood), the low-false-alarm point on 0/37 and 5/41.
* **Abstention.** At the balanced point the model abstains on 75 % of
  test_iid clips (62 % at the default low-false-alarm point):
  * 32 % have fewer than two frames with measurable flow;
  * 6 % have fewer than three valid masks;
  * the rest are within the calibrated margin of a threshold.

  On the clips it decides it is right on 57 % (smoke) and 74 % (fog) at
  the balanced point, and on 51 % and 74 % at the low-false-alarm point.
* **The GRU against the baselines.** It beats the motion-cue baseline A
  on AUROC, by a lot on fog. It is on par with a time-pooled logistic
  regression (0.60 / 0.73). It is not a stand-alone smoke detector; it is
  a calibrated, abstaining input for the fusion model.

**v1 against v2** (three-seed means; [`reports/ablation_sgs.md`](reports/ablation_sgs.md)):

| | v1 | v2 | v2, subgrid texture motion off |
|---|---|---|---|
| smoke AUROC test_iid / test_ood | 0.638 / 0.637 | 0.590 / 0.624 | 0.595 / 0.630 |
| fog AUROC test_iid / test_ood | 0.699 / 0.653 | 0.724 / 0.694 | 0.726 / 0.682 |
| false smoke on fog_only, low-false-alarm point, test_iid / test_ood | 4.5 / 11.0 % | 4.1 / 14.3 % | 4.5 / 14.0 % |

Engine v2 changed several things at once:

* subgrid texture motion;
* camera shake correlated in time;
* more shadows, camera artefacts and distant clouds in every class;
* more look-alike-heavy "neither" scenes.

Fog AUROC rose and smoke AUROC fell. The ablation re-rendered the same
simulations without subgrid texture motion, and it changes nothing beyond
the seed spread. The v2 test sets are harder than the v1 ones, so v1 and
v2 numbers are not directly comparable.

**Where the signal is lost (CFD-oracle diagnostic, never a model input;
[`reports/oracle_diagnostic.md`](reports/oracle_diagnostic.md)).** Computed
from the CFD fields, the same twelve features reach smoke AUROC 0.81
[0.74, 0.88] on test_iid (v1: 0.78); from the camera they reach 0.57 with
the same classifier. The velocity group is where most of it goes: 0.79
from the CFD, 0.50 from optical flow. Fog keeps its signal through the
camera (0.73 against 0.75), mostly through region geometry. Optical flow
follows the CFD velocity of the candidate region with correlation 0.83
horizontally and 0.41 vertically.

**Robustness** (re-renders of test_iid, exported model; deltas against
the nominal re-render):

* Heavy shake costs the most smoke AUROC (-0.14), heavy occlusion -0.05.
* Under the strongest imaging shift (noise, JPEG 12-25, low contrast)
  stabilisation fails its quality check. The model then abstains on every
  clip rather than reading camera motion as aerosol motion.
* Detector-like masks help slightly (+0.04 smoke), box masks change
  nothing.

**Leave-one-regime-out** (each regime held out of training in turn):

* Unseen look-alikes and fogs remain the weak point:
  * held out, drifting haze is called fog 92 % of the time;
  * steam fog is found 38 % of the time (AUROC 0.49);
  * drifting fog patches 34 %;
  * advected smoke remnants 17 % (AUROC 0.52).
* Advected fog banks (recall 0.91) and dissipating fog (recall 0.87)
  transfer best.

**Shortcut audit.** No context or render parameter predicts a label: all
folded AUROCs are at most 0.53. The validity flags alone predict fog with
pooled held-out AUROC 0.66, so they are flagged: fog scenes more often
have a valid mask and less often a stabilisable background. That is part
of the fog signal and is reported, not hidden.

### Far field, synthetic

**Data.** 1200 simulation slots. 62 CFD attempts were rejected and
re-drawn, and 1 slot kept no sequence. That leaves 3248 sequences:

* train 1799 (600 simulations), val_select 297, val_calib 456;
* test_iid 296 (148 simulations);
* test_ood 400 (200 simulations; wind 12-15 m/s, sources on 700-1000 m
  hills);
* plus 1177 robustness re-renders.

| | smoke AUROC | fog AUROC | false smoke on fog_only, balanced / low-false-alarm | ECE smoke / fog |
|---|---|---|---|---|
| C: CFD-pretrained GRU, test_iid | 0.55 [0.47, 0.62] | 0.67 [0.60, 0.74] | 68 % / 14 % [7, 20] | 0.09 / 0.08 |
| C, test_ood (held-out recipes) | 0.59 [0.53, 0.65] | 0.70 [0.64, 0.75] | 61 % / 9 % [4, 16] | 0.04 / 0.08 |
| A: motion-cue heuristic, test_iid | 0.58 | 0.61 | 85 % / 5 % | |
| pooled logistic, test_iid | 0.57 | 0.67 | 66 % / 10 % | |

* **Over seeds:** smoke 0.559 ± 0.008 and fog 0.687 ± 0.018 on test_iid;
  smoke 0.589 ± 0.012 and fog 0.692 ± 0.013 on test_ood. The
  low-false-alarm point gives 8.1 ± 4.9 % and 6.7 ± 2.5 % false smoke on
  fog_only, so it misses its 5 % target. On the fog_only clips the
  exported model decides on, it calls no smoke (0/25, 0/34).
* **On test_iid the GRU is no better on smoke than the four-cue
  heuristic A** (0.55 against 0.58); on test_ood (held-out recipes) it is
  ahead (0.59 against 0.54). On fog it is better (0.67 against 0.61 on
  test_iid). At the balanced point it abstains on 79 % of test_iid clips,
  33 % for too few flow frames.
* **The one-minute camera is the bottleneck, not the simulation**
  ([`reports/far/oracle_diagnostic.md`](reports/far/oracle_diagnostic.md)).
  From the CFD fields the features reach smoke 0.75 [0.66, 0.82] and fog
  0.81 [0.72, 0.87] on test_iid; from the camera, 0.57 and 0.64. Flow
  between frames a minute apart tracks the simulated horizontal velocity
  poorly (correlation 0.21; vertical 0.56) and is valid on only 30-39 % of
  frames. The velocity features from the camera are at chance for fog
  (0.50).
* **Stretch fix.** Over a 480 s texture period, km-scale shear drew the
  advected texture into streaks that real smoke and cloud do not show.
  Fading over-stretched texture (`render.texture.max_stretch`) removed
  them. It lowered synthetic smoke AUROC from 0.598 to 0.559 (seed means;
  fog unchanged), but real FIgLib AUROC did not move, so the streaks were
  a cue that exists only in the simulator.
* **Shortcut audit.** The validity flags alone predict fog at 0.61
  (flagged); no context or render parameter exceeds 0.54.
* **Leave-one-regime-out.** The new regimes transfer moderately: ridge
  cloud is found 70 % of the time (AUROC 0.78), and rain shafts are
  called smoke 50 % of the time when held out.

### CPU cost

One core, every library pinned to one thread; measured while other jobs
ran on the other cores
([`reports/benchmark_cpu.md`](reports/benchmark_cpu.md),
[`reports/far/benchmark_cpu.md`](reports/far/benchmark_cpu.md)):

| Stage | near field | far field |
|---|---|---|
| CFD simulation, accepted, incl. precursor | 6.4 s (about 560 per hour) | 8.6 s (about 420 per hour) |
| Render 12 observation frames (+ 12 lag frames near) | 167 ms | 119 ms |
| Feature extraction (12 x 14) | 80 ms | 75 ms |
| GRU training epoch, about 1800-2000 sequences | 0.21 s | 0.15 s |
| Inference, one sequence (features to decisions) | 1.0 ms | 0.8 ms |

The time step starts at 0.1 s near (0.5 s far) and is halved when the
expected peak speed would push the lattice Mach number past its limit:

* near field: 25 % of simulations ran at 0.1 s, 44 % at 0.05 s and 31 % at
  0.025 s;
* far field: 36 % at 0.5 s, 42 % at 0.25 s, 22 % at 0.125 s and under
  1 % at 0.0625 s.

The physical frame times are the same for all of them.

## How it fits together

```
 profile: configs/simulation.yaml (near) or configs/simulation_far.yaml (far)
        v
 scenario sampling (context first, class-independent)        src/cfd/scenarios.py
        |  wind, terrain, atmosphere, nuisances  ->  regime  ->  smoke / fog / distractor
        v
 2D CFD, 64 x 64 cells (near 320 m, far 3.2 km), CPU          src/cfd/{lbm,scalars,thermo,smoke,fog,simulation}.py
        |  smoke, fog liquid water, haze/dust/rain, advected texture coordinates,
        |  subgrid velocity u_sgs, at 12 observation times
        |  (near: 5 s apart + 12 lag times 1 s earlier; far: 60 s apart, no lag times)
        v
 camera rendering + candidate masks                           src/data/{render,masks}.py
        |  grayscale frames, texture moved by the flow and by u_sgs, detector-like
        |  masks, nuisances (correlated shake, occlusion, shadows, JPEG, noise, haze),
        |  the same for every class
        v
 motion features, 12 timesteps x 14                           src/data/features.py  <- real footage enters here
        |  DIS optical flow (near: 1 s lag pairs; far: consecutive one-minute frames),
        |  region geometry, validity flags
        v
 normaliser (train split only) -> GRU (14 -> 32) -> 16-d embedding -> smoke / fog logits
                                                              src/data/normalization.py, src/models/physics_gru.py
```

A model is trained on one profile; its export records the extractor
timing and flow mode, and inference checks real footage against them.

## The simulator

A vertical x-z section (64 x 64 cells; the camera sees the interior
without the absorbing layers) solved with a D2Q9 lattice-Boltzmann method:

* **Profiles:** near field (`configs/simulation.yaml`, the default): 5 m
  cells, 12 observations 5 s apart, each with a lag frame 1 s earlier.
  Far field (`configs/simulation_far.yaml`): 50 m cells, 12 observations
  60 s apart, no lag frames (flow between consecutive observations),
  wildfire-scale sources, km-scale fog and stratus, ridges, for camera
  networks that record one frame per minute. The code is scale-agnostic:
  physical constants come from the config, numerical widths scale with the
  cell size and window-relative times with the frame interval.
* **Flow:** regularised single-relaxation-time BGK collision with Guo
  forcing and a Smagorinsky subgrid viscosity (low base viscosity, so plumes
  become unsteady); Boussinesq buoyancy from potential temperature, water
  vapour and liquid-water loading relative to the ambient profiles.
  No-slip ground and terrain (bounce-back); velocity inlets and pressure
  outlets at the open sides and top; sponges in physical units; a gust force
  and an ambient-maintaining force keep the wind profile steady over the
  wall. The time step starts at 0.1 s (near) or 0.5 s (far) and is reduced
  when the expected peak speed requires it (Mach below 0.25). At every
  snapshot the run also stores the subgrid velocity scale
  u_sgs = C_s dx |S| of the Smagorinsky model.
* **Scalars:** finite-volume advection-diffusion (minmod MUSCL) of smoke,
  haze/dust (with settling), precipitation in rain-shaft scenes (fast
  settling), the moist-conserved variables theta_l and q_t
  (as anomalies about the ambient profiles, so the ambient state is exactly
  steady), and four texture coordinates. Every budget closes to round-off:
  inflow, outflow, sources, deposition, clipping and the weak-compressibility
  correction are all booked.
* **Moisture:** saturation adjustment diagnoses temperature, vapour and fog
  liquid water every step (latent heat included); fog forms where mixing or
  cooling supersaturates the air and evaporates where it is mixed with dry air.
* **Scenarios:** smoke (hot plume, smouldering, crosswind, pulsing, advected
  smoke remnant), fog (radiation, advected bank, valley fog with cold-air
  drainage, drifting patch, dissipating, steam fog), and look-alikes
  (drifting haze, settling dust bursts including moving vehicles, moving
  cloud shadows, camera artefacts, distant clouds). The far field adds a
  ridge cloud (fog: an orographic cap cloud that forms where humid wind is
  lifted over a ridge, with no prescribed liquid, anchored over the crest
  while the air moves through it) and a rain shaft (neither: fast-falling
  precipitation from a drifting cloud-base band, intermittent, sometimes
  evaporating as virga); both are off in the near field. Smoke is not always
  rising and fog is not always still: steam fog rises from a warm wet surface,
  smouldering smoke drifts along the ground, advected smoke and fog banks both
  enter from upwind. The look-alikes without aerosol (shadows, camera
  artefacts, distant clouds) are now each at least ~12 % of the "neither"
  scenes (2-5 % in v1).
* **Rendering:** the advected texture moves with the simulated air and, in
  addition, with the unresolved eddies: a smooth random displacement whose
  local velocity is proportional to u_sgs, wherever it is and whatever the
  class. Camera shake is an Ornstein-Uhlenbeck process at the frame times
  (correlation time 0.5-4 s near, 1-30 s far), so a lag frame 1 s earlier
  shakes almost with its observation while frames a minute apart do not.
* **Validation (every run):** finite fields, non-negative concentrations,
  Mach number, in-view density deviation, budget residuals, Boussinesq
  validity, and label support (the labelled aerosol must be visible in the
  camera view; unlabelled fog or smoke must be absent). Rejected runs are
  re-drawn and logged with their parameters.

## Data

Context (wind 0-7 m/s near, 0-12 m/s far, sometimes accelerating over the
window; terrain: hills, valleys, slopes of 5-40 m near, also ridges of
50-600 m far; atmosphere, spin-up, background haze, camera nuisances) is
drawn from the same distribution for every class, then a
compatible regime is chosen, so no context variable separates the classes by
construction (the evaluation audits this on the accepted data). Splits are by
simulation: `train`, `val_select` (early stopping, seed choice), `val_calib`
(temperature, thresholds, abstention), `test_iid`, and `test_ood` built from
held-out recipes in every class: faster wind (7-9 m/s near, 12-15 m/s far;
e.g. fast-moving fog) and sources on taller hills (45-80 m near,
700-1000 m far; unseen source locations and terrain). Seeds derive
from (split, category, index, attempt), so no random stream is shared across
splits.

Real footage enters as a manifest of videos or image folders with camera
ids, smoke and fog labels; an empty label means unknown (stored as NaN and
left out of that label's loss, calibration and metrics), so smoke-only
annotations are usable. Real splits are camera-disjoint.

## Features (per observation, 12 x 14)

| # | Feature | # | Feature |
|---|---|---|---|
| 1 | mean horizontal flow | 8 | area growth, d ln(area)/dt |
| 2 | mean vertical flow (+up) | 9 | centroid x velocity |
| 3 | horizontal flow std | 10 | centroid z velocity (+up) |
| 4 | vertical flow std | 11 | boundary roughness |
| 5 | divergence | 12 | source-anchor stability |
| 6 | mean abs vorticity | 13 | candidate mask valid |
| 7 | region area | 14 | optical flow valid |

Two timing modes, fixed by the profile a model was trained on:

* **Lag pairs (near field):** twelve observations 5 s apart span 55 s. Flow
  is measured over a 1 s lag pair at each observation (DIS optical flow),
  because flow over the 5 s observation spacing loses the texture
  correspondence.
* **Consecutive frames (far field):** twelve observations 60 s apart (each
  within 10 s of its target) span 11 minutes, with no lag frames; flow is
  measured between consecutive observations. A minute's displacement can
  exceed what DIS tracks from rest, so each pair is also tried from the
  displacement of the region's centroid and the start whose flow best
  reconstructs the frame is kept.

At the first observation the
between-observation derivatives (area growth, centroid velocity, anchor
stability) are 0; they have no flag of their own (the 14-feature layout
has only the two validity flags), which is unambiguous because they are
always the GRU's first step. The 1 s lag flow at t = 0 is valid; with
consecutive frames t = 0 has no flow (flow_valid 0). Camera shake is removed by a
similarity transform estimated from background corners outside the candidate
masks; when no estimate is possible (a frame-filling fog bank, an untextured
sky) the flow features of that clip are gated off rather than left to
measure camera motion (`FeatureConfig.require_stabilization`, to be disabled
only for rigid, static mounts). Lengths are in units of a region of
interest around the candidate masks, so features do not depend on how large
the scene appears in the image. The extractor's timing is exported with
the model: a near-field model refuses footage whose frame spacing cannot
provide 1 s pairs (for example one-minute HPWREN/FIgLib imagery) rather
than treating it as motion footage; a far-field model takes exactly that
imagery (it ignores lag frames, and abstains on features measured over lag
pairs).

## Usage

Everything runs on a CPU (the system targets one core; `--workers` only adds
throughput). Install PyTorch from the CPU wheel index, then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest -m "not slow"          # fast tests; drop -m to include the convergence checks
```

Stage A, synthetic pretraining and evaluation, near field (the commands
that produced `artifacts/` and `reports/`):

```bash
# 1. data: 250 simulations per category + 50 per category of held-out recipes,
#    robustness re-renders of test_iid, example figures
python scripts/generate_dataset.py --out data/synthetic --n-per-category 250 \
    --ood-per-category 50 --workers 4 --suites --examples reports/figures
python scripts/qa_dataset.py --data data/synthetic --out reports/qa_full.json

# 2. pretraining: three seeds, early stopping on val_select, calibration on val_calib
python scripts/train_physics.py --features data/synthetic/features.npz --out artifacts

# 3. evaluation (writes artifacts/evaluation.json and reports/), diagnostics, report
python scripts/evaluate_physics.py --features data/synthetic/features.npz --artifacts artifacts \
    --reports reports --robustness data/synthetic/features_robustness.npz \
    --summary data/synthetic/summary.json --loro
python scripts/oracle_diagnostic.py --data data/synthetic --out reports/oracle_diagnostic.json
python scripts/benchmark_cpu.py --out reports/benchmark_cpu.json
python scripts/summarize_results.py --evaluation reports/evaluation.json \
    --oracle reports/oracle_diagnostic.json --benchmark reports/benchmark_cpu.json \
    --qa reports/qa_full.json --out reports/results.md
```

Far field (one frame per minute): the same steps with `--config
configs/simulation_far.yaml`, into `data/synthetic_far` and `artifacts_far`
(the feature timing comes from the profile and is exported with the model):

```bash
python scripts/generate_dataset.py --config configs/simulation_far.yaml --out data/synthetic_far \
    --n-per-category 250 --ood-per-category 50 --workers 4 --suites --examples reports/figures_far
python scripts/qa_dataset.py --data data/synthetic_far --out reports/far/qa_full.json
python scripts/train_physics.py --features data/synthetic_far/features.npz --out artifacts_far
python scripts/evaluate_physics.py --features data/synthetic_far/features.npz --artifacts artifacts_far \
    --reports reports/far --robustness data/synthetic_far/features_robustness.npz \
    --summary data/synthetic_far/summary.json --loro
python scripts/oracle_diagnostic.py --data data/synthetic_far --out reports/far/oracle_diagnostic.json
python scripts/benchmark_cpu.py --config configs/simulation_far.yaml --out reports/far/benchmark_cpu.json \
    --features data/synthetic_far/features.npz --artifacts artifacts_far
# after the FIgLib Stage B below, which adds its section to reports/far/evaluation.json:
python scripts/summarize_results.py --evaluation reports/far/evaluation.json \
    --oracle reports/far/oracle_diagnostic.json --benchmark reports/far/benchmark_cpu.json \
    --qa reports/far/qa_full.json --out reports/far/results.md
```

Optional, not part of the v2 evaluation: a synthetic proxy set
(`evaluate_physics.py --proxy data/synthetic_proxy/features.npz`), a small,
separately seeded dataset rendered with detector / box masks and harsher
imaging, gives a pretrained-vs-scratch transfer proxy on a domain shift the
simulator itself produces (v1 result in [`reports/v1/results.md`](reports/v1/results.md);
not real-world evidence). For the far field the real FIgLib Stage B below
replaces it:

```bash
python scripts/generate_dataset.py --out data/synthetic_proxy --n-per-category 60 --ood-per-category 0 \
    --set seed=20270301 --set 'render.masks.source_probs={detector: 0.7, bbox: 0.3}' \
    --set 'render.noise_sigma=[0.03, 0.06]' --set 'render.jpeg_quality=[20, 60]' --set render.jpeg_prob=0.9 \
    --set 'render.contrast=[0.45, 0.9]' --set 'render.global_haze=[0.2, 0.45]' --workers 4
```

### Inference

```python
from src.inference.predict import PhysicsExpert

expert = PhysicsExpert.load("artifacts")               # near field; "artifacts_far" for one-frame-per-minute cameras
X, meta = expert.features_from_frames(frames, timestamps, masks, lag_frames, lag_timestamps)
emb = expert.encode(X)                                   # (B, 16) embedding for the fusion network
out = expert.predict(X, meta=meta)                       # operating_point="low_false_alarm" by default
out["decisions"]          # (B, 2) smoke, fog: 1 present, 0 absent, -1 abstain
out["calibration_source"] # "synthetic_val_calib" until real footage recalibrates it
```

Near-field model: `frames` are 12 observations about 5 s apart,
`lag_frames` one frame about 1 s before each. Far-field model: `frames` are
12 observations about 60 s apart (each within 10 s of its target) and
`lag_frames` / `lag_timestamps` are None (flow is measured between
consecutive observations; lag frames, if passed, are ignored). The
extractor settings always come from the exported model. `masks` are the
vision detector's candidate masks (or None for the built-in fallback
detector). `predict` abstains when the input is not measurable (too few
valid masks or flow pairs, out-of-range features, frame timing or flow mode
other than the model was trained on) or the calibrated logit is within the
fitted margin of the threshold; a label the model could not calibrate (no
known calibration target, e.g. fog after a smoke-only Stage B from scratch)
always abstains. The default operating point, `"low_false_alarm"`, uses the
smoke threshold fitted for at most 5 % false smoke on fog-only validation
clips; `operating_point="balanced"` called smoke on 70-92 % of the fog_only
clips it decided on in the v2 synthetic tests, so use it only where a missed
fire costs more than a fog false alarm. A model recalibrated on real
footage without fog labels (smoke-only Stage B) cannot fit the
low-false-alarm threshold and uses the balanced one under that name; its
`model_config.yaml` records this (`calibration.threshold_fit`).

### Real-footage detector

```bash
# appearance embeddings (CPU; keep --threads at the free cores: oversubscribing slows CLIP about 10x)
python scripts/embed_windows.py --manifest data/real/figlib/real.csv --artifacts artifacts_far \
    --out data/real/figlib/clip.npz                                     # frames 0/6/11, whole frame + 2 x 3
python scripts/embed_windows.py --manifest data/real/figlib/real.csv --artifacts artifacts_far \
    --grid 4x6 --frames 0,11 --no-full --out data/real/figlib/clip_fine.npz
# (the same for hpwren_fog, hpwren_fog2, hpwren_smoke, hpwren_smoke2; motion features from extract_features.py)

# 2026 fire windows around CAL FIRE incident starts, blind sheets, consolidated AI votes, manifest
python scripts/prepare_hpwren_fog.py select-fires --out data/real/hpwren_smoke2 --start 2026-07-16 --end 2026-10-18 \
    --max-km 70 --max-cameras 6 --min-acres 1 --offsets-min -60 20 45
python scripts/prepare_hpwren_fog.py download --out data/real/hpwren_smoke2
python scripts/prepare_hpwren_fog.py consolidate --out data/real/hpwren_smoke2 \
    --votes configs/real/hpwren_smoke2_votes.csv --labels configs/real/hpwren_smoke2_labels.csv --primary smoke
python scripts/prepare_hpwren_fog.py manifest --out data/real/hpwren_smoke2 \
    --labels configs/real/hpwren_smoke2_labels.csv --require smoke --source hpwren_smoke2

# train (selection and thresholds on dev cameras) and score the test cameras
python scripts/train_detector.py \
    --set figlib:data/real/figlib/real.csv:data/real/figlib/clip.npz:data/real/figlib/features.npz:data/real/figlib/clip_fine.npz \
    --set hpwren_fog:...  --set hpwren_fog2:...  --set hpwren_smoke:...  --set hpwren_smoke2:... \
    --out artifacts_detector --report reports/far/detector.json --final
```

```python
from src.detector.model import Detector

det = Detector.load("artifacts_detector")
out = det.predict(E, fine=Fe, operating_point="strict")   # E (N, 3, 7, 512) coarse, Fe (N, 2, 24, 512) fine
out["probs"]["smoke"], out["probs"]["fog"]
out["decisions"]["smoke"]   # operating points: "deploy" (<= 2.5 % of dev fog windows), "strict"
                            # (<= 5 % of all dev smoke-free windows), "balanced"; deploy and strict
                            # exist for smoke only, fog decisions always use the fog balanced threshold
```

v2 (not the default) needs two more inputs per window, then the same training with the v2 rules:

```bash
# full-resolution zoom crops of the top-4 tiles (tiles from the tile head; streams the originals)
python scripts/zoom_windows.py --manifest data/real/hpwren_smoke/real.csv --source hpwren \
    --selection zoom_selection.json --artifacts artifacts_far --out data/real/hpwren_smoke/clip_zoom.npz
# the same camera 14 min before frame 0 (HPWREN: fetch first, the archive keeps about 96 days)
python scripts/reference_frames.py fetch --manifest data/real/hpwren_smoke/real.csv --artifacts artifacts_far
python scripts/reference_frames.py embed --manifest data/real/hpwren_smoke/real.csv --source hpwren \
    --artifacts artifacts_far --out data/real/hpwren_smoke/clip_ref.npz
python scripts/train_detector.py --set <name>:<manifest>:<clip>:<features>:<fine>:<zoom>:<ref> ... \
    --deploy-recall 0.8 --fog-veto 0.95 --stack-fog-weight 3 --out artifacts_detector_v2 \
    --report reports/far/detector_v2.json --final
```

### Real footage: zero-shot checks and image statistics

`evaluate_real_zero_shot.py` runs the exported model unchanged (synthetic
thresholds, both operating points) on public real footage and lists, per
window, what was measurable, the probabilities, the decisions and the
abstention reasons: a sanity check of the real-footage path, not Stage B.
`image_stats.py` compares robust frame statistics of real and rendered clips
and says which `render:` ranges to widen (a diagnostic, never a model
input).

```bash
# FIRESENSE (near field; Zenodo 836749, CC BY 4.0): testpos03/04/05 are excluded with the reason
python scripts/prepare_firesense.py --out data/real/firesense
python scripts/extract_features.py --real-manifest data/real/firesense/real.csv --artifacts artifacts \
    --out data/real/firesense/features.npz
python scripts/evaluate_real_zero_shot.py --features data/real/firesense/features.npz --artifacts artifacts \
    --name firesense --description "..." --out reports/real_zero_shot_firesense.json

# HPWREN FIgLib (far field, one frame per minute; imagery courtesy of HPWREN, https://www.hpwren.ucsd.edu/):
# 250 fire sequences requested (229 usable), up to two windows before and two after the first plume
python scripts/prepare_figlib.py --out data/real/figlib --n 250 --pre-windows 2 --post-windows 2
python scripts/extract_features.py --real-manifest data/real/figlib/real.csv --artifacts artifacts_far \
    --out data/real/figlib/features.npz
python scripts/evaluate_real_zero_shot.py --features data/real/figlib/features.npz --artifacts artifacts_far \
    --name "figlib (all splits)" --description "..." --out reports/far/real_zero_shot_figlib.json
python - <<'PY'   # the test cameras on their own (the split Stage B reports on)
from src.data.dataset import FeatureSet
FeatureSet.load("data/real/figlib/features.npz").split("test_iid").save("data/real/figlib/features_test_iid.npz")
PY
python scripts/evaluate_real_zero_shot.py --features data/real/figlib/features_test_iid.npz \
    --artifacts artifacts_far --name "figlib (test_iid cameras)" --description "..." \
    --out reports/far/real_zero_shot_figlib_test.json

# real vs synthetic image statistics (noise, contrast, sharpness, JPEG blocking, flicker, shake; diagnostic).
# Far field: tune render ranges on the FIgLib train cameras, then check them on the held-out ones.
python scripts/image_stats.py --real-manifest data/real/firesense/real.csv --synthetic data/synthetic \
    --out reports/image_stats_firesense.json
python scripts/image_stats.py --real-manifest data/real/figlib/real.csv --real-splits train --one-per-video \
    --synthetic data/synthetic_far --out reports/far/image_stats_figlib_train.json
python scripts/image_stats.py --real-manifest data/real/figlib/real.csv \
    --real-splits val_select,val_calib,test_iid --one-per-video \
    --synthetic data/synthetic_far --out reports/far/image_stats_figlib.json
```

(`--description` holds the dataset paragraph at the top of each zero-shot
report; the committed reports keep theirs in their JSON.)

`prepare_figlib.py` downloads FIgLib sequences (about 81 frames, one per
minute, around the first visible plume), writes 12-frame windows 60 s apart
entirely before the plume (smoke 0) and starting at least 60 s after it
(smoke 1), leaves fog empty (unknown: FIgLib does not annotate it) and
assigns camera-disjoint splits by camera mount. Before a far-field model
exists, `extract_features.py --profile configs/simulation_far.yaml` takes
the timing from the profile instead of `--artifacts`.
`scripts/prepare_mivia.py` writes a manifest for MIVIA smoke videos
downloaded by a registered user (labels in
`configs/real/mivia_smoke_labels.csv`).

The HPWREN fog set (far field). The archive keeps about three months, so the
same windows can only be downloaded while they are retained. The window
list with its METAR evidence, the votes, the labels and the boxes are kept
in `configs/real/`:

```bash
python scripts/prepare_hpwren_fog.py select --out data/real/hpwren_fog --start 2026-07-20 --end 2026-10-18 \
    --n-fog 200 --n-clear 200
python scripts/prepare_hpwren_fog.py download --out data/real/hpwren_fog     # 12 frames per window + blind sheets
# blind labelling of data/real/hpwren_fog/_sheets/ with configs/real/hpwren_fog_labeling_guide.md
#   -> configs/real/hpwren_fog_votes.csv (two votes per sheet, a third where they disagree)
python scripts/prepare_hpwren_fog.py consolidate --out data/real/hpwren_fog \
    --votes configs/real/hpwren_fog_votes.csv --labels configs/real/hpwren_fog_labels.csv
python scripts/prepare_hpwren_fog.py manifest --out data/real/hpwren_fog --labels configs/real/hpwren_fog_labels.csv
python scripts/extract_features.py --real-manifest data/real/hpwren_fog/real.csv --artifacts artifacts_far \
    --out data/real/hpwren_fog/features.npz
python scripts/report_real_fog.py --features data/real/hpwren_fog/features.npz --hpwren-dir data/real/hpwren_fog \
    --labels configs/real/hpwren_fog_labels.csv --artifacts artifacts_far --figlib data/real/figlib/features.npz \
    --out reports/far/real_fog_hpwren.json
# region level: boxes drawn on grid sheets (configs/real/hpwren_fog_boxes.csv), kept where the box target
# matches the label (fog box on a fog window, look-alike box on a clear one), turned into per-frame masks
python scripts/region_boxes.py sheets --manifest data/real/hpwren_fog/real.csv --artifacts artifacts_far \
    --out data/real/boxes/sheets
python scripts/region_boxes.py masks --manifest data/real/hpwren_fog/real.csv --sheets-dir data/real/boxes/sheets \
    --boxes data/real/hpwren_fog/boxes_consistent.csv --out-manifest data/real/hpwren_fog/real_boxes.csv
python scripts/extract_features.py --real-manifest data/real/hpwren_fog/real_boxes.csv --artifacts artifacts_far \
    --out data/real/hpwren_fog/features_boxes.npz
python scripts/report_real_fog.py --features data/real/hpwren_fog/features_boxes.npz ... \
    --out reports/far/real_fog_hpwren_boxes.json
```

### Stage B, real footage

Run for the far-field model on FIgLib plus the HPWREN fog set (results
above; `reports/far/evaluation.json` has `"real": {"status": "complete"}`;
the earlier FIgLib-only run, fog `"no_labels"`, is kept in
`reports/far/stage_b_figlib_only.json`). Not run for the near-field model:
`reports/evaluation.json` keeps `"real": {"status": "pending"}` and
experiments B and D `null`, because no labelled real surveillance video
with compatible frame timing was available. Labels are handled per label:
FIgLib windows carry smoke labels and leave fog unknown. With labelled
footage:

```bash
# real.csv: video_path, camera_id, video_id, smoke, fog
#           [, split, camera_mount, start_s, stride_s, max_windows, window_id, mask_dir]
# smoke / fog: 0, 1, or empty = unknown
python scripts/extract_features.py --real-manifest real.csv --artifacts artifacts --out data/real/features.npz
python scripts/evaluate_physics.py --features data/synthetic/features.npz --artifacts artifacts \
    --reports reports --real data/real/features.npz     # trains B (scratch) and D (fine-tune), evaluates B, C, D

# far field: FIgLib (smoke labels) + HPWREN fog set (fog labels), merged with a leakage check;
# without --features the synthetic sections of the existing evaluation.json are kept
# (this produced the Stage B section of reports/far/evaluation.json; B and D go to runs/real_far_combined)
python scripts/merge_features.py --out data/real/far_combined/features.npz \
    data/real/figlib/features.npz data/real/hpwren_fog/features.npz
python scripts/evaluate_physics.py --artifacts artifacts_far --reports reports/far \
    --real data/real/far_combined/features.npz --real-out runs/real_far_combined
```

Splits are camera-disjoint (group = camera id; FIgLib assigns them per
camera mount). D fine-tunes the pretrained model at `finetune.lr` = 1e-4;
B trains the same architecture from random initialisation on the same real
splits at `training.lr`; both use the same epochs, patience, seeds and
normaliser policy, and the calibration is refitted on the real `val_calib`
split for every label with enough known targets. A label with no known
target (fog on FIgLib) keeps the pretrained calibration in D and always
abstains in B; it is reported as `"no_labels"` with the reason, never as 0,
and the real status is then `"partial"` (`"complete"` when every label was
evaluated). `scripts/train_physics.py --stage finetune|scratch` runs either
one on its own.

## Repository layout

```
configs/      simulation.yaml (near-field profile: CFD, scenarios, rendering, dataset),
              simulation_far.yaml (far-field one-frame-per-minute profile), training.yaml,
              real/mivia_smoke_labels.csv (per-video MIVIA labels),
              real/hpwren_fog_* (fog set: candidates, labelling guide, votes, labels, boxes),
              real/hpwren_fog2_*, real/hpwren_smoke_*, real/hpwren_smoke2_* (votes, labels)
src/cfd/      lbm, scalars, thermo, smoke, fog, scenarios, scenario_util, simulation, validation, results
src/data/     render, masks, features, feature_spec, normalization, dataset, pipeline, visualize
src/models/   physics_gru
src/training/ train, evaluate, metrics, baselines, audit
src/inference/predict (no training dependencies)
src/detector/ features (CLIP feature groups, fine tiles, zoom, change and reference scores),
              model (heads, tile and zoom heads, stacks, fog veto, Detector)
scripts/      generate_dataset, qa_dataset, extract_features, train_physics, evaluate_physics,
              oracle_diagnostic, benchmark_cpu, summarize_results, image_stats,
              prepare_firesense, prepare_figlib, prepare_mivia, prepare_hpwren_fog, region_boxes,
              merge_features, evaluate_real_zero_shot, report_real_fog, embed_windows, train_detector,
              zoom_windows, reference_frames
tests/        test_lbm, test_scalars, test_simulation, test_render, test_features, test_pipeline,
              test_model, test_training, test_image_stats, test_prepare_figlib,
              test_prepare_hpwren_fog, test_region_boxes, test_detector
docs/         spec.md (requirements), design.md (interface contract and design decisions)
artifacts/    exported near-field model (encoder, classifier, normaliser, config, evaluation);
              the far-field model goes to artifacts_far/, the real-footage detector to
              artifacts_detector/ (v1, the default; v2 in artifacts_detector_v2/)
reports/      near field: evaluation.json, results.md, QA, oracle diagnostic, benchmark, figures,
              FIRESENSE zero-shot and image statistics, subgrid-texture ablation;
              far/: the same for the far field plus the FIgLib zero-shot, Stage B and image
              statistics, and the real-footage detector (detector.json / .md); v1/: the
              engine v1 reports
```
