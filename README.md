# CFD-ML: a CFD-pretrained physics expert for wildfire smoke vs fog

**A small temporal network (GRU) trained to tell wildfire smoke from fog by
how it moves, pretrained on a 2D lattice-Boltzmann simulator of smoke, fog
and look-alikes, and producing a 16-dimensional motion embedding for a
multimodal wildfire detector. Synthetic pretraining only: its real-world
performance is untested.**

The physics expert is one of four parts of a larger system (vision, physics,
weather, fusion). It sees only motion features measured from camera frames:
optical flow, region area and growth, centroid motion, boundary roughness and
how stable the region's base is. It never sees the simulated velocity,
pressure or concentration fields; the simulator only renders camera-like
frames, and real footage goes through the same feature extractor.

> **Status: synthetic pretraining only. No real camera footage has been
> used for training or evaluation, so nothing here measures real-world
> accuracy.** Synthetic test scores show that the motion signal in the
> simulated data is learnable and how robust the model is to simulated
> nuisances; they are not evidence that the model reduces fog-related false
> alarms on real cameras. The real-data fine-tuning (Stage B) and the
> pretrained-vs-scratch comparison are implemented (see [Stage B](#stage-b-real-footage))
> and are pending labelled surveillance video.

## Results (synthetic only)

Full tables: [`reports/results.md`](reports/results.md) (generated from
[`reports/evaluation.json`](reports/evaluation.json)); data quality:
[`reports/qa_full.json`](reports/qa_full.json); CFD-oracle diagnostic:
[`reports/oracle_diagnostic.md`](reports/oracle_diagnostic.md); figures in
`reports/` and `reports/figures/` (example sequences per category and regime).

**Data.** 1200 simulation slots: 29 CFD attempts were rejected by the
numerical and label-support checks and re-drawn (5 % of both-class and
3.5 % of fog-only attempts, under 1 % of the rest); 2 valid simulations
(val_calib-smoke_only-0020, test_ood-both-0042) were dropped because none
of their render variants passed the label-visibility check. The remaining
1198 simulations give 3246 sequences: train 1800 (600 simulations),
val_select 300, val_calib 453 (151), test_iid 295 (148), test_ood 398 (199
simulations: wind 7-9 m/s, sources on 45-80 m hills), plus 1024 robustness
re-renders of the test_iid simulations.

**Experiment C, the CFD-pretrained GRU** (exported seed 0; 95 % intervals
from a bootstrap over simulations; mean ± sd over the three seeds in
`reports/results.md`):

| | smoke AUROC | fog AUROC | false smoke on fog_only (threshold only), balanced / low-false-alarm | ECE smoke / fog |
|---|---|---|---|---|
| test_iid | 0.64 [0.57, 0.71] | 0.69 [0.62, 0.76] | 46 % / 4 % [0, 9] | 0.07 / 0.07 |
| test_ood (held-out recipes) | 0.64 [0.58, 0.70] | 0.65 [0.57, 0.70] | 52 % / 12 % [5, 20] | 0.05 / 0.07 |
| A: motion-cue heuristic, test_iid | 0.56 | 0.49 | 45 % / 1 % | |
| pooled logistic (14 features x mean/std), test_iid | 0.62 | 0.73 | 68 % / 4 % | |

Seeds agree closely (smoke 0.638 ± 0.007, fog 0.699 ± 0.013 on test_iid).
The false-smoke column applies the threshold to every fog_only clip,
ignoring abstention; the low-false-alarm threshold (fitted that way on
val_calib) holds its 5 % target in distribution (4.1 %) but not on the
held-out recipes (12 %). Counting only the fog_only clips the model does
decide on, the balanced point calls smoke on 76 % of them (22/29 test_iid,
34/44 test_ood) and the low-false-alarm point on none (0/24, 0/31): use the
low-false-alarm point wherever a fog false alarm is costly. The model
abstains on 72 % of test_iid clips: 32 % because fewer than two frames
have measurable flow (mostly fog, haze and both-class scenes whose
background cannot be stabilised), 1 % because fewer than three frames have
a valid mask, and 39 % because a calibrated logit is within the margin of
its threshold. Accuracy on the clips it does decide is 0.61 (smoke) and
0.69 (fog).

**What this says.** The camera-level motion features carry a weak but real
signal. The GRU beats the motion-cue baseline A on AUROC (clearly on fog
and on test_ood, marginally on test_iid smoke) but not on false smoke on
fog_only, where A is equal or lower; it is on par with a time-pooled
logistic regression (better on smoke, worse on fog in distribution);
it largely keeps its ranking on the held-out wind and terrain recipes; its
calibration error is moderate (ECE about 0.07, no better than A's). It is
not a stand-alone smoke detector; it is a calibrated, abstaining input for
the fusion model.

**Where the signal is lost (CFD-oracle diagnostic, never a model input).**
The same twelve features computed from the CFD fields reach smoke AUROC
0.78 [0.71, 0.85] on test_iid against 0.60 for the camera features with the
same classifier. On test_iid the loss is mostly in the optical-flow group
(0.74 vs 0.55), above all in the spread of vertical velocity inside the
region (univariate AUROC 0.70 from the CFD, 0.56 from optical flow); the
region-mean vertical velocity itself is only a few cm/s. On the held-out
recipes the larger smoke loss is in the region group (0.73 vs 0.53, against
0.71 vs 0.60 for velocity). For fog the camera keeps almost all of it (0.73
vs 0.75 on test_iid), mainly through region geometry. Optical flow tracks the CFD velocity of the
candidate region with correlation 0.85 (horizontal) and 0.55 (vertical).

**Robustness and generalisation** (re-renders of test_iid; leave-one-
regime-out on test_iid):

* Heavy occlusion, heavy shake and degraded imaging cost about 0.10 smoke
  AUROC each; fog AUROC moves by less than 0.07. Under the strongest
  imaging shift (noise, JPEG 12-25, low contrast) background stabilisation
  fails its quality check and the model abstains on every clip rather than
  reading camera motion as aerosol motion.
* Detector-like masks do not hurt (smoke 0.71, fog 0.68); box masks cost
  0.05 smoke AUROC.
* Unseen regimes are the weak point: held out of training, drifting haze
  is called fog 82 % of the time, steam fog is found 44 % of the time
  (AUROC 0.48) and crosswind smoke 28 % of the time; hot plumes (recall
  0.70), dissipating fog (AUROC 0.74) and radiation fog (recall 0.80, only
  10 sequences) transfer best.

**Shortcut audit.** No context or render parameter predicts a label (all
folded AUROCs at most 0.53). The validity flags alone predict fog with
pooled held-out AUROC 0.60 (test_iid 0.64, test_ood 0.61, against the
GRU's 0.69 / 0.65; threshold 0.6, flagged): fog scenes more often have a
valid mask and less often a stabilisable background. That is a property of
the simulated and rendered fog scenes, carried by the flags rather than by
motion, and a good part of the fog signal; it is reported, not hidden.

**Synthetic transfer proxy (not real-world evidence).** On a separately
seeded synthetic set rendered with detector / box masks and harsher
imaging (240 simulations, re-split by simulation), the pretrained model
zero-shot (fog 0.70, smoke 0.57) and fine-tuned at lr 1e-4 (fog 0.67 ±
0.00, smoke 0.58 ± 0.00 over seeds) reach a higher fog AUROC than training
from scratch (0.60 ± 0.01) and the same smoke AUROC (0.57 ± 0.06); the
intervals on 48 test simulations overlap. B trains only on the proxy's
120 training simulations (337 sequences); C was pretrained on 600
simulations and D fine-tuned on the same 337 sequences. All three abstain
on about 90 % or more of the proxy test clips (heavy noise and JPEG defeat
stabilisation), so only the AUROCs are informative. This only shows the
Stage B code path works on a domain shift the simulator itself produces.

**Real-world status: pending.** Experiments B (real data from scratch) and
D (CFD-pretrained, fine-tuned on real data) are `null` in
`evaluation.json`: no labelled surveillance video with compatible frame
timing was available. Nothing here measures real-world accuracy or shows
a reduction of fog false alarms on real cameras.

**Single-core CPU cost** ([`reports/benchmark_cpu.md`](reports/benchmark_cpu.md);
every library pinned to one thread):

| Stage | Time |
|---|---|
| CFD simulation, accepted, incl. 30 s precursor | 5.6 s mean (about 640 per hour) |
| Render 12 + 12 lag frames (192 x 192) | 142 ms |
| Feature extraction (12 x 14) | 58 ms |
| GRU training epoch, 1800 sequences | 0.17 s (5202 parameters) |
| Inference, one sequence (features to decisions) | 0.9 ms |

The time step starts at the specified 0.1 s and is halved when the
expected peak speed would push the lattice Mach number past its limit:
25 % of the accepted simulations ran at 0.1 s, 44 % at 0.05 s and 31 % at
0.025 s (strong wind and hot plumes on 5 m cells); the physical frame
times are the same for all of them.

## How it fits together

```
 scenario sampling (context first, class-independent)        src/cfd/scenarios.py
        |  wind, terrain, atmosphere, nuisances  ->  regime  ->  smoke / fog / distractor
        v
 2D CFD, 320 m x 320 m, 64 x 64 cells, CPU                    src/cfd/{lbm,scalars,thermo,smoke,fog,simulation}.py
        |  smoke, fog liquid water, haze/dust, advected texture coordinates
        |  at 12 observation times (5 s apart) and 12 lag times (1 s earlier)
        v
 camera rendering + candidate masks                           src/data/{render,masks}.py
        |  grayscale frames, detector-like masks, nuisances (shake, occlusion,
        |  shadows, JPEG, noise, haze), the same for every class
        v
 motion features, 12 timesteps x 14                           src/data/features.py  <- real footage enters here
        |  DIS optical flow over 1 s lag pairs, region geometry, validity flags
        v
 normaliser (train split only) -> GRU (14 -> 32) -> 16-d embedding -> smoke / fog logits
                                                              src/data/normalization.py, src/models/physics_gru.py
```

## The simulator

A vertical x-z section (64 x 64 cells of 5 m; the camera sees the interior
without the absorbing layers) solved with a D2Q9 lattice-Boltzmann method:

* **Flow:** regularised single-relaxation-time BGK collision with Guo
  forcing and a Smagorinsky subgrid viscosity (low base viscosity, so plumes
  become unsteady); Boussinesq buoyancy from potential temperature, water
  vapour and liquid-water loading relative to the ambient profiles.
  No-slip ground and terrain (bounce-back); velocity inlets and pressure
  outlets at the open sides and top; sponges in physical units; a gust force
  and an ambient-maintaining force keep the wind profile steady over the
  wall. The time step starts at 0.1 s and is reduced when the expected peak
  speed requires it (Mach below 0.25).
* **Scalars:** finite-volume advection-diffusion (minmod MUSCL) of smoke,
  haze/dust (with settling), the moist-conserved variables theta_l and q_t
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
  cloud shadows, camera artefacts, distant clouds). Smoke is not always
  rising and fog is not always still: steam fog rises from a warm wet surface,
  smouldering smoke drifts along the ground, advected smoke and fog banks both
  enter from upwind.
* **Validation (every run):** finite fields, non-negative concentrations,
  Mach number, in-view density deviation, budget residuals, Boussinesq
  validity, and label support (the labelled aerosol must be visible in the
  camera view; unlabelled fog or smoke must be absent). Rejected runs are
  re-drawn and logged with their parameters.

## Data

Context (wind 0-7 m/s, sometimes accelerating over the window, terrain,
atmosphere, spin-up, background haze, camera nuisances) is drawn from the
same distribution for every class, then a
compatible regime is chosen, so no context variable separates the classes by
construction (the evaluation audits this on the accepted data). Splits are by
simulation: `train`, `val_select` (early stopping, seed choice), `val_calib`
(temperature, thresholds, abstention), `test_iid`, and `test_ood` built from
held-out recipes in every class: wind of 7-9 m/s (e.g. fast-moving fog) and
sources on 45-80 m hills (unseen source locations and terrain). Seeds derive
from (split, category, index, attempt), so no random stream is shared across
splits.

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

Twelve observations 5 s apart span 55 s. Flow is measured over a 1 s lag pair
at each observation (DIS optical flow), because flow over the 5 s observation
spacing loses the texture correspondence. At the first observation the
between-observation derivatives (area growth, centroid velocity, anchor
stability) are 0; they have no flag of their own (the 14-feature layout
has only the two validity flags), which is unambiguous because they are
always the GRU's first step, while the 1 s lag flow at t = 0 is valid. Camera shake is removed by a
similarity transform estimated from background corners outside the candidate
masks; when no estimate is possible (a frame-filling fog bank, an untextured
sky) the flow features of that clip are gated off rather than left to
measure camera motion (`FeatureConfig.require_stabilization`, to be disabled
only for rigid, static mounts). Lengths are in units of a region of
interest around the candidate masks, so features do not depend on how large
the scene appears in the image. Footage whose frame spacing cannot provide
1 s pairs (for example one-minute HPWREN/FIgLib imagery) is refused rather
than treated as motion footage.

## Usage

Everything runs on a CPU (the system targets one core; `--workers` only adds
throughput). Install PyTorch from the CPU wheel index, then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest -m "not slow"          # fast tests; drop -m to include the convergence checks
```

Stage A, synthetic pretraining and evaluation (the commands that produced
`artifacts/` and `reports/`):

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
    --summary data/synthetic/summary.json --loro --proxy data/synthetic_proxy/features.npz
python scripts/oracle_diagnostic.py --data data/synthetic --out reports/oracle_diagnostic.json
python scripts/benchmark_cpu.py --out reports/benchmark_cpu.json --features data/synthetic/features.npz
python scripts/summarize_results.py --evaluation reports/evaluation.json \
    --oracle reports/oracle_diagnostic.json --benchmark reports/benchmark_cpu.json \
    --qa reports/qa_full.json --out reports/results.md
```

The synthetic proxy set (`--proxy`) is a small, separately seeded dataset
rendered with detector / box masks and harsher imaging; it is used only for
the pretrained-vs-scratch transfer proxy and is not real-world evidence:

```bash
python scripts/generate_dataset.py --out data/synthetic_proxy --n-per-category 60 --ood-per-category 0 \
    --set seed=20270301 --set 'render.masks.source_probs={detector: 0.7, bbox: 0.3}' \
    --set 'render.noise_sigma=[0.03, 0.06]' --set 'render.jpeg_quality=[20, 60]' --set render.jpeg_prob=0.9 \
    --set 'render.contrast=[0.45, 0.9]' --set 'render.global_haze=[0.2, 0.45]' --workers 4
```

### Inference

```python
from src.inference.predict import PhysicsExpert

expert = PhysicsExpert.load("artifacts")
X, meta = expert.features_from_frames(frames, timestamps, masks, lag_frames, lag_timestamps)
emb = expert.encode(X)                                   # (B, 16) embedding for the fusion network
out = expert.predict(X, operating_point="balanced", meta=meta)
out["decisions"]          # (B, 2) smoke, fog: 1 present, 0 absent, -1 abstain
out["calibration_source"] # "synthetic_val_calib" until real footage recalibrates it
```

`frames` are 12 observations about 5 s apart, `lag_frames` one frame about
1 s before each; `masks` are the vision detector's candidate masks (or None
for the built-in fallback detector). `predict` abstains when the input is
not measurable (too few valid masks or flow pairs, out-of-range features,
incompatible frame timing) or the calibrated logit is within the fitted
margin of the threshold. `operating_point="low_false_alarm"` uses the smoke
threshold fitted for at most 5 % false smoke on fog-only validation clips.

### Stage B, real footage

Not run: no labelled real surveillance video with compatible frame timing was
available, so `evaluation.json` reports `"real": {"status": "pending"}` and
experiments B and D are `null`. With such footage:

```bash
# real.csv: video_path, camera_id, video_id, smoke, fog[, split, start_s, stride_s, max_windows, mask_dir]
python scripts/extract_features.py --real-manifest real.csv --out data/real/features.npz
python scripts/evaluate_physics.py --features data/synthetic/features.npz --artifacts artifacts \
    --reports reports --real data/real/features.npz     # trains B (scratch) and D (fine-tune), evaluates B, C, D
```

Splits are camera-disjoint (group = camera id). D fine-tunes the pretrained
model at `finetune.lr` = 1e-4; B trains the same architecture from random
initialisation on the same real splits at `training.lr`; both use the same
epochs, patience, seeds and normaliser policy, and the calibration is
refitted on the real `val_calib` split. `scripts/train_physics.py --stage
finetune|scratch` runs either one on its own.

## Repository layout

```
configs/      simulation.yaml (CFD, scenarios, rendering, dataset), training.yaml
src/cfd/      lbm, scalars, thermo, smoke, fog, scenarios, scenario_util, simulation, validation, results
src/data/     render, masks, features, feature_spec, normalization, dataset, pipeline, visualize
src/models/   physics_gru
src/training/ train, evaluate, metrics, baselines, audit
src/inference/predict (no training dependencies)
scripts/      generate_dataset, qa_dataset, extract_features, train_physics, evaluate_physics,
              oracle_diagnostic, benchmark_cpu, summarize_results
tests/        test_lbm, test_scalars, test_simulation, test_render, test_features, test_pipeline,
              test_model, test_training
docs/         spec.md (requirements), design.md (interface contract and design decisions)
artifacts/    exported model (encoder, classifier, normaliser, config, evaluation)
reports/      evaluation.json, results.md, QA, oracle diagnostic, benchmarks, figures
```
