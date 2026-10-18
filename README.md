# CFD-ML: a CFD-pretrained physics expert for wildfire smoke vs fog

**A small temporal network (GRU) that tells wildfire smoke from fog by how it
moves, pretrained on a 2D lattice-Boltzmann simulator of smoke, fog and
look-alikes, and producing a 16-dimensional motion embedding for a
multimodal wildfire detector.**

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
> pretrained-vs-scratch comparison are implemented and documented below, and
> are pending labelled surveillance video.

RESULTS_PLACEHOLDER

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

Context (wind 0-7 m/s, terrain, atmosphere, spin-up, background haze, camera
nuisances) is drawn from the same distribution for every class, then a
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
spacing loses the texture correspondence. Lengths are in units of a region of
interest around the candidate masks, so features do not depend on how large
the scene appears in the image. Footage whose frame spacing cannot provide
1 s pairs (for example one-minute HPWREN/FIgLib imagery) is refused rather
than treated as motion footage.

## Repository layout

```
configs/      simulation.yaml, training.yaml
src/cfd/      lbm, scalars, thermo, smoke, fog, scenarios, simulation, validation, results
src/data/     render, masks, features, feature_spec, normalization, dataset, pipeline, visualize
src/models/   physics_gru
src/training/ train, evaluate, metrics, baselines
src/inference/predict (no training dependencies)
scripts/      generate_dataset, extract_features, train_physics, evaluate_physics, benchmark_cpu
tests/        test_lbm, test_scalars, test_simulation, test_render, test_features, test_model, test_training
docs/         spec.md (requirements), design.md (interface contract)
artifacts/    exported model (encoder, classifier, normaliser, config, evaluation)
reports/      metrics, figures, benchmarks
```
