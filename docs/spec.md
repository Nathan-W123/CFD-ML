# Coding Agent Specification: CFD-Pretrained Physics Expert

(The requirements this repository implements, kept verbatim for reference.)

## 1. Objective

Implement, train, evaluate, and export a physics-based neural network that distinguishes wildfire smoke from fog using motion characteristics learned from computational fluid dynamics (CFD).

This model is one component of a larger multimodal wildfire detection system containing:

- Vision Expert — visual appearance
- Physics Expert — fluid motion and evolution
- Weather Expert — atmospheric context
- Fusion Network — final detection

**Scope:** Implement only the Physics Expert and its training pipeline.

The system must run on one CPU core without GPU acceleration.

### Success criteria

1. Produce physically plausible synthetic smoke and fog sequences.
2. Train a small temporal neural network on CFD-generated data.
3. Accept motion features extracted from real surveillance footage.
4. Produce a reusable 16-dimensional physics embedding.
5. Support further fine-tuning on real labeled footage.
6. Evaluate smoke/fog discrimination on held-out simulations and real data when available.

Do not claim real-world accuracy based on synthetic validation alone.

---

## 2. Technology stack

| Component | Implementation |
|---|---|
| CFD solver | Custom 2D Lattice Boltzmann Method |
| Flow discretization | D2Q9 |
| Scalar transport | Finite-difference advection–diffusion |
| Numerical acceleration | NumPy + Numba |
| Motion extraction | OpenCV |
| ML framework | PyTorch |
| Neural architecture | GRU |
| Training hardware | CPU only, one thread |
| Configuration | YAML |
| Dataset storage | Compressed NumPy arrays |
| Experiment tracking | CSV / JSON |
| Tests | pytest |

Use CPU-safe libraries, fixed random seeds, and reproducible configuration files.

Set PyTorch, BLAS, OpenMP, and Numba threading limits to one where supported.

---

## 3. CFD simulation

### 3.1 Numerical configuration

Implement a two-dimensional vertical cross-section (x-z plane).

| Parameter | Initial value |
|---|---|
| Grid | 64 × 64 |
| Approximate domain | 320 × 320 meters |
| Flow solver | D2Q9 LBM, BGK collision |
| Flow regime | Low-Mach, incompressible approximation |
| Buoyancy | Boussinesq approximation |
| Temperature | Transported scalar |
| Smoke concentration | Transported scalar |
| Water vapor | Transported scalar |
| Fog liquid water | Transported scalar |
| Simulation timestep | Start with 0.1 s |
| Simulated sequence | Approximately 60 seconds |
| Output frames | 12 |

The grid size is deliberately small for CPU execution. The domain is a simplified representation, not a high-fidelity wildfire simulation.

Choose the physical-to-lattice parameter mapping consistently. Enforce LBM stability constraints and use a smaller timestep if necessary.

### 3.2 Fluid equations

Implement incompressible flow with thermal buoyancy:

∇ · u = 0

∂u/∂t + (u · ∇)u = −∇p/ρ + ν∇²u + b

where:

- u = velocity field
- p = pressure
- ν = kinematic viscosity
- b = buoyancy acceleration

Use a Boussinesq thermal-buoyancy term based on local temperature relative to ambient temperature.

For the smoke concentration field:

∂C/∂t + u · ∇C = ∇ · (K∇C) + S

where:

- C = smoke concentration
- K = effective diffusion/mixing coefficient
- S = smoke emission source

Temperature must also evolve through advection, diffusion, and heating/cooling terms.

### 3.3 Smoke scenarios

Model smoke as a transported aerosol with an emission source and optional heating.

Randomize:

- Emission position and duration
- Heat-source strength
- Ambient wind velocity
- Wind direction and variability
- Emission rate
- Turbulent mixing
- Initial concentration
- Source intermittency

Generate multiple regimes:

1. Hot buoyant plumes
2. Weakly buoyant smoldering plumes
3. Crosswind-driven plumes
4. Intermittent or pulsing emissions
5. Cooled smoke transported from outside the scene

Do not make every smoke sample rise vertically. That would create an unrealistic classification shortcut.

### 3.4 Fog scenarios

Fog must not be simulated by simply relabeling smoke particles.

Implement simplified moisture transport:

- Water-vapor advection and diffusion
- Temperature-dependent saturation humidity
- Condensation when vapor exceeds saturation
- Evaporation when air becomes unsaturated
- Liquid-water transport
- Near-surface cooling

Use a saturation-adjustment approximation, with nonnegative moisture concentrations and water-mass accounting. Include latent-heat effects where computationally feasible.

Generate:

1. Near-ground radiation fog
2. Advected fog banks
3. Fog developing over uneven terrain
4. Localized fog patches
5. Fog dissipating or shifting in wind

Simulate already-established fog as well as evolving fog. Do not require an entire fog bank to form from clear air within a 60-second window.

### 3.5 Additional classes

Generate four labels using two binary targets:

| Scenario | Smoke target | Fog target |
|---|---:|---:|
| Smoke only | 1 | 0 |
| Fog only | 0 | 1 |
| Neither | 0 | 0 |
| Smoke and fog | 1 | 1 |

The mixed class can be generated by introducing an independent smoke source into a moist, fog-containing simulation.

Neither-class examples should include difficult visual distractors such as drifting haze-like structures and camera-motion artifacts rather than only empty frames.

### 3.6 Boundary conditions and validation

Implement:

- No-slip or appropriate near-ground boundary
- Inflow/outflow conditions for ambient wind
- Open or approximate outflow at the upper boundary
- Stable transport of concentration and moisture fields

Add numerical checks for:

- Nonnegative scalar concentrations
- Finite values
- Acceptable velocity and Mach numbers
- Mass conservation within expected source/sink and boundary-flow effects
- Grid convergence on selected test cases
- Plausible buoyant-plume behavior
- Fog condensation under supersaturated conditions

Reject unstable simulations rather than allowing them into training.

---

## 4. Synthetic video generation

The neural network must **not** receive exact CFD velocity, pressure, or concentration fields as inputs.

Those quantities will not be directly available from real-world cameras.

Instead, create synthetic camera-like sequences using the simulated smoke and fog concentrations.

### Rendering

1. Convert concentration fields into image opacity using a simplified optical-extinction model.
2. Render 12 grayscale frames per scenario.
3. Vary contrast, brightness, haze, image noise, and apparent spatial scale.
4. Generate candidate region masks.
5. Add realistic mask imperfections, partial occlusion, and small camera motion.
6. Optionally composite over varied natural landscape backgrounds.

Apply rendering variations to both smoke and fog. Do not associate unique colors, contrast settings, or background styles with either class.

### Output format

Each generated sequence must include:

- Unique simulation ID
- Scenario category and labels
- 12 rendered frames
- Frame timestamps
- Candidate masks
- Simulation parameters
- Random seed
- Numerical validity metadata

Use the same motion-feature extraction pipeline for synthetic and real footage.

---

## 5. Physics feature extraction

Use OpenCV optical flow, candidate-region masks, and geometric measurements.

For each frame, compute 12 physical-motion features:

| # | Feature | Definition |
|---|---|---|
| 1 | Horizontal flow | Mean horizontal optical-flow velocity |
| 2 | Vertical flow | Mean vertical optical-flow velocity, positive upward |
| 3 | Horizontal variability | Standard deviation of horizontal flow |
| 4 | Vertical variability | Standard deviation of vertical flow |
| 5 | Divergence | Mean flow divergence within candidate region |
| 6 | Vorticity | Mean absolute 2D flow curl |
| 7 | Region area | Candidate-mask area divided by image area |
| 8 | Area growth | Time derivative of log region area |
| 9 | Horizontal centroid motion | Change in centroid x-position per second |
| 10 | Vertical centroid motion | Change in centroid z-position per second |
| 11 | Boundary roughness | Perimeter-based shape irregularity |
| 12 | Source-anchor stability | Stability of the region's lower boundary over consecutive frames |

Normalize spatial measurements using frame dimensions and elapsed time.

Add two validity features:

13. Candidate mask valid
14. Optical flow valid

**Final neural-network input: 12 timesteps × 14 features.**

For each input window:

- Use 12 observations spanning approximately 60 seconds.
- Handle the first frame's missing temporal derivatives with zero values and invalidity flags.
- Handle empty candidate masks explicitly.
- Use training-set-only normalization statistics.
- Store preprocessing metadata with the trained model.

For real footage, stabilize the camera when necessary and obtain candidate masks from a detector or segmentation model.

Never substitute clean simulation masks during real-world inference.

---

## 6. Neural network architecture

Implement a small GRU encoder.

| Parameter | Specification |
|---|---|
| Input shape | (batch, 12, 14) |
| GRU layers | 1 |
| GRU hidden size | 32 |
| Direction | Unidirectional |
| Final hidden state | 32 dimensions |
| Dropout after GRU | 0.1 |
| Projection | Linear 32 → 16 |
| Projection normalization | LayerNorm |
| Embedding activation | GELU |
| Output embedding | 16 dimensions |
| Classification head | Linear 16 → 2 |
| Output activation | Sigmoid for inference |

The two outputs estimate smoke presence and fog presence independently. Use logits during training.

**The 16-dimensional embedding is the primary output for the future multimodal fusion network.**

Expose:

- `encode(features)` → physics embedding
- `predict(features)` → smoke/fog logits and probabilities

Do not require the classifier head for embedding extraction.

---

## 7. Training procedure

### Stage A — CFD pretraining

1. Generate valid synthetic simulations.
2. Render frames.
3. Extract physical-motion features.
4. Build labeled feature sequences.
5. Split data by independent simulation runs and parameter regimes.
6. Train the GRU using supervised learning.

Initial training configuration:

| Parameter | Value |
|---|---|
| Loss | BCEWithLogitsLoss |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 64 |
| Maximum epochs | 50 |
| Early stopping patience | 7 |
| Gradient clipping | 1.0 |
| Training runs | 3 random seeds |
| Compute | One CPU core |

Use class-balanced sampling or loss weighting if the dataset is imbalanced.

Use the validation set for model selection and threshold tuning.

Do not use the test set for tuning.

### Initial dataset target

Start with:

- 50 simulations per scenario category for pipeline debugging.
- Expand toward 250 independent simulations per category after the pipeline passes numerical and data-quality tests.
- Generate multiple visual/rendering variants from each simulation where useful.

The full initial target is approximately **1,000 independent CFD scenarios**, plus augmentations.

These numbers are provisional engineering targets, not a guarantee of classification performance or CPU runtime. Benchmark simulation throughput before scaling.

### Stage B — Real-world fine-tuning

When suitable labeled surveillance videos are available:

1. Extract the same 14 features.
2. Load the CFD-pretrained GRU.
3. Fine-tune using a learning rate of 0.0001.
4. Compare against a randomly initialized GRU trained on the same real data.
5. Evaluate on videos from camera locations not used in training.

Do not mix frames from one original video across training and testing.

Ensure training and evaluation frame intervals are compatible. For example, HPWREN FIgLib often contains one-minute imagery, which should not be treated as high-frame-rate motion footage.

If no suitable real training videos are available, complete synthetic pretraining and report that real-world validation is pending. Do not manufacture validation results.

---

## 8. Evaluation

Report separate synthetic and real-world results.

### Classification metrics

- Smoke precision
- Smoke recall
- Fog precision
- Fog recall
- Smoke/fog confusion
- F1 score per label
- False-smoke predictions on fog-only sequences
- Calibration of predicted probabilities

### Ablation experiments

Train and compare:

| Experiment | Description |
|---|---|
| A | Motion-feature heuristic baseline |
| B | GRU without CFD pretraining |
| C | CFD-pretrained GRU |
| D | CFD-pretrained GRU fine-tuned on real data |

Compare B and C only when training conditions make the comparison meaningful. For a fair pretraining experiment, B and C should receive the same real fine-tuning and evaluation data, and differing amounts of training data must be disclosed.

Test robustness against:

- Wind variation
- Weakly buoyant smoke
- Fast-moving fog
- Partial occlusion
- Camera shake
- Rendering changes
- Unseen source locations
- Unseen simulation parameter combinations

Choose final probability thresholds using validation data, and provide an uncertain/abstain option.

The critical final outcome is whether CFD pretraining improves discrimination on **unseen real camera footage**, not merely on similarly generated synthetic sequences.

---

## 9. Required repository structure

configs/{simulation,training}.yaml; src/cfd/{lbm,scalars,smoke,fog,scenarios}.py;
src/data/{render,features,dataset,normalization}.py; src/models/physics_gru.py;
src/training/{train,evaluate}.py; src/inference/predict.py;
scripts/{generate_dataset,extract_features,train_physics,evaluate_physics,benchmark_cpu}.py;
tests/{test_lbm,test_scalars,test_features,test_model,test_training}.py;
data/, artifacts/, reports/, requirements.txt, README.md.

Use clear interfaces between simulation, rendering, feature extraction, training, and inference.

---

## 10. Final deliverables

1. Functional 2D smoke/fog CFD simulator.
2. Reproducible synthetic dataset generation scripts.
3. Visualization examples for smoke, fog, both, and neither.
4. Shared synthetic/real motion-feature extractor.
5. Trained GRU checkpoint.
6. Fitted feature normalizer.
7. Validation and test metrics.
8. Confusion matrices and training curves.
9. CPU simulation and training benchmarks.
10. Documented inference API returning the 16-dimensional embedding.

Required model artifacts: physics_encoder.pt, physics_classifier.pt, feature_normalizer.json, model_config.yaml, evaluation.json.

Acceptance tests — the implementation is complete when:

- The simulator passes numerical sanity checks.
- The feature extractor produces finite arrays with shape (12, 14).
- Training completes on CPU with one thread.
- Fixed-seed runs are reproducible within reasonable numerical tolerance.
- The saved model can be reloaded without training dependencies.
- The encoder returns shape (batch, 16).
- The classifier returns shape (batch, 2).
- Held-out scenario evaluation runs without data leakage.
- Real camera features can be processed using the same interface.
- All required scripts are documented and executable.

Implementation priorities: (1) stable simulation and physically meaningful synthetic data; (2) correct real-compatible motion features and a working pretrained encoder; (3) reliable experimental evaluation; (4) real footage fine-tuning and integration readiness.

Avoid spending time on visualization polish, 3D CFD, large transformer architectures, distributed training, or end-to-end rendering networks until the core physics expert has been validated. Build the smallest technically defensible implementation first, benchmark it, and increase simulation fidelity only where the measurements justify doing so.

**Final objective:** Deliver a CFD-pretrained physics encoder that can be integrated into a multimodal wildfire detection system and empirically tested for reducing fog-related false wildfire alerts.
