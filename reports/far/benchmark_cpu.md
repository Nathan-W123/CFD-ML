| Stage | Single-core time |
|---|---|
| CFD simulation (accepted, incl. precursor) | 8.6 s/sim (417 sims/h) |
| Render 12 frames (no lag frames) | 119 ms |
| Feature extraction (12 x 14) | 75 ms |
| GRU training epoch (1799 sequences) | 0.15 s |
| Inference, one sequence (features -> probabilities) | 0.80 ms |
| Inference, batch of 256 | 3.9 ms |
