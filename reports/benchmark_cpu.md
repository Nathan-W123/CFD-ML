| Stage | Single-core time |
|---|---|
| CFD simulation (accepted, incl. precursor) | 5.6 s/sim (637 sims/h) |
| Render 12 + 12 lag frames | 142 ms |
| Feature extraction (12 x 14) | 58 ms |
| GRU training epoch (1800 sequences) | 0.17 s |
| Inference, one sequence (features -> probabilities) | 0.92 ms |
| Inference, batch of 256 | 3.0 ms |
