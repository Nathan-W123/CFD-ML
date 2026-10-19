| Stage | Single-core time |
|---|---|
| CFD simulation (accepted, incl. precursor) | 6.4 s/sim (565 sims/h) |
| Render 12 + 12 lag frames | 167 ms |
| Feature extraction (12 x 14) | 80 ms |
| GRU training epoch (2000 sequences) | 0.21 s |
| Inference, one sequence (features -> probabilities) | 0.99 ms |
| Inference, batch of 256 | 3.8 ms |
