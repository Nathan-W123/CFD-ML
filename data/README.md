# data/

Generated datasets live here and are not committed (see `.gitignore`); each
is fully reproducible from `configs/simulation.yaml` and the base seed.

```
python scripts/generate_dataset.py --out data/synthetic_debug --n-per-category 50 --ood-per-category 12
python scripts/generate_dataset.py --out data/synthetic --n-per-category 250 --ood-per-category 50
```

Layout of one dataset directory:

| Path | Content |
|---|---|
| `config_used.yaml` | the merged configuration used (sha256 in `summary.json`) |
| `sims/<split>/<sim_id>.npz` | accepted CFD simulation: smoke / fog LWC / tracer fields at 12 observation and 12 lag times, advected texture coordinates, terrain, every sampled parameter, seed, numerical validity record |
| `seqs/<split>/<seq_id>.npz` | rendered sequence: 12 frames + 12 lag frames (uint8), candidate masks, timestamps, labels, render parameters and seed, simulation parameters and validity |
| `features.npz` | `FeatureSet`: raw 12 x 14 motion features per sequence with labels, splits, groups and provenance |
| `manifest.csv` | one row per sequence |
| `rejected.csv` | every rejected simulation attempt and render retry, with its parameters and reasons |
| `summary.json` | counts, acceptance per category / regime / wind bin, timing |

Splits: `train`, `val_select` (early stopping, seed choice), `val_calib`
(temperature, thresholds, abstention), `test_iid`, `test_ood` (held-out
recipes: fast wind, elevated sources). A simulation and its variants never
span two splits.

Real footage (Stage B) is not included. `scripts/extract_features.py --real-manifest`
turns a CSV of labelled videos into a `FeatureSet` with camera ids as groups.

`backgrounds/` (optional): real landscape images for rendering, with a
`manifest.csv` (`filename,camera_id`). Images from cameras used for real
validation or testing must not be placed here.
