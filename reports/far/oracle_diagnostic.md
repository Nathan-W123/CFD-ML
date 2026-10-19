# CFD-oracle diagnostic (synthetic only; oracle features are never model inputs)

Data: `data/synthetic_far`; simulations per split: {'test_iid': 148, 'test_ood': 200, 'train': 600, 'val_calib': 152, 'val_select': 99}.

Pooled logistic regression, test AUROC [95 % grouped-bootstrap CI]:

| features | label | oracle test_iid | camera test_iid | oracle test_ood | camera test_ood |
|---|---|---|---|---|---|
| velocity | smoke | 0.658 [0.56, 0.75] | 0.577 [0.51, 0.64] | 0.728 [0.65, 0.79] | 0.536 [0.48, 0.59] |
| velocity | fog | 0.623 [0.53, 0.71] | 0.495 [0.43, 0.56] | 0.613 [0.54, 0.69] | 0.503 [0.44, 0.56] |
| region | smoke | 0.749 [0.67, 0.82] | 0.541 [0.47, 0.62] | 0.765 [0.70, 0.83] | 0.555 [0.50, 0.61] |
| region | fog | 0.789 [0.71, 0.86] | 0.645 [0.57, 0.71] | 0.748 [0.68, 0.81] | 0.688 [0.63, 0.75] |
| all | smoke | 0.746 [0.66, 0.82] | 0.572 [0.50, 0.65] | 0.771 [0.71, 0.83] | 0.554 [0.50, 0.60] |
| all | fog | 0.806 [0.72, 0.87] | 0.636 [0.56, 0.71] | 0.724 [0.65, 0.79] | 0.660 [0.60, 0.72] |

Univariate AUROC on the train split (time mean over valid steps, folded):

| feature | smoke oracle | smoke camera | fog oracle | fog camera |
|---|---|---|---|---|
| flow_u_mean | 0.503 | 0.505 | 0.508 | 0.514 |
| flow_v_mean | 0.569 | 0.503 | 0.609 | 0.504 |
| flow_u_std | 0.653 | 0.534 | 0.597 | 0.534 |
| flow_v_std | 0.639 | 0.541 | 0.572 | 0.522 |
| flow_divergence | 0.566 | 0.551 | 0.516 | 0.515 |
| flow_vorticity_abs | 0.652 | 0.542 | 0.635 | 0.531 |
| region_area | 0.56 | 0.537 | 0.547 | 0.623 |
| area_growth | 0.55 | 0.525 | 0.541 | 0.523 |
| centroid_dx | 0.516 | 0.524 | 0.508 | 0.516 |
| centroid_dz | 0.526 | 0.514 | 0.51 | 0.519 |
| boundary_roughness | 0.547 | 0.529 | 0.716 | 0.615 |
| anchor_stability | 0.6 | 0.579 | 0.582 | 0.577 |
