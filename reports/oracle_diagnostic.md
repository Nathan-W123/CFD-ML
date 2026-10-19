# CFD-oracle diagnostic (synthetic only; oracle features are never model inputs)

Data: `data/synthetic`; simulations per split: {'test_iid': 148, 'test_ood': 199, 'train': 600, 'val_calib': 151, 'val_select': 100}.

Pooled logistic regression, test AUROC [95 % grouped-bootstrap CI]:

| features | label | oracle test_iid | camera test_iid | oracle test_ood | camera test_ood |
|---|---|---|---|---|---|
| velocity | smoke | 0.744 [0.66, 0.82] | 0.552 [0.48, 0.62] | 0.707 [0.63, 0.77] | 0.603 [0.54, 0.67] |
| velocity | fog | 0.628 [0.54, 0.72] | 0.625 [0.55, 0.69] | 0.488 [0.41, 0.57] | 0.567 [0.51, 0.63] |
| region | smoke | 0.715 [0.63, 0.80] | 0.615 [0.55, 0.68] | 0.734 [0.66, 0.80] | 0.526 [0.46, 0.59] |
| region | fog | 0.727 [0.65, 0.81] | 0.733 [0.68, 0.79] | 0.768 [0.70, 0.84] | 0.680 [0.62, 0.74] |
| all | smoke | 0.783 [0.71, 0.85] | 0.600 [0.53, 0.67] | 0.745 [0.67, 0.81] | 0.590 [0.53, 0.65] |
| all | fog | 0.751 [0.67, 0.83] | 0.728 [0.67, 0.79] | 0.663 [0.59, 0.73] | 0.636 [0.57, 0.69] |

Univariate AUROC on the train split (time mean over valid steps, folded):

| feature | smoke oracle | smoke camera | fog oracle | fog camera |
|---|---|---|---|---|
| flow_u_mean | 0.529 | 0.522 | 0.54 | 0.524 |
| flow_v_mean | 0.596 | 0.518 | 0.503 | 0.513 |
| flow_u_std | 0.596 | 0.572 | 0.543 | 0.522 |
| flow_v_std | 0.698 | 0.558 | 0.567 | 0.565 |
| flow_divergence | 0.563 | 0.541 | 0.542 | 0.535 |
| flow_vorticity_abs | 0.561 | 0.569 | 0.506 | 0.549 |
| region_area | 0.55 | 0.518 | 0.576 | 0.625 |
| area_growth | 0.527 | 0.556 | 0.554 | 0.524 |
| centroid_dx | 0.508 | 0.531 | 0.532 | 0.504 |
| centroid_dz | 0.55 | 0.514 | 0.546 | 0.52 |
| boundary_roughness | 0.538 | 0.539 | 0.557 | 0.625 |
| anchor_stability | 0.555 | 0.558 | 0.524 | 0.547 |
