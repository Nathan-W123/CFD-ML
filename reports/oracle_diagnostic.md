# CFD-oracle diagnostic (synthetic only; oracle features are never model inputs)

Data: `data/synthetic`; simulations per split: {'test_iid': 148, 'test_ood': 199, 'train': 600, 'val_calib': 151, 'val_select': 100}.

Pooled logistic regression, test AUROC [95 % grouped-bootstrap CI]:

| features | label | oracle test_iid | camera test_iid | oracle test_ood | camera test_ood |
|---|---|---|---|---|---|
| velocity | smoke | 0.785 [0.71, 0.85] | 0.499 [0.43, 0.57] | 0.744 [0.67, 0.81] | 0.613 [0.55, 0.67] |
| velocity | fog | 0.610 [0.52, 0.69] | 0.624 [0.55, 0.69] | 0.516 [0.44, 0.60] | 0.569 [0.50, 0.64] |
| region | smoke | 0.760 [0.68, 0.84] | 0.560 [0.49, 0.62] | 0.761 [0.69, 0.82] | 0.579 [0.52, 0.64] |
| region | fog | 0.717 [0.63, 0.80] | 0.730 [0.66, 0.80] | 0.787 [0.72, 0.85] | 0.716 [0.66, 0.77] |
| all | smoke | 0.809 [0.74, 0.88] | 0.565 [0.50, 0.63] | 0.770 [0.70, 0.83] | 0.617 [0.56, 0.68] |
| all | fog | 0.753 [0.67, 0.83] | 0.725 [0.66, 0.79] | 0.690 [0.62, 0.76] | 0.686 [0.62, 0.75] |

Univariate AUROC on the train split (time mean over valid steps, folded):

| feature | smoke oracle | smoke camera | fog oracle | fog camera |
|---|---|---|---|---|
| flow_u_mean | 0.538 | 0.518 | 0.533 | 0.53 |
| flow_v_mean | 0.636 | 0.501 | 0.539 | 0.543 |
| flow_u_std | 0.663 | 0.564 | 0.615 | 0.516 |
| flow_v_std | 0.747 | 0.544 | 0.624 | 0.563 |
| flow_divergence | 0.588 | 0.546 | 0.521 | 0.556 |
| flow_vorticity_abs | 0.628 | 0.55 | 0.574 | 0.549 |
| region_area | 0.623 | 0.543 | 0.649 | 0.658 |
| area_growth | 0.542 | 0.555 | 0.541 | 0.519 |
| centroid_dx | 0.512 | 0.519 | 0.528 | 0.506 |
| centroid_dz | 0.55 | 0.501 | 0.542 | 0.501 |
| boundary_roughness | 0.596 | 0.577 | 0.613 | 0.664 |
| anchor_stability | 0.534 | 0.551 | 0.56 | 0.521 |
