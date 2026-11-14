# Projection and score specification v0.1

Status: draft implementation specified before any real-data model fitting or recognition outcomes, 2026-11-12. This document completes numerical conventions left open in the active protocol; it does not freeze the experiment or claim a new algorithm.

## Fitting and input

Rows are flattened whole/upper/lower GEIs, float64, with no feature-wise variance normalization or PCA whitening. Each view and region has its own fitting mean and projection. Only retained fitting people, NM01-04, may fit projections. NM05-06 are reserved for score normalization; BG/CL, validation and test observations cannot fit them. Callers must check verified manifests; argument guards are correctness checks, not security controls.

## PCA

For n fitting rows, Xc = X - mean_fit. Compute reduced deterministic SVD Xc = U diag(s) V^T using numpy.linalg.svd. Retain the requested first k right singular vectors, without whitening. Numerical rank counts s > max(n,d) * machine_epsilon * s_max. Reject infeasible dimensions instead of clipping. Covariance eigenvalues are s^2/(n-1). Canonicalize each column sign by making its largest absolute loading positive. This fixes sign ambiguity, not rotations in exactly repeated eigenspaces; save the actual learned arrays and environment for replay.

## PCA followed by regularized LDA

Learn the p-dimensional PCA precursor on the same permitted fitting rows. In its coordinates Z, with class counts n_c, means m_c and global m:

Sw = (1/n) sum_c sum_i_in_c (z_i-m_c)(z_i-m_c)^T.

Sb = (1/n) sum_c n_c (m_c-m)(m_c-m)^T.

Set a = trace(Sw)/p and B = Sw + lambda*a*I. Reject nonpositive a and nonpositive/nonfinite lambda. Solve Sb*v = rho*B*v using scipy.linalg.eigh(type=1), order descending rho, and retain k positive discriminant directions. Reject k > min(p,C-1,positive_numerical_rank). Positive rank uses p*epsilon*max(abs(rho)); no arbitrary null directions are filled in.

Generalized eigenvectors retain B-orthonormal scaling, v_i^T B v_j = delta_ij. Do not Euclidean-normalize each column or apply another eigenvalue weighting: these alter multi-dimensional cosine geometry. Compose W = V_PCA * V_LDA and canonicalize signs. Transform every later row by (x-mean_fit)*W. Save precursor dimension, lambda, scatter scale, classes, rank and spectrum with W and mean. The existing p={40,80}, lambda={.001,.01,.1}, B_feature={10,20,30} grids remain unchanged.

## Gallery and score

L2-normalize each projected NM01-04 clip, average those four vectors for a person/view, then L2-normalize the average as one gallery template. Reject zero/numerically degenerate rows or templates; do not add an undocumented epsilon to create them. Cosine distance is 1 - clip(unit_probe dot unit_template, -1, 1), with smaller values favorable. No probe-set centering or normalization is learned.

Regional distance mean/SD are fitted only from fitting-person NM05-06 impostor distances. Primary BG-selected and equal fusion share every region model and normalization. The explicitly separate secondary NM-selected control uses validation NM05-06 with those same already fixed regional models; it does not replace primary BG selection or admit CL tuning. Full-body selection uses its separate permitted validation grid as stated in the protocol.

## Verification and limits

Synthetic tests verify PCA reconstruction/orthogonality, generalized eigen-equations and B-orthonormality, translation/class-label invariance, infeasible ranks, forbidden fitting roles/conditions, exact gallery aggregation, cosine geometry, and isolation of the secondary control. These are mathematical tests, not gait outcomes. No trained real-data model exists from this stage.

Implementation: `src/gait_transfer/models.py`; selection/normalization: `src/gait_transfer/core.py`; tests: `tests/test_models.py`. Official numerical API references checked 2026-11-12: [NumPy SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html), [SciPy generalized symmetric eigenproblem](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html). These document solver behavior, not the novelty or effectiveness of our study conventions.
