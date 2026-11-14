# Transfer of Gait Region Weights from Bag Carrying to Coat Wearing

Author: Noor Saad. Source release: v1.0.0. Repository visibility: private, as explicitly requested by the author.

This repository contains the classical GEI/PCA/regularized-LDA implementation, synthetic tests, frozen study specifications and aggregate results for a person-disjoint bag-to-coat weight-transfer study. The original test is distinguished from ten overlapping post-test exploratory splits. The cancelled full-body GaitSet branch is excluded.

## Data-free verification

Using the environment recorded in requirements.txt:

    python verify_release.py
    python -m pytest -q
    python verify_aggregate_results.py

The reviewed snapshot passed 51 synthetic tests in a separate source directory using the recorded installed dependencies. These tests check mathematical/protocol behavior; the aggregate script checks summary arithmetic, not individual recognition predictions.

## Results and scope

The original LDA coat contrast (BG-selected minus equal fusion) was -3.19 percentage points. Its prespecified percentile interval excluded zero, while the post-test person-block sign-flip sensitivity gave p=0.109375. In ten overlapping exploratory splits, LDA coat contrasts were negative in five and identical in five. Lower-only exceeded BG-selected fusion on coats in all ten for both backbones. These splits are not independent confirmation or a new dataset.

## Reproduction boundaries

No source images, frame arrays, GEIs, fitted models or identity-level predictions are distributed. Dataset-dependent scripts retain the original path, role and hash checks and require the author's private authorized artifacts. This source release is not a one-command reproduction from raw data. See REPRODUCIBILITY.md and SOFTWARE.md. Access to a mirror does not establish original-provider bit identity or redistribution permission.

## Licensing and citation

No additional code reuse license has been selected for this private release. See LICENSE_STATUS.md. Dataset and dependency licenses remain separate. CITATION.cff identifies the software; no Zenodo DOI or published-paper DOI is asserted. The repository is not a public code-availability claim.
