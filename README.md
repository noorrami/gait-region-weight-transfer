# Transfer of Gait Region Weights from Bag Carrying to Coat Wearing

Author: Noor Saad. Source release: v1.2.0. Public implementation, tests, study specifications and aggregate findings.

The classical study tests bag-selected upper/lower GEI score fusion on coat identification with disjoint fitting, validation and test people. The original test is separate from ten overlapping post-test exploratory splits. The cancelled full-body GaitSet branch is excluded.

## Data-free checks

    python verify_release.py
    python -m pytest -q
    python verify_aggregate_results.py

The current suite contains 61 synthetic tests. They validate mathematical/protocol safeguards, not gait recognition accuracy.

## Complete reproduction from the authorized archive

Create a new Python 3.14 environment and install requirements-reproduction.txt. Supply the exact archive yourself, then run:

    python scripts/reproduce_from_archive.py --archive /path/to/casia-b-v1.zip --output /new/path/run-01

The output directory must not exist. The route rebuilds the sequence inventory, roles, all required GEIs, models, selection locks, original test, person-block sensitivity and all ten split fits/tests. It does not need the author's private manifests, images already extracted, trained models or previous scores. It retains integrity checks with newly generated reproduction hash chains. See RAW_REPRODUCTION.md.

The complete route was tested in a newly created virtual environment with freshly installed pinned dependencies on the recorded Windows host. Regenerated GEIs, score arrays and prediction rows matched retained originals; aggregate references also matched. See RAW_REPRODUCTION_REPORT.json. This is retrospective computational reproduction, not independent confirmation on new people or acquisition systems. Other OS/BLAS combinations were not automatically validated.

## Scientific findings

Original LDA CL BG-selected minus equal fusion was -3.19 percentage points. Its percentile interval excluded zero; the post-test person-block sign-flip sensitivity gave p=0.109375. Ten exploratory splits gave five negative and five identical LDA coat contrasts. Lower-only exceeded BG-selected on coats in all ten for both backbones. Overlapping splits are not independent replications.

## Data and licensing

The input is the pinned processed CASIA-B mirror, not verified bit-identical original-provider silhouettes. Dataset access/reuse conditions remain separate. No images, GEIs, fitted models or identity-level predictions are distributed. Keep generated reproduction workspaces private.

No additional code reuse license has been assigned; see LICENSE_STATUS.md. Version DOI: https://doi.org/10.5281/zenodo.24229763 . The original v1.1.0 archive remains at https://doi.org/10.5281/zenodo.24225532 . Cite the version in CITATION.cff. Software and methodological references are in SOFTWARE.md.

## Ordinal sensitivity supplement

See ORDINAL_ANALYSIS.md and rank_analysis/ for the new post-hoc analysis, all eleven split outcomes and the commands to regenerate them after the raw route. ORCID: https://orcid.org/0009-0008-5726-0942 . The original scientific source is unchanged; ordinal code is a separate addition.
