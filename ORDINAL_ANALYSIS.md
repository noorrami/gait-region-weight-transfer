# Post-hoc ordinal fusion and regional separation

This release adds a bounded sensitivity control for the original split and all ten existing exploratory splits, using both original backbones. It does not change the frozen original study, add independent people, refit projections or retune their hyperparameters. Rank-based classifier combination is established (Ho, Hull and Srihari, IEEE TPAMI 16(1), 66-75, 1994, DOI 10.1109/34.273716).

For each probe and region, distances become zero-based average ranks divided by gallery size minus one. Weight selection uses only validation BG with the original grid/tie preference. All split weights are locked before the new test phase. Doubled midranks and integer tenths coefficients avoid floating-point tie ambiguity. The smallest gallery ID resolves exact minimum-cost ties; tie counts are reported for every method/view.

The descriptive margin is (nearest-impostor distance - genuine distance)/candidate-distance range, with zero for constant rows. Genuine percentile ranks are also reported. Each quantity is averaged within person over views/repetitions before median/quartile summaries. Identity labels enter only the descriptive diagnostic and performance evaluation; no coat-driven calibration or tuning is performed. Ordinal fusion discards score spacing, changes the fusion rule and introduces ties, so it does not causally separate calibration from identity information.

## Reproduce

First run the complete raw-archive route from this version into a new directory, as described in RAW_REPRODUCTION.md. It also copies these new source files. From the generated directory, with the same installed environment, run:

    python scripts/run_rank_sensitivity.py select
    python scripts/run_rank_sensitivity.py evaluate
    python scripts/verify_rank_sensitivity.py
    python scripts/export_rank_sensitivity.py

The analysis output directory must not already exist. Selection must complete for all splits before evaluation. Outputs include private per-person predictions/diagnostics: keep the generated workspace private. The public rank_analysis directory contains only aggregate references, not the author's private input manifests or per-person outcomes. Recomputed aggregate_results.csv and aggregate_diagnostics.csv can be compared with those references.

The added analysis was independently replayed and then rerun on the score arrays produced by the earlier full cold-archive reproduction; all six comparison files matched byte for byte. The original cold-archive fit was not repeated for this source-only supplement, and its original report remains historical. This does not assert a new OS/host or independent-population validation. The local 66-file protection snapshot is optional in a generated reproduction workspace; all analysis input hashes are checked regardless.

## All outcomes

Original ordinal LDA: weight .9, BG selected/equal 59.04/54.26%, coat selected/equal 10.11/20.21%; lower-only coat 31.91%. Original ordinal PCA: weight .3, BG 39.89/37.23%, coat 16.49/12.77%; lower-only coat 34.04%. Across the ten dependent exploratory splits, LDA coat differences have 5 negative/5 zero signs (mean -2.90 pp), PCA 3 positive/4 negative/3 zero signs (mean -2.75 pp). Lower-only exceeds selected ordinal coat fusion in all ten splits for both backbones. No new significance test or best-backbone selection was performed. All conditions, views, ties and diagnostics are included in the aggregate files.
