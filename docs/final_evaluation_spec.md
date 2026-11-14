# Final evaluation execution specification v1

Written before first held-out recognition scores, 2026-11-12. Extends the frozen protocol with execution details only; no selected projection, normalization, weight, view, region, participant or endpoint is changed.

Require verified frozen-plan/source hashes, completed development state, selected-model lock and exact test-person manifest. Preserve all pre-test artifacts. Hash this specification, evaluation code and mathematical tests in `configs/evaluation.v1.json` before preparing held-out features. Output directories refuse overwriting v1. Any failed integrity operation is retained and documented; never silently exclude additional clips or people.

Prepare all ten supplied sequences per retained test person at 54/90, using the existing >=128 threshold, >=20 frames, original 64x64 canvas and whole-clip mean. No fitting, new normalization or model selection. Enroll each test person using the four locked-projection NM01-04 vectors as specified. Probe NM05-06, BG01-02 and CL01-02. Use identical feature/clip sets across all six comparisons and both backbones.

Retain raw full-body cosine scores and fitting-normalized upper/lower distances, gallery/probe IDs, predictions, exact ties and correctness indexed by person/view/repetition. Evaluate full-body, equal fusion, fixed BG-selected fusion, fixed NM-selected control, upper-only and lower-only. Report all per-view Rank-1 values and equal-view means, without choosing a favorable backbone/view.

Primary: PCA+regularized-LDA CL Rank-1 difference (BG-selected minus equal fusion), percentage points. Compute participant differences averaging two views and both probe repetitions within each person. Bootstrap these complete participant units with replacement, using NumPy default_rng(seed=20261111), 10,000 replicates. Use the 2.5th/97.5th percentiles with linear quantile interpolation for a 95% interval. No p-value or multiplicity claim is added. This specifies the interval method left implicit in the earlier protocol, before any test outcomes are known.

Apply the same paired-person procedure descriptively to CL differences versus full-body/upper/lower/NM-selected controls, the PCA-backbone CL contrasts, and BG/NM selected-minus-equal differences. Use the same seed/draw pattern across contrasts. Secondary intervals are unadjusted and descriptive. Gallery identities/templates and learned models remain fixed: intervals do not quantify retraining, gallery sampling, dataset-release changes or other populations.

Time feature threshold/GEI computation separately from read/decode/stack and output I/O. For projection/template/scoring, record one in-memory pass at one numerical thread; call it descriptive timing, not a repeated cold/warm latency benchmark. Keep model fitting time from the development record distinct.

Do not tune after test inspection. Preserve null/negative outcomes. Any later correction or new experiment needs a dated amendment identifying whether outcomes were already seen. Dataset preprocessing/provenance limitations and bounded literature review remain limitations after evaluation.
