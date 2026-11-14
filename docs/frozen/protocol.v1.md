# Study protocol draft v0.1

Title: **Transfer of Gait Region Weights from Bag Carrying to Coat Wearing**.
Adopted by the user on 2026-11-11. Academic registration/approval is not inferred from user adoption. No real-data experiment has run.

## Question, estimands and scope

Can upper/lower-body fusion weights selected using bag-carrying development data improve coat-wearing Rank-1 identification on unseen people without coat-driven selection?

The primary estimand is the paired difference in coat Rank-1 percentage points between bag-tuned regional fusion and equal regional fusion, averaged equally over 54 and 90 degree same-view evaluations, for the PCA-then-regularized-LDA backbone. PCA is a prespecified secondary replication. Do not select the better backbone on final test.

Distinct questions: (a) does bag-selected fusion help on held-out bag probes? (b) does that benefit transfer to coat probes? (c) does it exceed lower-region-only scoring, rather than simply selecting that region? Directional hypotheses predict improvement but permit a null or harmful effect. If the development-selected weight is 0.5, weighted and equal fusion coincide. If it is an endpoint, report that it selects one region. A positive point estimate alone is not proof of transfer or superiority.

## Professor requirement mapping

| Required item | Implementation plan |
|---|---|
| Full-body and upper/lower GEIs | Fixed GEI representation and horizontal division for every clip |
| PCA and LDA references | Full-body PCA and PCA+regularized LDA; explicitly state regularization |
| Normalized regional scores | Training-only normal-probe impostor distance mean/SD per region, view and representation |
| Equal versus bag-tuned weights | Same regional models, normalization, templates and clips; only weight differs |
| Freeze bag weight for coat | One weight per backbone, shared across both views, selected on validation BG only |
| Two prespecified views | 54 and 90 degrees; same-view identification, not a cross-view claim |
| People separated | Draft IDs 001-050 fit, 051-074 validation, 075-124 test |
| Identical clips | Shared inventory, exclusions and pair/score matrices for every method |
| Metrics | Rank-1 by view/condition, paired differences and feature extraction time |

## Data access, release and audit gate

Amendment 2026-11-12, before any recognition outcomes: the user authorized use of the downloaded processed Kaggle release `trnquanghuyn/casia-b`, version 1. Adopt it as a provenance-limited processed CASIA-B mirror, not as verified bit-identical original silhouettes or CASIA-B*. Source processing history remains unknown and the uploader license field is Unknown. Local access/integrity gates now refer to this pinned release, not proof of original completeness, unmodified segmentation or redistribution permission. Preserve these limitations in the manuscript. Do not mix another silhouette release into this study. Record provider URL, access conditions, archive/version, checksum and retrieval date.

Audit sequence IDs (person, condition, repetition, view), filenames versus directories, frame counts, readability, binary foreground encoding, empty masks, missing clips, spatial dimensions and duplicate content. Audit test counts/integrity without computing recognition scores or selecting favorable exclusions. Define exclusions before fitting; report condition/view/participant counts and use the same retained clips for every comparison. Do not treat frames as independent subjects. Preserve original files; derived GEIs go under data/processed with hashes and extraction settings.

## Split and experimental roles

Original assignments are 001-050 fitting / 051-074 validation / 075-124 test; all conditions, views and frames of each person remain in one role. Integrity exclusions never reassign people to another role. The first 74 development / last 50 test division resembles a common benchmark split, but using fewer people for projection fitting is not equivalent to papers training on all 74. Do not compare headline accuracy as if protocols match. Actual retained counts come from the verified manifest.

Before fitting, require at least 20 supplied frames in each required clip at both prescribed views. This audit-informed practical minimum rejects extremely short clips; it is not an empirically optimal threshold or proof of complete gait cycles. Fitting requires NM01-06; validation requires NM01-06 and BG01-02; test requires NM01-06, BG01-02 and CL01-02. Exclude a person within their assigned role if any required clip is missing/short. Do not exclude development identities based on unused CL clips. Every method shares retained people and clips. Record all reasons; no frame-count threshold search using recognition outcomes.

For fitting people: learn centering/PCA/LDA from NM01-04 only; use NM05-06 as a reserved fitting-person probe set for score normalization, against NM01-04 templates. This normalization is fitted once using cross-person impostor distances, not recomputed on validation/test or each probe's score distribution. All choices for zero SD must be specified; the primitive raises instead of hiding the problem.

For validation people: enroll with NM01-04, probe with BG01-02 to select dimensions, ridge and weight. No fitting-person gallery templates enter validation. No CL sequence in any role is used to choose regions, feature preparation, hyperparameters or weights. Integrity auditing is allowed. Validation BG is a selection set, not an unbiased estimate of final performance.

For test people: enroll with NM01-04, probe with BG01-02 and CL01-02; NM05-06 is a secondary clean-condition control. Enrollment uses the arithmetic mean of individually L2-normalized projected NM vectors, renormalized as one template per person/view/representation. This is not refitting a projection on test identities.

## GEI and regions

Amended preprocessing for the adopted mirror: preserve the supplied 64 rows x 64 columns without further resizing, cropping or alignment. Threshold uint8 grayscale at >=128 into foreground, then average all supplied frames in the whole clip using float64. This midpoint is fixed without recognition-driven selection; grayscale values are not asserted to be calibrated probabilities. No recovery of original geometry, anatomical alignment or gait cycles is claimed. Retain frame-number gaps without interpolation and log counts/gaps. Empty masks after thresholding cause an integrity-review failure; do not silently remove them. Only fitting NM and validation NM/BG features are prepared at this stage. Final-test GEIs/scores remain uncomputed until the test lock. Length-related confounding remains a limitation; a new sensitivity analysis requires a prospective decision rather than a favorable post-test choice.

Upper rows [0,32), lower [32,64). The same boundary is used for every method/condition/view. Region cutoffs are not tuned on coats. The fixed division approximates regions, not anatomical segmentation. Whole-body and region representations derive from exactly the same GEI.

## Representations, budget and scores

Fit each view independently; subjects and roles are identical. Feature budget B is selected on validation from {10,20,30}; full-body gets B coefficients, upper and lower B/2 each. This matches coefficient totals but not learned parameter count or computation, which are reported separately. PCA mean/SVD is fitting-only. LDA uses a fitting-only PCA precursor with dimensions in {40,80}, ridge in {0.001,0.01,0.1}, then output B or B/2. Reject infeasible candidates; do not clip dimensions silently. LDA outputs cannot exceed fitting classes minus one or numerical feasibility. Fix the within-scatter normalization/ridge convention before model implementation.

Projected vectors/templates use cosine distance (smaller is better). Reject zero vectors explicitly. For each region r and view v, z_rv=(d_rv-mu_rv)/sigma_rv using reserved fitting NM impostor distances. For a probe/template pair, d_fused=w*z_upper+(1-w)*z_lower. Equal fusion w=0.5. Candidate w in {0,0.1,...,1}. The same w is used at both views. No probe-dependent reweighting or coat-specific normalization.

Select shared representation hyperparameters with equal-fusion validation BG Rank-1 averaged over views; ties prefer smaller B, smaller precursor dimension, then stronger ridge. Freeze those region models and select w on the same validation BG criterion; weight ties within 1e-12 prefer closest to 0.5, then smaller w. This isolates weight choice from changing the representation, while recognizing both steps use the validation set. Full-body hyperparameters are independently validation-tuned with the same budget grid, not chosen from the test results. All intermediate choices and validation scores are retained.

The primary comparison uses identical region models for equal and tuned weights. Additional comparisons are full-body and each region alone; PCA backbone outcomes are secondary. Following the closest-work review, add a planned secondary normal-walking-selected weight control: select a weight using validation NM05-06 with the same already selected regional models and normalization, then freeze it for BG/CL evaluation. This isolates the weight-selection condition, does not use coat outcomes, and is not a reproduction of the 2017 nine-classifier decision fusion. The select_weight API implements the primary BG rule only; the separate select_normal_control_weight API now implements the planned validation NM05-06 control (2026-11-12); neither has run on real-data scores. No claim that this classical pipeline is state of the art without matched contemporary reference experiments.

## Metrics, uncertainty and timing

Numerical amendment 2026-11-12: `docs/projection_specification.md` fixes PCA centering/no whitening, numerical-rank rejection, sample-normalized within/between scatter, trace-scaled positive LDA ridge, generalized-eigenvector scaling and gallery/cosine conventions before real-data model fitting. The secondary NM05-06 control is now implemented separately from primary BG selection; no selection has run. These numerical conventions are also recorded in the configuration.

Rank-1 = correct top-1 probe identities / probe clips. Average within each participant first, then equally over participants and views; this equals micro averaging only for complete balanced data. Predeclare tie handling: sorted numeric gallery IDs and first minimum; log tie frequency because it can affect accuracy. Primary gain = 100*(Rank1_weighted_CL - Rank1_equal_CL); positive is favorable.

Report per-view NM/BG/CL scores and the prespecified mean, including all failures. For an initial uncertainty analysis, use paired probe-person bootstrap with 10,000 draws and a fixed seed, keeping all that person's conditions/views/repetitions together. Gallery and trained models remain fixed: this interval is conditional on this gallery and does not estimate uncertainty from retraining, gallery sampling or new acquisition systems. Avoid frame-wise/binomial iid intervals. The primary contrast has priority; secondary contrasts are descriptive unless multiplicity is addressed in an amended plan.

Time GEI extraction separately from projection fitting, projection/score inference and disk I/O. State hardware, threads, input sizes and cold/warm cache policy. No GPU is required for the proposed classical core, but actual runtime must be measured.

## Freezing, results and interpretation

Before real-data development: audit data, resolve preprocessing and exclusion rules, review closest literature, save manifests and freeze this experimental plan. Before final test: save selected model/configuration hashes, normalization, weight, gallery/probe rules and development results. No test-driven reruns selecting new weights, views, cutoffs or methods. Correct implementation errors transparently, retain superseded artifacts and label any affected re-evaluation.

A single old indoor dataset, two selected views and fixed coarse regions limit external validity. Positive Rank-1 gain does not establish security, universal clothing invariance or deployability. Negative results may still inform the transfer question, but publication value is not guaranteed. GEI, PCA, LDA and part-based fusion are established methods; novelty remains unresolved until the precise transfer protocol is compared to closest work.
