# Complete computational reproduction from the pinned archive

This route rebuilds the original experiment and all ten exploratory splits in a new directory. It does not require the author's private manifests, GEIs, projections or predictions. You must supply the exact authorized study archive; it is not downloaded or redistributed by this repository.

## Input

CASIA-B processed mirror used by this study, ZIP SHA256:

```text
58799af6895f8f5a849a2c04a472ba428839450c2a37135f2eafdb2490eb3dd3
```

This identifies the analyzed mirror, not a verified original-provider release or CASIA-B*. The route refuses a different ZIP. Use only data you are authorized to access and process.

## Install in a new environment

The measured environment is Python 3.14.5 on Windows 10 with the pinned versions in requirements.txt. Other operating systems or numerical libraries may produce small numerical differences and have not automatically been validated by this test.

Windows:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts/reproduce_from_archive.py --archive "D:\authorized-data\casia-b-v1.zip" --output "D:\reproduction-runs\run-01"
```

POSIX command structure (not an assertion of tested Linux/macOS numerical equivalence):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python scripts/reproduce_from_archive.py --archive /authorized-data/casia-b-v1.zip --output /reproduction-runs/run-01
```

The output directory must not exist. The program never overwrites a previous or failed run and does not support silently skipping partially completed stages. The source must be the complete versioned repository, including MANIFEST.json and reference aggregates.

## Executed stages

1. Verify source hashes and the complete archive hash; copy only the reviewed source/configuration files and the archive to the fresh workspace.
2. Build a fresh ZIP sequence inventory and derive role-specific integrity exclusions. Decode and CRC-check every study-used clip when building GEIs. The fixed archive hash binds this to the previously audited release; unused frames are not all re-decoded for a second full pixel audit.
3. Rebuild fitting/validation GEIs, create a retrospective reproduction-specific input lock, fit/select on allowed development data and independently reconstruct development scores.
4. Create fresh evaluation-specification hash links, rebuild test GEIs, score locked models, replay predictions and bootstrap distributions, and audit model/normalization/distance computations.
5. Recompute the post-test person-block sign-flip sensitivity.
6. Recreate the ten role assignments and required additional GEIs, lock their inputs, refit/select/test all ten, reconstruct their predictions and weight-stability distributions, and summarize every split.
7. Only after scoring finishes, compare role assignments, development selections, original test values, sensitivity and ten-split summaries with the frozen reference aggregates. Default numeric tolerance is 1e-10; unexpected differences fail the run rather than tuning the models to force agreement.

Original implementation files retain their checks. Only the new workspace's metadata hashes are rebased to newly generated files. Historical lock templates are preserved under reproduction/original_templates. The rerun is explicitly retrospective: it is not a newly preregistered experiment or an independent sample.

## Outputs and failure handling

The new workspace contains its own data, fitted models, scores, per-person predictions and result locks. Keep it private; these derived biometric artifacts are not automatically published.

- reproduction/state.json records completion or a preserved failure.
- reproduction/logs contains one log per stage.
- reproduction/verification.json is created only after the complete numerical comparison succeeds.
- results/test/v1 and results/robustness/v1 contain regenerated scientific artifacts.

If a stage fails, inspect its log and preserve the run. Do not delete checks, mix releases, overwrite old results or label a changed configuration as the same reproduction. Graphical manuscript formatting is a separate document-authoring step; this route reproduces the numerical study and verification artifacts.
