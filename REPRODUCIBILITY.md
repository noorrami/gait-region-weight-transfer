# Reproducibility scope

The complete raw-archive numerical route is scripts/reproduce_from_archive.py. It generates all required private manifests and input locks in a new workspace; it does not bypass historical checks or copy old GEIs/models/scores into computation. See RAW_REPRODUCTION.md for exact input, installation, stages and failure handling.

RAW_REPRODUCTION_REPORT.json records the tested fresh virtual environment, aggregate comparisons and post-run comparison to retained author artifacts. The source runner hash binds that report to this implementation. The operating system/host was the same as the original run; a new environment and cold data preparation do not constitute new sampling or external validation. The report's tested source-manifest SHA predates documentation-only release updates; scientific implementation hashes are unchanged.

Source images and derived biometric artifacts are excluded from GitHub. Generated outputs include such artifacts and must remain under authorized access. The ZIP inventory is rebuilt and every study-used clip is decoded/CRC-checked; unused frames are not all decoded again because the full archive hash matches the previously audited release.

The route reproduces numerical findings and verification artifacts. Manuscript layout and graphical rendering are separate authoring steps. Numeric tolerance is 1e-10; mismatches fail rather than silently changing settings. requirements-reproduction.txt pins the successfully installed packages; tested Python is 3.14.5 on Windows 10. Other platforms require their own validation.
