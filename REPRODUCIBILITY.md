# Reproducibility scope

The prepared candidate supports: inspection of the exact active scientific source; synthetic numerical/protocol tests; checking aggregate ten-split means/sign counts and the exact sensitivity fraction; inspection of versioned protocol and selection rules.

Full recognition replay is NOT included in the data-free check. It requires the pinned archive SHA256 58799af6895f8f5a849a2c04a472ba428839450c2a37135f2eafdb2490eb3dd3 and the corresponding authorized private sequence inventory, subject manifests, GEIs, frozen selection and result artifacts. Historical verification scripts retain those checks, and fail if inputs are absent or changed.

Recorded environment: Python 3.14.5, NumPy 2.5.2, SciPy 1.18.1, Pillow 12.3.0, pytest 9.1.1, Windows 10. Exact requirements describe the measured environment, not guaranteed wheels for every platform. No global dependency changes are performed by the release builder.

The package test report documents a new process in a separate source directory using the installed recorded dependencies. This is not a newly installed clean operating-system environment and not an independent full-data replication. Before publication, complete and test a raw-data manifest/freeze reconstruction recipe or disclose that limitation explicitly. Do not claim full computational reproduction from passing synthetic tests alone.

Raw data and original frozen artifacts remain immutable. For a new reproduction, use a new output directory and record that original outcomes are known. Never re-label a rerun as a new prospective experiment.
