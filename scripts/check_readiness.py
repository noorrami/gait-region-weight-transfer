"""Report study prerequisites; deliberately does not run a recognition model."""
import argparse
import json
import hashlib
from pathlib import Path


def readiness(root):
    frozen_path = root / "configs/study.frozen.json"
    config_path = frozen_path if frozen_path.is_file() else root / "configs/study.draft.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    blocked = [name for name, passed in config["gates"].items() if passed is not True]
    if config["status"] != "frozen":
        blocked.append("configuration_not_frozen")
    if not (root / "data/manifests/private/subject_partition.json").is_file():
        blocked.append("verified_subject_partition_artifact_missing")
    if not (root / "configs/study.frozen.json").is_file():
        blocked.append("frozen_configuration_artifact_missing")
    runtime_path = root / config.get("runtime_state_path", "results/development/v1/state.json")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else {}
    lock_path = runtime_path.parent / "selection.lock.json"
    lock_verified = False
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock_verified = (hashlib.sha256(config_path.read_bytes()).hexdigest() == lock.get("config_sha256")
            and bool(lock.get("artifacts")) and all((lock_path.parent / p).is_file() and
                hashlib.sha256((lock_path.parent / p).read_bytes()).hexdigest() == checksum
                for p, checksum in lock["artifacts"].items()))
        freeze_path = root / "configs/freeze_manifest.json"
        if lock_verified and freeze_path.is_file():
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            lock_verified = (hashlib.sha256(freeze_path.read_bytes()).hexdigest() == lock.get("freeze_manifest_sha256")
                and all((root / p).is_file() and hashlib.sha256((root / p).read_bytes()).hexdigest() == checksum
                        for p, checksum in freeze["artifacts"].items()))
        else:
            lock_verified = False
    test_state_path = root / "results/test/v1/state.json"
    test_state = json.loads(test_state_path.read_text(encoding="utf-8")) if test_state_path.is_file() else {}
    result_path = root / "results/test/v1/result_manifest.json"
    test_verified = False
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        test_verified = (lock_verified and bool(result.get("artifacts"))
            and hashlib.sha256(lock_path.read_bytes()).hexdigest() == result.get("selection_lock_sha256")
            and all((result_path.parent / p).is_file() and
                    hashlib.sha256((result_path.parent / p).read_bytes()).hexdigest() == checksum
                    for p, checksum in result["artifacts"].items()))
    return {"title": config["title"], "status": config["status"],
            "ready_for_experiment": not blocked, "unresolved_prerequisites": blocked,
            "real_data_training_run": runtime.get("development_executed", config["development_executed"]),
            "final_test_run": test_state.get("test_evaluated", runtime.get("test_evaluated", config["test_evaluated"])),
            "selected_model_lock_verified": lock_verified,
            "final_test_result_lock_verified": test_verified,
            "note": "Prerequisite report only; no scientific result or access-control guarantee"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-experiment", action="store_true")
    args = parser.parse_args()
    report = readiness(Path(__file__).resolve().parents[1])
    print(json.dumps(report, indent=2))
    raise SystemExit(2 if args.require_experiment and not report["ready_for_experiment"] else 0)
