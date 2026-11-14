"""Create integrity-only roles and development GEIs; never process test images."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gait_transfer.core import parse_frame_path, validate_partition
from gait_transfer.preparation import make_partition, processed_mirror_gei, required_sequences


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    config_path = ROOT / "configs/study.draft.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config["candidate_mirror"]["adopted_as_study_release"]:
        raise ValueError("Study release has not been adopted")
    policy = config["preparation_policy"]
    archive = ROOT / "data/raw/kaggle_casia_b_v1/casia-b-v1.zip"
    archive_hash = digest(archive)
    if archive_hash != policy["archive_sha256"]:
        raise ValueError("Pinned source archive hash mismatch")
    inventory = ROOT / "data/manifests/private/kaggle_casia_b_v1/sequence_inventory.csv"
    with inventory.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    partition = make_partition(rows, config["draft_subject_ids"], config["views_degrees"],
                               policy["minimum_frames"])
    retained = [p for ids in partition["roles"].values() for p in ids]
    validate_partition(partition["roles"], retained)
    if sorted(retained + [p["person_id"] for p in partition["excluded"]]) != list(range(1, 125)):
        raise ValueError("Eligibility does not account for all 124 people")
    partition.update({"status": "verified_integrity_manifest_not_frozen_protocol",
                      "source_archive_sha256": archive_hash,
                      "source_inventory_sha256": digest(inventory),
                      "config_sha256": digest(config_path), "policy": policy,
                      "recognition_outcomes_inspected": False})
    partition["input_configuration"] = config
    output = ROOT / "data/processed/kaggle_casia_b_v1/development_gei"
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / "provenance.json"
    # A failed rerun must not leave a success record for partially rewritten output.
    if record_path.exists():
        raise ValueError("Development preparation already exists; preserve versioned artifacts")
    records = []
    with zipfile.ZipFile(archive) as z:
        selected = {}
        for name in z.namelist():
            if not name.lower().endswith(".png"):
                continue
            relative = name[7:] if name.startswith("output/") else name
            item = parse_frame_path(relative)
            p, v = item["person_id"], item["view"]
            if v not in config["views_degrees"]:
                continue
            role = next((r for r in ("fit", "validation") if p in partition["roles"][r]), None)
            if role is None or (item["condition"], item["repetition"]) not in required_sequences(role):
                continue
            key = (role, p, item["condition"], item["repetition"], v)
            selected.setdefault(key, []).append((item["frame"], name))
        expected = sum(len(partition["roles"][r]) * len(required_sequences(r)) *
                       len(config["views_degrees"]) for r in ("fit", "validation"))
        if len(selected) != expected:
            raise ValueError("Development inventory mismatch")
        for key, members in sorted(selected.items()):
            role, person, condition, repetition, view = key
            members.sort()
            if len({n for n, _ in members}) != len(members):
                raise ValueError("Duplicate frame identity")
            frames = []
            for _, name in members:
                with Image.open(io.BytesIO(z.read(name))) as image:
                    if image.mode != "L" or image.size != (64, 64):
                        raise ValueError("Audited image properties changed")
                    frames.append(np.array(image))
            gei = processed_mirror_gei(np.stack(frames), min_frames=policy["minimum_frames"])
            target = output / f"{role}_{person:03d}_{condition}-{repetition:02d}_{view:03d}.npy"
            np.save(target, gei, allow_pickle=False)
            records.append({"role": role, "person_id": person, "condition": condition,
                            "repetition": repetition, "view": view, "frames": len(frames),
                            "gei": target.relative_to(ROOT).as_posix(), "sha256": digest(target)})
    manifest_path = ROOT / "data/manifests/private/subject_partition.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(partition, indent=2) + "\n", encoding="utf-8")
    record = {"status": "development_features_only_no_recognition_results",
              "source_archive_sha256": archive_hash, "config_sha256": digest(config_path),
              "partition_sha256": digest(manifest_path), "script_sha256": digest(Path(__file__)),
              "preprocessing_module_sha256": digest(ROOT / "src/gait_transfer/preparation.py"),
              "numpy_version": np.__version__, "shape": [64, 64],
              "input_configuration": config,
              "records": records, "test_geis_generated": False, "coat_geis_generated": False}
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"participants": {k: len(v) for k,v in partition["roles"].items()},
                      "excluded": [p["person_id"] for p in partition["excluded"]],
                      "development_geis": len(records), "test_processed": False}, indent=2))


if __name__ == "__main__":
    main()
