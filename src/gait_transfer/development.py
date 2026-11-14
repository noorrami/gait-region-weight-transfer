"""Development-only loading and deterministic candidate selection."""
import hashlib
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_features(root, provenance, partition):
    """Reject test/coats, unexpected clips, changed hashes and unsafe paths."""
    root = Path(root).resolve()
    loaded, keys = [], set()
    for record in provenance["records"]:
        role, person = record["role"], record["person_id"]
        condition, repetition, view = record["condition"], record["repetition"], record["view"]
        if role not in {"fit", "validation"} or person not in partition["roles"][role]:
            raise ValueError("Forbidden person or role in development features")
        if view not in [54, 90] or condition not in ({"nm"} if role == "fit" else {"nm", "bg"}):
            raise ValueError("Forbidden view/condition in development features")
        if repetition not in (range(1, 7) if condition == "nm" else (1, 2)):
            raise ValueError("Invalid sequence repetition")
        key = role, person, condition, repetition, view
        if key in keys:
            raise ValueError("Duplicate development clip")
        keys.add(key)
        path = (root / record["gei"]).resolve()
        if not path.is_relative_to(root) or sha256(path) != record["sha256"]:
            raise ValueError("Unsafe or changed GEI artifact")
        image = np.load(path, allow_pickle=False)
        if image.shape != (64, 64) or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("GEI properties differ from protocol")
        loaded.append((record, image))
    expected = {(role, person, condition, repetition, view)
                for role in ("fit", "validation") for person in partition["roles"][role]
                for condition in ({"nm"} if role == "fit" else {"nm", "bg"})
                for repetition in (range(1, 7) if condition == "nm" else (1, 2)) for view in (54, 90)}
    if keys != expected:
        raise ValueError("Incomplete development clip manifest")
    return loaded


def best_candidate(records):
    feasible = [r for r in records if r["status"] == "ok"]
    if not feasible:
        raise ValueError("No feasible development candidate")
    best = max(r["mean_rank1"] for r in feasible)
    tied = [r for r in feasible if abs(r["mean_rank1"] - best) <= 1e-12]
    return min(tied, key=lambda r: (r["budget"], r.get("precursor") or 0, -(r.get("ridge") or 0)))
