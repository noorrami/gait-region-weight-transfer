"""Integrity-only participant eligibility and fixed processed-mirror GEIs."""
import numpy as np

from .core import aligned_gei


def required_sequences(role):
    if role not in {"fit", "validation", "test"}:
        raise ValueError("Unknown participant role")
    sequences = [("nm", i) for i in range(1, 7)]
    if role != "fit":
        sequences += [("bg", i) for i in (1, 2)]
    if role == "test":
        sequences += [("cl", i) for i in (1, 2)]
    return sequences


def make_partition(rows, ranges, views, min_frames):
    """Use counts only; exclusions depend on each role's actual requirements."""
    if min_frames < 2 or not views or len(views) != len(set(views)):
        raise ValueError("Invalid integrity policy")
    lookup = {}
    for row in rows:
        key = tuple(int(row[k]) if k != "condition" else row[k]
                    for k in ("person_id", "condition", "repetition", "view"))
        if key in lookup:
            raise ValueError("Duplicate sequence identity")
        lookup[key] = int(row["frames"])
    roles, excluded, assigned = {}, [], set()
    for role, bounds in ranges.items():
        roles[role] = []
        for person in range(bounds["first"], bounds["last"] + 1):
            if person in assigned:
                raise ValueError("Person assigned to multiple roles")
            assigned.add(person)
            reasons = []
            for condition, repetition in required_sequences(role):
                for view in views:
                    count = lookup.get((person, condition, repetition, view), 0)
                    if count < min_frames:
                        reasons.append({"condition": condition, "repetition": repetition,
                                        "view": view, "frames": count,
                                        "reason": "missing" if count == 0 else "short"})
            if reasons:
                excluded.append({"person_id": person, "role": role, "reasons": reasons})
            else:
                roles[role].append(person)
    if set(roles) != {"fit", "validation", "test"} or any(not v for v in roles.values()):
        raise ValueError("Need three nonempty participant roles")
    return {"roles": roles, "excluded": excluded}


def processed_mirror_gei(frames, *, min_frames=20, threshold=128):
    """Preserve the supplied canvas; no alignment or cycle-recovery claim."""
    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 3 or frames.shape[1:] != (64, 64):
        raise ValueError("Require audited uint8 grayscale 64x64 mirror images")
    if len(frames) < min_frames or min_frames < 2 or threshold != 128:
        raise ValueError("Fixed frame-count and threshold policy violated")
    masks = frames >= threshold
    # Empty thresholded frames are an integrity failure, never silently removed.
    return aligned_gei(masks.astype(np.uint8))
