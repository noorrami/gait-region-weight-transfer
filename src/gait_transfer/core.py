"""Small explicit primitives; callers still need the frozen research protocol."""
from dataclasses import dataclass
from pathlib import PurePosixPath
import re

import numpy as np


def aligned_gei(frames):
    """Average already aligned binary silhouettes over a complete clip.

    Accept 0/1 or 0/255 masks, not arbitrary grayscale images. Alignment and
    data-dependent frame exclusion are deliberately not implemented here.
    """
    frames = np.asarray(frames)
    if frames.ndim != 3 or frames.shape[0] < 2 or min(frames.shape[1:]) < 2:
        raise ValueError("Need at least two aligned 2D silhouettes")
    if not np.isfinite(frames).all():
        raise ValueError("Silhouettes must be finite")
    values = set(np.unique(frames).tolist())
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError("Input is not a consistent binary mask encoding")
    masks = frames != 0
    if np.any(~masks.any(axis=(1, 2))):
        raise ValueError("Empty silhouettes require an explicit audit/exclusion policy")
    return masks.mean(axis=0, dtype=np.float64)


def body_regions(gei):
    gei = np.asarray(gei, dtype=float)
    if gei.ndim != 2 or min(gei.shape) < 2 or gei.shape[0] % 2 or not np.isfinite(gei).all():
        raise ValueError("A finite even-height GEI is required for fixed halves")
    boundary = gei.shape[0] // 2
    return gei[:boundary].copy(), gei[boundary:].copy()


def parse_frame_path(relative_path):
    """Parse documented CASIA-B layout, cross-checking redundant identifiers."""
    path = PurePosixPath(str(relative_path).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected a safe relative frame path")
    pattern = r"(\d{3})/(nm-0[1-6]|bg-0[12]|cl-0[12])/(\d{3})/(\d{3})-(nm-0[1-6]|bg-0[12]|cl-0[12])-(\d{3})-(\d+)\.png"
    match = re.fullmatch(pattern, path.as_posix(), re.IGNORECASE)
    if not match:
        raise ValueError("Frame path does not follow the declared release layout")
    person, sequence, view, file_person, file_sequence, file_view, frame = match.groups()
    if (person, sequence.lower(), view) != (file_person, file_sequence.lower(), file_view):
        raise ValueError("Directory and filename identifiers disagree")
    if not 1 <= int(person) <= 124 or int(view) not in range(0, 181, 18) or int(frame) < 1:
        raise ValueError("Identifier outside original CASIA-B specification")
    return {"person_id": int(person), "condition": sequence[:2].lower(),
            "repetition": int(sequence[-2:]), "view": int(view), "frame": int(frame)}


def validate_partition(roles, available_people):
    """Check exhaustive person-level roles; never infer roles from frames."""
    if set(roles) != {"fit", "validation", "test"}:
        raise ValueError("Require fit, validation and test roles")
    all_people = []
    for role, ids in roles.items():
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Empty role or duplicate person in {role}")
        all_people.extend(ids)
    if len(all_people) != len(set(all_people)):
        raise ValueError("Person leakage across roles")
    if set(all_people) != set(available_people):
        raise ValueError("Partition does not match audited available people")
    return True


def _distance_matrix(distances):
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or min(distances.shape) == 0 or not np.isfinite(distances).all():
        raise ValueError("Nonempty finite distance matrix required")
    return distances


@dataclass(frozen=True)
class DistanceNormalization:
    mean: float
    std: float

    def transform(self, distances):
        return (_distance_matrix(distances) - self.mean) / self.std


def fit_distance_normalization(distances, probe_people, gallery_people, *, role, condition):
    if role != "fit" or condition != "nm":
        raise ValueError("Normalization may only use reserved fitting-person NM probes")
    distances = _distance_matrix(distances)
    probes, gallery = np.asarray(probe_people), np.asarray(gallery_people)
    if probes.ndim != 1 or gallery.ndim != 1 or distances.shape != (len(probes), len(gallery)):
        raise ValueError("Labels and score matrix differ")
    impostors = distances[probes[:, None] != gallery[None, :]]
    if len(impostors) < 2:
        raise ValueError("Insufficient fitting impostor distances")
    mean, std = float(impostors.mean()), float(impostors.std(ddof=0))
    if std <= np.finfo(float).eps:
        raise ValueError("Degenerate distance normalization")
    return DistanceNormalization(mean, std)


def rank1(distances, probe_people, gallery_people):
    """Person-macro Rank-1 and probe successes, numeric gallery-ID tie order."""
    distances = _distance_matrix(distances)
    probes, gallery = np.asarray(probe_people, dtype=int), np.asarray(gallery_people, dtype=int)
    if probes.ndim != 1 or gallery.ndim != 1 or distances.shape != (len(probes), len(gallery)):
        raise ValueError("Labels and distance matrix differ")
    if len(set(gallery.tolist())) != len(gallery) or not set(probes.tolist()) <= set(gallery.tolist()):
        raise ValueError("Need unique gallery templates and a genuine template for every probe")
    order = np.argsort(gallery, kind="stable")
    predictions = gallery[order][np.argmin(distances[:, order], axis=1)]
    correct = predictions == probes
    per_person = [correct[probes == person].mean() for person in np.unique(probes)]
    return float(np.mean(per_person)), correct


def fuse(upper, lower, weight):
    upper, lower = _distance_matrix(upper), _distance_matrix(lower)
    if upper.shape != lower.shape or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Matching matrices and finite weight in [0,1] required")
    return weight * upper + (1 - weight) * lower


def select_weight(view_inputs, candidates, *, role, condition):
    """Select one common weight by mean view score on validation BG only.

    view_inputs items: (normalized_upper, normalized_lower, probe_ids, gallery_ids).
    All region models/normalizations must already be frozen by the caller.
    """
    if role != "validation" or condition != "bg":
        raise ValueError("Weight selection is restricted to validation bag probes")
    return _select_shared_weight(view_inputs, candidates)


def select_normal_control_weight(view_inputs, candidates, *, role, condition, probe_sequences_by_view):
    """Prespecified secondary control; never permits coat-driven selection."""
    if role != "validation" or condition != "nm":
        raise ValueError("Normal control requires validation NM probes")
    if len(probe_sequences_by_view) != len(view_inputs):
        raise ValueError("Need sequence labels for every view")
    for inputs, sequences in zip(view_inputs, probe_sequences_by_view):
        if len(sequences) != len(inputs[2]) or not set(sequences) <= {"nm-05", "nm-06"}:
            raise ValueError("Normal control may use only NM05-06 probes")
    return _select_shared_weight(view_inputs, candidates)


def _select_shared_weight(view_inputs, candidates):
    if not view_inputs or len(candidates) == 0:
        raise ValueError("Need views and candidate weights")
    records = []
    for weight in candidates:
        scores = [rank1(fuse(upper, lower, weight), probes, gallery)[0]
                  for upper, lower, probes, gallery in view_inputs]
        records.append({"weight": float(weight), "mean_rank1": float(np.mean(scores))})
    best = max(record["mean_rank1"] for record in records)
    tied = [record for record in records if abs(record["mean_rank1"] - best) <= 1e-12]
    selected = min(tied, key=lambda record: (abs(record["weight"] - 0.5), record["weight"]))
    return selected["weight"], records
