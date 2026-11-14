"""Prespecified classical projections; no automatic fitting or data loading."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


def _matrix(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("Require a nonempty finite matrix")
    return x


def _fit_input(x, role, sequence_ids):
    x = _matrix(x)
    if role != "fit" or len(sequence_ids) != len(x):
        raise ValueError("Projection fitting requires fitting-person data and sequence labels")
    if not set(sequence_ids) <= {"nm-01", "nm-02", "nm-03", "nm-04"}:
        raise ValueError("Projection fitting is restricted to NM01-04")
    if len(x) < 3:
        raise ValueError("Insufficient fitting observations")
    return x


def _dimension(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("Dimension must be a positive integer")
    return int(value)


def _sign_columns(components):
    components = components.copy()
    pivots = np.argmax(np.abs(components), axis=0)
    components *= np.where(components[pivots, np.arange(components.shape[1])] < 0, -1, 1)
    return components


@dataclass(frozen=True)
class Projection:
    mean: np.ndarray
    components: np.ndarray
    spectrum: np.ndarray
    method: str
    metadata: dict

    def transform(self, x):
        x = _matrix(x)
        if x.shape[1] != len(self.mean):
            raise ValueError("Projection input feature count differs")
        return (x - self.mean) @ self.components


def fit_pca(x, dimension, *, role, sequence_ids):
    x = _fit_input(x, role, sequence_ids)
    dimension = _dimension(dimension)
    mean = x.mean(axis=0)
    centered = x - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    tolerance = max(centered.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > tolerance))
    if dimension > rank:
        raise ValueError(f"PCA dimension {dimension} exceeds numerical rank {rank}")
    return Projection(mean, _sign_columns(vt[:dimension].T),
                      singular[:dimension] ** 2 / (len(x) - 1), "pca",
                      {"numerical_rank": rank, "whiten": False,
                       "rank_tolerance": float(tolerance), "samples": len(x)})


def fit_pca_lda(x, people, dimension, precursor_dimension, ridge, *, role, sequence_ids):
    x = _fit_input(x, role, sequence_ids)
    dimension, precursor_dimension = _dimension(dimension), _dimension(precursor_dimension)
    people = np.asarray(people)
    if people.ndim != 1 or len(people) != len(x):
        raise ValueError("Class labels and observations differ")
    classes, counts = np.unique(people, return_counts=True)
    if len(classes) < 2 or np.any(counts < 2):
        raise ValueError("Need at least two classes and two observations per class")
    if dimension > min(precursor_dimension, len(classes) - 1):
        raise ValueError("LDA dimension exceeds precursor or classes-minus-one limit")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("Require finite positive ridge")
    pca = fit_pca(x, precursor_dimension, role=role, sequence_ids=sequence_ids)
    z = pca.transform(x)
    mean = z.mean(axis=0)
    sw = np.zeros((precursor_dimension, precursor_dimension))
    sb = np.zeros_like(sw)
    for person in classes:
        group = z[people == person]
        group_mean = group.mean(axis=0)
        residual = group - group_mean
        sw += residual.T @ residual
        delta = group_mean - mean
        sb += len(group) * np.outer(delta, delta)
    sw /= len(x)
    sb /= len(x)
    scale = float(np.trace(sw) / precursor_dimension)
    if scale <= 0:
        raise ValueError("Degenerate within-class scatter")
    regularized = sw + ridge * scale * np.eye(precursor_dimension)
    eigenvalues, eigenvectors = eigh(sb, regularized, type=1, check_finite=True)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    tolerance = max(sb.shape) * np.finfo(float).eps * max(abs(eigenvalues).max(), np.finfo(float).tiny)
    positive_rank = int(np.sum(eigenvalues > tolerance))
    if dimension > positive_rank:
        raise ValueError("Requested LDA dimension exceeds positive discriminant rank")
    # Preserve generalized-eigenvector scaling: columns are B-orthonormal,
    # where B = Sw + ridge * trace(Sw)/p * I. No additional column whitening.
    components = _sign_columns(pca.components @ eigenvectors[:, :dimension])
    return Projection(pca.mean, components, eigenvalues[:dimension], "pca_then_regularized_lda",
                      {"samples": len(x), "classes": len(classes),
                       "precursor_dimension": precursor_dimension, "ridge": float(ridge),
                       "within_scatter_scale": scale, "positive_discriminant_rank": positive_rank,
                       "scatter_divisor": "number_of_fitting_samples",
                       "eigenvector_scaling": "regularized_within_scatter_orthonormal"})


def unit_rows(x):
    x = _matrix(x)
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("Zero or numerically degenerate feature vector")
    return x / norms[:, None]


def gallery_templates(projected, people, sequence_ids):
    projected = unit_rows(projected)
    people = np.asarray(people)
    if people.ndim != 1 or len(people) != len(projected) or len(sequence_ids) != len(people):
        raise ValueError("Gallery metadata differs from feature rows")
    sequence_ids = np.asarray(sequence_ids)
    ids = np.unique(people)
    templates = []
    for person in ids:
        selected = people == person
        if sorted(sequence_ids[selected].tolist()) != ["nm-01", "nm-02", "nm-03", "nm-04"]:
            raise ValueError("Each gallery person requires exactly NM01-04")
        templates.append(projected[selected].mean(axis=0))
    return ids, unit_rows(np.asarray(templates))


def cosine_distances(probes, gallery):
    probes, gallery = unit_rows(probes), unit_rows(gallery)
    if probes.shape[1] != gallery.shape[1]:
        raise ValueError("Probe/gallery feature counts differ")
    return 1 - np.clip(probes @ gallery.T, -1, 1)
