"""Post-hoc ordinal fusion and label-free score transformation primitives."""
import numpy as np


def doubled_midranks(distances):
    """Return twice each zero-based midrank; use exact equality for ties."""
    d = np.asarray(distances, dtype=float)
    if d.ndim != 2 or d.shape[1] < 2 or not np.isfinite(d).all():
        raise ValueError("Require finite probe-by-gallery distances with at least two candidates")
    result = np.empty(d.shape, dtype=np.int64)
    for row, values in enumerate(d):
        order = np.argsort(values, kind="stable")
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            result[row, order[start:end]] = start + end - 1
            start = end
    return result


def rank_cost(upper, lower, tenths):
    u, l = np.asarray(upper), np.asarray(lower)
    if (u.shape != l.shape or u.ndim != 2 or not np.issubdtype(u.dtype, np.integer)
            or not np.issubdtype(l.dtype, np.integer) or not isinstance(tenths, (int, np.integer))
            or not 0 <= tenths <= 10):
        raise ValueError("Require matching integer midranks and integer weight tenths in 0..10")
    return tenths * u + (10 - tenths) * l


def predict(cost, gallery_ids):
    d, ids = np.asarray(cost), np.asarray(gallery_ids)
    if d.ndim != 2 or d.shape[1] != len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid candidate scores or gallery identities")
    order = np.argsort(ids, kind="stable")
    predictions = ids[order[np.argmin(d[:, order], axis=1)]]
    ties = np.sum(d == d.min(axis=1, keepdims=True), axis=1)
    return predictions, ties


def person_accuracy(correct, people):
    c, p = np.asarray(correct, dtype=float), np.asarray(people)
    return float(np.mean([c[p == person].mean() for person in np.unique(p)]))


def select_rank_weight(inputs, *, role, condition):
    if role != "validation" or condition != "bg":
        raise ValueError("Rank weight selection requires validation BG only")
    records = []
    for k in range(11):
        rates = []
        for u, l, probes, gallery in inputs:
            predictions, _ = predict(rank_cost(u, l, k), gallery)
            rates.append(person_accuracy(predictions == probes, probes))
        records.append({"tenths": k, "mean_rank1": float(np.mean(rates)), "per_view_rank1": rates})
    best = max(r["mean_rank1"] for r in records)
    tied = [r for r in records if abs(r["mean_rank1"] - best) <= 1e-12]
    return min(tied, key=lambda r: (abs(r["tenths"] - 5), r["tenths"]))["tenths"], records


def probe_diagnostics(distances, probe_ids, gallery_ids):
    """Labels identify the true template only; never return calibration parameters."""
    d = np.asarray(distances, dtype=float)
    ranks = doubled_midranks(d)
    probes, gallery = np.asarray(probe_ids), np.asarray(gallery_ids)
    if len(probes) != len(d) or not set(probes) <= set(gallery) or len(set(gallery)) != len(gallery):
        raise ValueError("Require one genuine candidate per probe")
    genuine = np.array([np.flatnonzero(gallery == p)[0] for p in probes])
    rows = np.arange(len(d))
    impostors = d.copy()
    impostors[rows, genuine] = np.inf
    gap = impostors.min(axis=1) - d[rows, genuine]
    span = np.ptp(d, axis=1)
    margin = np.divide(gap, span, out=np.zeros_like(gap), where=span > 0)
    percentile = ranks[rows, genuine] / (2 * (d.shape[1] - 1))
    return margin, percentile, span == 0
