"""Synthetic mathematical tests; no biometric findings are generated here."""
import numpy as np
import pytest
from gait_transfer.rank_sensitivity import doubled_midranks, rank_cost, predict, select_rank_weight, probe_diagnostics


def test_midrank_ties_and_order():
    np.testing.assert_array_equal(doubled_midranks([[4, 1, 1, 3]]), [[6, 1, 1, 4]])


def test_positive_affine_and_nonlinear_monotone_invariance():
    d = np.array([[1, 3, 2], [4, 4, 1]], dtype=float)
    for other in [7*d + 19, np.exp(d)]:
        np.testing.assert_array_equal(doubled_midranks(d), doubled_midranks(other))


def test_exact_fusion_tie_uses_numeric_identity():
    cost = rank_cost(np.array([[0, 2]]), np.array([[2, 0]]), 5)
    prediction, ties = predict(cost, [8, 3])
    assert prediction.tolist() == [3] and ties.tolist() == [2]


def test_development_selection_tie_and_role_guard():
    inputs = [(np.array([[0, 2]]), np.array([[0, 2]]), np.array([1]), np.array([1, 2]))]
    k, records = select_rank_weight(inputs, role="validation", condition="bg")
    assert k == 5 and len(records) == 11
    with pytest.raises(ValueError):
        select_rank_weight(inputs, role="test", condition="cl")


def test_scale_free_margin_signs_and_constant_rows():
    d = np.array([[0, 2, 4], [3, 0, 4], [1, 1, 1]], dtype=float)
    a = probe_diagnostics(d, [1, 1, 1], [1, 2, 3])
    b = probe_diagnostics(11*d - 7, [1, 1, 1], [1, 2, 3])
    np.testing.assert_allclose(a[0], [.5, -.75, 0])
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], [0, .5, .5])
    assert a[2].tolist() == [False, False, True]


def test_nonfinite_or_undersized_inputs_rejected():
    for d in [[[np.nan, 1]], [[2]]]:
        with pytest.raises(ValueError):
            doubled_midranks(d)
