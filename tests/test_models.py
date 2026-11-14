import numpy as np
import pytest

from gait_transfer.models import (fit_pca, fit_pca_lda, gallery_templates,
                                 cosine_distances, unit_rows)
from gait_transfer.core import select_normal_control_weight, select_weight


def fixture():
    rng = np.random.default_rng(19)
    people = np.repeat(np.arange(4), 4)
    x = rng.normal(size=(16, 8)) + np.repeat(rng.normal(size=(4, 8)) * 3, 4, axis=0)
    return x, people, [f"nm-{i:02d}" for _ in range(4) for i in range(1,5)]


def test_pca_reconstructs_centered_subspace_and_rejects_infeasible_rank():
    x, _, seq = fixture()
    model = fit_pca(x, 8, role="fit", sequence_ids=seq)
    np.testing.assert_allclose(model.transform(x) @ model.components.T, x-x.mean(0), atol=1e-12)
    np.testing.assert_allclose(model.components.T @ model.components, np.eye(8), atol=1e-12)
    with pytest.raises(ValueError, match="rank"):
        fit_pca(np.ones((16,8)), 1, role="fit", sequence_ids=seq)


@pytest.mark.parametrize("role,sequence", [("test","nm-01"),("validation","nm-01"),
                                        ("fit","nm-05"),("fit","bg-01"),("fit","cl-01")])
def test_fit_guards_reject_forbidden_roles_and_sequences(role, sequence):
    x, _, _ = fixture()
    with pytest.raises(ValueError):
        fit_pca(x, 2, role=role, sequence_ids=[sequence]*len(x))


def test_lda_satisfies_generalized_eigen_equation_and_ridge_normalization():
    x, people, seq = fixture()
    ridge=0.01
    model=fit_pca_lda(x,people,3,8,ridge,role="fit",sequence_ids=seq)
    centered=x-x.mean(0)
    sw=np.zeros((8,8)); sb=np.zeros_like(sw)
    for p in np.unique(people):
        group=centered[people==p]; mean=group.mean(0)
        sw+=(group-mean).T@(group-mean)
        sb+=len(group)*np.outer(mean,mean)
    sw/=len(x);sb/=len(x)
    b=sw+ridge*np.trace(sw)/8*np.eye(8)
    np.testing.assert_allclose(sb@model.components,
                               (b@model.components)*model.spectrum,rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(model.components.T@b@model.components,np.eye(3),atol=1e-10)
    # Relabeling class IDs and translating all features preserve distances.
    shifted=fit_pca_lda(x+50,people+100,3,8,ridge,role="fit",sequence_ids=seq)
    np.testing.assert_allclose(cosine_distances(model.transform(x),model.transform(x)),
                               cosine_distances(shifted.transform(x+50),shifted.transform(x+50)),atol=1e-10)
    with pytest.raises(ValueError,match="classes-minus-one"):
        fit_pca_lda(x,people,4,8,ridge,role="fit",sequence_ids=seq)


def test_gallery_averages_unit_clips_then_renormalizes_and_keeps_numeric_ids():
    x=np.array([[2.,0],[0,1],[2,0],[0,1],[-1.,0],[0,-1],[-1,0],[0,-1]])
    ids,templates=gallery_templates(x,[2]*4+[1]*4,[f"nm-{i:02d}" for _ in range(2) for i in range(1,5)])
    np.testing.assert_array_equal(ids,[1,2])
    np.testing.assert_allclose(templates,[[-2**-0.5,-2**-0.5],[2**-0.5,2**-0.5]])
    np.testing.assert_allclose(cosine_distances(templates,templates),[[0,2],[2,0]],atol=1e-15)
    with pytest.raises(ValueError,match="Zero"):
        unit_rows([[0,0]])


def test_normal_control_is_separate_from_primary_and_rejects_coats():
    good=np.array([[0.,1.],[1.,0.]])
    bad=1-good
    inputs=[(good,bad,[1,2],[1,2])]
    selected,_=select_normal_control_weight(inputs,[0.,0.5,1.],role="validation",condition="nm",
                                          probe_sequences_by_view=[["nm-05","nm-06"]])
    assert selected==1.
    with pytest.raises(ValueError):
        select_normal_control_weight(inputs,[0.5],role="validation",condition="cl",
                                     probe_sequences_by_view=[["cl-01","cl-02"]])
    with pytest.raises(ValueError):
        select_normal_control_weight(inputs,[0.5],role="validation",condition="nm",
                                     probe_sequences_by_view=[["nm-01","nm-02"]])
    with pytest.raises(ValueError):
        select_weight(inputs,[0.5],role="validation",condition="nm")
