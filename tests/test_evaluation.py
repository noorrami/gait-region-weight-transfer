import numpy as np
import pytest
from gait_transfer.evaluation import outcomes, paired_person_difference


def test_bootstrap_uses_people_not_correlated_views_or_frames():
    a=np.zeros((3,2,2),dtype=bool);b=a.copy();a[0]=True
    report,boot=paired_person_difference(a,b,seed=7,replicates=10000)
    assert report['difference_pp']==pytest.approx(100/3)
    # Each replicate samples three whole persons, so only multiples of 100/3 occur.
    assert np.allclose(boot/(100/3),np.rint(boot/(100/3)))
    report2,boot2=paired_person_difference(np.repeat(a,2,axis=2),np.repeat(b,2,axis=2),seed=7,replicates=10000)
    np.testing.assert_array_equal(boot,boot2)
    assert report2['participants']==3


def test_paired_identical_outcomes_have_exact_zero_difference_and_interval():
    x=np.array([[[1,0],[0,1]],[[0,0],[1,1]]])
    report,boot=paired_person_difference(x,x,seed=1)
    assert report['difference_pp']==0 and report['ci95_percentile_pp']==[0,0]
    assert not boot.any()
    with pytest.raises(ValueError):paired_person_difference(x,x[:1],seed=1)


def test_numeric_gallery_ties_and_macro_averaging():
    score,pred,correct,ties=outcomes([[0,0],[0,1],[0,1]],[1,2,2],[2,1])
    np.testing.assert_array_equal(pred,[1,2,2])
    np.testing.assert_array_equal(ties,[2,1,1])
    assert score==1
