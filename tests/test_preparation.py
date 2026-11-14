import numpy as np
import pytest

from gait_transfer.preparation import make_partition, processed_mirror_gei, required_sequences


def test_role_specific_integrity_requirements_do_not_use_fit_coats():
    ranges = {r: {"first": i, "last": i} for i,r in enumerate(("fit", "validation", "test"), 1)}
    rows = [dict(person_id=i, condition=c, repetition=n, view=v, frames=20)
            for i,r in enumerate(ranges, 1) for c,n in required_sequences(r) for v in (54,90)]
    partition = make_partition(rows, ranges, [54,90], 20)
    assert partition["roles"] == {"fit": [1], "validation": [2], "test": [3]}
    assert not partition["excluded"]
    # The same short-clip rule must detect loss of an actually required clip.
    rows[0]["frames"] = 19
    with pytest.raises(ValueError, match="nonempty"):
        make_partition(rows, ranges, [54,90], 20)


def test_fixed_threshold_preserves_canvas_and_arithmetic_mean():
    fixture = np.zeros((20,64,64), dtype=np.uint8)
    fixture[:,10:50,20:40] = 128
    fixture[:10,10:50,20:40] = 127
    fixture[:,0,0] = 255
    gei = processed_mirror_gei(fixture)
    assert gei.shape == (64,64)
    assert np.all(gei[10:50,20:40] == 0.5)
    assert gei.sum() == 401


def test_empty_thresholded_frames_fail_instead_of_silent_deletion():
    with pytest.raises(ValueError, match="Empty"):
        processed_mirror_gei(np.full((20,64,64),127,dtype=np.uint8))
    with pytest.raises(ValueError, match="policy"):
        processed_mirror_gei(np.zeros((19,64,64),dtype=np.uint8))
