import json
from pathlib import Path

import numpy as np
import pytest

from gait_transfer.core import aligned_gei, body_regions, fit_distance_normalization, fuse, parse_frame_path, rank1, select_weight, validate_partition


def test_gei_matches_hand_calculation_and_encoding():
    frames = np.array([[[1,0],[0,0]], [[1,1],[0,0]]], dtype=np.uint8)
    expected = np.array([[1,.5],[0,0]])
    np.testing.assert_array_equal(aligned_gei(frames), expected)
    np.testing.assert_array_equal(aligned_gei(frames * 255), expected)


@pytest.mark.parametrize("frames", [np.zeros((2,2,2)), np.ones((1,2,2)), np.full((2,2,2),.5), np.full((2,2,2),np.nan)])
def test_bad_masks_are_rejected(frames):
    with pytest.raises(ValueError):
        aligned_gei(frames)


def test_regions_cover_gei_without_overlap():
    gei = np.arange(24).reshape(6,4)
    upper, lower = body_regions(gei)
    np.testing.assert_array_equal(np.vstack((upper,lower)), gei)
    upper[0,0] = -100
    assert gei[0,0] == 0


def test_valid_frame_parse_and_identifier_crosscheck():
    assert parse_frame_path('001/bg-01/054/001-bg-01-054-001.png') == {
        'person_id':1,'condition':'bg','repetition':1,'view':54,'frame':1}
    with pytest.raises(ValueError, match='disagree'):
        parse_frame_path('001/bg-01/054/002-bg-01-054-001.png')


@pytest.mark.parametrize("path", ['../001/bg-01/054/001-bg-01-054-001.png', '125/bg-01/054/125-bg-01-054-001.png', '001/bg-01/055/001-bg-01-055-001.png'])
def test_invalid_paths_rejected(path):
    with pytest.raises(ValueError):
        parse_frame_path(path)


def test_partition_is_person_disjoint_and_exhaustive():
    assert validate_partition({'fit':[1,2],'validation':[3],'test':[4]}, [1,2,3,4])
    with pytest.raises(ValueError, match='leakage'):
        validate_partition({'fit':[1,2],'validation':[2,3],'test':[4]}, [1,2,3,4])
    with pytest.raises(ValueError, match='match'):
        validate_partition({'fit':[1],'validation':[2],'test':[3]}, [1,2,3,4])


def test_normalization_uses_impostors_only_and_is_immutable():
    distances = np.array([[0,2,4],[6,0,8],[10,12,0]], dtype=float)
    normalization = fit_distance_normalization(distances,[1,2,3],[1,2,3],role='fit',condition='nm')
    assert normalization.mean == 7
    assert normalization.std == pytest.approx(np.std([2,4,6,8,10,12]))
    normalized = normalization.transform(np.ones((2,3))*100)
    assert normalized.shape == (2,3)
    assert normalization.mean == 7


@pytest.mark.parametrize("role,condition", [('validation','bg'),('test','nm'),('fit','cl')])
def test_normalization_rejects_wrong_data_role(role,condition):
    with pytest.raises(ValueError, match='fitting-person'):
        fit_distance_normalization([[0,2],[3,0]],[1,2],[1,2],role=role,condition=condition)


def test_rank1_macro_not_frame_or_probe_weighted_and_ties_sorted():
    score, correct = rank1([[0,1],[0,1],[1,0]], [1,1,1], [1,2])
    assert score == pytest.approx(2/3)
    score, _ = rank1([[0,1],[0,1],[0,1],[0,1]], [1,1,1,2], [1,2])
    assert score == .5  # 3 correct probes of person 1, zero of person 2
    score, _ = rank1([[0,0]], [1], [2,1])
    assert score == 1  # numeric person 1 wins an exact tie regardless of input order


def test_weight_ties_prefer_equal_and_input_order_does_not_matter():
    matrix = np.array([[0,1],[1,0]], dtype=float)
    views = [(matrix,matrix,[1,2],[1,2])] * 2
    weight, records = select_weight(views,[1,.5,0],role='validation',condition='bg')
    assert weight == .5
    assert len(records) == 3


@pytest.mark.parametrize("role,condition", [('test','bg'),('validation','cl'),('fit','bg')])
def test_weight_selection_cannot_use_coats_or_test(role,condition):
    with pytest.raises(ValueError, match='validation bag'):
        select_weight([], [0,.5,1],role=role,condition=condition)


def test_weighted_scoring_selects_useful_region_on_fixture_only():
    upper = np.array([[0,3],[3,0]], dtype=float)
    lower = np.array([[3,0],[0,3]], dtype=float)
    weight, _ = select_weight([(upper,lower,[1,2],[1,2])],[0,.5,1],role='validation',condition='bg')
    assert weight == 1
    np.testing.assert_array_equal(fuse(upper,lower,1), upper)
    np.testing.assert_array_equal(fuse(upper,lower,0), lower)


def test_draft_config_keeps_primary_test_closed():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'configs/study.draft.json').read_text(encoding='utf-8'))
    assert config['status'] == 'draft_not_frozen'
    assert config['test_evaluated'] is False
    assert config['coat_used_for_selection'] is False
    assert config['views_degrees'] == [54,90]
    assert config['gates']['protocol_frozen'] is False
    assert config['model_selection_frozen'] is False


def test_flags_alone_cannot_declare_experiment_ready(tmp_path):
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('readiness',root/'scripts/check_readiness.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((root/'configs/study.draft.json').read_text(encoding='utf-8'))
    config['gates'] = {name:True for name in config['gates']}
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/study.draft.json').write_text(json.dumps(config),encoding='utf-8')
    report = module.readiness(tmp_path)
    assert not report['ready_for_experiment']
    assert 'verified_subject_partition_artifact_missing' in report['unresolved_prerequisites']
    assert 'frozen_configuration_artifact_missing' in report['unresolved_prerequisites']
