"""Synthetic mathematical checks only; no gait-result claims."""
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('sensitivity',Path(__file__).resolve().parents[1]/'scripts/sensitivity_person_flip.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_zero_vectors_have_p_one():
    assert module.exact_person_flip([0,0,0])['two_sided_p']==1

def test_three_aligned_signs():
    assert module.exact_person_flip([-1,-1,-1])['two_sided_p']==.25

def test_magnitude_not_only_sign_count():
    # Cancellation under [2,-1] has an absolute sum of one; all assignments are >= one.
    assert module.exact_person_flip([2,-1])['two_sided_p']==1

def test_zeros_and_global_sign_preserve_p():
    a=module.exact_person_flip([2,1,1])
    b=module.exact_person_flip([-2,-1,-1,0,0])
    assert a['two_sided_p']==b['two_sided_p']==.25
    assert b['people']==5 and b['nonzero_people']==3

def test_fractional_or_empty_data_rejected():
    with pytest.raises(ValueError):module.exact_person_flip([.5,0])
    with pytest.raises(ValueError):module.exact_person_flip([])
