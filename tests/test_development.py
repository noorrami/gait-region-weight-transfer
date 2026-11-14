import copy
import numpy as np
import pytest
from gait_transfer.development import best_candidate, load_features, sha256


def test_candidate_selection_obeys_budget_precursor_ridge_ties_and_rejections():
    records=[dict(status='ok',budget=20,precursor=40,ridge=.1,mean_rank1=.9),
             dict(status='ok',budget=10,precursor=80,ridge=.1,mean_rank1=.9),
             dict(status='ok',budget=10,precursor=40,ridge=.001,mean_rank1=.9),
             dict(status='ok',budget=10,precursor=40,ridge=.1,mean_rank1=.9),
             dict(status='rejected',budget=10,precursor=40,ridge=.1,mean_rank1=1.)]
    assert best_candidate(records)==records[3]


def test_loader_requires_complete_roles_and_rejects_coats_test_and_changed_hash(tmp_path):
    partition={'roles':{'fit':[1],'validation':[2],'test':[3]}}
    records=[]
    for role,person in [('fit',1),('validation',2)]:
        for condition in (['nm'] if role=='fit' else ['nm','bg']):
            for repetition in (range(1,7) if condition=='nm' else [1,2]):
                for view in [54,90]:
                    p=tmp_path/f'{role}_{condition}_{repetition}_{view}.npy'
                    np.save(p,np.ones((64,64))*.5)
                    records.append(dict(role=role,person_id=person,condition=condition,
                        repetition=repetition,view=view,gei=p.name,sha256=sha256(p)))
    assert len(load_features(tmp_path,{'records':records},partition))==28
    for field,value in [('role','test'),('condition','cl'),('sha256','invalid')]:
        bad=copy.deepcopy(records);bad[0][field]=value
        with pytest.raises(ValueError): load_features(tmp_path,{'records':bad},partition)
    with pytest.raises(ValueError,match='Incomplete'):
        load_features(tmp_path,{'records':records[:-1]},partition)
