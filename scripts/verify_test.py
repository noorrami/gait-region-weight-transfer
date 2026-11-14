"""Independent artifact/prediction audit; no fitting, tuning or source modification."""
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gait_transfer.evaluation import digest,verify_lock


def main():
    verify_lock(ROOT)
    output=ROOT/'results/test/v1'
    manifest=json.loads((output/'result_manifest.json').read_text())
    for name,sha in manifest['artifacts'].items():assert digest(output/name)==sha
    assert manifest['evaluation_spec_sha256']==digest(ROOT/'configs/evaluation.v2.json')
    report=json.loads((output/'summary.json').read_text())
    selection=json.loads((ROOT/'results/development/v1/selection.json').read_text())
    with (output/'predictions.csv').open() as f:
        rows=list(csv.DictReader(f))
    csv_rows={tuple(r[k] for k in ('backbone','condition','view','person_id','repetition','method')):r for r in rows}
    if len(csv_rows)!=len(rows):raise ValueError('Duplicate CSV prediction identity')
    checks=0;ties=0
    with np.load(output/'scores.npz',allow_pickle=False) as scores,np.load(output/'correctness.npz',allow_pickle=False) as correctness:
        np.testing.assert_array_equal(correctness['people'],report['people'])
        for backbone in selection:
            for condition in ('nm','bg','cl'):
                for method,w in [('full',None),('equal',.5),('bag_selected',selection[backbone]['upper_weight_bg']),
                                 ('normal_selected',selection[backbone]['upper_weight_nm_control']),('upper_only',1.),('lower_only',0.)]:
                    all_correct=[]
                    for vi,view in enumerate(report['views']):
                        prefix=f'{backbone}_full_{view}_{condition}'
                        ids=scores[prefix+'_probe_ids'];gids=scores[f'{backbone}_full_{view}_gallery_ids']
                        if w is None:d=scores[prefix+'_distances']
                        else:d=w*scores[f'{backbone}_upper_{view}_{condition}_distances']+(1-w)*scores[f'{backbone}_lower_{view}_{condition}_distances']
                        # Numeric gallery tie rule, independently implemented.
                        ordered=np.argsort(gids)
                        predictions=gids[ordered][d[:,ordered].argmin(axis=1)]
                        gold=predictions==ids
                        cube=gold.reshape(len(report['people']),2)
                        np.testing.assert_array_equal(cube,correctness[f'{backbone}__{condition}__{method}'][:,vi,:])
                        accuracy=np.mean([gold[ids==p].mean() for p in report['people']])
                        assert abs(accuracy-report['rank1'][backbone][condition][method]['per_view_rank1'][vi])<1e-12
                        top_ties=(d==d.min(axis=1,keepdims=True)).sum(axis=1)
                        for j,(p,predicted) in enumerate(zip(ids,predictions)):
                            rep=(5 if condition=='nm' else 1)+j%2
                            row=csv_rows[backbone,condition,str(view),str(p),str(rep),method]
                            assert int(row['predicted_person'])==int(predicted)
                            assert int(row['correct'])==int(gold[j])
                            assert int(row['number_top_ties'])==int(top_ties[j])
                            checks+=1
                        ties+=int(np.sum(top_ties>1));all_correct.append(float(accuracy))
                    assert abs(np.mean(all_correct)-report['rank1'][backbone][condition][method]['mean_rank1'])<1e-12
        with np.load(output/'bootstrap.npz',allow_pickle=False) as bootstrap:
            for name,contrast in report['contrasts'].items():
                # Use the stored primary/secondary key without ambiguous backbone splitting.
                marker=next(c for c in ('cl','bg','nm') if f'_{c}_bag_selected_minus_' in name)
                backbone,control=name.split(f'_{marker}_bag_selected_minus_')
                a=correctness[f'{backbone}__{marker}__bag_selected'].astype(float)
                b=correctness[f'{backbone}__{marker}__{control}'].astype(float)
                individual=100*(a-b).mean(axis=(1,2))
                assert abs(individual.mean()-contrast['difference_pp'])<1e-12
                draws=np.random.default_rng(contrast['seed']).integers(len(individual),size=(10000,len(individual)))
                expected=individual[draws].mean(axis=1)
                np.testing.assert_array_equal(expected,bootstrap[name])
                np.testing.assert_allclose(np.percentile(expected,[2.5,97.5]),contrast['ci95_percentile_pp'],atol=1e-12)
    assert len(rows)==checks
    for r in rows:
        assert int(r['correct'])==int(r['person_id']==r['predicted_person'])
    result={'status':'verified','predictions_checked':checks,'contrast_bootstraps_verified':len(report['contrasts']),
        'comparisons_by_view_checked':72,'exact_top_ties_across_all_predictions':ties,
        'source_result_manifest_sha256':digest(output/'result_manifest.json'),
        'note':'Audits saved score matrices, correctness cubes, summaries and bootstrap replay; not new model selection.'}
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
