"""Independent post-test reconstruction of every split's test matrices/predictions."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gait_transfer.robustness import check_plan,branch_output,verify_split_lock,sha
def unit(x):return x/np.linalg.norm(x,axis=1,keepdims=True)
def main():
    check_plan(ROOT);result={'status':'verified','splits':10,'test_matrices_reconstructed':0,
      'predictions_checked':0,'contrasts_checked':0,'validation_stability_distributions_checked':0,
      'maximum_absolute_matrix_residual':0.}
    cache={}
    def image(r):
        p=r['gei']
        if p not in cache:
            assert sha(ROOT/p)==r['sha256'];cache[p]=np.load(ROOT/p,allow_pickle=False)
        return cache[p]
    for index in range(10):
        plan,_=verify_split_lock(ROOT,index)
        base=branch_output(ROOT,index);out=base/'test';dev=base/'development'
        lock=json.loads((out/'result.lock.json').read_text())
        for name,digest in lock['artifacts'].items():assert sha(out/name)==digest
        report=json.loads((out/'summary.json').read_text())
        part=json.loads((ROOT/plan['partition_path']).read_text());people=part['roles']['test']
        assert not(set(part['roles']['fit'])&set(people) or set(part['roles']['validation'])&set(people))
        records=json.loads((ROOT/plan['test_feature_provenance']).read_text())['records']
        selection=json.loads((dev/'selection.json').read_text())
        with (out/'predictions.csv').open() as f:rows=list(csv.DictReader(f))
        lookup={tuple(r[k] for k in ['backbone','condition','view','person_id','repetition','method']):r for r in rows}
        assert len(lookup)==len(rows)
        with np.load(out/'scores.npz',allow_pickle=False) as scores,np.load(out/'correctness.npz',allow_pickle=False) as cubes:
            for b,ch in selection.items():
                for view in [54,90]:
                    D={}
                    for region in ['full','upper','lower']:
                        stem=f'{b}_{region}_{view:03d}'
                        with np.load(dev/(stem+'.npz'),allow_pickle=False) as m:mean,W=m['mean'],m['components']
                        meta=json.loads((dev/(stem+'.json')).read_text())
                        selected=sorted([r for r in records if r['view']==view],key=lambda r:(r['person_id'],r['condition'],r['repetition']))
                        def project(rows):
                            values=np.stack([(image(r)[:32] if region=='upper' else image(r)[32:] if region=='lower' else image(r)).reshape(-1) for r in rows])
                            return unit((values-mean)@W)
                        enrollment=[r for r in selected if r['condition']=='nm' and r['repetition']<=4]
                        ids=np.array([r['person_id'] for r in enrollment]);z=project(enrollment)
                        G=unit(np.stack([z[ids==p].mean(axis=0) for p in people]))
                        for c in ['nm','bg','cl']:
                            probes=[r for r in selected if r['condition']==c and (c!='nm' or r['repetition']>=5)]
                            d=1-np.clip(project(probes)@G.T,-1,1)
                            if meta['normalization'] is not None:
                                norm=meta['normalization'];d=(d-norm['mean'])/norm['std']
                            expected=scores[f'{b}_{region}_{view}_{c}_distances']
                            error=float(np.max(np.abs(d-expected)));result['maximum_absolute_matrix_residual']=max(error,result['maximum_absolute_matrix_residual'])
                            np.testing.assert_allclose(d,expected,atol=1e-11,rtol=1e-9)
                            D[region,c]=d;result['test_matrices_reconstructed']+=1
                    for c in ['nm','bg','cl']:
                        for method,w in [('full',None),('equal',.5),('bag_selected',ch['upper_weight_bg']),
                                         ('normal_selected',ch['upper_weight_nm_control']),('upper_only',1.),('lower_only',0.)]:
                            d=D['full',c] if w is None else w*D['upper',c]+(1-w)*D['lower',c]
                            predicted=np.array(people)[d.argmin(axis=1)];gold=predicted==np.repeat(people,2)
                            vi=[54,90].index(view)
                            np.testing.assert_array_equal(gold.reshape(len(people),2),cubes[f'{b}__{c}__{method}'][:,vi,:])
                            assert abs(gold.mean()-report['rank1'][b][c][method]['per_view_rank1'][vi])<1e-12
                            for j,p in enumerate(np.repeat(people,2)):
                                rep=([5,6] if c=='nm' else [1,2])[j%2]
                                row=lookup[b,c,str(view),str(p),str(rep),method]
                                assert int(row['predicted_person'])==predicted[j] and int(row['correct'])==gold[j]
                                result['predictions_checked']+=1
                for c in ['nm','bg','cl']:
                    for method,r in report['rank1'][b][c].items():
                        assert abs(cubes[f'{b}__{c}__{method}'].mean()-r['mean_rank1'])<1e-12
                    for control in ['full','equal','lower_only']:
                        d=100*(cubes[f'{b}__{c}__bag_selected'].astype(float)-cubes[f'{b}__{c}__{control}']).mean()
                        assert abs(d-report['contrasts_pp'][f'{b}__{c}__bag_minus_{control}'])<1e-12
                        result['contrasts_checked']+=1
                with np.load(dev/f'{b}_validation_scores.npz',allow_pickle=False) as z:
                    weights=plan['upper_region_weight_grid'];values=[]
                    for w in weights:
                        per_view=[]
                        for v in [54,90]:
                            d=w*z[f'upper_{v}_bg_distance']+(1-w)*z[f'lower_{v}_bg_distance']
                            ids=z[f'upper_{v}_bg_probe_ids'];gids=z[f'upper_{v}_gallery_ids']
                            per_view.append((gids[d.argmin(axis=1)]==ids).reshape(len(part['roles']['validation']),2).mean(axis=1))
                        values.append(np.mean(per_view,axis=0))
                    values=np.array(values);n=values.shape[1]
                    draws=np.random.default_rng(plan['split_seed']+100).integers(n,size=(1000,n))
                    means=values[:,draws].mean(axis=2);frequency={str(w):0 for w in weights}
                    for j in range(1000):
                        best=means[:,j].max()
                        choices=[i for i,x in enumerate(means[:,j]) if abs(x-best)<=1e-12]
                        k=min(choices,key=lambda i:(abs(weights[i]-.5),weights[i]))
                        frequency[str(weights[k])]+=1
                    assert frequency==report['weight_stability'][b]['frequencies']
                    result['validation_stability_distributions_checked']+=1
        print(f'Independently verified split {index+1}/10',flush=True)
    result['unique_geis_hash_checked']=len(cache)
    result['original_test_was_known']=True
    (ROOT/'results/robustness/v1/verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
