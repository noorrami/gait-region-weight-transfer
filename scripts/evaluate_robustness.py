"""Evaluate locked exploratory split and summarize validation-weight stability."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gait_transfer.robustness import branch_output,verify_split_lock,sha
from gait_transfer.models import Projection,gallery_templates,cosine_distances
from gait_transfer.evaluation import outcomes
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main(index):
    plan,_=verify_split_lock(ROOT,index)
    base=branch_output(ROOT,index);dev=base/'development'
    part=json.loads((ROOT/plan['partition_path']).read_text())
    people=sorted(part['roles']['test']);views=[54,90]
    provenance=json.loads((ROOT/plan['test_feature_provenance']).read_text())
    expected={(p,c,n,v) for p in people for c in ['nm','bg','cl']
              for n in (range(1,7) if c=='nm' else [1,2]) for v in views}
    loaded={}
    for r in provenance['records']:
        key=(r['person_id'],r['condition'],r['repetition'],r['view'])
        if key not in expected or key in loaded or sha(ROOT/r['gei'])!=r['sha256']:raise ValueError('Invalid split test features')
        loaded[key]=np.load(ROOT/r['gei'],allow_pickle=False)
    assert set(loaded)==expected
    selection=json.loads((dev/'selection.json').read_text())
    out=base/'test';out.mkdir(exist_ok=False)
    summaries={};arrays={};cubes={};predictions=[];stability={}
    started=time.perf_counter()
    for backbone,ch in selection.items():
        summaries[backbone]={c:{} for c in ['nm','bg','cl']}
        def batch(region,view,condition,reps):
            keys=[(p,condition,n,view) for p in people for n in reps]
            images=np.stack([loaded[k] for k in keys])
            images=images[:,:32] if region=='upper' else images[:,32:] if region=='lower' else images
            return images.reshape(len(images),-1),np.array([k[0] for k in keys]),[f'{condition}-{k[2]:02d}' for k in keys],keys
        for vi,view in enumerate(views):
            distances={}
            for region in ['full','upper','lower']:
                stem=f'{backbone}_{region}_{view:03d}'
                with np.load(dev/(stem+'.npz'),allow_pickle=False) as z:
                    model=Projection(z['mean'],z['components'],z['spectrum'],backbone,{})
                metadata=json.loads((dev/(stem+'.json')).read_text())
                x,ids,seqs,_=batch(region,view,'nm',range(1,5))
                gids,G=gallery_templates(model.transform(x),ids,seqs)
                np.testing.assert_array_equal(gids,people)
                arrays[f'{backbone}_{region}_{view}_gallery_ids']=gids
                for c in ['nm','bg','cl']:
                    x,ids,seqs,keys=batch(region,view,c,[5,6] if c=='nm' else [1,2])
                    d=cosine_distances(model.transform(x),G);norm=metadata['normalization']
                    if norm is not None:d=(d-norm['mean'])/norm['std']
                    distances[region,c]=d
                    arrays[f'{backbone}_{region}_{view}_{c}_distances']=d
                    arrays[f'{backbone}_{region}_{view}_{c}_probe_ids']=ids
            for c in ['nm','bg','cl']:
                ids=np.repeat(people,2);reps=[5,6] if c=='nm' else [1,2]
                for method,w in [('full',None),('equal',.5),('bag_selected',ch['upper_weight_bg']),
                                 ('normal_selected',ch['upper_weight_nm_control']),('upper_only',1.),('lower_only',0.)]:
                    d=distances['full',c] if w is None else w*distances['upper',c]+(1-w)*distances['lower',c]
                    accuracy,pred,correct,ties=outcomes(d,ids,gids)
                    key=f'{backbone}__{c}__{method}'
                    if key not in cubes:cubes[key]=np.zeros((len(people),2,2),dtype=np.uint8)
                    cubes[key][:,vi,:]=correct.reshape(len(people),2)
                    summaries[backbone][c].setdefault(method,{'per_view_rank1':[]})['per_view_rank1'].append(accuracy)
                    for j in range(len(ids)):
                        predictions.append([backbone,c,view,int(ids[j]),reps[j%2],method,int(pred[j]),int(correct[j])])
        # Bootstrap probe people on the development grid with fixed gallery/models.
        val_people=sorted(part['roles']['validation']);n=len(val_people)
        grid=[]
        with np.load(dev/f'{backbone}_validation_scores.npz',allow_pickle=False) as z:
            for w in plan['upper_region_weight_grid']:
                per_view=[]
                for view in views:
                    d=w*z[f'upper_{view}_bg_distance']+(1-w)*z[f'lower_{view}_bg_distance']
                    ids=z[f'upper_{view}_bg_probe_ids'];gids=z[f'upper_{view}_gallery_ids']
                    _,_,correct,_=outcomes(d,ids,gids)
                    np.testing.assert_array_equal(ids,np.repeat(val_people,2))
                    per_view.append(correct.reshape(n,2).mean(axis=1))
                grid.append(np.stack(per_view).mean(axis=0))
        grid=np.stack(grid)
        draw=np.random.default_rng(plan['split_seed']+100).integers(n,size=(1000,n))
        bootstrap=grid[:,draw].mean(axis=2)
        candidates=plan['upper_region_weight_grid'];selected=[]
        for j in range(1000):
            best=bootstrap[:,j].max();ties=[i for i in range(11) if abs(bootstrap[i,j]-best)<=1e-12]
            selected.append(min(ties,key=lambda i:(abs(candidates[i]-.5),candidates[i])))
        stability[backbone]={'selected_upper_weight':ch['upper_weight_bg'],'replicates':1000,
            'frequencies':{str(w):int(selected.count(i)) for i,w in enumerate(candidates)},
            'scope':'Conditional validation probe-person resampling, fixed selected models and gallery'}
        for c,methods in summaries[backbone].items():
            for method,r in methods.items():r['mean_rank1']=float(np.mean(r['per_view_rank1']))
    contrasts={}
    for b in selection:
        for c in ['nm','bg','cl']:
            for control in ['equal','lower_only','full']:
                a=cubes[f'{b}__{c}__bag_selected'].astype(float)
                d=cubes[f'{b}__{c}__{control}'].astype(float)
                contrasts[f'{b}__{c}__bag_minus_{control}']=float(100*(a-d).mean())
    report={'status':'complete_post_test_exploratory','split_index':index,'seed':plan['split_seed'],
            'people':people,'role_counts':{r:len(ids) for r,ids in part['roles'].items()},
            'views':views,'rank1':summaries,'contrasts_pp':contrasts,'weight_stability':stability,
            'development_selection_lock_sha256':sha(dev/'selection.lock.json'),
            'wall_seconds':time.perf_counter()-started}
    write(out/'summary.json',report)
    np.savez_compressed(out/'scores.npz',**arrays)
    np.savez_compressed(out/'correctness.npz',people=np.array(people),**cubes)
    with (out/'predictions.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['backbone','condition','view','person_id','repetition','method','predicted_person','correct']);writer.writerows(predictions)
    write(out/'result.lock.json',{'status':'exploratory_split_complete',
       'artifacts':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
    print(f'Split {index}: evaluation completed; {len(people)} test people.',flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
