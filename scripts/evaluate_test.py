"""Evaluate fixed held-out identities using locked projections/weights; never fit/select."""
import csv
import datetime
import json
import os
from pathlib import Path
import sys
import time

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gait_transfer.evaluation import digest,verify_lock,outcomes,paired_person_difference
from gait_transfer.models import Projection,gallery_templates,cosine_distances


def write_json(path,data):path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


def main():
    plan,_=verify_lock(ROOT)
    spec_path=ROOT/'configs/evaluation.v2.json';spec=json.loads(spec_path.read_text())
    for relative,sha in spec['implementation_artifacts'].items():
        if digest(ROOT/relative)!=sha:raise ValueError('Evaluation implementation changed')
    if digest(ROOT/'results/development/v1/selection.lock.json')!=spec['selection_lock_sha256']:
        raise ValueError('Evaluation specification/selection mismatch')
    features_path=ROOT/'data/processed/kaggle_casia_b_v1/test_gei_v1/provenance.json'
    provenance=json.loads(features_path.read_text())
    if provenance['evaluation_spec_sha256']!=spec['accepted_feature_preparation_spec_sha256'] or provenance['plan_sha256']!=digest(ROOT/'configs/study.frozen.json'):
        raise ValueError('Test preparation used a different specification')
    if provenance['selection_lock_sha256']!=spec['selection_lock_sha256']:
        raise ValueError('Test preparation preceded this selection lock')
    partition_path=ROOT/'data/manifests/private/subject_partition.json'
    if provenance['partition_sha256']!=digest(partition_path):raise ValueError('Test partition changed')
    people=sorted(json.loads(partition_path.read_text())['roles']['test']);views=plan['views_degrees']
    expected={(p,c,n,v) for p in people for c in ('nm','bg','cl')
              for n in (range(1,7) if c=='nm' else (1,2)) for v in views}
    loaded={}
    for r in provenance['records']:
        key=r['person_id'],r['condition'],r['repetition'],r['view']
        path=(ROOT/r['gei']).resolve()
        if key not in expected or key in loaded or not path.is_relative_to(ROOT.resolve()) or digest(path)!=r['sha256']:
            raise ValueError('Unexpected, duplicate or changed test feature')
        image=np.load(path,allow_pickle=False)
        if image.shape!=(64,64) or not np.isfinite(image).all() or image.min()<0 or image.max()>1:
            raise ValueError('Invalid test GEI')
        loaded[key]=image
    if set(loaded)!=expected:raise ValueError('Incomplete frozen test set')
    output=ROOT/'results/test/v1';output.mkdir(parents=True,exist_ok=False)
    write_json(output/'state.json',{'status':'running','started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'evaluation_spec_sha256':digest(spec_path),'selection_lock_sha256':spec['selection_lock_sha256']})
    dev=ROOT/'results/development/v1';selected=json.loads((dev/'selection.json').read_text())
    summaries={};cubes={};arrays={};prediction_rows=[];timings=[]
    person_index={p:i for i,p in enumerate(people)}
    def batch(region,view,condition,repetitions):
        keys=[(p,condition,n,view) for p in people for n in repetitions]
        images=np.stack([loaded[k] for k in keys])
        if region=='upper':images=images[:,:32]
        elif region=='lower':images=images[:,32:]
        return images.reshape(len(images),-1),np.array([k[0] for k in keys]),[f'{condition}-{k[2]:02d}' for k in keys],keys

    for backbone,selection in selected.items():
        summaries[backbone]={c:{} for c in ('nm','bg','cl')}
        for view_index,view in enumerate(views):
            distances={};probe_metadata={}
            for region in ('full','upper','lower'):
                name=f'{backbone}_{region}_{view:03d}'
                with np.load(dev/f'{name}.npz',allow_pickle=False) as values:
                    model=Projection(values['mean'],values['components'],values['spectrum'],backbone,{})
                metadata=json.loads((dev/f'{name}.json').read_text())
                gallery,ids,seqs,_=batch(region,view,'nm',range(1,5))
                t=time.perf_counter();gids,templates=gallery_templates(model.transform(gallery),ids,seqs)
                template_seconds=time.perf_counter()-t
                np.testing.assert_array_equal(gids,people)
                arrays[f'{backbone}_{region}_{view}_gallery_ids']=gids
                for condition in ('nm','bg','cl'):
                    x,ids,seqs,keys=batch(region,view,condition,(5,6) if condition=='nm' else (1,2))
                    t=time.perf_counter();d=cosine_distances(model.transform(x),templates)
                    norm=metadata['normalization']
                    if norm is not None:d=(d-norm['mean'])/norm['std']
                    elapsed=time.perf_counter()-t
                    timings.append({'backbone':backbone,'region':region,'view':view,'condition':condition,
                        'probe_projection_scoring_normalization_seconds':elapsed,'probe_count':len(x),
                        'gallery_template_seconds':template_seconds})
                    distances[region,condition]=d;probe_metadata[condition]=(ids,keys)
                    prefix=f'{backbone}_{region}_{view}_{condition}'
                    arrays[prefix+'_distances']=d;arrays[prefix+'_probe_ids']=ids
            for condition in ('nm','bg','cl'):
                ids,keys=probe_metadata[condition]
                for method,w in [('full',None),('equal',.5),('bag_selected',selection['upper_weight_bg']),
                                 ('normal_selected',selection['upper_weight_nm_control']),('upper_only',1.),('lower_only',0.)]:
                    d=distances['full',condition] if w is None else w*distances['upper',condition]+(1-w)*distances['lower',condition]
                    score,pred,correct,ties=outcomes(d,ids,gids)
                    cube_key=backbone,condition,method
                    if cube_key not in cubes:cubes[cube_key]=np.zeros((len(people),len(views),2),dtype=bool)
                    for j,key in enumerate(keys):
                        p,_,repetition,_=key;rep_index=repetition-(5 if condition=='nm' else 1)
                        cubes[cube_key][person_index[p],view_index,rep_index]=correct[j]
                        prediction_rows.append([backbone,condition,view,p,repetition,int(pred[j]),int(correct[j]),int(ties[j]),method])
                    item=summaries[backbone][condition].setdefault(method,{'per_view_rank1':[], 'exact_ties_by_view':[]})
                    item['per_view_rank1'].append(score);item['exact_ties_by_view'].append(int(np.sum(ties>1)))
        for condition in summaries[backbone].values():
            for item in condition.values():item['mean_rank1']=float(np.mean(item['per_view_rank1']))
    contrasts={};bootstrap_arrays={}
    for backbone in summaries:
        for condition in ('cl','bg','nm'):
            controls=['equal','full','upper_only','lower_only','normal_selected'] if condition=='cl' else ['equal']
            for control in controls:
                key=f'{backbone}_{condition}_bag_selected_minus_{control}'
                contrast,bootstrap=paired_person_difference(cubes[backbone,condition,'bag_selected'],cubes[backbone,condition,control],
                    seed=plan['seed'],replicates=plan['bootstrap_draft']['replicates'])
                contrast['primary']=(backbone==plan['primary_backbone'] and condition=='cl' and control=='equal')
                contrast['secondary_interpretation']='descriptive_unadjusted' if not contrast['primary'] else 'prespecified_primary'
                contrasts[key]=contrast;bootstrap_arrays[key]=bootstrap
    primary_key=f"{plan['primary_backbone']}_cl_bag_selected_minus_equal"
    report={'status':'completed_locked_test_v1','completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'people':people,'views':views,'participants':len(people),'probe_clips_per_condition':len(people)*len(views)*2,
        'rank1':summaries,'contrasts':contrasts,'primary_contrast':primary_key,
        'notes':['Primary interval resamples paired probe people with fixed gallery and trained models; no retraining/gallery uncertainty.',
                 'Secondary contrast intervals are descriptive and not adjusted for multiplicity.',
                 'No hyperparameter or weight selection occurred on test data.']}
    write_json(output/'summary.json',report)
    write_json(output/'timing.json',{'records':timings,'numpy':np.__version__,'numerical_threads':1,
        'policy':'One warm in-memory pass; decoding and disk I/O excluded; descriptive timings, not repeated latency benchmark'})
    np.savez(output/'scores.npz',**arrays)
    np.savez(output/'correctness.npz',people=np.array(people),**{'__'.join(k):v for k,v in cubes.items()})
    np.savez(output/'bootstrap.npz',**bootstrap_arrays)
    with (output/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['backbone','condition','view','person_id','repetition','predicted_person','correct','number_top_ties','method']);writer.writerows(prediction_rows)
    artifacts=['summary.json','timing.json','scores.npz','correctness.npz','bootstrap.npz','predictions.csv']
    write_json(output/'result_manifest.json',{'evaluation_spec_sha256':digest(spec_path),
        'selection_lock_sha256':spec['selection_lock_sha256'],'test_feature_provenance_sha256':digest(features_path),
        'artifacts':{name:digest(output/name) for name in artifacts},'test_evaluated':True})
    write_json(output/'state.json',{'status':'complete','test_evaluated':True,'model_selection_changed':False})
    print(json.dumps({'primary':contrasts[primary_key],'rank1':summaries},indent=2))


if __name__=='__main__':main()
