"""Recompute selected validation scores from saved projections without refitting."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gait_transfer.development import sha256, load_features


def main():
    output=ROOT/'results/development/v1'
    lock=json.loads((output/'selection.lock.json').read_text())
    assert sha256(ROOT/'configs/study.frozen.json')==lock['config_sha256']
    assert sha256(ROOT/'configs/freeze_manifest.json')==lock['freeze_manifest_sha256']
    freeze=json.loads((ROOT/'configs/freeze_manifest.json').read_text())
    for relative,checksum in freeze['artifacts'].items(): assert sha256(ROOT/relative)==checksum
    for relative,checksum in lock['artifacts'].items(): assert sha256(output/relative)==checksum
    plan=json.loads((ROOT/'configs/study.frozen.json').read_text())
    provenance=json.loads((ROOT/plan['development_feature_provenance']).read_text())
    partition=json.loads((ROOT/'data/manifests/private/subject_partition.json').read_text())
    loaded=load_features(ROOT,provenance,partition)
    selections=json.loads((output/'selection.json').read_text())
    report={'status':'verified','test_images_or_scores_accessed':False,
            'source_lock_sha256':sha256(output/'selection.lock.json'),
            'score_matrices_reconstructed':0,'comparisons':[]}
    for backbone,selection in selections.items():
        with np.load(output/f'{backbone}_validation_scores.npz',allow_pickle=False) as saved:
            for view in plan['views_degrees']:
                distances={}
                for region in ('full','upper','lower'):
                    name=f'{backbone}_{region}_{view:03d}'
                    with np.load(output/f'{name}.npz',allow_pickle=False) as model:
                        mean,components=model['mean'],model['components']
                    metadata=json.loads((output/f'{name}.json').read_text())
                    rows=sorted([(r,im) for r,im in loaded if r['role']=='validation' and r['view']==view],
                                key=lambda pair:(pair[0]['person_id'],pair[0]['condition'],pair[0]['repetition']))
                    def project(image):
                        image=image[:32] if region=='upper' else image[32:] if region=='lower' else image
                        v=(image.ravel()-mean)@components
                        return v/np.linalg.norm(v)
                    gallery_ids=np.array(sorted(partition['roles']['validation']))
                    templates=[]
                    for person in gallery_ids:
                        vectors=[project(im) for r,im in rows if r['person_id']==person and r['condition']=='nm' and r['repetition']<=4]
                        assert len(vectors)==4
                        average=np.mean(vectors,axis=0);templates.append(average/np.linalg.norm(average))
                    for condition in ('bg','nm'):
                        probes=[(r,im) for r,im in rows if r['condition']==condition and (condition=='bg' or r['repetition']>=5)]
                        ids=np.array([r['person_id'] for r,_ in probes])
                        d=1-np.clip(np.stack([project(im) for _,im in probes])@np.stack(templates).T,-1,1)
                        if metadata['normalization'] is not None:
                            n=metadata['normalization'];d=(d-n['mean'])/n['std']
                        prefix=f'{region}_{view}_{condition}'
                        np.testing.assert_allclose(d,saved[f'{prefix}_distance'],rtol=1e-9,atol=1e-11)
                        np.testing.assert_array_equal(ids,saved[f'{prefix}_probe_ids'])
                        np.testing.assert_array_equal(gallery_ids,saved[f'{region}_{view}_gallery_ids'])
                        distances[region,condition]=d
                        report['score_matrices_reconstructed']+=1
                for condition in ('bg','nm'):
                    for method,w in [('full',None),('equal',.5),('bag_selected',selection['upper_weight_bg']),
                                     ('normal_selected',selection['upper_weight_nm_control']),('upper_only',1.),('lower_only',0.)]:
                        d=distances['full',condition] if w is None else w*distances['upper',condition]+(1-w)*distances['lower',condition]
                        ids=saved[f'full_{view}_{condition}_probe_ids']
                        predictions=gallery_ids[np.argmin(d,axis=1)]
                        accuracy=float(np.mean([np.mean(predictions[ids==p]==p) for p in gallery_ids]))
                        expected=selection['validation_only_comparisons'][condition][method]['per_view_rank1'][plan['views_degrees'].index(view)]
                        assert abs(accuracy-expected)<1e-12
                        report['comparisons'].append({'backbone':backbone,'view':view,'condition':condition,'method':method,
                            'rank1':accuracy,'exact_top_ties':int(np.sum(np.sum(d==d.min(axis=1,keepdims=True),axis=1)>1))})
    path=ROOT/'results/development/v1/verification.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='comparisons'},indent=2))


if __name__=='__main__':main()
