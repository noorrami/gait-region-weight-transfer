"""Post-test computational audit of the locked pipeline, not new model selection."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gait_transfer.evaluation import verify_lock
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
verify_lock(ROOT)
partition=read('data/manifests/private/subject_partition.json')
groups={k:set(v) for k,v in partition['roles'].items()}
assert [len(groups[k]) for k in ['fit','validation','test']]==[49,23,47]
assert not(groups['fit']&groups['validation'] or groups['fit']&groups['test'] or groups['validation']&groups['test'])
dev=read('data/processed/kaggle_casia_b_v1/development_gei/provenance.json')
test=read('data/processed/kaggle_casia_b_v1/test_gei_v1/provenance.json')
records=dev['records']+[dict(r,role='test') for r in test['records']]
seen=set();loaded=[]
for r in records:
 key=(r['person_id'],r['condition'],r['repetition'],r['view'])
 assert key not in seen;seen.add(key)
 assert r['person_id'] in groups[r['role']]
 assert r['frames']>=20 and r['view'] in [54,90]
 assert sha(r['gei'])==r['sha256']
 a=np.load(ROOT/r['gei'],allow_pickle=False)
 assert a.shape==(64,64) and a.dtype==np.float64
 assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1
 loaded.append((r,a))
for role,people in groups.items():
 for person in people:
  actual={(r['condition'],r['repetition'],r['view']) for r,a in loaded if r['person_id']==person}
  expected={('nm',j,v) for j in range(1,7) for v in [54,90]}
  if role!='fit':expected|={('bg',j,v) for j in [1,2] for v in [54,90]}
  if role=='test':expected|={('cl',j,v) for j in [1,2] for v in [54,90]}
  assert actual==expected

max_error=0.;matrix_checks=0;normalization_checks=0;model_checks=0
def close(actual,expected):
 global max_error
 max_error=max(max_error,float(np.max(np.abs(actual-expected))))
 np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-11)
def unit(a):
 norm=np.linalg.norm(a,axis=1,keepdims=True)
 assert (norm>1e-12).all()
 return a/norm

with np.load(ROOT/'results/test/v1/scores.npz',allow_pickle=False) as saved:
 for backbone in ['pca','pca_then_regularized_lda']:
  for view in [54,90]:
   for region in ['full','upper','lower']:
    stem=f'results/development/v1/{backbone}_{region}_{view:03d}'
    with np.load(ROOT/(stem+'.npz'),allow_pickle=False) as f:
     mean,W=f['mean'],f['components']
    meta=read(stem+'.json')
    fitting=sorted([(r,a) for r,a in loaded if r['role']=='fit' and r['view']==view and r['condition']=='nm' and r['repetition']<=4],
                   key=lambda t:(t[0]['person_id'],t[0]['repetition']))
    def vectors(rows):
     return np.stack([(a[:32] if region=='upper' else a[32:] if region=='lower' else a).reshape(-1) for r,a in rows])
    X=vectors(fitting)
    assert X.shape[0]==196 and len({r['person_id'] for r,a in fitting})==49
    close(X.mean(axis=0),mean)
    if backbone=='pca':close(W.T@W,np.eye(W.shape[1]))
    else:
     Y=(X-mean)@W
     ids=np.array([r['person_id'] for r,a in fitting])
     R=Y-np.stack([Y[ids==i].mean(axis=0) for i in ids])
     metric=R.T@R/len(R)+meta['model']['ridge']*meta['model']['within_scatter_scale']*(W.T@W)
     close(metric,np.eye(W.shape[1]))
    model_checks+=1
    def project(rows):return unit((vectors(rows)-mean)@W)
    def gallery(rows):
     enrollment=[(r,a) for r,a in rows if r['condition']=='nm' and r['repetition']<=4]
     ids=np.array(sorted({r['person_id'] for r,a in enrollment}))
     z=project(enrollment);eids=np.array([r['person_id'] for r,a in enrollment])
     assert all(np.count_nonzero(eids==i)==4 for i in ids)
     return ids,unit(np.stack([z[eids==i].mean(axis=0) for i in ids]))
    if meta['normalization'] is not None:
     gid,G=gallery(fitting)
     normal=[(r,a) for r,a in loaded if r['role']=='fit' and r['view']==view and r['condition']=='nm' and r['repetition']>=5]
     D=1-np.clip(project(normal)@G.T,-1,1)
     pid=np.array([r['person_id'] for r,a in normal])
     impostors=D[pid[:,None]!=gid[None,:]]
     close(np.array([impostors.mean(),impostors.std(ddof=0)]),
           np.array([meta['normalization']['mean'],meta['normalization']['std']]))
     normalization_checks+=1
    rows=sorted([(r,a) for r,a in loaded if r['role']=='test' and r['view']==view],
                key=lambda t:(t[0]['person_id'],t[0]['condition'],t[0]['repetition']))
    gid,G=gallery(rows)
    prefix=f'{backbone}_{region}_{view}'
    np.testing.assert_array_equal(gid,saved[prefix+'_gallery_ids'])
    for condition in ['nm','bg','cl']:
     probes=[(r,a) for r,a in rows if r['condition']==condition and (condition!='nm' or r['repetition']>=5)]
     assert len(probes)==94
     D=1-np.clip(project(probes)@G.T,-1,1)
     if meta['normalization'] is not None:D=(D-meta['normalization']['mean'])/meta['normalization']['std']
     np.testing.assert_array_equal([r['person_id'] for r,a in probes],saved[prefix+'_'+condition+'_probe_ids'])
     close(D,saved[prefix+'_'+condition+'_distances'])
     matrix_checks+=1

report={'status':'passed','scope':'Post-test computational verification only; no fitting, tuning, new contrast or generalization experiment',
 'roles_person_disjoint':True,'retained_people':{k:len(v) for k,v in groups.items()},
 'gei_hash_shape_range_and_clip_policy_checks':len(records),
 'model_means_and_orthonormal_metric_checks':model_checks,
 'fitting_only_impostor_normalizations_recomputed':normalization_checks,
 'test_distance_matrices_reconstructed':matrix_checks,
 'maximum_absolute_numeric_residual':max_error,'tolerance':{'relative':1e-9,'absolute':1e-11},
 'sources':{p:sha(p) for p in [
 'configs/freeze_manifest.json','results/development/v1/selection.lock.json',
 'results/test/v1/result_manifest.json','scripts/audit_manuscript_evidence.py']},
 'limits':['Does not establish original-provider bit identity or legal reuse rights',
 'Does not prove absence of every design error or malicious leakage',
 'Does not estimate uncertainty across retraining, partitions or galleries']}
out=ROOT/'results/audit/manuscript_revision';out.mkdir(parents=True,exist_ok=True)
(out/'computational_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
