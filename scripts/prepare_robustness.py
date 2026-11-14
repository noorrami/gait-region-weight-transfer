"""Prepare outcome-independent repeated splits and immutable GEI cache."""
import csv,hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gait_transfer.preparation import required_sequences,processed_mirror_gei
from gait_transfer.core import parse_frame_path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def assign_roles(seed,lookup):
    people=np.random.default_rng(seed).permutation(np.arange(1,125)).tolist()
    assigned={'fit':people[:50],'validation':people[50:74],'test':people[74:]}
    retained={r:[] for r in assigned};excluded=[]
    for role,ids in assigned.items():
        for person in sorted(ids):
            bad=[{'condition':c,'repetition':n,'view':v,'frames':lookup.get((person,c,n,v),0)}
                 for c,n in required_sequences(role) for v in [54,90] if lookup.get((person,c,n,v),0)<20]
            if bad:excluded.append({'person_id':person,'role':role,'reasons':bad})
            else:retained[role].append(person)
    assert len({p for ids in assigned.values() for p in ids})==124
    return {'assigned':assigned,'roles':retained,'excluded':excluded}
def main():
    root=ROOT/'configs/robustness/v1'
    if root.exists():raise ValueError('Preserve existing exploration; do not overwrite')
    root.mkdir(parents=True)
    inventory=ROOT/'data/manifests/private/kaggle_casia_b_v1/sequence_inventory.csv'
    with inventory.open() as f:rows=list(csv.DictReader(f))
    lookup={(int(r['person_id']),r['condition'],int(r['repetition']),int(r['view'])):int(r['frames']) for r in rows}
    original=json.loads((ROOT/'configs/study.frozen.json').read_text())
    master={'status':'post_test_exploratory_plan_before_new_split_outcomes',
      'date':'2026-11-13','original_test_already_seen':True,'original_primary_unchanged':True,
      'split_seeds':list(range(2026111301,2026111311)),'assigned_sizes':{'fit':50,'validation':24,'test':50},
      'eligibility':'Original role-specific NM/BG/CL requirements; >=20 frames at 54 and 90; no refill/reassignment',
      'preprocessing':'Same immutable mirror, uint8>=128, 64x64 whole-clip mean; same 32-row halves',
      'selection':'Original grids/ties; fitting NM01-04; normalization fitting NM05-06; validation BG only; NM secondary control',
      'weight_stability':{'replicates':1000,'seed_offset':100,'unit':'validation probe person',
       'scope':'Conditional on chosen regional models and fixed validation gallery; not retraining bootstrap'},
      'reporting':'All ten splits, all six controls and views. Median/range/mean and paired split differences; no pooled independence or selected best run',
      'ci_policy':'No new confirmatory cross-split CI/p-value; overlapping splits are dependent',
      'deep_policy':'Resource assessment only until training budget and exact setup are established; no pretrained CASIA-B leakage',
      'baseline_plan_sha256':sha(ROOT/'configs/study.frozen.json'),'inventory_sha256':sha(inventory)}
    needed=set()
    for i,seed in enumerate(master['split_seeds']):
        part=assign_roles(seed,lookup)
        write(root/f'split_{i:02d}.partition.json',part)
        for role,ids in part['roles'].items():
            needed|={(p,c,n,v) for p in ids for c,n in required_sequences(role) for v in [54,90]}
        plan={**original,'exploratory':True,'original_outcomes_known':True,'split_index':i,'split_seed':seed,
              'partition_path':f'configs/robustness/v1/split_{i:02d}.partition.json',
              'development_feature_provenance':f'configs/robustness/v1/split_{i:02d}.development.json',
              'test_feature_provenance':f'configs/robustness/v1/split_{i:02d}.test.json'}
        write(root/f'split_{i:02d}.json',plan)
    write(root/'plan.json',master)
    cache=ROOT/'data/processed/robustness_gei_v1';cache.mkdir(parents=True,exist_ok=False)
    records={};loaded_records=[]
    for rel in ['data/processed/kaggle_casia_b_v1/development_gei/provenance.json',
                'data/processed/kaggle_casia_b_v1/test_gei_v1/provenance.json']:
        loaded_records+=json.loads((ROOT/rel).read_text())['records']
    for r in loaded_records:
        key=(r['person_id'],r['condition'],r['repetition'],r['view'])
        if key in needed:
            if sha(ROOT/r['gei'])!=r['sha256']:raise ValueError('Changed original GEI')
            records[key]={k:v for k,v in r.items() if k!='role'}
    missing=needed-set(records)
    archive=ROOT/'data/raw/kaggle_casia_b_v1/casia-b-v1.zip'
    if sha(archive)!='58799af6895f8f5a849a2c04a472ba428839450c2a37135f2eafdb2490eb3dd3':raise ValueError('Archive changed')
    print(f'Planned 10 splits; reuse {len(records)} GEIs; prepare {len(missing)} additional clips.',flush=True)
    with zipfile.ZipFile(archive) as z:
        selected={k:[] for k in missing}
        for name in z.namelist():
            if not name.endswith('.png'):continue
            parts=name.split('/')
            if len(parts)!=5 or parts[0]!='output':continue
            try:
                condition,rep=parts[2].split('-')
                key=(int(parts[1]),condition,int(rep),int(parts[3]))
            except ValueError:continue
            if key in selected:selected[key].append(name)
        for i,(key,names) in enumerate(sorted(selected.items())):
            names.sort(key=lambda n:parse_frame_path(n[7:])['frame'])
            frames=np.stack([np.asarray(Image.open(io.BytesIO(z.read(n)))) for n in names])
            gei=processed_mirror_gei(frames)
            p,c,n,v=key;target=cache/f'{p:03d}_{c}-{n:02d}_{v:03d}.npy'
            np.save(target,gei,allow_pickle=False)
            records[key]={'person_id':p,'condition':c,'repetition':n,'view':v,
                          'frames':len(names),'gei':str(target.relative_to(ROOT)).replace(chr(92),'/'),'sha256':sha(target)}
            if (i+1)%100==0:print(f'Prepared {i+1}/{len(selected)} additional GEIs',flush=True)
    if set(records)!=needed:raise ValueError('Feature cache incomplete')
    write(cache/'provenance.json',{'original_archive_sha256':sha(archive),'post_test_exploratory':True,
          'records':[records[k] for k in sorted(records)],'new_clips':len(missing),'reused_clips':len(records)-len(missing)})
    for i in range(10):
        part=json.loads((root/f'split_{i:02d}.partition.json').read_text())
        for kind,roles in [('development',['fit','validation']),('test',['test'])]:
            out=[]
            for role in roles:
                for p in part['roles'][role]:
                    for c,n in required_sequences(role):
                        for v in [54,90]:out.append({**records[p,c,n,v],'role':role})
            write(root/f'split_{i:02d}.{kind}.json',{'records':out,'post_test_exploratory':True,
                  'partition_sha256':sha(root/f'split_{i:02d}.partition.json')})
    print('All split manifests and GEI provenance prepared; no recognition outcomes computed.',flush=True)
if __name__=='__main__':main()
