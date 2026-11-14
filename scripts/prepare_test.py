"""Extract only fixed test GEIs after verifying pre-test locks; no recognition."""
import csv
import datetime
import io
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gait_transfer.core import parse_frame_path
from gait_transfer.preparation import processed_mirror_gei,required_sequences
from gait_transfer.evaluation import digest,verify_lock


def main():
    plan,_=verify_lock(ROOT)
    evaluation=json.loads((ROOT/'configs/evaluation.v1.json').read_text())
    for name,sha in evaluation['implementation_artifacts'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Evaluation implementation changed')
    partition=json.loads((ROOT/'data/manifests/private/subject_partition.json').read_text())
    archive=ROOT/'data/raw/kaggle_casia_b_v1/casia-b-v1.zip'
    if digest(archive)!=plan['preparation_policy']['archive_sha256']:
        raise ValueError('Source archive hash differs')
    people=set(partition['roles']['test']);views=plan['views_degrees']
    with (ROOT/'data/manifests/private/kaggle_casia_b_v1/sequence_inventory.csv').open() as stream:
        counts={(int(r['person_id']),r['condition'],int(r['repetition']),int(r['view'])):int(r['frames'])
                for r in csv.DictReader(stream)}
    expected={(p,c,n,v) for p in people for c,n in required_sequences('test') for v in views}
    output=ROOT/'data/processed/kaggle_casia_b_v1/test_gei_v1'
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();io_seconds=0.;gei_seconds=0.;records=[]
    with zipfile.ZipFile(archive) as z:
        selected={}
        for member in z.infolist():
            name=member.filename
            if not name.lower().endswith('.png'):continue
            relative=name[7:] if name.startswith('output/') else name
            parts=relative.split('/')
            if len(parts)!=4 or not parts[0].isdigit() or int(parts[0]) not in people:continue
            if not parts[2].isdigit() or int(parts[2]) not in views:continue
            item=parse_frame_path(relative)
            key=(item['person_id'],item['condition'],item['repetition'],item['view'])
            if key not in expected:raise ValueError('Unexpected test sequence')
            selected.setdefault(key,[]).append((item['frame'],name))
        if set(selected)!=expected:raise ValueError('Test sequence set differs from frozen requirements')
        for key,members in sorted(selected.items()):
            members.sort()
            if len(members)!=counts[key] or len({n for n,_ in members})!=len(members):
                raise ValueError('Test frame inventory mismatch')
            t=time.perf_counter();frames=[]
            for _,name in members:
                with Image.open(io.BytesIO(z.read(name))) as image:
                    if image.mode!='L' or image.size!=(64,64):raise ValueError('Test image properties differ')
                    frames.append(np.array(image))
            frames=np.stack(frames);io_seconds+=time.perf_counter()-t
            t=time.perf_counter()
            gei=processed_mirror_gei(frames,min_frames=plan['preparation_policy']['minimum_frames'])
            gei_seconds+=time.perf_counter()-t
            p,c,n,v=key;target=output/f'test_{p:03d}_{c}-{n:02d}_{v:03d}.npy'
            np.save(target,gei,allow_pickle=False)
            records.append({'person_id':p,'condition':c,'repetition':n,'view':v,'frames':len(frames),
                            'gei':target.relative_to(ROOT).as_posix(),'sha256':digest(target)})
            if len(records)%200==0:print(f'Test integrity preparation: {len(records)}/{len(expected)} GEIs',flush=True)
    report={'status':'test_features_prepared_no_recognition','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'records':records,'source_archive_sha256':digest(archive),
            'plan_sha256':digest(ROOT/'configs/study.frozen.json'),
            'evaluation_spec_sha256':digest(ROOT/'configs/evaluation.v1.json'),
            'selection_lock_sha256':digest(ROOT/'results/development/v1/selection.lock.json'),
            'partition_sha256':digest(ROOT/'data/manifests/private/subject_partition.json'),
            'wall_seconds':time.perf_counter()-started,'read_decode_stack_seconds':io_seconds,'binary_gei_seconds':gei_seconds,
            'timing_note':'One pass; I/O/decode separated from threshold/mean; output I/O excluded from compute bucket; not a repeated inference benchmark',
            'test_recognition_computed':False}
    (output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'verified_test_geis':len(records),'people':len(people),'recognition_computed':False}))


if __name__=='__main__':main()
