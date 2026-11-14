"""Hash exploratory code/manifests before the first new split model fit."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
root=ROOT/'configs/robustness/v1'
if (root/'plan.lock.json').exists():raise ValueError('Exploratory plan already locked')
# Complete the metadata link before any fitting, with no feature/content changes.
for i in range(10):
 for kind in ['development','test']:
  p=root/f'split_{i:02d}.{kind}.json';x=json.loads(p.read_text())
  x['partition_sha256']=sha(root/f'split_{i:02d}.partition.json')
  p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
paths=list(root.glob('*.json'))+[
 ROOT/p for p in ['scripts/prepare_robustness.py','scripts/run_robustness_development.py',
 'scripts/evaluate_robustness.py','scripts/freeze_robustness.py','src/gait_transfer/robustness.py',
 'src/gait_transfer/core.py','src/gait_transfer/models.py','src/gait_transfer/development.py',
 'data/processed/robustness_gei_v1/provenance.json','tests/test_robustness.py']]
manifest={'status':'post_test_exploration_locked_before_first_new_fit','date':'2026-11-13',
 'original_test_already_seen':True,'artifacts':{
 str(p.relative_to(ROOT)).replace(chr(92),'/'):sha(p) for p in paths}}
(root/'plan.lock.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print('Ten-split exploratory plan and implementation locked; original experiment unchanged.')
