"""Isolated post-test robustness branch; never changes original frozen artifacts."""
import hashlib,json
from pathlib import Path
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def branch_output(root,index):
    if not isinstance(index,int) or not 0<=index<10:raise ValueError('Expected split 0..9')
    return Path(root)/f'results/robustness/v1/split_{index:02d}'
def check_plan(root):
    root=Path(root)
    manifest=json.loads((root/'configs/robustness/v1/plan.lock.json').read_text())
    for relative,digest in manifest['artifacts'].items():
        if sha(root/relative)!=digest:raise ValueError('Exploration input changed: '+relative)
    return manifest
def verify_split_lock(root,index):
    root=Path(root);check_plan(root)
    plan_path=root/f'configs/robustness/v1/split_{index:02d}.json'
    plan=json.loads(plan_path.read_text())
    dev=branch_output(root,index)/'development'
    lock=json.loads((dev/'selection.lock.json').read_text())
    if lock['config_sha256']!=sha(plan_path):raise ValueError('Selection plan mismatch')
    for name,digest in lock['artifacts'].items():
        if sha(dev/name)!=digest:raise ValueError('Selection artifact mismatch')
    if lock['coat_used_for_selection'] or lock['test_evaluated']:raise ValueError('Invalid selection chronology')
    return plan,lock
