"""Locked-artifact checks and paired person bootstrap; never selects models."""
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify_lock(root):
    root=Path(root)
    freeze_path=root/'configs/freeze_manifest.json'
    plan_path=root/'configs/study.frozen.json'
    lock_path=root/'results/development/v1/selection.lock.json'
    freeze=json.loads(freeze_path.read_text())
    lock=json.loads(lock_path.read_text())
    if digest(plan_path)!=lock['config_sha256'] or digest(freeze_path)!=lock['freeze_manifest_sha256']:
        raise ValueError('Frozen plan or manifest changed')
    for relative,checksum in freeze['artifacts'].items():
        if digest(root/relative)!=checksum:raise ValueError(f'Frozen input changed: {relative}')
    for relative,checksum in lock['artifacts'].items():
        if digest(lock_path.parent/relative)!=checksum:raise ValueError(f'Locked model/result changed: {relative}')
    state=json.loads((lock_path.parent/'state.json').read_text())
    if state.get('status')!='complete' or not state.get('model_selection_frozen'):
        raise ValueError('Development selection not complete')
    if lock.get('coat_used_for_selection') or lock.get('test_evaluated'):
        raise ValueError('Invalid pre-test selection lock')
    return json.loads(plan_path.read_text()),lock


def paired_person_difference(first, second, *, seed, replicates=10000):
    """Input axes are person, view, repetition; CI conditions on fixed gallery/models."""
    first,second=np.asarray(first),np.asarray(second)
    if (first.shape!=second.shape or first.ndim!=3 or first.shape[0]<2 or
        min(first.shape[1:])<1 or replicates<2 or
        not np.isin(first,[0,1]).all() or not np.isin(second,[0,1]).all()):
        raise ValueError('Require matched binary person/view/repetition outcomes')
    delta=100*(first.astype(float)-second.astype(float)).mean(axis=(1,2))
    draws=np.random.default_rng(seed).integers(0,len(delta),size=(replicates,len(delta)))
    bootstrap=delta[draws].mean(axis=1)
    low,high=np.quantile(bootstrap,[.025,.975],method='linear')
    return {'difference_pp':float(delta.mean()),'ci95_percentile_pp':[float(low),float(high)],
            'unit':'probe_person','participants':len(delta),'replicates':replicates,'seed':seed,
            'gallery_and_models_fixed':True},bootstrap


def outcomes(distances, people, gallery):
    d=np.asarray(distances,dtype=float);people=np.asarray(people);gallery=np.asarray(gallery)
    if (d.shape!=(len(people),len(gallery)) or not np.isfinite(d).all() or
        len(np.unique(gallery))!=len(gallery) or not set(people)<=set(gallery)):
        raise ValueError('Invalid score matrix/identities')
    order=np.argsort(gallery,kind='stable')
    prediction=gallery[order][np.argmin(d[:,order],axis=1)]
    correct=prediction==people
    ties=np.sum(d==d.min(axis=1,keepdims=True),axis=1)
    rank1=float(np.mean([correct[people==p].mean() for p in np.unique(people)]))
    return rank1,prediction,correct,ties
