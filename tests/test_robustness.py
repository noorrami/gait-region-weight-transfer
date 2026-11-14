"""Synthetic role-allocation checks; these are not recognition findings."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare_robustness',ROOT/'scripts/prepare_robustness.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def lookup():
 return {(p,c,n,v):20 for p in range(1,125) for c in ['nm','bg','cl']
         for n in (range(1,7) if c=='nm' else [1,2]) for v in [54,90]}
def test_seeded_person_disjoint_allocation():
 a=m.assign_roles(2026111301,lookup());b=m.assign_roles(2026111301,lookup())
 assert a==b and [len(a['roles'][r]) for r in ['fit','validation','test']]==[50,24,50]
 assert len({p for ids in a['roles'].values() for p in ids})==124
def test_unused_coat_does_not_exclude_fit_person():
 l=lookup();p=m.assign_roles(2026111301,l)['roles']['fit'][0]
 del l[p,'cl',1,54]
 assert p in m.assign_roles(2026111301,l)['roles']['fit']
def test_bad_required_clip_excludes_without_refill():
 l=lookup();a=m.assign_roles(2026111301,l);p=a['roles']['test'][0]
 l[p,'cl',1,54]=19;b=m.assign_roles(2026111301,l)
 assert len(b['roles']['test'])==49 and p not in b['roles']['test']
 assert b['excluded'][0]['person_id']==p
def test_ten_seed_partitions_differ():
 l=lookup()
 assert len({tuple(m.assign_roles(s,l)['roles']['test']) for s in range(2026111301,2026111311)})==10
