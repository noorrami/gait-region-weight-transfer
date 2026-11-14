"""Synthetic safeguards for retrospective reproduction; not gait recognition tests."""
import importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reproduction_runner',ROOT/'scripts/reproduce_from_archive.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_output_requires_fresh_directory_and_preserves_archive(tmp_path,monkeypatch):
 source=tmp_path/'source';source.mkdir();archive=tmp_path/'fixture.zip';archive.write_bytes(b'synthetic fixture')
 monkeypatch.setattr(m,'PINNED_ARCHIVE',m.sha(archive))
 target=tmp_path/'new_run'
 assert m.check_output(source,archive,target)[2]==target
 assert not target.exists() and archive.read_bytes()==b'synthetic fixture'
 target.mkdir()
 with pytest.raises(ValueError,match='must be new'):m.check_output(source,archive,target)

def test_changed_archive_is_rejected_before_output_creation(tmp_path):
 source=tmp_path/'source';source.mkdir();archive=tmp_path/'fixture.zip';archive.write_bytes(b'not the pinned data')
 target=tmp_path/'new_run'
 with pytest.raises(ValueError,match='Wrong or changed'):m.check_output(source,archive,target)
 assert not target.exists()

def test_source_manifest_rejects_changed_or_unsafe_inputs(tmp_path):
 (tmp_path/'code.py').write_text('synthetic fixture')
 manifest={'artifacts':{'code.py':m.sha(tmp_path/'code.py')}}
 (tmp_path/'MANIFEST.json').write_text(json.dumps(manifest))
 assert m.verify_source(tmp_path)==manifest
 (tmp_path/'code.py').write_text('changed')
 with pytest.raises(ValueError,match='Source snapshot changed'):m.verify_source(tmp_path)
 manifest={'artifacts':{'../outside':'fixture'}};(tmp_path/'MANIFEST.json').write_text(json.dumps(manifest))
 with pytest.raises(ValueError,match='Unsafe'):m.verify_source(tmp_path)

def test_result_comparison_preserves_shapes_and_rejects_nonfinite_values():
 n,r=m.compare_tree({'weights':[.5,.7],'status':'matched'}, {'weights':[.5,.7+1e-12],'status':'matched'})
 assert n==2 and r<1e-10
 with pytest.raises(ValueError,match='differs'):m.compare_tree([.5],[.6])
 with pytest.raises(ValueError,match='List length'):m.compare_tree([1,2],[1])
 with pytest.raises(ValueError,match='Invalid numeric'):m.compare_tree(.5,float('nan'))
