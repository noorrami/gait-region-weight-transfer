import importlib.util
import io
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


def load_auditor():
    path=Path(__file__).resolve().parents[1]/'scripts/audit_casia_archive.py'
    spec=importlib.util.spec_from_file_location('archive_audit',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archive_audit_reports_missing_coverage_duplicates_and_unsafe_members(tmp_path):
    image=io.BytesIO()
    Image.fromarray(np.array([[0,255],[0,255]],dtype=np.uint8)).save(image,format='PNG')
    archive=tmp_path/'fixture.zip'
    with zipfile.ZipFile(archive,'w') as output:
        output.writestr('output/001/nm-01/054/001-nm-01-054-001.png',image.getvalue())
        output.writestr('output/002/nm-01/054/002-nm-01-054-001.png',image.getvalue())
        output.writestr('../unsafe.png',image.getvalue())
    report=load_auditor().audit(archive,tmp_path/'audit')
    assert report['images_decoded']==2
    assert not report['expected_structure_present']
    assert len(report['missing_sequences'])==13638
    assert report['cross_person_duplicate_groups']==1
    assert report['mask_encodings']=={'binary_0_255':2}
    assert not report['approved_for_training']
    assert not (tmp_path/'unsafe.png').exists()


def test_corrupt_png_is_reported_not_silently_accepted(tmp_path):
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as output:
        output.writestr('001/bg-01/090/001-bg-01-090-001.png',b'not-an-image')
    report=load_auditor().audit(archive,tmp_path/'audit')
    assert report['images_decoded']==0
    assert len(report['decode_or_crc_errors'])==1
    assert not report['all_images_decoded']
