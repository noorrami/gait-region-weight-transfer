"""Inventory a CASIA-B mirror directly in ZIP; no extraction or recognition."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from gait_transfer.core import parse_frame_path


def audit(archive, output, *, pixels=True):
    archive, output = Path(archive).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sequences = defaultdict(list)
    files, unexpected, duplicate_keys, seen = [], [], [], set()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if member.is_dir():
                continue
            path = PurePosixPath(member.filename)
            if path.is_absolute() or '..' in path.parts or stat.S_ISLNK(member.external_attr >> 16):
                unexpected.append({'path':member.filename,'reason':'unsafe_path_or_link'})
                continue
            parts = path.parts
            # The user-specified Kaggle release has the observed wrapper output/.
            relative = '/'.join(parts[1:]) if parts and parts[0]=='output' else path.as_posix()
            try:
                record = parse_frame_path(relative)
            except ValueError as error:
                unexpected.append({'path':member.filename,'reason':str(error)})
                continue
            frame_key = (record['person_id'],record['condition'],record['repetition'],record['view'],record['frame'])
            if frame_key in seen:
                duplicate_keys.append(member.filename)
            seen.add(frame_key)
            sequence_key = frame_key[:4]
            sequences[sequence_key].append(record['frame'])
            files.append((member,record))
        expected = {(person,condition,repetition,view)
                    for person in range(1,125) for condition,count in [('nm',6),('bg',2),('cl',2)]
                    for repetition in range(1,count+1) for view in range(0,181,18)}
        missing = sorted(expected-set(sequences))
        gaps = []
        with (output/'sequence_inventory.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.writer(handle)
            writer.writerow(['person_id','condition','repetition','view','frames','first_frame','last_frame','missing_within_range'])
            for key, indices in sorted(sequences.items()):
                unique=sorted(set(indices))
                gap=unique[-1]-unique[0]+1-len(unique)
                if gap:
                    gaps.append({'sequence':key,'missing_within_range':gap})
                writer.writerow([*key,len(indices),unique[0],unique[-1],gap])
        index={'archive':str(archive),'png_frames':len(files),'people':sorted({k[0] for k in sequences}),
               'views':sorted({k[3] for k in sequences}),'sequence_count':len(sequences),
               'expected_sequence_count':len(expected),'missing_sequences':missing,
               'unexpected_members':unexpected,'duplicate_frame_keys':duplicate_keys,
               'frame_index_gaps':gaps,'condition_sequences':dict(Counter(k[1] for k in sequences)),
               'expected_structure_present':not missing and not unexpected and not duplicate_keys,
               'original_release_identity_verified':False,'processing_history_verified':False,
               'recognition_performed':False,'pixel_audit_requested':pixels}
        (output/'index_summary.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
        print(json.dumps({k:index[k] for k in ['png_frames','sequence_count','expected_sequence_count','condition_sequences','expected_structure_present']},indent=2),flush=True)
        print('People:',len(index['people']),'views:',len(index['views']),'missing sequences:',len(missing),'gaps:',len(gaps),flush=True)
        if not pixels:
            return index
        shapes,modes,encodings=Counter(),Counter(),Counter()
        errors,empty=[],[]
        pixel_first, duplicate_groups={},{}
        sampled=False
        with gzip.open(output/'frame_inventory.csv.gz','wt',newline='',encoding='utf-8') as handle:
            writer=csv.writer(handle)
            writer.writerow(['archive_member','person_id','condition','repetition','view','frame','file_sha256','pixel_sha256','width','height','mode','mask_encoding'])
            for number,(member,record) in enumerate(files,1):
                try:
                    raw=source.read(member)  # ZIP CRC verification happens here.
                    with Image.open(io.BytesIO(raw)) as image:
                        image.load()
                        width,height,mode=image.width,image.height,image.mode
                        histogram=image.convert('L').histogram()
                        levels={i for i,count in enumerate(histogram) if count}
                        encoding='binary_0_255' if levels <= {0,255} else ('binary_0_1' if levels <= {0,1} else 'nonbinary_grayscale')
                        if histogram[0]==width*height:
                            empty.append(member.filename)
                        canonical=f'{width}:{height}:RGBA:'.encode()+image.convert('RGBA').tobytes()
                        pixel_hash=hashlib.sha256(canonical).hexdigest()
                    shapes[f'{width}x{height}']+=1
                    modes[mode]+=1
                    encodings[encoding]+=1
                    if pixel_hash in pixel_first:
                        first=pixel_first[pixel_hash]
                        group=duplicate_groups.setdefault(pixel_hash,{'count':1,'people':{first[0]},'examples':[first[1]]})
                        group['count']+=1
                        group['people'].add(record['person_id'])
                        if len(group['examples'])<3:
                            group['examples'].append(member.filename)
                    else:
                        pixel_first[pixel_hash]=(record['person_id'],member.filename)
                    writer.writerow([member.filename,record['person_id'],record['condition'],record['repetition'],record['view'],record['frame'],hashlib.sha256(raw).hexdigest(),pixel_hash,width,height,mode,encoding])
                    if not sampled and record['person_id']==1 and record['condition']=='nm' and record['view']==54:
                        (output/'sample_fit_person_nm_054.png').write_bytes(raw)
                        sampled=True
                except Exception as error:
                    errors.append({'path':member.filename,'error':type(error).__name__,'message':str(error)[:160]})
                if number % 25000 == 0:
                    print(f'Audited {number:,}/{len(files):,} frames',flush=True)
        duplicates=[{'pixel_sha256':digest,'count':group['count'],'people':sorted(group['people']),'examples':group['examples']}
                    for digest,group in duplicate_groups.items()]
        (output/'duplicate_pixel_groups.json').write_text(json.dumps(duplicates,indent=2),encoding='utf-8')
        result={**index,'images_decoded':sum(shapes.values()),'image_shapes':dict(shapes),'image_modes':dict(modes),
                'mask_encodings':dict(encodings),'decode_or_crc_errors':errors,'empty_frames':empty,
                'duplicate_pixel_groups':len(duplicates),'cross_person_duplicate_groups':sum(len(g['people'])>1 for g in duplicates),
                'all_images_decoded':sum(shapes.values())==len(files) and not errors,
                'approved_for_training':False,'dataset_gates_changed':False}
        (output/'audit_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({k:result[k] for k in ['images_decoded','image_shapes','image_modes','mask_encodings','duplicate_pixel_groups','cross_person_duplicate_groups','all_images_decoded']},indent=2),flush=True)
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--index-only',action='store_true')
    args=parser.parse_args()
    audit(args.archive,args.output,pixels=not args.index_only)
