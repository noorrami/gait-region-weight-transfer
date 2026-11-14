"""Rebuild the complete classical study from the pinned archive in a new workspace.

This is retrospective computational reproduction, not independent confirmation.
No old GEIs, fitted projections, score matrices or private manifests are inputs.
"""
import argparse,hashlib,importlib.metadata,json,math,os,platform,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
SOURCE=Path(__file__).resolve().parents[1]
PINNED_ARCHIVE='58799af6895f8f5a849a2c04a472ba428839450c2a37135f2eafdb2490eb3dd3'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def verify_source(source):
 manifest=read(source/'MANIFEST.json')
 for relative,digest in manifest['artifacts'].items():
  p=Path(relative)
  if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe source manifest path')
  if sha(source/p)!=digest:raise ValueError('Source snapshot changed: '+relative)
 return manifest
def check_output(source,archive,output):
 source=source.resolve();archive=archive.resolve();output=output.resolve()
 if output.exists():raise ValueError('Output must be new; existing or failed runs are never overwritten')
 if source==output or source.is_relative_to(output):raise ValueError('Output must not contain or replace the source snapshot')
 if not archive.is_file() or sha(archive)!=PINNED_ARCHIVE:raise ValueError('Wrong or changed input archive')
 return source,archive,output
def compare_tree(expected,actual,path='root',tolerance=1e-10):
 count=0;maximum=0.
 if isinstance(expected,dict):
  for key,value in expected.items():
   if key not in actual:raise ValueError('Missing reproduced field: '+path+'/'+key)
   n,residual=compare_tree(value,actual[key],path+'/'+key,tolerance);count+=n;maximum=max(maximum,residual)
 elif isinstance(expected,list):
  if len(expected)!=len(actual):raise ValueError('List length differs: '+path)
  for i,(a,b) in enumerate(zip(expected,actual)):
   n,residual=compare_tree(a,b,f'{path}/{i}',tolerance);count+=n;maximum=max(maximum,residual)
 elif isinstance(expected,bool) or expected is None or isinstance(expected,str):
  if expected!=actual:raise ValueError('Non-numeric result differs: '+path)
 elif isinstance(expected,(int,float)):
  if not isinstance(actual,(int,float)) or not math.isfinite(float(actual)):raise ValueError('Invalid numeric field: '+path)
  maximum=abs(float(expected)-float(actual));count=1
  if maximum>tolerance:raise ValueError(f'Reproduced value differs at {path}: {expected} vs {actual}')
 else:raise TypeError('Unsupported comparison field: '+path)
 return count,maximum

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--archive',required=True,help='Authorized original pinned ZIP, read only')
 parser.add_argument('--output',required=True,help='New isolated output directory')
 args=parser.parse_args()
 source,archive,root=check_output(SOURCE,Path(args.archive),Path(args.output))
 manifest=verify_source(source);started=time.perf_counter();root.mkdir(parents=True)
 provenance=root/'reproduction';provenance.mkdir();logs=provenance/'logs';logs.mkdir()
 context={'status':'retrospective_computational_reproduction','original_results_already_known':True,
  'started_utc':datetime.now(timezone.utc).isoformat(),'source_manifest_sha256':sha(source/'MANIFEST.json'),
  'runner_sha256':sha(Path(__file__)),'archive_sha256':PINNED_ARCHIVE,
  'old_features_models_scores_used_as_inputs':False,'scientific_configuration_changed':False,
  'inventory_policy':'Fresh ZIP index; full pinned-archive hash; decode/CRC all study-used clips, not a repeated pixel audit of unused frames',
  'python':sys.version,'platform':platform.platform(),
  'isolated_virtual_environment':sys.prefix!=sys.base_prefix,
  'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','Pillow','pytest']},
  'stages_completed':[]}
 write(provenance/'state.json',context)
 # Copy explicitly reviewed source only, never data/results or a recursive workspace.
 for relative in manifest['artifacts']:
  target=root/relative
  if relative.startswith('configs/robustness/'):
   target=provenance/'original_templates'/relative
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/relative,target)
 shutil.copyfile(source/'MANIFEST.json',provenance/'source_manifest.json')
 for name in ['study.frozen.json','freeze_manifest.json','evaluation.v1.json','evaluation.v2.json']:
  target=provenance/'original_templates'/'configs'/name;target.parent.mkdir(parents=True,exist_ok=True)
  shutil.copyfile(source/'configs'/name,target)
 target=root/'data/raw/kaggle_casia_b_v1/casia-b-v1.zip';target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(archive,target)
 if sha(target)!=PINNED_ARCHIVE:raise ValueError('Archive copy failed integrity check')
 env=os.environ.copy()
 for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:env[k]='1'
 def run(label,script,*arguments):
  print('START '+label,flush=True)
  with (logs/(label+'.log')).open('w',encoding='utf-8') as log:
   result=subprocess.run([sys.executable,str(root/script),*map(str,arguments)],cwd=root,env=env,
                         stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:
   raise RuntimeError(f'Stage {label} failed. Preserve this run and inspect {logs/(label+".log")}')
  context['stages_completed'].append(label);write(provenance/'state.json',context)
  print(f'DONE {label} ({time.perf_counter()-started:.1f}s total)',flush=True)
 try:
  inventory=root/'data/manifests/private/kaggle_casia_b_v1'
  run('01_inventory','scripts/audit_casia_archive.py','--archive',target,'--output',inventory,'--index-only')
  run('02_development_geis','scripts/prepare_development.py')
  part=read(root/'data/manifests/private/subject_partition.json')
  expected_part=read(source/'reference/participant_roles.json')
  compare_tree(expected_part,{'roles':part['roles'],'excluded_ids':sorted(r['person_id'] for r in part['excluded'])})
  # Rebase metadata in the new workspace only. Every check remains active.
  plan=read(root/'configs/study.frozen.json');plan['reproduction_context']=context.copy()
  write(root/'configs/study.frozen.json',plan)
  original_freeze=read(provenance/'original_templates/configs/freeze_manifest.json')
  freeze={'status':'retrospective_reproduction_input_lock','locked_utc':datetime.now(timezone.utc).isoformat(),
   'original_outcomes_already_known':True,'before_reproduction_fit':True,
   'original_template_sha256':sha(provenance/'original_templates/configs/freeze_manifest.json'),
   'artifacts':{relative:sha(root/relative) for relative in original_freeze['artifacts']}}
  write(root/'configs/freeze_manifest.json',freeze)
  run('03_development_fit','scripts/run_development.py')
  run('04_development_verification','scripts/verify_development.py')
  template=read(provenance/'original_templates/configs/evaluation.v2.json')
  spec={k:template[k] for k in ['bootstrap']}
  spec.update({'version':1,'original_outcomes_already_known':True,'before_reproduction_test_scoring':True,
   'plan_sha256':sha(root/'configs/study.frozen.json'),
   'selection_lock_sha256':sha(root/'results/development/v1/selection.lock.json'),
   'implementation_artifacts':{name:sha(root/name) for name in template['implementation_artifacts']}})
  write(root/'configs/evaluation.v1.json',spec)
  run('05_test_geis','scripts/prepare_test.py')
  spec={**spec,'version':2,'accepted_feature_preparation_spec_sha256':sha(root/'configs/evaluation.v1.json'),
        'supersedes_spec_sha256':sha(root/'configs/evaluation.v1.json')}
  write(root/'configs/evaluation.v2.json',spec)
  run('06_original_test','scripts/evaluate_test.py')
  run('07_original_verification','scripts/verify_test.py')
  run('08_computational_audit','scripts/audit_manuscript_evidence.py')
  run('09_person_sensitivity','scripts/sensitivity_person_flip.py')
  run('10_split_geis_and_roles','scripts/prepare_robustness.py')
  run('11_split_lock','scripts/freeze_robustness.py')
  splitlock=read(root/'configs/robustness/v1/plan.lock.json')
  splitlock.update(status='retrospective_reproduction_split_lock',original_outcomes_already_known=True,
                   reproduction_locked_utc=datetime.now(timezone.utc).isoformat())
  write(root/'configs/robustness/v1/plan.lock.json',splitlock)
  run('12_all_ten_split_fits_and_tests','scripts/run_robustness.py')
  run('13_split_verification','scripts/verify_robustness.py')
  run('14_split_summary','scripts/report_robustness.py')
  # Reference outcomes are read only after all new fitting/selection/testing is complete.
  expected=read(source/'aggregate_results.json');actual=read(root/'results/test/v1/summary.json')
  numeric=0;maximum=0.
  comparisons=[('original',expected['original'],actual),
   ('development',read(source/'reference/development_selection.json'),read(root/'results/development/v1/selection.json')),
   ('sensitivity',expected['sensitivity'],read(root/'results/audit/person_sensitivity_v1/sensitivity.json')),
   ('exploratory',expected['exploratory'],read(root/'results/robustness/v1/summary.json'))]
  for label,a,b in comparisons:
   count,residual=compare_tree(a,b,label);numeric+=count;maximum=max(maximum,residual)
  verify_source(source)
  outputs=['data/manifests/private/subject_partition.json','results/development/v1/selection.lock.json',
   'results/test/v1/result_manifest.json','results/test/v1/summary.json','results/audit/person_sensitivity_v1/sensitivity.json',
   'configs/robustness/v1/plan.lock.json','results/robustness/v1/summary.json','results/robustness/v1/verification.json']
  for i in range(10):outputs.extend(f'results/robustness/v1/split_{i:02d}/{p}' for p in ['development/selection.lock.json','test/result.lock.json'])
  report={'status':'full_raw_reproduction_matched','original_outcomes_already_known':True,
   'new_independent_confirmation':False,'old_features_models_scores_used_as_inputs':False,
   'numeric_fields_matched':numeric,'maximum_numeric_difference':maximum,'tolerance':1e-10,
   'participant_roles_matched':True,'original_test_reproduced':True,'exploratory_splits_reproduced':10,
   'sensitivity_p':read(root/'results/audit/person_sensitivity_v1/sensitivity.json')['two_sided_p'],
   'source_snapshot_unchanged':True,'wall_seconds':time.perf_counter()-started,
   'environment':{k:context[k] for k in ['python','platform','isolated_virtual_environment','packages']},
   'source_manifest_sha256':sha(source/'MANIFEST.json'),'archive_sha256':PINNED_ARCHIVE,
   'output_hashes':{p:sha(root/p) for p in outputs}}
  write(provenance/'verification.json',report)
  context.update(status='complete_matched',ended_utc=datetime.now(timezone.utc).isoformat())
  write(provenance/'state.json',context)
  print(json.dumps(report,indent=2),flush=True)
 except BaseException as error:
  context.update(status='failed_preserved',error=str(error));write(provenance/'state.json',context);raise

if __name__=='__main__':main()
