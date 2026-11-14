"""Execute exactly ten locked exploratory fits/evaluations; never choose a split."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'results/robustness/v1';out.mkdir(parents=True,exist_ok=True)
state=out/'state.json'
if state.exists():raise ValueError('Execution already started; inspect state instead of overwriting')
started=time.perf_counter()
for i in range(10):
 p=out/f'split_{i:02d}';p.mkdir()
 state.write_text(json.dumps({'status':'running','split':i,'completed':i})+'\n')
 for script,name in [('run_robustness_development.py','development.log'),('evaluate_robustness.py','evaluation.log')]:
  with (p/name).open('w',encoding='utf-8') as f:
   subprocess.run([sys.executable,str(ROOT/'scripts'/script),str(i)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(f'Completed split {i+1}/10 ({time.perf_counter()-started:.1f} seconds)',flush=True)
state.write_text(json.dumps({'status':'complete','completed':10,'seconds':time.perf_counter()-started})+'\n')
