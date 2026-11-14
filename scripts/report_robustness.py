"""Summarize every locked exploratory split; no independence-based test."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'results/robustness/v1'
verification=json.loads((base/'verification.json').read_text());assert verification['status']=='verified'
rows=[];summaries={}
for i in range(10):
 test=json.loads((base/f'split_{i:02d}/test/summary.json').read_text())
 dev=json.loads((base/f'split_{i:02d}/development/selection.json').read_text())
 row={'split':i,'seed':test['seed'],**{f'n_{r}':n for r,n in test['role_counts'].items()}}
 for b,label in [('pca_then_regularized_lda','lda'),('pca','pca')]:
  row[label+'_weight']=dev[b]['upper_weight_bg']
  for c in ['nm','bg','cl']:
   for m in ['full','equal','bag_selected','normal_selected','upper_only','lower_only']:
    row[f'{label}_{c}_{m}']=100*test['rank1'][b][c][m]['mean_rank1']
   for control in ['equal','lower_only']:
    row[f'{label}_{c}_delta_{control}']=test['contrasts_pp'][f'{b}__{c}__bag_minus_{control}']
  freq=test['weight_stability'][b]['frequencies']
  row[label+'_selected_weight_bootstrap_frequency']=freq[str(dev[b]['upper_weight_bg'])]/1000
 rows.append(row)
for label in ['lda','pca']:
 summaries[label]={}
 for c in ['nm','bg','cl']:
  summaries[label][c]={}
  for control in ['equal','lower_only']:
   values=np.array([r[f'{label}_{c}_delta_{control}'] for r in rows])
   summaries[label][c][control]={'mean_pp':float(values.mean()),'median_pp':float(np.median(values)),
    'min_pp':float(values.min()),'max_pp':float(values.max()),'positive':int((values>1e-12).sum()),
    'zero':int((np.abs(values)<=1e-12).sum()),'negative':int((values< -1e-12).sum())}
 summaries[label]['selected_weight_counts']={str(w):int(sum(r[label+'_weight']==w for r in rows)) for w in np.arange(11)/10}
 summaries[label]['conditional_selected_weight_frequency']=[r[label+'_selected_weight_bootstrap_frequency'] for r in rows]
report={'status':'verified_post_test_exploratory','splits':10,'overlapping_splits_not_independent':True,
        'rows':rows,'descriptive_summary':summaries,'verification':verification}
(base/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
with (base/'split_results.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(summaries,indent=2))
print('split n_fit n_val n_test lda_w lda_cl_delta_equal lda_cl_delta_lower')
for r in rows:print(r['split'],r['n_fit'],r['n_validation'],r['n_test'],r['lda_weight'],round(r['lda_cl_delta_equal'],3),round(r['lda_cl_delta_lower_only'],3))
