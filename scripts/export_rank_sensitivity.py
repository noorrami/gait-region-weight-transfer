"""Export all ordinal-analysis aggregate rows; never export per-person outcomes."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/audit/rank_sensitivity_v1'
r=json.loads((OUT/'summary.json').read_text())
rows=[]
for a in r['rows']:
    item={k:a[k] for k in ['split','backbone','condition','people','upper_weight','selected_minus_equal_pp','selected_minus_lower_pp','old_distance_selected_minus_equal_pp']}
    for method,result in a['rank1'].items():
        item[method+'_rank1_percent']=100*result['mean']
        for j,v in enumerate([54,90]):
            item[f'{method}_rank1_{v}_percent']=100*result['per_view'][j]
            item[f'{method}_top_ties_{v}']=result['top_ties'][j]
    rows.append(item)
with (OUT/'aggregate_results.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
items=[]
for a in r['diagnostics']:
    item={k:a[k] for k in ['split','backbone','condition','region','people','constant_probe_rows']}
    for prefix,field in [('margin','person_mean_margin_q25_median_q75'),('genuine_percentile','person_mean_genuine_percentile_q25_median_q75')]:
        item.update({prefix+'_'+name:value for name,value in zip(['q25','median','q75'],a[field])})
    items.append(item)
with (OUT/'aggregate_diagnostics.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
compact={}
for b in ['pca_then_regularized_lda','pca']:
    rs=[a for a in r['rows'] if a['split']!='original' and a['condition']=='cl' and a['backbone']==b]
    delta=np.array([a['selected_minus_equal_pp'] for a in rs])
    compact[b]={'coat_delta_mean_pp':float(delta.mean()),'coat_delta_range_pp':[float(delta.min()),float(delta.max())],
                'positive_negative_zero_splits':[int((delta>1e-12).sum()),int((delta< -1e-12).sum()),int((np.abs(delta)<=1e-12).sum())],
                'lower_only_better_than_selected_splits':sum(a['selected_minus_lower_pp']< -1e-12 for a in rs),
                'selected_minus_lower_mean_pp':float(np.mean([a['selected_minus_lower_pp'] for a in rs])),
                'weights':[a['upper_weight'] for a in rs]}
(OUT/'compact_summary.json').write_text(json.dumps(compact,indent=2)+'\n',encoding='utf-8')
print(json.dumps(compact,indent=2))
