"""Check published aggregate arithmetic; not a replay of individual predictions."""
import json
from pathlib import Path
x=json.loads((Path(__file__).parent/'aggregate_results.json').read_text())
rows=x['exploratory']['rows']
assert len(rows)==10 and x['exploratory']['overlapping_splits_not_independent']
for b in ['lda','pca']:
    for control in ['equal','lower_only']:
        values=[r[f'{b}_cl_delta_{control}'] for r in rows]
        expected=x['exploratory']['descriptive_summary'][b]['cl'][control]
        assert abs(sum(values)/len(values)-expected['mean_pp'])<1e-10
        assert sum(v>1e-12 for v in values)==expected['positive']
        assert sum(v < -1e-12 for v in values)==expected['negative']
        assert sum(abs(v)<=1e-12 for v in values)==expected['zero']
s=x['sensitivity']
assert s['extreme_assignments']/s['distinct_nonzero_sign_assignments']==s['two_sided_p']
print('Ten-split aggregate arithmetic and sensitivity fraction verified.')
print('This does not reconstruct recognition scores, person outcomes or raw-data results.')
