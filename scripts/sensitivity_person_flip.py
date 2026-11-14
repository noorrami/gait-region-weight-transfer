"""Exact post-test sensitivity for one locked contrast, never model selection."""
import hashlib,itertools,json,math
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def exact_person_flip(integer_differences):
    """Integer summed differences, one four-probe block per participant."""
    values=np.asarray(integer_differences)
    if values.ndim!=1 or not len(values) or not np.all(values==values.astype(int)):
        raise ValueError('Require a nonempty integer person-difference vector')
    values=values.astype(int);observed=int(values.sum())
    weights=[abs(int(x)) for x in values if x]
    distribution=Counter({0:1})
    for weight in weights:
        following=Counter()
        for total,count in distribution.items():
            following[total-weight]+=count;following[total+weight]+=count
        distribution=following
    total=2**len(weights)
    extreme=sum(count for value,count in distribution.items() if abs(value)>=abs(observed))
    assert sum(distribution.values())==total
    return {'people':len(values),'nonzero_people':len(weights),
            'positive_people':int((values>0).sum()),'negative_people':int((values<0).sum()),
            'zero_people':int((values==0).sum()),'observed_summed_probe_difference':observed,
            'distinct_nonzero_sign_assignments':total,'extreme_assignments':extreme,
            'two_sided_p':extreme/total,'null_sum_counts':dict(sorted(distribution.items()))}

def main():
    from scipy.stats import permutation_test
    a_path=ROOT/'results/test/v1/correctness.npz'
    summary_path=ROOT/'results/test/v1/summary.json'
    locked=json.loads(summary_path.read_text())['contrasts']['pca_then_regularized_lda_cl_bag_selected_minus_equal']
    with np.load(a_path,allow_pickle=False) as z:
        a=z['pca_then_regularized_lda__cl__bag_selected'].astype(int)
        b=z['pca_then_regularized_lda__cl__equal'].astype(int)
    assert a.shape==b.shape==(47,2,2)
    delta=(a-b).sum(axis=(1,2))
    result=exact_person_flip(delta)
    assert math.isclose(100*int(delta.sum())/(4*len(delta)),locked['difference_pp'],abs_tol=1e-12)
    # Independent enumeration of all 128 effective assignments and SciPy cross-check.
    nonzero=delta[delta!=0]
    brute=[sum(sign*int(value) for sign,value in zip(signs,nonzero))
           for signs in itertools.product([-1,1],repeat=len(nonzero))]
    assert Counter(brute)==Counter(result['null_sum_counts'])
    check=permutation_test((nonzero,),np.sum,permutation_type='samples',
                          vectorized=False,n_resamples=np.inf,alternative='two-sided')
    assert math.isclose(check.pvalue,result['two_sided_p'],abs_tol=1e-15)
    gain=int(((a==1)&(b==0)).sum());loss=int(((a==0)&(b==1)).sum())
    n=gain+loss;k=min(gain,loss)
    probe_p=min(1.,2*sum(math.comb(n,j) for j in range(k+1))/2**n)
    result.update({
      'status':'post_test_sensitivity_not_preregistered','date':'2026-11-13',
      'contrast':'pca_then_regularized_lda_cl_bag_selected_minus_equal',
      'difference_pp':100*int(delta.sum())/(4*len(delta)),
      'person_summed_difference_histogram':{str(i):int((delta==i).sum()) for i in sorted(set(delta))},
      'unit':'person','fixed_models_and_gallery':True,
      'crosschecks':{'brute_force_enumeration':True,'scipy_permutation_test_p':float(check.pvalue)},
      'probe_independence_diagnostic_only':{'gain':gain,'loss':loss,'exact_binomial_p':probe_p,
        'warning':'Ignores within-person dependence; not adopted as inferential result'},
      'original_percentile_ci_unchanged':locked['ci95_percentile_pp'],
      'assumptions':'Independent person-block exchangeability/sign symmetry; no causal randomization claim',
      'limits':'Does not incorporate retraining, split or gallery sampling; bootstrap CI is not obtained by inversion of this test',
      'source_hashes':{
        str(p.relative_to(ROOT)).replace(chr(92),'/'):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [a_path,summary_path,ROOT/'configs/sensitivity.posthoc.v1.json',Path(__file__)]}})
    out=ROOT/'results/audit/person_sensitivity_v1';out.mkdir(parents=True,exist_ok=True)
    (out/'sensitivity.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
