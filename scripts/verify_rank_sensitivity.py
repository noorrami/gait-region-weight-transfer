"""Independently replay the ordinal analysis with SciPy ranks and direct diagnostics."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/audit/rank_sensitivity_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(upper, lower, ids, gallery, k):
    u = np.rint(2 * (rankdata(upper, method="average", axis=1) - 1)).astype(np.int64)
    l = np.rint(2 * (rankdata(lower, method="average", axis=1) - 1)).astype(np.int64)
    cost = k * u + (10-k) * l
    order = np.argsort(gallery)
    pred = gallery[order[np.argmin(cost[:, order], axis=1)]]
    acc = np.mean([np.mean(pred[ids == p] == p) for p in np.unique(ids)])
    ties = np.sum(cost == cost.min(axis=1, keepdims=True), axis=1)
    return pred, float(acc), ties


def main():
    for name, expected in read(OUT / "plan.lock.json")["artifacts"].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in read(OUT / "result.lock.json")["artifacts"].items():
        assert sha(OUT / name) == expected, name
    selections = read(OUT / "selection.json")
    result = read(OUT / "summary.json")
    index = {(r["split"], r["backbone"], r["condition"]): r for r in result["rows"]}
    diagnostic_index = {(r["split"], r["backbone"], r["condition"], r["region"]): r for r in result["diagnostics"]}
    with (OUT / "predictions.private.csv").open(newline="", encoding="utf-8") as f:
        predictions = {(r['split'],r['backbone'],r['condition'],int(r['view']),int(r['person']),int(r['repetition_index']),r['method']):r for r in csv.DictReader(f)}
    with (OUT / "diagnostics.private.csv").open(newline="", encoding="utf-8") as f:
        diagnostic_rows = {(r['split'],r['backbone'],r['condition'],int(r['view']),int(r['person']),int(r['repetition_index']),r['region']):r for r in csv.DictReader(f)}
    checked_predictions = checked_margins = checked_grids = 0
    for label in selections:
        dev = ROOT / ("results/development/v1" if label == "original" else f"results/robustness/v1/{label}/development")
        test = ROOT / ("results/test/v1" if label == "original" else f"results/robustness/v1/{label}/test")
        for b, chosen in selections[label].items():
            with np.load(dev / f"{b}_validation_scores.npz", allow_pickle=False) as z:
                rates = []
                for k in range(11):
                    per_view = [score(z[f"upper_{v}_bg_distance"],z[f"lower_{v}_bg_distance"],z[f"upper_{v}_bg_probe_ids"],z[f"upper_{v}_gallery_ids"],k)[1] for v in [54,90]]
                    rates.append(float(np.mean(per_view)))
                    np.testing.assert_allclose(per_view, chosen['grid'][k]['per_view_rank1'], atol=1e-15, rtol=0)
                    checked_grids += 1
                expected = min([k for k,a in enumerate(rates) if abs(a-max(rates))<=1e-12],key=lambda k:(abs(k-5),k))
                assert expected == chosen['upper_weight_tenths']
            with np.load(test / "scores.npz", allow_pickle=False) as z:
                for c in ["nm","bg","cl"]:
                    row = index[label,b,c]
                    direct = {r:[[],[]] for r in ["upper","lower"]}
                    for vi,v in enumerate([54,90]):
                        u,l = [z[f"{b}_{r}_{v}_{c}_distances"] for r in ["upper","lower"]]
                        ids,gallery = z[f"{b}_upper_{v}_{c}_probe_ids"],z[f"{b}_upper_{v}_gallery_ids"]
                        for m,k in [('equal',5),('bag_selected',expected),('upper_only',10),('lower_only',0)]:
                            pred,accuracy,ties = score(u,l,ids,gallery,k)
                            assert abs(row['rank1'][m]['per_view'][vi]-accuracy)<1e-15
                            assert row['rank1'][m]['top_ties'][vi] == int((ties>1).sum())
                            for j,p in enumerate(ids):
                                saved = predictions[label,b,c,v,int(p),j%2,m]
                                assert int(saved['predicted'])==pred[j] and int(saved['top_ties'])==ties[j]
                                assert int(saved['correct'])==int(pred[j]==p)
                                checked_predictions += 1
                        for r,d in [('upper',u),('lower',l)]:
                            for j,p in enumerate(ids):
                                a = list(d[j]); pos = list(gallery).index(p)
                                genuine = a[pos]; impostor = min(a[:pos]+a[pos+1:]); span=max(a)-min(a)
                                margin = (impostor-genuine)/span if span else 0.
                                percentile=(sum(x<genuine for x in a)+(sum(x==genuine for x in a)-1)/2)/(len(a)-1)
                                saved=diagnostic_rows[label,b,c,v,int(p),j%2,r]
                                assert abs(float(saved['margin'])-margin)<1e-14
                                assert abs(float(saved['genuine_percentile'])-percentile)<1e-14
                                direct[r][0].append(margin);direct[r][1].append(percentile)
                                checked_margins+=1
                    for r,values in direct.items():
                        saved=diagnostic_index[label,b,c,r]
                        for field,items in zip(['person_mean_margin_q25_median_q75','person_mean_genuine_percentile_q25_median_q75'],values):
                            means=np.array(items).reshape(2,len(gallery),2).mean(axis=(0,2))
                            np.testing.assert_allclose(np.quantile(means,[.25,.5,.75]),saved[field],atol=1e-14,rtol=0)
                    delta = 100*(row['rank1']['bag_selected']['mean']-row['rank1']['equal']['mean'])
                    assert abs(delta-row['selected_minus_equal_pp'])<1e-14
    assert checked_predictions==len(predictions) and checked_margins==len(diagnostic_rows)
    protected_path=ROOT/'reproductions/original_artifacts_before_v1.json'
    protected=read(protected_path) if protected_path.exists() else read(OUT/'plan.lock.json')['artifacts']
    for name,expected in protected.items():assert sha(ROOT/name)==expected,name
    verification={'status':'independently_verified','validation_grid_points_checked':checked_grids,
                  'prediction_rows_checked':checked_predictions,'diagnostic_rows_checked':checked_margins,
                  'aggregate_result_rows_checked':len(index),'aggregate_diagnostic_rows_checked':len(diagnostic_index),
                  'original_protected_files_unchanged':len(protected) if protected_path.exists() else None,
                  'locked_inputs_unchanged':True,'protected_local_snapshot_available':protected_path.exists(),
                  'independent_rank_implementation':'scipy.stats.rankdata with average ties; integer fusion costs',
                  'selection_lock_sha256':sha(OUT/'selection.lock.json'),'result_lock_sha256':sha(OUT/'result.lock.json'),
                  'verifier_sha256':sha(Path(__file__))}
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
