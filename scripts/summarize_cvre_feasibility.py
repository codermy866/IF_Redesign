"""Paired, center/outcome-stratified case bootstrap of the finite CVRE screen."""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cvre_feasibility_v1'
FOLDS=('shiyan','enshi','wuhan','jingzhou','xiangyang')


def ap(y,p,c):
    return np.mean([average_precision_score(y[c==z],p[c==z]) for z in np.unique(c) if len(np.unique(y[c==z]))==2])


def main():
    rows=[];summaries=[]
    for fold in FOLDS:
        root=OUT/fold/'seed_20260905'
        report=json.loads((root/'complete.json').read_text());summaries.append(report)
        d=np.load(root/'predictions.npz',allow_pickle=False)
        y,c=d['y'],d['centers'];rng=np.random.default_rng(20260906)
        strata=[np.where((c==z)&(y==v))[0] for z in np.unique(c) for v in np.unique(y)]
        strata=[x for x in strata if len(x)]
        comparisons=[('robust','information'),('robust','full'),('robust','random_robust_matched_cost'),('robust_no_cost','random'),('robust_no_cost','expected_attention')]
        for left,right in comparisons:
            delta=float(ap(y,d[left],c)-ap(y,d[right],c))
            values=[];risk_values=[]
            def risk(p):
                p=np.clip(p,1e-6,1-1e-6)
                return -(y*np.log(p)+(1-y)*np.log(1-p))
            risk_delta=risk(d[left])-risk(d[right])
            for _ in range(1000):
                ix=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])
                values.append(ap(y[ix],d[left][ix],c[ix])-ap(y[ix],d[right][ix],c[ix]))
                risk_values.append(float(risk_delta[ix].mean()))
            rows.append({'fold':fold,'contrast':left+' - '+right,'delta_macro_auprc':delta,'case_bootstrap_ci95':np.quantile(values,[.025,.975]).tolist(),'mean_logloss_delta':float(risk_delta.mean()),'logloss_delta_ci95':np.quantile(risk_values,[.025,.975]).tolist()})
    policies=list(summaries[0]['policies'])
    table={k:{'median_macro_auprc':float(np.median([s['policies'][k]['macro_auprc'] for s in summaries])),
              'mean_fold_oct_cost':float(np.mean([s['policies'][k]['mean_oct_units'] for s in summaries]))} for k in policies}
    def diffs(name):return [r['delta_macro_auprc'] for r in rows if r['contrast']==name]
    gates={'robust_beats_information_4_of_5':sum(v>0 for v in diffs('robust - information'))>=4,
           'median_full_delta_at_least_minus_002':float(np.median(diffs('robust - full')))>=-.02,
           'mean_oct_cost_below_4':table['robust']['mean_fold_oct_cost']<4,
           'median_matched_random_delta_positive':float(np.median(diffs('robust - random_robust_matched_cost')))>0}
    result={'n_folds':5,'n_seeds':1,'test_labels_evaluated':False,'policies':table,'paired_contrasts':rows,'gates':gates,'continue_to_large_training':all(gates.values()),'full_median_ap':float(np.median([s['full']['macro_auprc'] for s in summaries]))}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# CVRE feasibility screen','', 'Five source-validation folds, one fixed seed; frozen predictor. Reused development cohort.','', '| Policy | Median macro-AUPRC | Mean fold OCT units |','|---|---:|---:|']
    for k,v in table.items():lines.append(f"| {k} | {v['median_macro_auprc']:.4f} | {v['mean_fold_oct_cost']:.4f} |")
    lines.extend(['',f"Full-evidence median macro-AUPRC: {result['full_median_ap']:.4f}", '',f"Advance to large training: {result['continue_to_large_training']}",'', 'The expectation uses training-case completions; repeat-frame sensitivity is a proxy. Neither identifies nuisance-only medical counterfactuals. Normalized costs are not clinical time. Folds overlap; bootstrap intervals are descriptive within each fold.'])
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
