#!/usr/bin/env python3
"""Diagnostic-only audit of frozen held-out patient predictions.

Do not fit models, choose thresholds or reverse scores on test labels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

REQ = ('patient_id', 'center', 'model', 'y_cin2', 'prob')


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    d = {'N': len(y), 'positives': int(y.sum()), 'prevalence': float(y.mean())}
    d['nll'] = float(log_loss(y, p, labels=[0, 1]))
    if len(set(y)) == 2:
        d.update(auc=float(roc_auc_score(y, p)),
                 auprc=float(average_precision_score(y, p)),
                 reversed_auc=float(roc_auc_score(y, 1.0-p)))
    else:
        d.update(auc=np.nan, auprc=np.nan, reversed_auc=np.nan)
    return d


def audit(df, anchor, out):
    missing = sorted(set(REQ)-set(df.columns))
    if missing:
        raise ValueError(f'Required columns missing: {missing}')
    d = df.copy()
    d['patient_id'] = d['patient_id'].astype(str)
    d['center'] = d['center'].astype(str)
    d['model'] = d['model'].astype(str)
    if d[list(REQ)].isnull().any().any():
        raise ValueError('Required fields cannot contain nulls')
    d['prob'] = pd.to_numeric(d['prob'], errors='raise')
    d['y_cin2'] = pd.to_numeric(d['y_cin2'], errors='raise')
    if (not np.isfinite(d['prob']).all() or not d['prob'].between(0, 1).all()
            or not d['y_cin2'].isin([0, 1]).all()):
        raise ValueError('Probabilities must be finite in [0,1]; labels must be 0/1')
    if d.duplicated(['patient_id','model']).any():
        raise ValueError('Duplicate patient+model; patient_id must be global across centers')
    if d.groupby('patient_id')['center'].nunique().max() > 1 or d.groupby('patient_id')['y_cin2'].nunique().max() > 1:
        raise ValueError('Patient has conflicting center/labels')
    models = sorted(d['model'].unique())
    if anchor not in models: raise ValueError('Anchor absent')
    base_ids = set(d.loc[d['model']==anchor,'patient_id'])
    for model in models:
        if set(d.loc[d['model']==model,'patient_id']) != base_ids:
            raise ValueError(f'{model}: unmatched patients compared with anchor; do not silently subset')
    d['prob'] = d['prob'].clip(1e-7,1-1e-7)
    main = pd.DataFrame([dict(model=m, **metrics(g.y_cin2,g.prob)) for m,g in d.groupby('model')])
    center = pd.DataFrame([dict(model=m,center=c, **metrics(g.y_cin2,g.prob))
                           for (m,c),g in d.groupby(['model','center'])])
    wide = d.pivot(index='patient_id',columns='model',values='prob')
    meta = d[d.model==anchor].set_index('patient_id')
    comp=[]
    a = wide[anchor].to_numpy()
    al = np.log(a/(1-a))
    for m in models:
        if m==anchor: continue
        b = wide[m].to_numpy()
        bl = np.log(b/(1-b))
        y = meta.loc[wide.index,'y_cin2'].to_numpy()
        center_index = meta.loc[wide.index,'center'].to_numpy()
        comp.append(dict(model=m, anchor=anchor,
                         paired_auc_delta=metrics(y,b)['auc']-metrics(y,a)['auc'],
                         mean_abs_logit_residual=float(np.mean(np.abs(bl-al))),
                         p95_abs_logit_residual=float(np.quantile(np.abs(bl-al),.95)),
                         ranking_spearman=float(pd.Series(a).corr(pd.Series(b),method='spearman')),
                         threshold_flip_fraction_05=float(np.mean((a>=.5)!=(b>=.5))),
                         worse_center_count=int(sum(
                            metrics(y[center_index==c],b[center_index==c])['auc']
                            < metrics(y[center_index==c],a[center_index==c])['auc']
                            for c in np.unique(center_index)
                            if len(np.unique(y[center_index==c])) == 2)),
                         comparable_centers=int(sum(len(np.unique(y[center_index==c])) == 2
                                                    for c in np.unique(center_index)) )))
    details=[]
    for field in ['valid_oct_sites','has_colpo','reference_type','histology_only']:
        if field not in d.columns: continue
        for (model,value),g in d.groupby(['model',field],dropna=False):
            details.append(dict(subgroup=field, value=str(value),model=model,**metrics(g.y_cin2,g.prob)))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    main.to_csv(out/'summary.csv',index=False)
    center.to_csv(out/'by_center.csv',index=False)
    pd.DataFrame(comp).to_csv(out/'paired_diagnostics.csv',index=False)
    pd.DataFrame(details).to_csv(out/'by_subgroup.csv',index=False)
    issues=[]
    for row in main.itertuples():
        if np.isfinite(row.auc) and row.auc < 0.5:
            issues.append(f'{row.model}: AUROC below 0.5; verify label direction, logits and class mapping using SOURCE data')
    if len(d['center'].unique())<2:issues.append('No cross-center analysis possible')
    report=dict(patients=len(base_ids),centers=sorted(d.center.unique().tolist()),models=models,
                note='Diagnostics only; no hyperparameters, signs or thresholds chosen on target outcomes.',
                warnings=issues)
    (out/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(main.to_string(index=False))
    print('\nPaired prediction diagnostics:\n'+pd.DataFrame(comp).to_string(index=False))
    print('\nSaved to:',out)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('predictions',type=Path)
    p.add_argument('--anchor',default='anchor')
    p.add_argument('--out',type=Path,default=Path('rime_auc_audit'))
    a=p.parse_args()
    audit(pd.read_csv(a.predictions),a.anchor,a.out)


if __name__=='__main__': main()
