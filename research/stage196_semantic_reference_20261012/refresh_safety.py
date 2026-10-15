"""Recompute only safety metadata after the documented CIN3 worksheet-join fix."""
from pathlib import Path
import json
import shutil
import joblib
import numpy as np
import pandas as pd
import core
import paired_statistics as stats
from run_factorial import semantic_features

ROOT=Path(__file__).resolve().parent


def main():
    dest=ROOT/'four_center_cache_pilot'
    if not (dest/'COMPLETE.json').exists():raise RuntimeError('Wait for existing source-frozen pilot')
    if (dest/'SAFETY_LABEL_CORRECTION.json').exists():return
    all_rows=json.loads((ROOT/'private/COHORT.json').read_text())
    rows=sorted([r for r in all_rows if r['center']!='wuhan' and r['old_input'] is not None],key=lambda r:r['key'])
    folds=joblib.load(dest/'private/SOURCE_FROZEN_MODELS.joblib')
    cache=json.loads((ROOT/'private/CACHE_INDEX.json').read_text());features=[]
    for row in rows:
        with np.load(cache[row['key']]['path'],allow_pickle=False) as f:
            features.append(dict(delta=f['delta'][0],mask=f['mask'].astype(bool),A=f['A'][0],native=f['q'][0],hidden=f['hidden'][0]))
    delta=np.stack([f['delta'] for f in features]);mask=np.stack([f['mask'] for f in features])
    hidden=np.stack([f['hidden'] for f in features]);native=np.array([f['native'] for f in features]);y=np.array([r['y'] for r in rows])
    safety=[]
    for center,fold in folds.items():
        test=fold['test'];train=fold['train']
        predictions={'CLINICAL':fold['clinical'][test],'JEV_NATIVE':native[test],'CLINICAL_JEV_SCALAR':fold['base'][test],
            'SEMANTIC_PCA26':fold['base'][test]+core.RHO*semantic_features(hidden[test],mask[test],fold['semantic']['state'])@fold['semantic']['head']['w']}
        predictions.update({name:core.predict(head,delta[test],mask[test],fold['base'][test]) for name,head in fold['heads'].items()})
        label3=np.array([-1 if rows[i]['cin3'] is None else rows[i]['cin3'] for i in test])
        for name,z in predictions.items():
            for target in (.9,.95):
                threshold=stats.source_threshold(y[train],fold['source_oof'][name],target);referred=z>=threshold
                tp=int((referred&(y[test]==1)).sum());tn=int((~referred&(y[test]==0)).sum())
                n3=int((label3==1).sum());detected=int((referred&(label3==1)).sum());lo,hi=stats.binomial_interval(detected,n3)
                safety.append(dict(center=center,method=name,source_CIN2_sensitivity_target=target,source_selected_threshold=threshold,
                    N=len(test),CIN2_sensitivity=tp/int(y[test].sum()),specificity=tn/int((y[test]==0).sum()),
                    CIN3_known=int((label3>=0).sum()),CIN3_unknown=int((label3<0).sum()),CIN3_positive=n3,CIN3_detected=detected,
                    CIN3_missed=n3-detected,CIN3_sensitivity=detected/n3 if n3 else None,CIN3_CI_low=lo,CIN3_CI_high=hi,
                    acceptable_harm_margin='NOT_SPECIFIED; no clinical safety pass claimed'))
    old=dest/'TABLE4_SOURCE_THRESHOLD_CIN3_SAFETY.csv'
    archive=dest/'snapshots';archive.mkdir(exist_ok=True)
    shutil.copyfile(old,archive/'TABLE4_BEFORE_CIN3_JOIN_FIX.csv')
    pd.DataFrame(safety).to_csv(old,index=False)
    (dest/'SAFETY_LABEL_CORRECTION.json').write_text(json.dumps(dict(
        primary_predictions_refitted=False,primary_CIN2_performance_unchanged=True,
        thresholds_reselected=False,source_frozen_models_unchanged=True,
        corrected='CIN3 labels only; pathology-bearing worksheet join',
        known=804,positive=82,unknown=89),indent=2)+'\n')
    print('CIN3 safety refreshed; source models and primary results unchanged.',flush=True)


if __name__=='__main__':main()
