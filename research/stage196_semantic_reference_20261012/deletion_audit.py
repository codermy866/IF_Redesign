"""Label-blind matched OCT deletion selection and exact-versus-fixed evaluation."""
import os
os.umask(0o077)
from pathlib import Path
import argparse
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import core
from run_factorial import serial

ROOT=Path(__file__).resolve().parent
METHODS=('M1_MOMENT','M2_MOMENT_CFV','M3_LAPLACE','M4_LAPLACE_CFV')


def queue():
    rows=sorted(json.loads((ROOT/'private/COHORT.json').read_text()),key=lambda r:r['key'])
    folds=joblib.load(ROOT/'five_center/private/SOURCE_FROZEN_MODELS.joblib')
    cache=json.loads((ROOT/'private/CACHE_INDEX.json').read_text());tasks=[]
    for outer,fold in folds.items():
        patients=sorted(fold['test'],key=lambda i:hashlib.sha256((rows[i]['key']+'196').encode()).hexdigest())[:4]
        for index in patients:
            row=rows[index];key=row['key'];clinical=float(fold['clinical'][index]);beta=fold['base_alpha']/fold['A_scale']
            with np.load(cache[key]['path'],allow_pickle=False) as f:
                delta=f['delta'][0][None];mask=f['mask'][None].astype(bool);A=float(f['A'][0])
            valid=np.flatnonzero(mask[0,:12]);assert len(valid)>=2
            base=np.array([clinical+beta*A]);full=core.predict(fold['heads']['M4_LAPLACE_CFV'],delta,mask,base)[0]
            value=[]
            for slot in valid:
                altered=mask.copy();altered[:,slot]=False
                pred=core.predict(fold['heads']['M4_LAPLACE_CFV'],delta,altered,base-beta*delta[:,slot])[0]
                value.append((1 if full>=0 else -1)*(full-pred))
            ranking=valid[np.argsort(value,kind='stable')];rng=np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8],16)+196)
            choices={'high_single':[int(ranking[-1])],'low_single':[int(ranking[0])],
                     'random_single':[int(rng.choice(valid))],
                     'high_OCT_group':sorted(map(int,ranking[-2:])),
                     'low_OCT_group':sorted(map(int,ranking[:2])),
                     'random_OCT_group':sorted(map(int,rng.choice(valid,2,replace=False))),
                     'remove_colposcopy':list(range(12,20)), 'remove_OCT':list(range(12))}
            for condition,slots in choices.items():
                tasks.append(dict(key=key,center=outer,condition=condition,slots=slots))
    (ROOT/'private/EXACT_DELETION_QUEUE.json').write_text(json.dumps(tasks,indent=2)+'\n')
    (ROOT/'DELETION_PROTOCOL.json').write_text(json.dumps(dict(
        patients_per_center=4,patient_selection='fixed SHA order, no outcome use',
        ranking='M4 predicted-decision-aligned fixed-contribution margin, not true outcome value',
        modalities_matched=True,OCT_group_size=2,anatomical_adjacency_assumed=False,
        exact_full_prefix_recomputed=True,scope='small mechanism/approximation audit, not powered clinical validation',
        cases=len(tasks)),indent=2)+'\n')


def summarize():
    rows=sorted(json.loads((ROOT/'private/COHORT.json').read_text()),key=lambda r:r['key']);lookup={r['key']:i for i,r in enumerate(rows)}
    folds=joblib.load(ROOT/'five_center/private/SOURCE_FROZEN_MODELS.joblib')
    cache=json.loads((ROOT/'private/CACHE_INDEX.json').read_text());tasks=json.loads((ROOT/'private/EXACT_DELETION_QUEUE.json').read_text())
    result=[]
    for task in tasks:
        key=task['key'];i=lookup[key];patient=rows[i];fold=folds[task['center']]
        beta=fold['base_alpha']/fold['A_scale'];clinical=float(fold['clinical'][i]);y=patient['y']
        with np.load(cache[key]['path'],allow_pickle=False) as f:
            d=f['delta'][0][None];mask=f['mask'][None].astype(bool);A=float(f['A'][0])
        name=hashlib.sha256(json.dumps(dict(key=key,slots=task['slots']),sort_keys=True).encode()).hexdigest()[:24]+'.npz'
        with np.load(ROOT/'private/exact_deletions'/name,allow_pickle=False) as f:
            exact_d=f['delta'][0][None];exact_mask=f['mask'][None].astype(bool);exact_A=float(f['A'][0])
        approximate_mask=mask.copy();approximate_mask[:,task['slots']]=False
        full_base=np.array([clinical+beta*A]);exact_base=np.array([clinical+beta*exact_A])
        approx_base=np.array([clinical+beta*(A-d[:,task['slots']].sum())])
        for method in METHODS:
            head=fold['heads'][method]
            full=float(core.predict(head,d,mask,full_base)[0])
            exact=float(core.predict(head,exact_d,exact_mask,exact_base)[0])
            approximate=float(core.predict(head,d,approximate_mask,approx_base)[0])
            nll=lambda z:float(np.logaddexp(0,z)-y*z)
            result.append(dict(patient_key=key,center=task['center'],condition=task['condition'],method=method,
                full_NLL=nll(full),exact_deleted_NLL=nll(exact),exact_NLL_change=nll(exact)-nll(full),
                approximation_logit_error=abs(exact-approximate),approximation_NLL_error=abs(nll(exact)-nll(approximate))))
    df=pd.DataFrame(result);df.to_csv(ROOT/'private/DELETION_PATIENT_RESULTS.csv',index=False)
    summary=df.groupby(['method','condition']).agg(N=('patient_key','nunique'),
        mean_exact_NLL_change=('exact_NLL_change','mean'),mean_logit_approximation_error=('approximation_logit_error','mean'),
        max_logit_approximation_error=('approximation_logit_error','max')).reset_index()
    summary.to_csv(ROOT/'five_center/TABLE4_EXACT_DELETION_AND_APPROXIMATION.csv',index=False)
    pairs=[]
    for method in METHODS:
        for size,hi,lo,random in [('single','high_single','low_single','random_single'),
                                 ('OCT_group','high_OCT_group','low_OCT_group','random_OCT_group')]:
            pivot=df[df.method==method].pivot(index=['center','patient_key'],columns='condition',values='exact_NLL_change')
            differences=pivot[hi]-pivot[random];rng=np.random.default_rng(196)
            boot=np.zeros(3000)
            for center in pivot.index.get_level_values('center').unique():
                vals=differences.loc[center].to_numpy();draw=rng.integers(len(vals),size=(3000,len(vals)))
                boot+=vals[draw].sum(1)/len(differences)
            lower,upper=np.quantile(boot,[.025,.975])
            pairs.append(dict(method=method,deletion=size,patients=len(pivot),high_minus_random=float(differences.mean()),
                              CI_low=float(lower),CI_high=float(upper),high_minus_low=float((pivot[hi]-pivot[lo]).mean()),
                              inference='small matched mechanism audit; conditional patient uncertainty'))
    pd.DataFrame(pairs).to_csv(ROOT/'five_center/DELETION_VALUE_DIAGNOSTIC.csv',index=False)
    (ROOT/'EXACT_DELETION_STATUS.json').write_text(json.dumps(dict(state='COMPLETE',patients=df.patient_key.nunique(),
        model_sensitive_deletion_not_clinical_causality=True,per_view_outcome_value_learning_established=False),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['queue','summarize']);a=p.parse_args()
    queue() if a.stage=='queue' else summarize()
