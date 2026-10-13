"""Completed matched baseline controls; no RIME factorial claims."""
import runtime
from runtime import HERE,BASE,save,config
import numpy as np,pandas as pd,joblib
from scipy.special import expit,logit
from rime_dataset import load_patients
from summarize import metrics

def main():
    rows=load_patients();y=np.array([r['y'] for r in rows]);g=np.array([r['center'] for r in rows]);names=['CLINICAL','ANCHOR','NO_JEV_THREE_MODAL_LINEAR','FROZEN_JEV_SEMANTIC_LINEAR_PROBE']
    p={n:np.zeros(len(rows)) for n in names};th={n:np.zeros(len(rows)) for n in names}
    previous=pd.read_csv(BASE/'196_controlled_cfv_jev_20261012/five_center/private/PATIENT_PREDICTIONS.csv').set_index('patient_key')
    folds=joblib.load(BASE/'196_controlled_cfv_jev_20261012/five_center/private/SOURCE_FROZEN_MODELS.joblib')
    for outer in sorted(set(g)):
        tr=np.flatnonzero(g!=outer);te=np.flatnonzero(g==outer)
        for name in ['CLINICAL','ANCHOR']:
            source_name='SEMANTIC_PCA26' if name=='ANCHOR' else name
            p[name][te]=previous.loc[[rows[i]['key'] for i in te],source_name]
            source=folds[outer]['source_oof'][source_name];th[name][te]=np.quantile(source[y[tr]==1],.05,method='lower')
        for name in names[2:]:
            with np.load(HERE/'private/baselines'/outer/(name+'.npz')) as f:
                assert np.array_equal(f['test'],te) and np.array_equal(f['source_indices'],tr)
                p[name][te]=expit(f['seed_logits']).mean(0)
                source=logit(np.clip(expit(f['source_seed_logits']).mean(0),1e-12,1-1e-12));th[name][te]=np.quantile(source[f['source_y']==1],.05,method='lower')
    boot={name:{m:[] for m in metrics(y,p[name],th[name])} for name in names};rng=np.random.default_rng(199)
    strata=[np.flatnonzero((g==c)&(y==v)) for c in sorted(set(g)) for v in [0,1]]
    for b in range(3000):
        at=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata if len(s)])
        for name in names:
            for metric,value in metrics(y[at],p[name][at],th[name][at]).items():boot[name][metric].append(value)
    table=[];centers=[];private=[]
    for name in names:
        record=dict(method=name,N=len(y),**metrics(y,p[name],th[name]),scope='cohort matched internal development; backbone/head/preprocessing differ')
        for metric,values in boot[name].items():record[metric+'_CI_low'],record[metric+'_CI_high']=np.quantile(values,[.025,.975])
        table.append(record)
        for center in sorted(set(g)):
            at=g==center;centers.append(dict(method=name,center=center,N=int(at.sum()),**metrics(y[at],p[name][at],th[name][at])))
        for i,row in enumerate(rows):private.append(dict(key=row['key'],center=g[i],method=name,y=int(y[i]),probability=p[name][i],source_threshold=th[name][i]))
    pd.DataFrame(table).to_csv(HERE/'TABLE_BASELINES_MAIN.csv',index=False)
    pd.DataFrame(centers).to_csv(HERE/'TABLE_BASELINES_CENTERS.csv',index=False)
    pd.DataFrame(private).to_parquet(HERE/'private/BASELINE_PATIENT_PREDICTIONS.parquet',index=False)
    save(HERE/'BASELINE_REPORT_COMPLETE.json',dict(state='COMPLETE',N=984,bootstrap=3000,seeds=[17,29,43],
        linear_controls_deterministic_identical_seed_predictions=True,core_factorial_not_evaluated=True,independent_confirmation=False))

if __name__=='__main__':main()
