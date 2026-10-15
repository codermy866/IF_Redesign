"""Add only complete five-fold comparators; incomplete methods stay in status."""
import runtime
from runtime import HERE,config,save
import json,time,numpy as np,pandas as pd
from scipy.special import expit,logit
from report import metrics,auc,bootstrap
from data import load_patients

def main():
    rows=load_patients();y=np.array([r['y'] for r in rows]);g=np.array([r['center'] for r in rows]);cfg=config()
    rc=json.loads((HERE/'RECENT_CONFIG.json').read_text());status=[];records=[];centres=[];seeds=[];cost=[];prob={};threshold={}
    expected=[('TRACK_'+track+'_'+kind,'recent_runs') for track in rc['tracks'] for kind in rc['kinds']+rc['simple'] if not(track=='B' and kind in rc['simple'])]
    expected +=[(n,'system_runs') for n in ['SYS_CLINICAL','SYS_OCT','SYS_COLPO','SYS_CONCAT_MLP','SYS_LATE_LOGIT','SYS_LATE_PROB']]
    for name,subdir in expected:
        complete=[c for c in cfg['centers'] if (HERE/'private'/subdir/c/(name+'.npz')).exists()]
        status.append(dict(method=name,state='COMPLETE' if len(complete)==5 else 'IN_PROGRESS_OR_NOT_RUN',completed_folds=len(complete),total_folds=5))
        if len(complete)!=5:continue
        pp=np.zeros((3,len(rows)));tt=np.zeros(len(rows));params=[];seconds=[];caps=0
        for c in cfg['centers']:
            folder=HERE/'private'/subdir/c
            with np.load(folder/(name+'.npz')) as f:
                te=np.flatnonzero(g==c);source=np.flatnonzero(g!=c)
                assert np.array_equal(f['test'],te) and set(f['source_indices'])==set(source)
                assert np.array_equal(f['source_y'],y[f['source_indices']])
                pp[:,te]=expit(f['seed_logits']);z=logit(np.clip(expit(f['source_seed_logits']).mean(0),1e-12,1-1e-12))
                tt[te]=np.quantile(z[f['source_y']==1],.05,method='lower')
            diag=folder/(name+'_REFIT_DIAGNOSTICS.json')
            if diag.exists():
                d=json.loads(diag.read_text());params.extend(r['parameters'] for r in d);seconds.extend(r['fit_seconds'] for r in d)
                sel=json.loads((folder/(name+'_SELECTION_LOCK.json')).read_text());caps+=sel.get('source_fit_caps_still_improving',0)
            else:
                d=json.loads((folder/(name+'_DIAGNOSTICS.json')).read_text());seconds.append(d['seconds']);params.extend(r['parameters'] for r in d['seeds'] if r['parameters'] is not None)
        p=pp.mean(0);prob[name]=p;threshold[name]=tt
        records.append(dict(method=name,track='A' if name.startswith('TRACK_A') else 'B' if name.startswith('TRACK_B') else 'System',N=len(rows),positive=int(y.sum()),prevalence=float(y.mean()),**metrics(y,p,tt),macro_AUROC=float(np.mean([auc(y[g==c],p[g==c]) for c in cfg['centers']])),source_fit_cap_warning=caps>0,independent_confirmation=False))
        for c in cfg['centers']:
            at=g==c;centres.append(dict(method=name,center=c,N=int(at.sum()),**metrics(y[at],p[at],tt[at])))
        for seed,pseed in zip(cfg['seeds'],pp):seeds.append(dict(method=name,seed=seed,**metrics(y,pseed,tt),stochastic=not name.startswith('SYS_') or name=='SYS_CONCAT_MLP'))
        cost.append(dict(method=name,parameters_min=min(params) if params else None,parameters_max=max(params) if params else None,recorded_fit_seconds=sum(seconds),source_fit_caps_still_improving=caps,source_cap_caveat='Do not claim fully converged superiority if cap warnings persist'))
    pd.DataFrame(status).to_csv(HERE/'tables/TABLE_BASELINE_STATUS.csv',index=False)
    core=pd.read_csv(HERE/'tables/TABLE_MAIN_RESULTS.csv')
    # Track B and system references have separate sections, never a mechanism row.
    for name,rr in [('RECENT_AND_SYSTEM',records),('RECENT_CENTERS',centres),('RECENT_SEEDS',seeds),('RECENT_COST',cost)]:pd.DataFrame(rr).to_csv(HERE/'tables'/('TABLE_'+name+'.csv'),index=False)
    combined=pd.concat([core[core.method.isin(['P0_ANCHOR','P3_PALE'])],pd.DataFrame(records)],ignore_index=True)
    combined.to_csv(HERE/'tables/TABLE_2_SYSTEM_AND_RECENT.csv',index=False)
    save(HERE/'BASELINE_REPORT_STATE.json',dict(time=time.time(),complete_methods=[r['method'] for r in records],incomplete_methods=[r['method'] for r in status if r['completed_folds']<5],Flow_MIL='CODE_NOT_CONFIRMED_NOT_RUN',tracks_separate=True,submission_claim=False))
    # Inferential tests are added only once the declared comparator family closes.
    if all(s['completed_folds']==5 for s in status):
        primary=pd.read_parquet(HERE/'private/PATIENT_PAIRED_PREDICTIONS.parquet')
        df=primary[primary.method=='P3_PALE'];lookup={r['patient_alias']:r for r in df.to_dict('records')}
        import hmac,hashlib
        salt=(HERE/'private/PATIENT_ALIAS_SALT.key').read_bytes()
        prob['P3_PALE']=np.array([lookup[hmac.new(salt,r['key'].encode(),hashlib.sha256).hexdigest()[:24]]['probability'] for r in rows])
        threshold['P3_PALE']=np.zeros(len(rows))
        # AUROC/NLL/Brier contrasts do not depend on the threshold placeholder.
        contrasts={'P3_PALE-'+n:('P3_PALE',n) for n in prob if n!='P3_PALE'}
        pd.DataFrame(bootstrap(y,g,prob,threshold,contrasts)).to_csv(HERE/'tables/TABLE_RECENT_PAIRED_CONTRASTS.csv',index=False)
    print('Baseline report:',len(records),'complete methods;',sum(s['completed_folds']<5 for s in status),'incomplete',flush=True)
if __name__=='__main__':main()
