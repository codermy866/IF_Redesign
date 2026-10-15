"""Patient-aligned paired inference; no best-seed reporting."""
import runtime
from runtime import HERE,save,config
import argparse,json
import numpy as np,pandas as pd,joblib,torch
from scipy.special import expit,logit
from scipy.stats import beta
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
from rime_dataset import load_patients

CORE=['A_ATTENTION_MOMENT','B_ATTENTION_LAPLACE','C_MEASURE_MOMENT','D_RIME_LAPLACE']
def metrics(y,p,thresholds):
    z=logit(np.clip(p,1e-12,1-1e-12));d=z>=thresholds
    return dict(AUROC=roc_auc_score(y,p),AUPRC=average_precision_score(y,p),NLL=float(np.mean(np.logaddexp(0,z)-y*z)),
        Brier=float(np.mean((y-p)**2)),F1=f1_score(y,d,zero_division=0),
        Sensitivity=float(d[y==1].mean()),Specificity=float((~d[y==0]).mean()))

def main(phase):
    cfg=config();rows=load_patients();y=np.array([r['y'] for r in rows]);g=np.array([r['center'] for r in rows])
    names=CORE.copy()
    centres=sorted(set(g))
    if all((HERE/'private/head_runs'/c/'B1_SOURCE_SELECTED_WARMUP.npz').exists() for c in centres):names+=['B1_SOURCE_SELECTED_WARMUP']
    if phase in ['B1','B2','B5','confirmation'] and all((HERE/'private/head_runs'/c/'B2_GEOMETRY.npz').exists() for c in centres):names+=['B2_GEOMETRY','B2_LOCAL','B2_LOCAL_CAPACITY_MATCHED','B2_CONTEXT_SHUFFLED']
    if phase in ['controls','B1','B2','B5','confirmation'] and all((HERE/'private/head_runs'/c/'MEASURE_LSE56.npz').exists() for c in centres):names+=['ATTENTION_DIVERSITY_LAPLACE','MEASURE_LSE56']
    if phase=='B5':names+=['B5_VALUE_AUXILIARY','B5_SHUFFLED_AUXILIARY','B5_STRATIFIED_SHUFFLED','B5_PATIENT_AUXILIARY','B5_POSTHOC_ONLY']
    raw_names=['ORIGINAL_RAW_ATTENTION_MOMENT','ORIGINAL_RAW_ATTENTION_LAPLACE'] if all((runtime.RECOVERY/'private/head_runs'/c/'B_ATTENTION_LAPLACE.npz').exists() for c in centres) else []
    folds=joblib.load(runtime.BASE/'196_controlled_cfv_jev_20261012/five_center/private/SOURCE_FROZEN_MODELS.joblib')
    baseline_names=['NO_JEV_THREE_MODAL_LINEAR','FROZEN_JEV_SEMANTIC_LINEAR_PROBE'] if (HERE/'BASELINES_COMPLETE.json').exists() else []
    ps={n:np.full((3,len(y)),np.nan) for n in names+['ANCHOR','CLINICAL']+baseline_names+raw_names};ths={n:np.full(len(y),np.nan) for n in ps}
    for outer in sorted(set(g)):
        tr=np.flatnonzero(g!=outer);te=np.flatnonzero(g==outer)
        part=joblib.load(HERE/'private/fold_features'/outer/'outer.joblib')
        for name in ['ANCHOR','CLINICAL']:
            z=part['anchor']['anchor'][te] if name=='ANCHOR' else folds[outer]['clinical'][te]
            source=folds[outer]['source_oof']['SEMANTIC_PCA26' if name=='ANCHOR' else 'CLINICAL']
            ps[name][:,te]=expit(z);ths[name][te]=np.quantile(source[y[tr]==1],.05,method='lower')
        for name in names+baseline_names+raw_names:
            folder='baselines' if name in baseline_names else 'head_runs'
            filename={'B5_POSTHOC_ONLY':'D_RIME_LAPLACE','ORIGINAL_RAW_ATTENTION_MOMENT':'A_ATTENTION_MOMENT','ORIGINAL_RAW_ATTENTION_LAPLACE':'B_ATTENTION_LAPLACE'}.get(name,name)
            base=runtime.RECOVERY if name in raw_names else HERE
            with np.load(base/'private'/folder/outer/(filename+'.npz')) as f:
                assert np.array_equal(f['test'],te) and np.array_equal(f['source_indices'],tr)
                ps[name][:,te]=expit(f['seed_logits'])
                source=logit(np.clip(expit(f['source_seed_logits']).mean(0),1e-12,1-1e-12))
                ths[name][te]=np.quantile(source[f['source_y']==1],.05,method='lower')
    if any(not np.isfinite(v).all() for v in ps.values()):raise RuntimeError('Incomplete patient predictions')
    probabilities={n:p.mean(0) for n,p in ps.items()};point={n:metrics(y,p,ths[n]) for n,p in probabilities.items()}
    contrasts={'A2_MINUS_A1':{'B_ATTENTION_LAPLACE':1,'A_ATTENTION_MOMENT':-1},
        'A3_MINUS_A1':{'C_MEASURE_MOMENT':1,'A_ATTENTION_MOMENT':-1},
        'A4_MINUS_A2':{'D_RIME_LAPLACE':1,'B_ATTENTION_LAPLACE':-1},
        'A4_MINUS_A3':{'D_RIME_LAPLACE':1,'C_MEASURE_MOMENT':-1},
        'A4_MINUS_A1':{'D_RIME_LAPLACE':1,'A_ATTENTION_MOMENT':-1},
        'INTERACTION':{'D_RIME_LAPLACE':1,'C_MEASURE_MOMENT':-1,'B_ATTENTION_LAPLACE':-1,'A_ATTENTION_MOMENT':1},
        'A4_MINUS_ANCHOR':{'D_RIME_LAPLACE':1,'ANCHOR':-1}}
    if 'B1_SOURCE_SELECTED_WARMUP' in names:contrasts['B1_MINUS_JOINT']={'B1_SOURCE_SELECTED_WARMUP':1,'D_RIME_LAPLACE':-1}
    if 'B2_GEOMETRY' in names:
        for n in ['B2_GEOMETRY','B2_LOCAL','B2_LOCAL_CAPACITY_MATCHED','B2_CONTEXT_SHUFFLED']:contrasts['CONDITIONAL_MINUS_'+n]={'D_RIME_LAPLACE':1,n:-1}
    if raw_names:
        contrasts['DEDUP_MINUS_RAW_ATTENTION_MOMENT']={'A_ATTENTION_MOMENT':1,'ORIGINAL_RAW_ATTENTION_MOMENT':-1}
        contrasts['DEDUP_MINUS_RAW_ATTENTION_LAPLACE']={'B_ATTENTION_LAPLACE':1,'ORIGINAL_RAW_ATTENTION_LAPLACE':-1}
    if 'MEASURE_LSE56' in names:
        contrasts['A4_MINUS_ATTENTION_DIVERSITY']={'D_RIME_LAPLACE':1,'ATTENTION_DIVERSITY_LAPLACE':-1}
        contrasts['A4_MINUS_LSE56']={'D_RIME_LAPLACE':1,'MEASURE_LSE56':-1}
    if phase=='B5':
        for n in ['D_RIME_LAPLACE','B5_SHUFFLED_AUXILIARY','B5_STRATIFIED_SHUFFLED','B5_PATIENT_AUXILIARY','ANCHOR']:contrasts['B5_VALUE_MINUS_'+n]={'B5_VALUE_AUXILIARY':1,n:-1}
    strata=[np.flatnonzero((g==c)&(y==v)) for c in sorted(set(g)) for v in [0,1]]
    rng=np.random.default_rng(cfg['bootstrap_seed']);boot={n:{m:[] for m in point[n]} for n in probabilities}
    cb={(n,c):[] for n in probabilities for c in sorted(set(g))}
    for repeat in range(cfg['bootstrap_replicates']):
        at=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata if len(s)])
        for n,p in probabilities.items():
            mm=metrics(y[at],p[at],ths[n][at])
            for key,value in mm.items():boot[n][key].append(value)
            for c in sorted(set(g)):
                cc=at[g[at]==c];cb[n,c].append(roc_auc_score(y[cc],p[cc]))
    table=[];centers=[];seeds=[];private=[];safety=[];calibration=[]
    for n,p in probabilities.items():
        record=dict(method=n,N=len(y),CIN2_positive=int(y.sum()),prevalence=float(y.mean()),**point[n],
            evaluation='five-centre reused-cohort internal development',independent_confirmation=False)
        for m,values in boot[n].items():record[m+'_CI_low'],record[m+'_CI_high']=np.quantile(values,[.025,.975])
        aucs=[]
        for c in sorted(set(g)):
            at=g==c;mm=metrics(y[at],p[at],ths[n][at]);aucs.append(mm['AUROC'])
            lo,hi=np.quantile(cb[n,c],[.025,.975]);centers.append(dict(method=n,center=c,N=int(at.sum()),**mm,AUROC_CI_low=lo,AUROC_CI_high=hi))
            label3=np.array([-1 if r['cin3'] is None else r['cin3'] for r in rows])[at];detected=logit(np.clip(p[at],1e-12,1-1e-12))>=ths[n][at]
            n3=int((label3==1).sum());d3=int(((label3==1)&detected).sum())
            safety.append(dict(method=n,center=c,CIN3_known=int((label3>=0).sum()),CIN3_unknown=int((label3<0).sum()),CIN3_positive=n3,CIN3_detected=d3,CIN3_missed=n3-d3,
                CIN3_sensitivity=d3/n3 if n3 else None,CI_low=float(beta.ppf(.025,d3,n3-d3+1)) if d3 else 0.,CI_high=float(beta.ppf(.975,d3+1,n3-d3)) if d3<n3 else 1.,clinical_safety_pass=False))
            bins=np.minimum((p[at]*10).astype(int),9)
            for bin_id in range(10):
                inside=bins==bin_id
                if inside.any():calibration.append(dict(method=n,center=c,bin=bin_id,N=int(inside.sum()),mean_probability=float(p[at][inside].mean()),observed_rate=float(y[at][inside].mean())))
        record.update(macro_AUROC=float(np.mean(aucs)),worst_center_AUROC=float(min(aucs)));table.append(record)
        for seed,pp in zip(cfg['seeds'],ps[n]):seeds.append(dict(method=n,seed=seed,**metrics(y,pp,ths[n])))
        for i,r in enumerate(rows):private.append(dict(key=r['key'],center=g[i],y=int(y[i]),cin3=r['cin3'],method=n,probability=p[i],source_threshold=ths[n][i],seed17=ps[n][0,i],seed29=ps[n][1,i],seed43=ps[n][2,i]))
    paired=[]
    for name,coef in contrasts.items():
        observed=sum(v*point[n]['AUROC'] for n,v in coef.items());samples=sum(v*np.asarray(boot[n]['AUROC']) for n,v in coef.items())
        lo,hi=np.quantile(samples,[.025,.975]);q=.05/(2*cfg['new_comparison_family_size']);al,ah=np.quantile(samples,[q,1-q])
        cds={c:sum(v*roc_auc_score(y[g==c],probabilities[n][g==c]) for n,v in coef.items()) for c in sorted(set(g))}
        paired.append(dict(contrast=name,AUROC_delta=observed,CI_low=lo,CI_high=hi,Bonferroni_CI_low=al,Bonferroni_CI_high=ah,
            positive_centers=sum(v>0 for v in cds.values()),**{c+'_delta':v for c,v in cds.items()},support=bool(al>0 and sum(v>0 for v in cds.values())>=4),
            multiplicity_family='23 planned new development contrasts, including 2 prior CV1 comparisons; original Stage197 primary family unchanged',family_size=cfg['new_comparison_family_size']))
    pd.DataFrame(table).to_csv(HERE/f'TABLE_{phase.upper()}_MAIN.csv',index=False)
    pd.DataFrame(paired).to_csv(HERE/f'TABLE_{phase.upper()}_CONTRASTS.csv',index=False)
    pd.DataFrame(centers).to_csv(HERE/f'TABLE_{phase.upper()}_CENTERS.csv',index=False)
    pd.DataFrame(seeds).to_csv(HERE/f'TABLE_{phase.upper()}_SEEDS.csv',index=False)
    pd.DataFrame(safety).to_csv(HERE/f'TABLE_{phase.upper()}_CIN3.csv',index=False)
    pd.DataFrame(calibration).to_csv(HERE/f'TABLE_{phase.upper()}_CALIBRATION.csv',index=False)
    pd.DataFrame(private).to_parquet(HERE/f'private/{phase.upper()}_PATIENT_PREDICTIONS.parquet',index=False)
    save(HERE/(phase.upper()+'_CONTRIBUTION_REVIEW.json'),dict(state='COMPLETED_ACTUAL_RESULTS',contrasts=paired,
        interaction_is_diagnostic_not_required=True,independent_confirmation=False,
        next_priority='B1 optimizer then B2 conditioning, without changing core preprocessing; B5 only after legal native targets',
        retain_candidate_by_source_only=True))
    (HERE/(phase.upper()+'_CONTRIBUTION_REVIEW.md')).write_text('# Actual contribution evidence\n\n'+pd.DataFrame(paired).to_csv(index=False)+'\nAll three seeds retained. Bootstrap conditions on models and historical development; only five centres. No independent cohort.\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['core','controls','B1','B2','B5','confirmation'],required=True);main(p.parse_args().phase)
