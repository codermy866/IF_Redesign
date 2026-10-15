"""Source-only center CV then locked patient-level factorial evaluation.

The four-center cache pilot and the selected five-center study are separate.
Both condition on ONE frozen released JEV backbone; adaptation is not selected
using these comparisons. No main-method pointer is overwritten.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.umask(0o077)
from pathlib import Path
import argparse
import hashlib
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import core
import paired_statistics as stats

ROOT=Path(__file__).resolve().parent


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(serial(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)


def clinical_frame(rows):
    return pd.DataFrame([dict(age=np.nan if r['age'] is None else r['age'],hpv=r['hpv'],tct=r['tct']) for r in rows])


def fit_prior(frame,y,train):
    transform=ColumnTransformer([
        ('age',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),['age']),
        ('categories',OneHotEncoder(handle_unknown='ignore'),['hpv','tct'])])
    prior=Pipeline([('preprocess',transform),('logistic',LogisticRegression(C=1.,max_iter=2000,random_state=196))])
    prior.fit(frame.iloc[train],y[train]);return prior


def native_scale(values):return max(float(np.quantile(np.abs(values),.9)),.05)


def fit_head_from_matrix(x,base,y,g,lam):
    from scipy.optimize import minimize
    opt=minimize(lambda w:core.objective(w,x,base,y,g,lam,'NONE'),np.zeros(26),jac=True,
                 method='L-BFGS-B',options=dict(maxiter=1000,maxls=100,ftol=1e-11,gtol=1e-7))
    if not np.isfinite(opt.fun) or (not opt.success and np.max(abs(opt.jac))>1e-4):raise RuntimeError('Reference head convergence')
    return dict(w=opt.x,lambda_=lam,rho=core.RHO,parameters=26)


def semantic_state(hidden,mask):
    _,_,vt=np.linalg.svd(hidden-hidden.mean(0),full_matrices=False)
    basis=vt[:26].T;std=np.maximum((hidden@basis).std(0),.005)
    return dict(basis=basis,std=std)


def semantic_features(hidden,mask,state):
    x=np.clip(hidden@state['basis']/state['std'],-16,16)
    return np.where(mask.any(1)[:,None],x,0.)


def main(cohort):
    audit=json.loads((ROOT/'DATA_AUDIT.json').read_text())
    if audit['state']!='AUDITED_WITH_LIMITATIONS' or audit['cross_patient_shared_image_hashes']:
        raise RuntimeError('Global image leakage audit not cleared')
    assert json.loads((ROOT/'CORE_VERIFICATION.json').read_text())['state']=='PASS'
    protocol=json.loads((ROOT/'DESIGN_LOCK.json').read_text())
    for name,digest in protocol['code_dependencies'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest, 'Frozen code changed'
    all_rows=json.loads((ROOT/'private/COHORT.json').read_text())
    rows=sorted([r for r in all_rows if cohort=='five' or (r['center']!='wuhan' and r['old_input'] is not None)],key=lambda r:r['key'])
    dest=ROOT/('five_center' if cohort=='five' else 'four_center_cache_pilot');dest.mkdir(exist_ok=True)
    if (dest/'COMPLETE.json').exists():return
    cache=json.loads((ROOT/'private/CACHE_INDEX.json').read_text())
    loaded=[]
    for r in rows:
        path=Path(cache[r['key']]['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==cache[r['key']]['sha256']
        with np.load(path,allow_pickle=False) as f:
            loaded.append({name:f[name][0] for name in ('delta','A','I','q','hidden','only_o','only_k')}|{'mask':f['mask'].astype(bool)})
    delta=np.stack([r['delta'] for r in loaded]);mask=np.stack([r['mask'] for r in loaded])
    native=np.asarray([r['q'] for r in loaded]);A=np.asarray([r['A'] for r in loaded])
    hidden=np.stack([r['hidden'] for r in loaded])
    y=np.asarray([r['y'] for r in rows]);g=np.asarray([r['center'] for r in rows]);frame=clinical_frame(rows)
    folds={};selection=[];oof=np.zeros((len(rows),len(core.ARMS)+4));names=['CLINICAL','JEV_NATIVE','CLINICAL_JEV_SCALAR','SEMANTIC_PCA26']+list(core.ARMS)
    assert oof.shape[1]==len(names)
    started=time.monotonic()
    # EVERY source state is frozen before any outer metric is calculated.
    for outer in sorted(np.unique(g)):
        train=np.flatnonzero(g!=outer);test=np.flatnonzero(g==outer)
        assert not set(train)&set(test)
        cv=[];baseline_cv={a:np.zeros(len(rows)) for a in protocol['base_alpha_grid']}
        clinical_cv=np.zeros(len(rows))
        for held in sorted(np.unique(g[train])):
            tr=np.flatnonzero((g!=outer)&(g!=held));va=np.flatnonzero(g==held)
            prior=fit_prior(frame,y,tr);c_tr=prior.decision_function(frame.iloc[tr]);c_va=prior.decision_function(frame.iloc[va])
            a_scale=native_scale(A[tr]);clinical_cv[va]=c_va
            for alpha in baseline_cv:baseline_cv[alpha][va]=c_va+alpha*A[va]/a_scale
            cv.append(dict(tr=tr,va=va,c_tr=c_tr,c_va=c_va,a_scale=a_scale))
        selected_alpha=max(baseline_cv,key=lambda a:(core.metrics(y[train],baseline_cv[a][train])['AUROC'],
            -core.metrics(y[train],baseline_cv[a][train])['NLL'],-a))
        prior=fit_prior(frame,y,train);clinical=prior.decision_function(frame)
        a_scale=native_scale(A[train]);visual_scale=selected_alpha/a_scale;base=clinical+visual_scale*A
        fold=dict(train=train,test=test,prior=prior,clinical=clinical,base=base,
                  base_alpha=selected_alpha,A_scale=a_scale,heads={},source_oof={
                      'CLINICAL':clinical_cv[train],'JEV_NATIVE':native[train],
                      'CLINICAL_JEV_SCALAR':baseline_cv[selected_alpha][train]})
        for arm,(kind,constraint) in core.ARMS.items():
            candidates=[]
            for lam in core.LAMBDAS:
                source_oof=np.zeros(len(rows))
                for part in cv:
                    tr,va=part['tr'],part['va'];beta=selected_alpha/part['a_scale']
                    head=core.fit(delta[tr],mask[tr],part['c_tr']+beta*A[tr],y[tr],g[tr],kind,constraint,lam,beta)
                    source_oof[va]=core.predict(head,delta[va],mask[va],part['c_va']+beta*A[va])
                candidates.append(dict(lambda_=lam,source_oof=source_oof,**core.metrics(y[train],source_oof[train])))
            best=max(candidates,key=lambda r:(r['AUROC'],-r['NLL'],r['lambda_']))
            fold['heads'][arm]=core.fit(delta[train],mask[train],base[train],y[train],g[train],kind,constraint,best['lambda_'],visual_scale)
            fold['source_oof'][arm]=best['source_oof'][train]
            selection.append(dict(fold=outer,method=arm,selected_lambda=best['lambda_'],base_alpha=selected_alpha,
                                  AUROC=best['AUROC'],NLL=best['NLL'],parameters=26,source_only=True))
        # A representative early semantic fusion control, matched at 26 coefficients.
        choices=[]
        semantic_cv=[]
        for part in cv:
            tr,va=part['tr'],part['va'];state=semantic_state(hidden[tr],mask[tr])
            semantic_cv.append((part,semantic_features(hidden[tr],mask[tr],state),semantic_features(hidden[va],mask[va],state)))
        for lam in core.LAMBDAS:
            source_oof=np.zeros(len(rows))
            for part,xt,xv in semantic_cv:
                tr,va=part['tr'],part['va'];beta=selected_alpha/part['a_scale']
                head=fit_head_from_matrix(xt,part['c_tr']+beta*A[tr],y[tr],g[tr],lam)
                source_oof[va]=part['c_va']+beta*A[va]+core.RHO*xv@head['w']
            choices.append(dict(lambda_=lam,source_oof=source_oof,**core.metrics(y[train],source_oof[train])))
        best=max(choices,key=lambda r:(r['AUROC'],-r['NLL'],r['lambda_']))
        sem=semantic_state(hidden[train],mask[train]);xh=semantic_features(hidden,mask,sem)
        fold['semantic']=dict(state=sem,head=fit_head_from_matrix(xh[train],base[train],y[train],g[train],best['lambda_']))
        fold['source_oof']['SEMANTIC_PCA26']=best['source_oof'][train]
        folds[outer]=fold
        save(dest/'STATE.json',dict(state='SOURCE_CENTER_CV',fold=outer,completed_folds=len(folds),seconds=time.monotonic()-started))
        print('SOURCE_FROZEN',cohort,outer,'alpha',selected_alpha,flush=True)
    model_path=dest/'private/SOURCE_FROZEN_MODELS.joblib';model_path.parent.mkdir(exist_ok=True);joblib.dump(folds,model_path)
    save(dest/'SOURCE_METHOD_LOCK.json',dict(state='ALL_SOURCE_MODELS_FROZEN',backbone='FROZEN_RELEASED_JEV27B',
        adapter_strategy_selection_completed=False,source_only_inner_CV=True,
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),historical_label_exposure=True,
        independent_confirmation=False,cohort=cohort))
    pd.DataFrame(selection).to_csv(dest/'SOURCE_SELECTION.csv',index=False)
    center_rows=[];safety=[];calibration=[]
    for outer,fold in folds.items():
        test=fold['test'];train=fold['train'];z={'CLINICAL':fold['clinical'][test],
            'JEV_NATIVE':native[test],'CLINICAL_JEV_SCALAR':fold['base'][test],
            'SEMANTIC_PCA26':fold['base'][test]+core.RHO*semantic_features(hidden[test],mask[test],fold['semantic']['state'])@fold['semantic']['head']['w']}
        z.update({arm:core.predict(head,delta[test],mask[test],fold['base'][test]) for arm,head in fold['heads'].items()})
        for j,name in enumerate(names):
            oof[test,j]=z[name];center_rows.append(dict(center=outer,method=name,**core.metrics(y[test],z[name])))
            probability=expit(z[name]);bins=np.minimum((10*probability).astype(int),9)
            for k in range(10):
                at=bins==k
                if at.any():calibration.append(dict(center=outer,method=name,bin=k,N=int(at.sum()),mean_probability=float(probability[at].mean()),observed_positive_rate=float(y[test][at].mean())))
            for target in (.90,.95):
                threshold=stats.source_threshold(y[train],fold['source_oof'][name],target)
                referred=z[name]>=threshold;tp=int((referred&(y[test]==1)).sum());tn=int((~referred&(y[test]==0)).sum())
                label3=np.asarray([-1 if rows[i]['cin3'] is None else rows[i]['cin3'] for i in test])
                positive3=label3==1;known3=label3>=0;n3=int(positive3.sum());detected=int((referred&positive3).sum());lo,hi=stats.binomial_interval(detected,n3)
                safety.append(dict(center=outer,method=name,source_CIN2_sensitivity_target=target,source_selected_threshold=threshold,
                    N=len(test),CIN2_sensitivity=tp/int(y[test].sum()),specificity=tn/int((y[test]==0).sum()),
                    CIN3_known=int(known3.sum()),CIN3_unknown=int((~known3).sum()),CIN3_positive=n3,CIN3_detected=detected,
                    CIN3_missed=n3-detected,CIN3_sensitivity=detected/n3 if n3 else None,CIN3_CI_low=lo,CIN3_CI_high=hi,
                    acceptable_harm_margin='NOT_SPECIFIED; no clinical safety pass claimed'))
    all_scores=dict(zip(names,oof.T));table=[dict(method=name,**core.metrics(y,z),scope='reused selected cohort internal development',independent_confirmation=False) for name,z in all_scores.items()]
    pd.DataFrame(table).to_csv(dest/'TABLE1_BASELINES.csv',index=False)
    centers=pd.DataFrame(center_rows);centers.to_csv(dest/'TABLE2_LOCO_CENTERS.csv',index=False)
    pd.DataFrame(safety).to_csv(dest/'TABLE4_SOURCE_THRESHOLD_CIN3_SAFETY.csv',index=False)
    pd.DataFrame(calibration).to_csv(dest/'TABLE4_CALIBRATION_BINS.csv',index=False)
    keys=['M1_MOMENT','M2_MOMENT_CFV','M3_LAPLACE','M4_LAPLACE_CFV']
    boot=stats.paired_bootstrap(y,g,all_scores,protocol['bootstrap_repeats'],196)
    values={row['method']:row['AUROC'] for row in table}
    contrasts={'Laplace_without_CFV':{keys[2]:1,keys[0]:-1},'CFV_with_Moment':{keys[1]:1,keys[0]:-1},
        'CFV_with_Laplace':{keys[3]:1,keys[2]:-1},'Full_vs_Moment':{keys[3]:1,keys[0]:-1},
        'Interaction':{keys[3]:1,keys[2]:-1,keys[1]:-1,keys[0]:1},
        'Laplace_vs_FOCAL':{keys[2]:1,'FOCAL':-1},
        'CFV_vs_SimpleConsistency':{keys[3]:1,'LAPLACE_SIMPLE_CONSISTENCY':-1}}
    contrast_rows=[]
    for label,coefficients in contrasts.items():
        point=sum(coeff*values[name] for name,coeff in coefficients.items());samples=sum(coeff*boot[name] for name,coeff in coefficients.items())
        center_delta={c:sum(coeff*float(centers[(centers.center==c)&(centers.method==name)].AUROC.iloc[0]) for name,coeff in coefficients.items()) for c in np.unique(g)}
        lower,upper=np.quantile(samples,[.025,.975]);family_lower,family_upper=np.quantile(samples,[.05/(2*len(contrasts)),1-.05/(2*len(contrasts))])
        contrast_rows.append(dict(contrast=label,AUROC_delta=point,CI_low=lower,CI_high=upper,
            Bonferroni_CI_low=family_lower,Bonferroni_CI_high=family_upper,
            positive_centers=sum(v>0 for v in center_delta.values()),centers=len(center_delta),
            conditional_support=bool(family_lower>0 and sum(v>0 for v in center_delta.values())>=int(np.ceil(.75*len(center_delta)))),
            inference='conditional on fitted models; historical model-selection uncertainty not included',**{c+'_delta':v for c,v in center_delta.items()}))
    pd.DataFrame(contrast_rows).to_csv(dest/'TABLE3_FACTORIAL_CONTRASTS.csv',index=False)
    # Center-level results/leave-one-center-out pooled changes are separate from patient bootstrap.
    sensitivity=[]
    for dropped in np.unique(g):
        at=g!=dropped;v={name:core.metrics(y[at],all_scores[name][at])['AUROC'] for name in keys}
        sensitivity.append(dict(excluded_center=dropped,**v,interaction=(v[keys[3]]-v[keys[2]])-(v[keys[1]]-v[keys[0]])))
    pd.DataFrame(sensitivity).to_csv(dest/'CENTER_SENSITIVITY.csv',index=False)
    private=pd.DataFrame(dict(patient_key=[r['key'] for r in rows],center=g,y=y,**{name:expit(z) for name,z in all_scores.items()}))
    private.to_csv(dest/'private/PATIENT_PREDICTIONS.csv',index=False)
    save(dest/'CLAIM_STATUS.json',dict(state='CONTROLLED_FACTORIAL_COMPLETE',cohort=cohort,N=len(rows),
        upstream='FROZEN_RELEASED_JEV27B',upstream_adaptation_comparison_complete=False,
        factorial=contrast_rows,CFV_definition='center-risk non-inferiority soft penalty; not a new per-view value learner',
        exact_deletion_mechanism_complete=False,independent_confirmation=False,
        CIN3_labels='strict research parsing with unresolved mask, not human adjudication',
        final_method_frozen=False,main_pointer_unchanged=True))
    save(dest/'COMPLETE.json',dict(state='COMPLETE',N=len(rows),folds=len(folds),seconds=time.monotonic()-started))
    print('COMPLETE',cohort,len(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',choices=['four','five'],required=True)
    main(p.parse_args().cohort)
