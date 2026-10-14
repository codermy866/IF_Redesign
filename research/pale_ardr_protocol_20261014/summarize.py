"""Patient-aligned complete-matrix reporting, safety and an explicit stop verdict."""
import runtime
from runtime import HERE,OLD,save,config
import json,hmac,hashlib,os
import numpy as np,pandas as pd,torch
from scipy.special import expit,logit
from scipy.stats import beta
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
from rime_dataset import load_patients

def metrics(y,p,t):
    z=logit(np.clip(p,1e-12,1-1e-12));d=z>=t
    return dict(AUROC=roc_auc_score(y,p),AUPRC=average_precision_score(y,p),NLL=float(np.mean(np.logaddexp(0,z)-y*z)),
                Brier=float(np.mean((p-y)**2)),Sensitivity=float(d[y==1].mean()),Specificity=float((~d[y==0]).mean()),F1=f1_score(y,d,zero_division=0))

def not_run(reason):
    for name,cols in {'ARDR_MAIN_COMPARISON.csv':['method','AUROC','AUPRC','NLL','Brier','status'],
        'ARDR_CENTER_RESULTS.csv':['method','center','AUROC','NLL'], 'ARDR_SEED_RESULTS.csv':['method','seed','AUROC','NLL'],
        'ARDR_ANCHOR_RISK_DELTA.csv':['method','center','NLL_delta_vs_anchor'],
        'ARDR_DECISION_CHANGE.csv':['method','anchor_correct_to_wrong','anchor_wrong_to_correct'],
        'ARDR_CIN3_SAFETY.csv':['method','known','unknown','positive','detected','missed'],
        'ARDR_PAIRED_BOOTSTRAP.csv':['contrast','metric','delta','CI_low','CI_high']}.items():
        pd.DataFrame(columns=cols).to_csv(HERE/name,index=False)
    (HERE/'ARDR_MECHANISM_ANALYSIS.md').write_text('# Source-only stop\n\n'+reason+'\n\nOuter comparisons were not run. Generalization and CIN3+ outcome changes remain Inconclusive. No additional architecture or parameter search.\n')
    (HERE/'ARDR_FINAL_VERDICT.md').write_text('# Final admission decision\n\nARDR不纳入主方法。\n\n'+reason+'\n\nSource pilot evidence is retained in ARDR_PILOT_RESULTS.csv; unrun patient comparisons are empty with explicit state, not fabricated zero scores. PALE historical results remain unchanged.\n')
    save(HERE/'FINAL_VERDICT.json',dict(decision='ARDR不纳入主方法',status='Not Supported for admission; Inconclusive for unrun outer comparisons',reason=reason,complete_outer_matrix=False))

def main():
    cfg=config();rows=load_patients();y=np.array([r['y'] for r in rows]);g=np.array([r['center'] for r in rows]);label3=np.array([-1 if r['cin3'] is None else r['cin3'] for r in rows]);centers=cfg['centers'];methods=cfg['methods']
    if not all((HERE/'private/runs'/c/'COMPLETE.json').exists() for c in centers):
        reason=json.loads((HERE/'PILOT_GATE.json').read_text())['state'] if (HERE/'PILOT_GATE.json').exists() else 'Experiment incomplete; no complete patient comparison available'
        not_run(reason);return
    N=len(rows);seed_p={n:np.full((3,N),np.nan) for n in methods+['ANCHOR']};threshold={n:np.full(N,np.nan) for n in seed_p};train_rows=[];optimism=[]
    for c in centers:
        folder=HERE/'private/runs'/c;te=np.flatnonzero(g==c);tr=np.flatnonzero(g!=c)
        optimism.extend(dict(outer=c,**r) for r in json.loads((folder/'ANCHOR_OPTIMISM_AUDIT.json').read_text()))
        for name in methods:
            with np.load(folder/(name+'.npz')) as f:
                assert np.array_equal(f['test'],te) and set(f['source_indices'])==set(tr) and np.array_equal(f['source_y'],y[f['source_indices']])
                seed_p[name][:,te]=expit(f['seed_logits']);seed_p['ANCHOR'][:,te]=expit(f['anchor_logits'])
                source=logit(np.clip(expit(f['source_seed_logits']).mean(0),1e-12,1-1e-12))
                threshold[name][te]=np.quantile(source[f['source_y']==1],.05,method='lower')
                threshold['ANCHOR'][te]=np.quantile(f['source_anchor_logits'][f['source_y']==1],.05,method='lower')
            for md in json.loads((folder/(name+'_REFIT_DIAGNOSTICS.json')).read_text()):
                for idx,src in enumerate(md['training_centres']):train_rows.append(dict(outer=c,method=name,seed=md['seed'],source_center=src,
                    source_model_NLL=md['training_model_risk'][idx],source_anchor_NLL=md['training_anchor_risk'][idx],source_risk_delta=md['training_risk_delta'][idx],
                    epsilon=md['candidate'].get('epsilon',0.),constraint_satisfied=md['training_risk_delta'][idx]<=md['candidate'].get('epsilon',0.)+1e-8,
                    mean_abs_delta=md['source_mean_abs_delta'],nonzero_fraction=md['source_nonzero_delta_fraction'],fit_seconds=md['fit_seconds']))
    assert all(np.isfinite(p).all() for p in seed_p.values())
    probabilities={n:p.mean(0) for n,p in seed_p.items()};main_rows=[];center_rows=[];seed_rows=[];changes=[];safety=[];private=[]
    salt_path=HERE/'private/PREDICTION_ALIAS_SALT.key'
    if not salt_path.exists():salt_path.write_bytes(os.urandom(32))
    salt=salt_path.read_bytes();anchor=probabilities['ANCHOR'];az=logit(np.clip(anchor,1e-12,1-1e-12));ad=az>=threshold['ANCHOR']
    for name,p in probabilities.items():
        z=logit(np.clip(p,1e-12,1-1e-12));delta=z-az;m=metrics(y,p,threshold[name]);aucs=[];nlls=[]
        for c in centers:
            at=g==c;mm=metrics(y[at],p[at],threshold[name][at]);aucs.append(mm['AUROC']);nlls.append(mm['NLL'])
            center_rows.append(dict(method=name,center=c,N=int(at.sum()),positive=int(y[at].sum()),**mm))
        main_rows.append(dict(method=name,N=N,positive=int(y.sum()),**m,macro_AUROC=float(np.mean(aucs)),worst_center_AUROC=float(min(aucs)),
            macro_center_NLL=float(np.mean(nlls)),worst_center_NLL=float(max(nlls)),mean_absolute_logit_update=float(abs(delta).mean()),
            nonzero_update_fraction=float((abs(delta)>1e-4).mean()),evaluation='FR1 shared repair; repeated-cohort five-centre internal-external development',independent_confirmation=False))
        for seed,pp in zip(cfg['seeds'],seed_p[name]):seed_rows.append(dict(method=name,seed=seed,**metrics(y,pp,threshold[name])))
        for scope,t in [('anchor_fixed_source_threshold',threshold['ANCHOR']),('same_source95_policy_model_specific_threshold',threshold[name])]:
            d=z>=t;correct_anchor=ad==y;correct_model=d==y
            changes.append(dict(method=name,decision_scope=scope,anchor_correct_to_wrong=int((correct_anchor&~correct_model).sum()),
                anchor_wrong_to_correct=int((~correct_anchor&correct_model).sum()),mean_abs_logit_update=float(abs(delta).mean()),nonzero_fraction=float((abs(delta)>1e-4).mean())))
        for c in ['POOLED']+centers:
            at=np.ones(N,bool) if c=='POOLED' else g==c;d=z[at]>=threshold[name][at];ll=label3[at];n=int((ll==1).sum());hit=int(((ll==1)&d).sum())
            safety.append(dict(method=name,center=c,known=int((ll>=0).sum()),unknown=int((ll<0).sum()),positive=n,detected=hit,missed=n-hit,
                CIN3_sensitivity=hit/n if n else None,CI_low=float(beta.ppf(.025,hit,n-hit+1)) if hit else 0.,CI_high=float(beta.ppf(.975,hit+1,n-hit)) if hit<n else 1.,
                CIN2_sensitivity=metrics(y[at],p[at],threshold[name][at])['Sensitivity'],source95_target_achieved=bool(metrics(y[at],p[at],threshold[name][at])['Sensitivity']>=.95),
                clinical_safety_guarantee=False,endpoint='original strict locked parser; 108 unknown; hist_class discrepancy retained'))
        for i,row in enumerate(rows):private.append(dict(patient_alias=hmac.new(salt,str(row['key']).encode(),hashlib.sha256).hexdigest()[:24],center=g[i],CIN2=int(y[i]),CIN3=int(label3[i]),method=name,
            probability=p[i],anchor_probability=anchor[i],source_threshold=threshold[name][i],anchor_source_threshold=threshold['ANCHOR'][i],delta_logit=delta[i],
            seed17=seed_p[name][0,i],seed29=seed_p[name][1,i],seed43=seed_p[name][2,i]))
    frame=pd.DataFrame(main_rows);cf=pd.DataFrame(center_rows);sf=pd.DataFrame(safety);tf=pd.DataFrame(train_rows)
    frame.to_csv(HERE/'ARDR_MAIN_COMPARISON.csv',index=False);cf.to_csv(HERE/'ARDR_CENTER_RESULTS.csv',index=False);pd.DataFrame(seed_rows).to_csv(HERE/'ARDR_SEED_RESULTS.csv',index=False)
    pd.DataFrame(changes).to_csv(HERE/'ARDR_DECISION_CHANGE.csv',index=False);sf.to_csv(HERE/'ARDR_CIN3_SAFETY.csv',index=False)
    pd.DataFrame(private).to_parquet(HERE/'private/PATIENT_PREDICTIONS.parquet',index=False);tf.to_csv(HERE/'ARDR_SOURCE_CONSTRAINTS.csv',index=False);pd.DataFrame(optimism).to_csv(HERE/'ANCHOR_OPTIMISM_AUDIT.csv',index=False)
    risks=[]
    for c in centers:
        sub=cf[cf.center==c].set_index('method');ardr=sub.loc['A5_ARDR']
        risks.append(dict(center=c,**{'NLL_ARDR_minus_'+n:float(ardr.NLL-sub.loc[n].NLL) for n in methods[:-1]+['ANCHOR']},
            source_ARDR_mean_risk_delta=float(tf[(tf.outer==c)&(tf.method=='A5_ARDR')].source_risk_delta.mean()),
            source_ARDR_constraint_satisfied_fraction=float(tf[(tf.outer==c)&(tf.method=='A5_ARDR')].constraint_satisfied.mean())))
    pd.DataFrame(risks).to_csv(HERE/'ARDR_ANCHOR_RISK_DELTA.csv',index=False)
    # Paired centre-by-label patient bootstrap conditions on selected fitted
    # models; equal-centre NLL and pooled NLL are kept separate.
    strata=[np.flatnonzero((g==c)&(y==v)) for c in centers for v in [0,1]];rng=np.random.default_rng(cfg['bootstrap_seed'])
    comparison=methods[:-1]+['ANCHOR'];boot={(n,k):[] for n in comparison for k in ['AUROC','AUPRC','NLL','Brier','macro_center_NLL']}
    p5=probabilities['A5_ARDR']
    for _ in range(cfg['bootstrap_replicates']):
        at=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata if len(s)]);yy=y[at]
        five=metrics(yy,p5[at],threshold['A5_ARDR'][at]);five_macro=np.mean([metrics(y[at[g[at]==c]],p5[at[g[at]==c]],threshold['A5_ARDR'][at[g[at]==c]])['NLL'] for c in centers])
        for n in comparison:
            other=metrics(yy,probabilities[n][at],threshold[n][at])
            for k in ['AUROC','AUPRC','NLL','Brier']:boot[n,k].append(five[k]-other[k])
            macro=np.mean([metrics(y[at[g[at]==c]],probabilities[n][at[g[at]==c]],threshold[n][at[g[at]==c]])['NLL'] for c in centers]);boot[n,'macro_center_NLL'].append(five_macro-macro)
    stats=[];q=.05/(2*cfg['paired_family_size']);point=frame.set_index('method')
    for (n,k),values in boot.items():
        stats.append(dict(contrast='A5_ARDR_minus_'+n,metric=k,delta=float(point.loc['A5_ARDR',k]-point.loc[n,k]),
            CI_low=float(np.quantile(values,.025)),CI_high=float(np.quantile(values,.975)),Bonferroni_CI_low=float(np.quantile(values,q)),Bonferroni_CI_high=float(np.quantile(values,1-q)),
            replicates=cfg['bootstrap_replicates'],scope='conditional-on-fitted-models development; no history/selection/centre-population correction'))
    stat=pd.DataFrame(stats);stat.to_csv(HERE/'ARDR_PAIRED_BOOTSTRAP.csv',index=False)
    f=point.loc['A5_ARDR'];a0=point.loc['A0_PALE'];rules=cfg['final_admission'];checks=[]
    risk=pd.DataFrame(risks)
    satisfaction=float(tf[tf.method=='A5_ARDR'].constraint_satisfied.mean())
    checks.append(dict(item='Source constraint satisfaction',supported=satisfaction>=rules['source_constraint_satisfaction_fraction_min'],observed=satisfaction))
    checks.append(dict(item='Held-centre NLL vs PALE',supported=bool((risk.NLL_ARDR_minus_A0_PALE<0).sum()>=3 and f.worst_center_NLL<=a0.worst_center_NLL and stat[(stat.contrast=='A5_ARDR_minus_A0_PALE')&(stat.metric=='NLL')].CI_high.iloc[0]<0),observed=int((risk.NLL_ARDR_minus_A0_PALE<0).sum())))
    for n in methods[1:-1]:
        upper=stat[(stat.contrast=='A5_ARDR_minus_'+n)&(stat.metric=='NLL')].Bonferroni_CI_high.iloc[0]
        checks.append(dict(item='Independent value vs '+n,supported=bool(upper<0 and (risk['NLL_ARDR_minus_'+n]<0).sum()>=3),observed=float(upper)))
    checks.append(dict(item='Non-collapsed local increment',supported=bool(f.mean_absolute_logit_update>=max(.01,.1*a0.mean_absolute_logit_update) and f.nonzero_update_fraction>=.5),observed=float(f.mean_absolute_logit_update)))
    checks.append(dict(item='Discrimination retained vs repaired PALE',supported=bool(f.AUROC>=a0.AUROC-.01 and f.macro_AUROC>=a0.macro_AUROC-.01),observed=float(f.AUROC-a0.AUROC)))
    sp=sf[(sf.method=='A5_ARDR')&(sf.center=='POOLED')].iloc[0];bp=sf[(sf.method=='A0_PALE')&(sf.center=='POOLED')].iloc[0]
    extra=[int(sf[(sf.method=='A5_ARDR')&(sf.center==c)].missed.iloc[0]-sf[(sf.method=='A0_PALE')&(sf.center==c)].missed.iloc[0]) for c in centers]
    checks.append(dict(item='No extra known CIN3 misses vs repaired PALE',supported=bool(sp.missed<=bp.missed and max(extra)<=1),observed=int(sp.missed-bp.missed)))
    admitted=all(r['supported'] for r in checks);decision='ARDR纳入主方法' if admitted else 'ARDR不纳入主方法'
    save(HERE/'FINAL_VERDICT.json',dict(decision=decision,complete_outer_matrix=True,checks=checks,
        statuses={r['item']:'Supported' if r['supported'] else 'Not Supported' for r in checks},independent_confirmation=False,
        historical_point8074_scope='legacy reproduction, not the comparator after shared FR1 correction',further_architecture_search=False))
    table=pd.DataFrame(checks).to_csv(index=False)
    (HERE/'ARDR_FINAL_VERDICT.md').write_text('# Final source-relative risk admission\n\n'+decision+'。\n\n'+table+'\nAll six methods use common FR1 repair. Historical results are preserved separately. Full results are internal-external development, not independent confirmation or clinical safety. No more new methods will be added.\n')
    (HERE/'ARDR_MECHANISM_ANALYSIS.md').write_text('# ARDR mechanism evidence\n\n'+pd.DataFrame(risks).to_csv(index=False)+'\n'+table+'\nTrain constraints are soft and use in-sample frozen training anchors; source validation anchors are recipient-excluded. Anchor optimism is separately audited. Unknown CIN3 outcomes are retained. Paired effects include all centres and seeds.\n')
    if admitted:
        (HERE/'METHOD_ARDR.tex').write_text('\\section{Anchor-Relative Decision Risk Optimization}\nWe retain the frozen JEV anchor, local residual MLP and signed readout. The additional source-only soft penalty is $\\gamma C^{-1}\\sum_c[\\max(0,R_c(z)-R_c(a)-\\epsilon)]^2$. Anchors are detached. This soft empirical constraint does not guarantee target-centre or clinical safety. Shared source-only FR1 preprocessing and all comparison histories are disclosed.\n')
        (HERE/'EXPERIMENTS_ARDR.tex').write_text('\\section{Experiments}\nFive-centre development involved984patients and303CIN2+ cases, with seeds17/29/43 and3000paired patient resamples.\n'+frame[['method','AUROC','AUPRC','NLL','Brier']].to_latex(index=False))
        (HERE/'HIGHLIGHTS_ARDR.txt').write_text('Source-centre empirical risk is assessed relative to a frozen patient anchor.\nOne unchanged evidence architecture isolates the optimization comparison.\nRecipient-excluded source validation and direct risk-regularization controls test added value.\n')
    save(HERE/'STATISTICS_COMPLETE.json',dict(state='COMPLETE',methods=methods,seeds=cfg['seeds'],N=N,decision=decision,all_held_centres=True,training_changed=False))
    print(frame[['method','AUROC','AUPRC','NLL','Brier','macro_AUROC']].to_string(index=False));print(decision,flush=True)
if __name__=='__main__':main()
