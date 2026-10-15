"""Real patient-paired readouts. Nothing here chooses a model or threshold."""
import runtime
from runtime import HERE,PREVIOUS,config,save,sha
import argparse,json,time,hmac,hashlib,os
import numpy as np,pandas as pd,torch
from scipy.special import expit,logit
from scipy.stats import rankdata,beta
from sklearn.metrics import roc_auc_score,average_precision_score
from data import load_patients,prepare,inputs,source_scale
from heads import make_head

def auc(y,p):
    positive=y==1;n=positive.sum();return float((rankdata(p)[positive].sum()-n*(n+1)/2)/(n*(len(y)-n)))
def ap(y,p):
    order=np.argsort(-p,kind='stable');yy=y[order];ss=p[order];at=np.r_[np.flatnonzero(np.diff(ss)),len(y)-1]
    tp=np.cumsum(yy)[at];new=np.diff(np.r_[0,tp]);return float((new*tp/(at+1)).sum()/yy.sum())
def metrics(y,p,t):
    z=logit(np.clip(p,1e-12,1-1e-12));d=z>=t
    return dict(AUROC=auc(y,p),AUPRC=ap(y,p),NLL=float(np.mean(np.logaddexp(0,z)-y*z)),Brier=float(np.mean((p-y)**2)),Sensitivity=float(d[y==1].mean()),Specificity=float((~d[y==0]).mean()))

def bootstrap(y,g,prob,threshold,contrasts):
    cfg=config();rng=np.random.default_rng(cfg['bootstrap_seed']);blocks=[np.flatnonzero((g==c)&(y==label)) for c in cfg['centers'] for label in [0,1]]
    draws={name:[] for name in contrasts};names=sorted(set(n for pair in contrasts.values() for n in pair))
    for _ in range(cfg['bootstrap_replicates']):
        at=np.concatenate([rng.choice(b,len(b),replace=True) for b in blocks]);yy=y[at];gg=g[at]
        value={}
        for name in names:
            pp=prob[name][at];m=metrics(yy,pp,threshold[name][at]);m['macro_AUROC']=float(np.mean([auc(yy[gg==c],pp[gg==c]) for c in cfg['centers']]))
            value[name]=m
        for name,(a,b) in contrasts.items():draws[name].append([value[a][k]-value[b][k] for k in ['AUROC','AUPRC','NLL','Brier','macro_AUROC']])
    records=[];alpha=.05/max(1,len(contrasts))
    for name,(a,b) in contrasts.items():
        aa=metrics(y,prob[a],threshold[a]);bb=metrics(y,prob[b],threshold[b])
        aa['macro_AUROC']=float(np.mean([auc(y[g==c],prob[a][g==c]) for c in cfg['centers']]))
        bb['macro_AUROC']=float(np.mean([auc(y[g==c],prob[b][g==c]) for c in cfg['centers']]))
        draw=np.array(draws[name])
        for ki,key in enumerate(['AUROC','AUPRC','NLL','Brier','macro_AUROC']):
            lo,hi=np.quantile(draw[:,ki],[.025,.975]);adjlo,adjhi=np.quantile(draw[:,ki],[alpha/2,1-alpha/2])
            records.append(dict(contrast=name,metric=key,delta=aa[key]-bb[key],CI_low=lo,CI_high=hi,adjusted_CI_low=adjlo,adjusted_CI_high=adjhi,replicates=cfg['bootstrap_replicates'],family_size=len(contrasts),conditioning='fixed fitted models, centre x label patient resampling; not seed or new-centre uncertainty'))
    return records

def main(phase):
    torch.set_num_threads(1);cfg=config();rows=load_patients();y=np.array([r['y'] for r in rows]);g=np.array([r['center'] for r in rows]);cin3=np.array([-1 if r['cin3'] is None else r['cin3'] for r in rows]);N=len(rows)
    methods=[n for n in cfg['variants'] if all((HERE/'private/runs'/c/(n+'.npz')).exists() for c in cfg['centers'])]
    assert all(m in methods for m in cfg['core'])
    seed_p={m:np.full((3,N),np.nan) for m in methods+['P0_ANCHOR']};threshold={m:np.full(N,np.nan) for m in seed_p}
    compute=[];epochzero=[]
    for c in cfg['centers']:
        te=np.flatnonzero(g==c);tr=np.flatnonzero(g!=c);folder=HERE/'private/runs'/c
        for method in methods:
            with np.load(folder/(method+'.npz')) as f:
                assert np.array_equal(f['test'],te) and set(f['source_indices'])==set(tr)
                assert np.array_equal(f['source_y'],y[f['source_indices']])
                seed_p[method][:,te]=expit(f['seed_logits']);seed_p['P0_ANCHOR'][:,te]=expit(f['anchor_logits'])
                source=logit(np.clip(expit(f['source_seed_logits']).mean(0),1e-12,1-1e-12))
                threshold[method][te]=np.quantile(source[f['source_y']==1],.05,method='lower')
                threshold['P0_ANCHOR'][te]=np.quantile(f['source_anchor_logits'][f['source_y']==1],.05,method='lower')
            diag=json.loads((folder/(method+'_REFIT_DIAGNOSTICS.json')).read_text())
            sourcefits=list((HERE/'private/cases'/c).glob(method+'_*.pt')) if (HERE/'private/cases'/c).exists() else []
            source_seconds=0.;source_meta=[]
            for path in sourcefits:
                ck=torch.load(path,map_location='cpu',weights_only=False);md=ck['metadata'];source_seconds+=md['fit_seconds'];source_meta.append(md)
                epochzero.append(dict(outer=c,method=method,seed=md['seed'],best_epoch=md['best_epoch'],epochs_run=md['epochs_run'],epoch_zero=md['best_epoch']==0))
            if method=='P3_PALE':
                # Measured existing selection cost; reuse is not cost-free modeling.
                for path in (PREVIOUS/'private/cases'/c).glob('A0_PALE_*.pt'):
                    md=torch.load(path,map_location='cpu',weights_only=False)['metadata'];source_seconds+=md['fit_seconds']
            part=prepare(c,'outer');scale=source_scale(part,cfg['variants'][method]);inp=inputs(part,te,'cpu',scale)
            ck=torch.load(folder/f'{method}_seed17_OUTER.pt',map_location='cpu',weights_only=False)
            model=make_head(inp['b'].shape[-1],cfg['variants'][method]).double();model.load_state_dict(ck['state']);model.eval()
            with torch.no_grad():
                for _ in range(5):model(**inp)
                start=time.perf_counter()
                for _ in range(20):model(**inp)
                elapsed=(time.perf_counter()-start)/20
            measured_peaks=[md['peak_cuda_bytes'] for md in source_meta+diag if 'peak_cuda_bytes' in md]
            compute.append(dict(outer=c,method=method,trainable_parameters=diag[0]['parameters'],allocated_parameters=diag[0]['allocated_parameters'],source_selection_fit_seconds=source_seconds,refit_seconds=sum(md['fit_seconds'] for md in diag),source_fits=len(sourcefits),head_CPU_batch_inference_seconds=elapsed,head_CPU_ms_per_patient=1000*elapsed/len(te),inference_patients=len(te),peak_training_cuda_bytes=max(measured_peaks) if measured_peaks else None,reused=method=='P3_PALE',cost_scope='Measured local-head/source search only; missing historical CUDA peak stays unknown, not zero; frozen upstream extraction cost separate'))
    assert all(np.isfinite(p).all() for p in seed_p.values());prob={m:p.mean(0) for m,p in seed_p.items()}
    mainrows=[];centres=[];seeds=[];safety=[];pairs=[]
    saltpath=HERE/'private/PATIENT_ALIAS_SALT.key'
    if not saltpath.exists():saltpath.write_bytes(os.urandom(32))
    salt=saltpath.read_bytes()
    for method,p in prob.items():
        cc=[]
        for c in cfg['centers']:
            at=g==c;m=metrics(y[at],p[at],threshold[method][at]);cc.append(m)
            centres.append(dict(method=method,center=c,N=int(at.sum()),positive=int(y[at].sum()),prevalence=float(y[at].mean()),**m))
        mainrows.append(dict(method=method,N=N,positive=int(y.sum()),prevalence=float(y.mean()),**metrics(y,p,threshold[method]),macro_AUROC=float(np.mean([m['AUROC'] for m in cc])),macro_NLL=float(np.mean([m['NLL'] for m in cc])),worst_center_AUROC=float(min(m['AUROC'] for m in cc)),worst_center_NLL=float(max(m['NLL'] for m in cc)),protocol='FR1/NLL-first, frozen PALE development comparison',independent_confirmation=False))
        for seed,pp in zip(cfg['seeds'],seed_p[method]):
            seeds.append(dict(method=method,seed=seed,**metrics(y,pp,threshold[method]),macro_AUROC=float(np.mean([auc(y[g==c],pp[g==c]) for c in cfg['centers']])),threshold_scope='Same ensemble-derived legal source threshold; seeds assessed separately'))
        z=logit(np.clip(p,1e-12,1-1e-12));a=logit(np.clip(prob['P0_ANCHOR'],1e-12,1-1e-12))
        gain=np.logaddexp(0,a)-y*a-(np.logaddexp(0,z)-y*z)
        for i,row in enumerate(rows):pairs.append(dict(patient_alias=hmac.new(salt,str(row['key']).encode(),hashlib.sha256).hexdigest()[:24],center=g[i],CIN2=int(y[i]),CIN3=int(cin3[i]),method=method,probability=float(p[i]),anchor_probability=float(prob['P0_ANCHOR'][i]),delta=float(z[i]-a[i]),loss_gain=float(gain[i])))
        for c in ['POOLED']+cfg['centers']:
            at=np.ones(N,bool) if c=='POOLED' else g==c;d=z[at]>=threshold[method][at];labels=cin3[at];n=int((labels==1).sum());hit=int(((labels==1)&d).sum())
            safety.append(dict(method=method,center=c,known=int((labels>=0).sum()),unknown=int((labels<0).sum()),positive=n,detected=hit,missed=n-hit,sensitivity=hit/n if n else None,CI_low=float(beta.ppf(.025,hit,n-hit+1)) if hit else 0.,CI_high=float(beta.ppf(.975,hit+1,n-hit)) if hit<n else 1.))
    for name,records in [('MAIN_RESULTS',mainrows),('CENTER_RESULTS',centres),('SEED_RESULTS',seeds),('CIN3_SAFETY',safety),('COMPUTE_COST',compute),('EPOCH_ZERO',epochzero)]:pd.DataFrame(records).to_csv(HERE/'tables'/('TABLE_'+name+'.csv'),index=False)
    pd.DataFrame(pairs).to_parquet(HERE/'private/PATIENT_PAIRED_PREDICTIONS.parquet',index=False)
    contrasts={name:tuple(name.split('-')) for name in cfg['primary_contrasts']}
    boot=bootstrap(y,g,prob,threshold,contrasts)
    if phase=='all':
        secondary={f'P3_PALE-{n}':('P3_PALE',n) for n in methods if n not in cfg['core']}
        if secondary:boot+=bootstrap(y,g,prob,threshold,secondary)
        contrasts.update(secondary)
    pd.DataFrame(boot).to_csv(HERE/'tables/TABLE_PAIRED_CONTRASTS.csv',index=False)
    claims=[]
    for contrast,(a,b) in contrasts.items():
        au=next(r for r in boot if r['contrast']==contrast and r['metric']=='AUROC');nl=next(r for r in boot if r['contrast']==contrast and r['metric']=='NLL')
        positive=sum(auc(y[g==c],prob[a][g==c])>auc(y[g==c],prob[b][g==c]) for c in cfg['centers'])
        support='SUPPORTED_WITHIN_DEVELOPMENT' if au['adjusted_CI_low']>0 and positive>=3 else 'NOT_SUPPORTED' if au['adjusted_CI_high']<=0 else 'INCONCLUSIVE'
        claims.append(dict(contrast=contrast,status=support,AUROC_delta=au['delta'],AUROC_adjusted_CI=[au['adjusted_CI_low'],au['adjusted_CI_high']],NLL_delta=nl['delta'],positive_centres=positive,criterion='predeclared adjusted AUROC CI and >=3 positive centres; NLL and macro AUC separately reported'))
    save(HERE/'RQ_VERDICTS.json',dict(phase=phase,claims=claims,negative_results_retained=True,independent_confirmation=False))
    pd.DataFrame(mainrows).to_csv(HERE/'tables/TABLE_3_CORE_AND_ORDER.csv',index=False)
    pd.DataFrame([r for r in mainrows if r['method'] in ['P0_ANCHOR','P2_MLP_MOMENT','P3_PALE','C1_MLP_MEAN','C2_MLP_MEAN_STD','C4_MLP_LSE56','C5_MLP_LEARNED56','C7_SINGLE_SCALE','C8_MAGNITUDE']]).to_csv(HERE/'tables/TABLE_4_READOUTS.csv',index=False)
    save(HERE/(phase.upper()+'_REPORT_COMPLETE.json'),dict(state='COMPLETE',methods=list(prob),patients=N,replicates=cfg['bootstrap_replicates'],phase=phase,time=time.time()))
    print('Real paired report complete:',phase,len(prob),'methods',flush=True)
    if (HERE/'figures.py').exists():
        import figures;figures.main()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['core','all'],required=True);a=p.parse_args()
    rng=np.random.default_rng(17);yy=np.tile([0,1],10);pp=np.round(rng.random(20),1)
    assert abs(auc(yy,pp)-roc_auc_score(yy,pp))<1e-12 and abs(ap(yy,pp)-average_precision_score(yy,pp))<1e-12
    main(a.phase)
