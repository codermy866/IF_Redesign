"""Finite source-only candidate selection and one frozen outer refit per arm."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('OMP_NUM_THREADS','4')
import runtime
from runtime import HERE,save,config,verify,sha
import argparse,json,time,hashlib
import numpy as np,pandas as pd,torch,joblib
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from data import prepare,inputs
from training import fit
from rime_dataset import load_patients

def case_key(method,candidate,lam,seed,held):
    return method+'_'+hashlib.sha256(json.dumps(dict(candidate=candidate,lam=lam,seed=seed,held=held),sort_keys=True).encode()).hexdigest()[:20]

def fit_case(outer,held,method,candidate,lam,seed,device,rows,y,g):
    part=prepare(outer,held);tr=part['train'];va=np.flatnonzero(g==held)
    assert not np.isin(tr,np.flatnonzero((g==held)|(g==outer))).any()
    key=case_key(method,candidate,lam,seed,held);path=HERE/'private/cases'/outer/(key+'.pt')
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert np.array_equal(saved['train'],tr) and np.array_equal(saved['validation'],va)
        return saved
    ti=inputs(part,tr,device);vi=inputs(part,va,device)
    head,md=fit(ti,vi,y[tr],g[tr],y[va],lam,seed,method,candidate)
    with torch.no_grad():prediction=head(**vi).cpu().numpy()
    saved=dict(state={k:v.detach().cpu() for k,v in head.state_dict().items()},metadata=md,prediction=prediction,
        anchor=part['anchor']['anchor'][va],train=tr,validation=va,config_sha256=sha(HERE/'CONFIG.json'),preprocessing='FR1')
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(saved,path)
    save(HERE/f'{outer}_PROGRESS.json',dict(state='SOURCE_CV',method=method,candidate=candidate,lam=lam,seed=seed,held=held,pid=os.getpid(),time=time.time()))
    return saved

def evaluate_candidate(outer,helds,method,candidate,lam,device,rows,y,g):
    source=np.flatnonzero(g!=outer);oof=np.full((3,len(rows)),np.nan);anchor=np.full(len(rows),np.nan);epochs=[];fits=[]
    at=np.concatenate([np.flatnonzero(g==h) for h in helds])
    for si,seed in enumerate(config()['seeds']):
        for held in helds:
            saved=fit_case(outer,held,method,candidate,lam,seed,device,rows,y,g);va=saved['validation'];oof[si,va]=saved['prediction'];anchor[va]=saved['anchor']
            epochs.append(saved['metadata']['best_epoch']);fits.append(saved['metadata'])
    prob=expit(oof[:,at]).mean(0);z=np.log(np.clip(prob,1e-12,None)/np.clip(1-prob,1e-12,None));yy=y[at]
    nll=[];brier=[];centers=[]
    for held in helds:
        mask=g[at]==held;nll.append(float(np.mean(np.logaddexp(0,z[mask])-yy[mask]*z[mask])));brier.append(float(np.mean((prob[mask]-yy[mask])**2)))
        centers.append(dict(held_source=held,NLL=nll[-1],Brier=brier[-1],anchor_NLL=float(np.mean(np.logaddexp(0,anchor[at][mask])-yy[mask]*anchor[at][mask]))))
    residual=oof[:,at]-anchor[at][None,:]
    return dict(method=method,candidate=candidate,lambda_=lam,mean_source_center_NLL=float(np.mean(nll)),mean_source_center_Brier=float(np.mean(brier)),
        source_AUROC=float(roc_auc_score(yy,prob)),source_AUPRC=float(__import__('sklearn.metrics',fromlist=['average_precision_score']).average_precision_score(yy,prob)),
        source_anchor_AUPRC=float(__import__('sklearn.metrics',fromlist=['average_precision_score']).average_precision_score(yy,expit(anchor[at]))),
        source_indices=at,source_seed_logits=oof[:,at],source_anchor_logits=anchor[at],fixed_refit_epochs=int(np.median(epochs)),
        mean_absolute_logit_update=float(abs(residual).mean()),nonzero_update_fraction=float((abs(residual)>1e-4).mean()),
        source_constraint_satisfaction=float(np.mean([m['constraint_satisfaction_fraction'] for m in fits])),
        fit_seconds=float(sum(m['fit_seconds'] for m in fits)),source_centres=centers,source_only=True)

def best_of(candidates):
    return min(candidates,key=lambda c:(c['mean_source_center_NLL'],c['mean_source_center_Brier'],-c['source_AUROC'],c['fixed_refit_epochs']))

def public_candidate(item):return {k:v for k,v in item.items() if k not in ['source_seed_logits','source_anchor_logits','source_indices']}

def run(outer,device,pilot=False):
    torch.set_num_threads(4);verify();rows=load_patients();y=np.array([r['y'] for r in rows]);g=np.array([r['center'] for r in rows]);cfg=config()
    helds=[cfg['pilot_held_source']] if pilot else sorted(set(g[g!=outer]));folder=HERE/'private'/('pilot' if pilot else 'runs')/outer;folder.mkdir(parents=True,exist_ok=True)
    base=[evaluate_candidate(outer,helds,'A0_PALE',{},lam,device,rows,y,g) for lam in cfg['base_lambdas']]
    selected=best_of(base);lam=selected['lambda_'];choices={'A0_PALE':selected}
    save(folder/'A0_PALE_SOURCE_CONFIGURATIONS.json',[public_candidate(c) for c in base])
    for method in cfg['methods'][1:]:
        candidates=[evaluate_candidate(outer,helds,method,candidate,lam,device,rows,y,g) for candidate in cfg['candidates'][method]]
        choices[method]=best_of(candidates);save(folder/(method+'_SOURCE_CONFIGURATIONS.json'),[public_candidate(c) for c in candidates])
    for method,selected in choices.items():save(folder/(method+'_SELECTION_LOCK.json'),public_candidate(selected)|dict(seeds=cfg['seeds'],selection_rule=cfg['selection']))
    if pilot:
        results=[]
        for method in cfg['methods']:
            candidates=json.loads((folder/(method+'_SOURCE_CONFIGURATIONS.json')).read_text())
            results.extend(dict(outer=outer,method=method,candidate=json.dumps(c['candidate'],sort_keys=True),lambda_=c['lambda_'],
                source_NLL=c['mean_source_center_NLL'],source_Brier=c['mean_source_center_Brier'],source_AUROC=c['source_AUROC'],source_AUPRC=c['source_AUPRC'],
                mean_abs_delta=c['mean_absolute_logit_update'],nonzero_fraction=c['nonzero_update_fraction'],source_constraint_satisfaction=c['source_constraint_satisfaction'],
                fit_seconds=c['fit_seconds'],chosen=c['candidate']==choices[method]['candidate'] and c['lambda_']==choices[method]['lambda_'],no_outer_results=True) for c in candidates)
        pd.DataFrame(results).to_csv(HERE/'ARDR_PILOT_RESULTS.csv',index=False)
        a0=choices['A0_PALE'];ardr=json.loads((folder/'A5_ARDR_SOURCE_CONFIGURATIONS.json').read_text())
        collapsed=lambda c:c['mean_absolute_logit_update']<=max(.01,.1*a0['mean_absolute_logit_update'])
        all_collapsed=all(collapsed(c) for c in ardr)
        gain=any(c['mean_source_center_NLL']<a0['mean_source_center_NLL']-.001 and c['source_AUPRC']>c['source_anchor_AUPRC'] for c in ardr)
        go=not(all_collapsed and not gain)
        save(HERE/'PILOT_GATE.json',dict(state='PASS_FULL_FIXED_MATRIX' if go else 'STOP_ANCHOR_COLLAPSE_WITHOUT_SOURCE_BENEFIT',
             all_ARDR_candidates_collapsed=all_collapsed,source_risk_and_discrimination_gain_present=gain,outer_labels_used=False,
             source_only=True,configuration_search_not_expanded=True,full_matrix_allowed=go,
             throughput_seconds=sum(r['fit_seconds'] for r in results),fit_cases=sum(len(cfg['candidates'].get(m,base))*3 for m in cfg['methods'])))
        print('PILOT_COMPLETE',go,flush=True);return
    tr=np.flatnonzero(g!=outer);te=np.flatnonzero(g==outer);part=prepare(outer,'outer');ti=inputs(part,tr,device);vi=inputs(part,te,device)
    assert np.array_equal(part['train'],tr)
    for method,selected in choices.items():
        result=folder/(method+'.npz')
        if result.exists():continue
        z=[];refit=[]
        for seed in cfg['seeds']:
            path=folder/f'{method}_seed{seed}_OUTER.pt'
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=False)
                from if_redesign.rime.readout import RIMEHead
                head=RIMEHead(ti['b'].shape[1],.3,'mlp','laplace').to(ti['h']);head.load_state_dict(saved['state']);head.eval();md=saved['metadata']
                assert np.array_equal(saved['train'],tr)
            else:
                head,md=fit(ti,None,y[tr],g[tr],None,selected['lambda_'],seed,method,selected['candidate'],fixed_epochs=selected['fixed_refit_epochs'])
                torch.save(dict(state={k:v.detach().cpu() for k,v in head.state_dict().items()},metadata=md,train=tr,source_selection_frozen=True),path)
            with torch.no_grad():z.append(head(**vi).cpu().numpy())
            refit.append(md)
        assert np.array_equal(selected['source_indices'],np.concatenate([np.flatnonzero(g==c) for c in helds]))
        np.savez_compressed(result,test=te,seed_logits=np.stack(z),anchor_logits=part['anchor']['anchor'][te],
            source_indices=selected['source_indices'],source_seed_logits=selected['source_seed_logits'],source_y=y[selected['source_indices']],
            source_g=g[selected['source_indices']],source_anchor_logits=selected['source_anchor_logits'])
        save(folder/(method+'_REFIT_DIAGNOSTICS.json'),refit)
        print('OUTER_SAVED',outer,method,flush=True)
    # Same-source optimism audit: full source anchor in-sample versus legal
    # leave-source-centre-out predictions, both on the exact same patients.
    at=choices['A0_PALE']['source_indices'];oof=choices['A0_PALE']['source_anchor_logits'];ins=part['anchor']['anchor'][at]
    record=[]
    for c in helds:
        mask=g[at]==c;yy=y[at][mask]
        record.append(dict(center=c,N=int(mask.sum()),in_sample_anchor_NLL=float(np.mean(np.logaddexp(0,ins[mask])-yy*ins[mask])),
            out_of_fold_anchor_NLL=float(np.mean(np.logaddexp(0,oof[mask])-yy*oof[mask])),
            scope='training-anchor optimism; crossfit validation only, no target guarantee'))
    save(folder/'ANCHOR_OPTIMISM_AUDIT.json',record)
    save(folder/'COMPLETE.json',dict(state='COMPLETE',outer=outer,methods=cfg['methods'],seeds=cfg['seeds'],preprocessing='FR1'))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--outer',required=True);parser.add_argument('--device',default='cuda:0');parser.add_argument('--pilot',action='store_true');a=parser.parse_args()
    run(a.outer,a.device,a.pilot)
