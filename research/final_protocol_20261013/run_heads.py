"""Fixed core followed by one-axis, source-only controls."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import argparse,json,time
import runtime
from runtime import HERE,save,verify,config
import numpy as np,torch,joblib
from rime_dataset import load_patients,prepare_fold
from factorial_inputs import tensor_inputs,shuffled_scoring_context
import core_heads
from train_control import train_control

def prob_logits(values):
    from scipy.special import expit,logit
    return logit(np.clip(expit(values).mean(0),1e-12,1-1e-12))

def run_control(outer,phase,device):
    verify();torch.set_num_threads(4);cfg=config();rows=load_patients()
    y=np.asarray([r['y'] for r in rows]);g=np.asarray([r['center'] for r in rows])
    tr=np.flatnonzero(g!=outer);test=np.flatnonzero(g==outer)
    directory=HERE/'private/head_runs'/outer
    parts={p.stem:joblib.load(p) for p in (HERE/'private/fold_features'/outer).glob('*.joblib')}
    lock=json.loads((directory/'D_RIME_LAPLACE_SOURCE_SELECTION_LOCK.json').read_text())
    tau,lam=lock['selected_tau'],lock['selected_lambda']
    if phase=='B1':arms={'B1_SOURCE_SELECTED_WARMUP':[(w,'conditional') for w in cfg['warmup_grid']]}
    elif phase=='B2':arms={f'B2_{mode.upper()}':[(0,mode)] for mode in cfg['B2_variants'] if mode!='conditional'}
    else:raise ValueError(phase)
    for name,candidates in arms.items():
        if (directory/(name+'.npz')).exists():continue
        choices=[]
        for warmup,mode in candidates:
            oofs=[];epochs=[]
            for seed in cfg['seeds']:
                oof=np.full(len(rows),np.nan)
                for held in sorted(set(g[tr])):
                    fit=tr[g[tr]!=held];val=tr[g[tr]==held]
                    baseline=directory/f'D_RIME_LAPLACE_tau{tau}_lam{lam}_seed{seed}_inner{held}.pt'
                    file=baseline if warmup==0 and mode=='conditional' else directory/f'{name}_warmup{warmup}_seed{seed}_inner{held}.pt'
                    if file.exists():saved=torch.load(file,map_location='cpu',weights_only=False)
                    else:
                        assert np.array_equal(parts[held]['train'],fit)
                        ti=tensor_inputs(parts[held],fit,'D_RIME_LAPLACE',tau,device,rows)
                        vi=tensor_inputs(parts[held],val,'D_RIME_LAPLACE',tau,device,rows)
                        if mode=='context_shuffled':
                            ti,vi,audit=shuffled_scoring_context(ti,vi,seed)
                            save(directory/f'{name}_seed{seed}_inner{held}_CONTEXT_SHUFFLE.json',audit)
                        head,md=train_control(ti,vi,y,g,tau,lam,seed,mode=mode,warmup=warmup,
                            max_epochs=cfg['max_epochs'],patience=cfg['patience'])
                        with torch.no_grad():z=head(**{k:v for k,v in vi.items() if k!='indices'}).cpu().numpy()
                        saved=dict(state={k:v.cpu() for k,v in head.state_dict().items()},metadata=md,
                            prediction=z,train=fit,validation=val)
                        torch.save(saved,file)
                    assert np.array_equal(saved['train'],fit) and np.array_equal(saved['validation'],val)
                    oof[val]=saved['prediction'];epochs.append(saved['metadata']['best_epoch'])
                    save(HERE/f'{phase}_{outer}_STATE.json',dict(method=name,warmup=warmup,score_mode=mode,seed=seed,held=held,source_only=True))
                oofs.append(oof)
            oofs=np.stack(oofs);z=prob_logits(oofs)
            from sklearn.metrics import roc_auc_score
            choices.append(dict(warmup=warmup,score_mode=mode,source_AUROC=roc_auc_score(y[tr],z[tr]),
                source_NLL=float(np.mean(np.logaddexp(0,z[tr])-y[tr]*z[tr])),fixed_refit_epochs=int(np.median(epochs)),source_seed_logits=oofs))
        save(directory/(name+'_ALL_SOURCE_CONFIGURATIONS.json'),[
            {**c,'source_indices':tr,'source_seed_logits':c['source_seed_logits'][:,tr]} for c in choices])
        best=max(choices,key=lambda c:(c['source_AUROC'],-c['source_NLL'],-c['warmup']))
        save(directory/(name+'_SOURCE_SELECTION_LOCK.json'),{k:v for k,v in best.items() if k!='source_seed_logits'}|dict(
            selected_tau=tau,selected_lambda=lam,effective_search_configurations=len(choices),tau_lambda_from_core_source_only=True))
        ti=tensor_inputs(parts['outer'],tr,'D_RIME_LAPLACE',tau,device,rows)
        vi=tensor_inputs(parts['outer'],test,'D_RIME_LAPLACE',tau,device,rows)
        logits=[]
        for seed in cfg['seeds']:
            seed_ti,seed_vi=ti,vi
            if best['score_mode']=='context_shuffled':
                seed_ti,seed_vi,audit=shuffled_scoring_context(ti,vi,seed)
                save(directory/f'{name}_seed{seed}_OUTER_CONTEXT_SHUFFLE.json',audit)
            if best['warmup']==0 and best['score_mode']=='conditional':
                file=directory/f'D_RIME_LAPLACE_seed{seed}_OUTER.pt'
                saved=torch.load(file,map_location='cpu',weights_only=False)
                head=core_heads.restore(saved,ti['b'].shape[-1],device)
            else:
                file=directory/f'{name}_seed{seed}_OUTER.pt'
                # No test labels in fitting or in the refit epoch count.
                head,md=train_control(seed_ti,None,y,g,tau,lam,seed,mode=best['score_mode'],warmup=best['warmup'],
                    fixed_epochs=best['fixed_refit_epochs'])
                torch.save(dict(state={k:v.cpu() for k,v in head.state_dict().items()},metadata=md,train=tr,
                    source_selection_frozen=True),file)
            with torch.no_grad():logits.append(head(**{k:v for k,v in seed_vi.items() if k!='indices'}).cpu().numpy())
        np.savez_compressed(directory/(name+'.npz'),test=test,seed_logits=np.stack(logits),source_indices=tr,
            source_seed_logits=best['source_seed_logits'][:,tr],source_y=y[tr],source_g=g[tr])
    save(directory/(phase+'_COMPLETE.json'),dict(state='COMPLETE',outer=outer,methods=list(arms)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True);p.add_argument('--phase',choices=['core','controls','B1','B2'],required=True)
    p.add_argument('--device',default='cpu');a=p.parse_args();verify()
    if a.phase=='core':core_heads.fit_fold(a.fold,a.device)
    elif a.phase=='controls':core_heads.fit_fold(a.fold,a.device,methods={'ATTENTION_DIVERSITY_LAPLACE':('attention','laplace'),'MEASURE_LSE56':('measure','lse')})
    else:run_control(a.fold,a.phase,a.device)
