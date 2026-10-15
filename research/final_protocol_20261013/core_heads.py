"""Finite source-only CV, three seeds, fixed outer evaluation and full negatives."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4';os.umask(0o077)
import sys,argparse,json,itertools,copy,time
import numpy as np
import torch
import pandas as pd
from rime_common import *
sys.path.insert(0,str(Path(__file__).resolve().parent/'code'))
from rime_dataset import load_patients,prepare_fold
from factorial_inputs import tensor_inputs
from if_redesign.rime.training import train_head
from if_redesign.rime.readout import RIMEHead
sys.path.insert(0,str(OLD));import core;import paired_statistics as stats
import joblib

METHODS={'A_ATTENTION_MOMENT':('attention','moment'),'B_ATTENTION_LAPLACE':('attention','laplace'),'C_MEASURE_MOMENT':('measure','moment'),'D_RIME_LAPLACE':('measure','laplace')}

def restore(checkpoint,context_dim,device):
    md=checkpoint['metadata'];head=RIMEHead(context_dim,md['tau'],md['pooling'],md['readout']).to(device=device,dtype=torch.float64)
    if md['pooling']=='caprmil':
        import importlib.util
        spec=importlib.util.spec_from_file_location('caprmil_author',DEST/'author_reference/caprmil.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        head.caprmil=mod.TransolverBlock(dim=8,num_heads=2,head_dim=4,cluster_num=4,mlp_ratio=2,dropout=0.,last_layer=False).to(device=device,dtype=torch.float64)
    head.load_state_dict(checkpoint['state']);head.eval();return head
def fit_fold(outer,device,methods=None):
    torch.set_num_threads(4)
    cfg=json.loads((DEST/'CONFIG.json').read_text());rows=load_patients()
    y=np.array([r['y'] for r in rows]);g=np.array([r['center'] for r in rows]);train=np.flatnonzero(g!=outer);test=np.flatnonzero(g==outer)
    prepare_fold(outer);partsdir=DEST/'private/fold_features'/outer
    parts={p.stem:joblib.load(p) for p in partsdir.glob('*.joblib')}
    dest=DEST/'private/head_runs'/outer;dest.mkdir(parents=True,exist_ok=True)
    predictions=[];selection=[]
    for name,(pooling,readout) in (methods or METHODS).items():
        resultpath=dest/(name+'.npz')
        if resultpath.exists():continue
        candidates=[];input_cache={}
        # Identical nine-configuration limits; for fixed poolers tau is inactive
        # and exact duplicate settings are not run or represented as extra search.
        taus=cfg['taus'] if pooling not in ['mean','mlp','caprmil','geometry','pca_refit'] or name=='B3_CLUSTER_LAPLACE' else [.3]
        for tau,lam in itertools.product(taus,cfg['lambdas']):
            seed_oof=[];epochs=[];seed_records=[]
            for seed in cfg['seeds']:
                oof=np.full(len(rows),np.nan)
                for held in sorted(set(g[train])):
                    tr=train[g[train]!=held];va=train[g[train]==held];part=parts[held]
                    assert np.array_equal(part['train'],tr)
                    file=dest/f'{name}_tau{tau}_lam{lam}_seed{seed}_inner{held}.pt'
                    if file.exists():
                        saved=torch.load(file,map_location='cpu',weights_only=False);oof[va]=saved['prediction'];epochs.append(saved['metadata']['best_epoch']);continue
                    cache_key=(held,tau)
                    if cache_key not in input_cache:
                        input_cache[cache_key]=(tensor_inputs(part,tr,name,tau,device,rows),tensor_inputs(part,va,name,tau,device,rows))
                    ti,vi=input_cache[cache_key]
                    started=time.monotonic();head,md=train_head(ti,vi,y,g,pooling,readout,tau,lam,seed,max_epochs=cfg['max_epochs'],patience=cfg['patience'],attention_diversity_weight=cfg['attention_diversity_weight'] if name=='ATTENTION_DIVERSITY_LAPLACE' else 0.)
                    md['fit_seconds']=time.monotonic()-started
                    with torch.no_grad():z=head(**{k:v for k,v in vi.items() if k!='indices'}).cpu().numpy()
                    torch.save(dict(state={k:v.cpu() for k,v in head.state_dict().items()},metadata=md,prediction=z,train=tr,validation=va),file)
                    oof[va]=z;epochs.append(md['best_epoch']);seed_records.append(md)
                    save(DEST/'HEAD_STATE.json',dict(state='SOURCE_CV',outer=outer,method=name,seed=seed,tau=tau,lambda_=lam,held_source=held))
                seed_oof.append(oof)
            probability=np.mean(1/(1+np.exp(-np.stack(seed_oof))),axis=0)
            mean=np.log(np.clip(probability,1e-12,1-1e-12)/np.clip(1-probability,1e-12,1-1e-12))
            candidates.append(dict(tau=tau,lambda_=lam,source_oof=mean,seed_source_oof=np.stack(seed_oof),
                fixed_refit_epochs=int(np.median(epochs)),**core.metrics(y[train],mean[train])))
        save(dest/(name+'_ALL_SOURCE_CONFIGURATIONS.json'),[
            {**c,'source_indices':train,'source_oof':c['source_oof'][train],
             'seed_source_oof':c['seed_source_oof'][:,train]} for c in candidates])
        best=max(candidates,key=lambda r:(r['AUROC'],-r['NLL'],r['lambda_'],-r['tau']))
        save(dest/(name+'_SOURCE_SELECTION_LOCK.json'),dict(selected_tau=best['tau'],selected_lambda=best['lambda_'],
            source_AUROC=best['AUROC'],source_NLL=best['NLL'],source_only=True,
            fixed_refit_epochs=best['fixed_refit_epochs'],seeds=cfg['seeds'],effective_search_configurations=len(candidates)))
        source=np.stack(best['seed_source_oof'])[:,train];seed_test=[]
        input_cache.clear()
        ti=tensor_inputs(parts['outer'],train,name,best['tau'],device,rows)
        vi=tensor_inputs(parts['outer'],test,name,best['tau'],device,rows)
        for seed in cfg['seeds']:
            file=dest/f'{name}_seed{seed}_OUTER.pt'
            if file.exists():saved=torch.load(file,map_location='cpu',weights_only=False);head=restore(saved,ti['b'].shape[-1],device);md=saved['metadata']
            else:
                head,md=train_head(ti,None,y,g,pooling,readout,best['tau'],best['lambda_'],seed,fixed_epochs=best['fixed_refit_epochs'],attention_diversity_weight=cfg['attention_diversity_weight'] if name=='ATTENTION_DIVERSITY_LAPLACE' else 0.)
                torch.save(dict(state={k:v.cpu() for k,v in head.state_dict().items()},metadata=md,train=train,source_selection_frozen=True),file)
            with torch.no_grad():seed_test.append(head(**{k:v for k,v in vi.items() if k!='indices'}).cpu().numpy())
        np.savez_compressed(resultpath,test=test,seed_logits=np.stack(seed_test),source_indices=train,source_seed_logits=source,source_y=y[train],source_g=g[train])
        print('OUTER_SAVED',outer,name,flush=True)
    save(dest/('CONTROLS_COMPLETE.json' if methods else 'COMPLETE.json'),dict(state='COMPLETE',outer=outer,methods=list(methods or METHODS),test_patients=len(test)))
