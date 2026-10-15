"""Common exact numerical atoms across ALL four core cells.

Only local visibility/context for scoring is altered in controls. Original
source raw-frame coordinate scaling, native context and anchor stay intact.
"""
import numpy as np,torch
from rime_dataset import tensor_inputs as legacy_inputs
from if_redesign.rime.readout import coordinates

def tensor_inputs(part,indices,method,tau,device='cpu',rows=None,frame_budget=10):
    mapped={'A_ATTENTION_MOMENT':'C_MEASURE_MOMENT','B_ATTENTION_LAPLACE':'D_RIME_LAPLACE',
        'ATTENTION_DIVERSITY_LAPLACE':'D_RIME_LAPLACE','MEASURE_LSE56':'D_RIME_LAPLACE'}.get(method,method)
    result=legacy_inputs(part,indices,mapped,tau,device,rows,frame_budget)
    if method=='MEASURE_LSE56':
        if frame_budget!=10:raise ValueError('Readout control uses full frozen source scale')
        with torch.no_grad():raw=coordinates(torch.tensor(part['h'],dtype=torch.double),'lse').numpy()
        fit=part['train'];valid=part['valid'][fit];count=valid.sum(-1);sites=(count>0).sum(-1)
        weights=valid/count.clip(1)[...,None]/sites.clip(1)[:,None,None]/len(fit)
        mean=(raw[fit]*weights[...,None]).sum((0,1,2));scale=np.maximum(np.sqrt(((raw[fit]-mean)**2*weights[...,None]).sum((0,1,2))),.005)
        result['coordinate_scale']=torch.tensor(scale,dtype=torch.double,device=device)
    return result

def shuffled_scoring_context(train,validation,seed):
    rng=np.random.default_rng(seed);n=len(train['indices'])
    if n<2:raise ValueError('Context shuffle requires at least two source patients')
    order=rng.permutation(n);donors=np.roll(order,1);mapping=np.empty(n,int);mapping[order]=donors
    val_donors=rng.integers(n,size=len(validation['indices']))
    ti=dict(train);vi=dict(validation)
    ti['b']=train['b'][torch.as_tensor(mapping,device=train['b'].device)]
    vi['b']=train['b'][torch.as_tensor(val_donors,device=train['b'].device)]
    audit=dict(train_recipients=train['indices'],train_donors=train['indices'][mapping],validation_recipients=validation['indices'],
        validation_donors=train['indices'][val_donors],labels_used=False,anchor_unchanged=True,
        inference='recipient scorer context replaced from legal source pool; no validation/test patient provides donor context')
    return ti,vi,audit
