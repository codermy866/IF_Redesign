"""Meaningful gradient, capacity and source-only projection regression checks."""
import numpy as np
import torch
from if_redesign.rime.training import train_head
from if_redesign.rime.readout import RIMEHead

def test_patient_heads_learn_without_changing_anchor():
    rng=np.random.default_rng(29);n=16
    h=torch.tensor(rng.normal(size=(n,2,3,8))*.2,dtype=torch.double)
    d=h[:,:,:,None]-h[:,:,None,:];k=torch.exp(-d.square().sum(-1))
    anchor=torch.linspace(-.3,.3,n,dtype=torch.double);saved=anchor.clone()
    valid=torch.ones((n,2,3),dtype=torch.bool);valid[0,0,2]=False
    inputs=dict(h=h,k=k,valid=valid,b=torch.randn(n,26,dtype=torch.double),anchor=anchor,
        coordinate_scale=torch.ones(56,dtype=torch.double),indices=np.arange(n))
    y=np.arange(n)%2;groups=np.repeat(['a','b','c','d'],4)
    for pooling in ['measure','attention','gmp','mlp','pca_refit']:
        head,meta=train_head(inputs,inputs,y,groups,pooling,'laplace',.3,.003,17,max_epochs=3,patience=3)
        assert meta['source_only'] and meta['parameters']>0
        assert torch.isfinite(head(**{k:v for k,v in inputs.items() if k!='indices'})).all()
    torch.testing.assert_close(anchor,saved)

def test_factorial_capacity_and_stronger_mlp():
    for context_dim in [26,34,42]:
        counts=[]
        for pool,readout in [('attention','moment'),('attention','laplace'),('measure','moment'),('measure','laplace')]:
            head=RIMEHead(context_dim,.3,pool,readout)
            counts.append(sum(p.numel() for p in head.parameters() if p.requires_grad))
        assert len(set(counts))==1
        head=RIMEHead(context_dim,.3,'mlp','laplace')
        assert sum(p.numel() for p in head.parameters() if p.requires_grad)>=counts[0]

def test_semantic_refit_regularizes_effective_weights_and_preserves_start():
    head=RIMEHead(26,.3,'pca_refit').double()
    initial=torch.linspace(-.2,.3,26,dtype=torch.double)
    head.pca_weights.data.copy_(initial)
    anchor=torch.tensor([.7],dtype=torch.double)
    inputs=dict(h=torch.zeros(1,1,1,8,dtype=torch.double),k=torch.ones(1,1,1,1,dtype=torch.double),
        valid=torch.ones(1,1,1,dtype=torch.bool),b=torch.zeros(1,26,dtype=torch.double),anchor=anchor,
        coordinate_scale=torch.ones(56,dtype=torch.double),semantic_x=torch.randn(1,26,dtype=torch.double),initial_semantic_weight=initial)
    torch.testing.assert_close(head(**inputs),anchor)
    penalty=sum((p*p).sum() for p in head.parameters() if p.requires_grad)
    torch.testing.assert_close(penalty,(initial*initial).sum())
