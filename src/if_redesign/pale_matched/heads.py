"""Prespecified replacements of one PALE decision path, no new primary model."""
import torch
from torch import nn
from if_redesign.rime.readout import RIMEHead,coordinates

def readout_dim(kind):return {'mean':8,'std':16,'single':24,'magnitude':32}.get(kind,56)
def zero_readout(h,kind,learned=None):
    if kind in ('laplace','moment'):return coordinates(h,kind)
    if kind=='mean':return h
    if kind=='single':return torch.stack([h]+[-torch.expm1(-2*(s*h).clamp_min(0)) for s in [1.,-1.]],-1).flatten(-2)
    if kind=='magnitude':return torch.stack([h]+[-torch.expm1(-t*h.abs()) for t in [.5,2.,8.]],-1).flatten(-2)
    if kind=='learned':return learned(h)-learned(torch.zeros_like(h))
    raise ValueError(kind)

def masked_mean(h,valid):return (h*valid[...,None]).sum(-2)/valid.sum(-1).clamp_min(1)[...,None]

def summarize_sites(h,valid,spec,adapt,learned):
    order=spec['order'];kind=spec['readout']
    if order=='mean_adapt_phi':h=masked_mean(h,valid).unsqueeze(-2);valid=valid.any(-1).unsqueeze(-1)
    if adapt is not None:
        h=torch.tanh(h+adapt(h)) if spec['adapt']=='residual' else torch.tanh(adapt(h))
    if order=='adapt_mean_phi':h=masked_mean(h,valid).unsqueeze(-2);valid=valid.any(-1).unsqueeze(-1)
    if kind=='std':
        mean=masked_mean(h,valid)
        variance=masked_mean((h-mean[...,None,:]).square(),valid)
        # Smooth zero-reference std avoids undefined derivative at variance zero.
        return torch.cat([mean,torch.sqrt(variance+1e-12)-1e-6],-1)
    if kind=='lse':
        terms=[masked_mean(h,valid)]
        for s in [1.,-1.]:
            for t in [.5,2.,8.]:terms.append(torch.log1p(masked_mean(torch.expm1(t*(s*h).clamp_min(0)),valid))/t)
        return torch.stack(terms,-1).flatten(-2)
    phi=zero_readout(h,kind,learned)
    if spec.get('pool')=='max':
        out=phi.masked_fill(~valid[...,None],-torch.inf).amax(-2)
        return torch.where(valid.any(-1)[...,None],out,torch.zeros_like(out))
    return masked_mean(phi,valid)

class VariantHead(RIMEHead):
    def __init__(self,context_dim,spec):
        super().__init__(context_dim,.3,'mlp','laplace')
        self.spec=spec;dim=readout_dim(spec['readout'])
        if spec['adapt']=='none':del self.local_mlp
        if dim!=56:self.g=nn.Parameter(torch.zeros(dim));self.beta=nn.Parameter(torch.zeros(dim))
        if spec['readout']=='learned':self.learned=nn.Sequential(nn.Linear(8,16),nn.Tanh(),nn.Linear(16,56))
        if spec.get('site') in ('mean','flat'):self.g.requires_grad_(False)
    def forward(self,h,k,valid,b,anchor,coordinate_scale,return_details=False):
        spec=self.spec
        if spec.get('site')=='flat':h=h.flatten(1,2).unsqueeze(1);valid=valid.flatten(1,2).unsqueeze(1)
        v=summarize_sites(h,valid,spec,getattr(self,'local_mlp',None),getattr(self,'learned',None))/coordinate_scale
        site_valid=valid.any(-1);v=v*site_valid[...,None]
        safe=site_valid.clone();safe[:,0]|=~site_valid.any(-1)
        alpha=torch.softmax((v@self.g).masked_fill(~safe,-torch.inf),-1)
        r=(alpha[...,None]*v).sum(-2)
        z=(anchor if spec.get('anchor',True) else torch.zeros_like(anchor))+r@self.beta
        return (z,dict(v=v,alpha=alpha)) if return_details else z

def make_head(context_dim,spec):
    # Exact P3 implementation and initialization, not an approximate rewrite.
    if spec==dict(adapt='residual',order='adapt_phi_mean',readout='laplace',site='attention',anchor=True,ranking=.1,weighting='centre'):
        return RIMEHead(context_dim,.3,'mlp','laplace')
    return VariantHead(context_dim,spec)
