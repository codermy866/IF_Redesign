"""56-dimensional zero-anchored readouts and shared hierarchical residual head."""
import torch
from torch import nn
from .measure import measure

def coordinates(h,kind='laplace'):
    if kind=='laplace':
        terms=[h]
        for sign in [1.,-1.]:
            for t in [.5,2.,8.]:terms.append(-torch.expm1(-t*(sign*h).clamp_min(0)))
        return torch.stack(terms,dim=-1).flatten(-2)
    if kind=='moment':
        p0=torch.ones_like(h);p1=h;zero0=1.;zero1=0.;terms=[p1]
        for degree in range(2,8):
            p=((2*degree-1)*h*p1-(degree-1)*p0)/degree
            zero=-(degree-1)*zero0/degree
            terms.append(p-zero)
            p0,p1=p1,p
            zero0,zero1=zero1,zero
        return torch.stack(terms,-1).flatten(-2)
    if kind=='lse':
        # Singleton responses for a common source raw-frame scale. Actual
        # weighted LogSumExp is evaluated after constructing the same measure.
        return torch.stack([h]+[(sign*h).clamp_min(0) for sign in [1.,-1.] for _ in [.5,2.,8.]],-1).flatten(-2)
    raise ValueError(kind)

class RIMEHead(nn.Module):
    def __init__(self,context_dim,tau=.3,pooling='measure',readout='laplace'):
        super().__init__();self.tau=tau;self.pooling=pooling;self.readout=readout
        self.a=nn.Parameter(torch.zeros(8));self.U=nn.Parameter(torch.randn(8,2)*.01)
        self.V=nn.Parameter(torch.randn(context_dim,2)*.01)
        self.g=nn.Parameter(torch.zeros(56));self.beta=nn.Parameter(torch.zeros(56))
        if pooling=='mlp':
            # Round upward so this ordinary MLP has at least the same number
            # of effective selector coefficients as the conditional QP head.
            width=(24+2*context_dim+15)//16
            self.local_mlp=nn.Sequential(nn.Linear(8,width,bias=False),nn.Tanh(),nn.Linear(width,8,bias=False))
        if pooling=='pca_refit':self.pca_weights=nn.Parameter(torch.zeros(26))
        if pooling not in ('measure','identity','attention'):
            for p in (self.a,self.U,self.V):p.requires_grad_(False)
        if pooling=='pca_refit':
            self.g.requires_grad_(False);self.beta.requires_grad_(False)
    def scores(self,h,b):
        return torch.tanh(h@self.a+((h@self.U)*(b@self.V)[:,None,None,:]).sum(-1))
    def forward(self,h,k,valid,b,anchor,coordinate_scale,return_details=False,
                semantic_x=None,initial_semantic_weight=None):
        if self.pooling=='pca_refit':
            sx=b[:,:26] if semantic_x is None else semantic_x
            initial=torch.zeros_like(self.pca_weights) if initial_semantic_weight is None else initial_semantic_weight
            return anchor+sx@(self.pca_weights-initial)
        if self.pooling=='mlp':h=torch.tanh(h+self.local_mlp(h))
        if self.pooling=='caprmil':
            flat=h.flatten(0,1);vm=valid.flatten(0,1);result=torch.zeros_like(flat)
            # Bucket by available frame count: the author's block never sees
            # artificial padded instances or invented spatial coordinates.
            for count in torch.unique(vm.sum(-1)):
                n=int(count)
                if n==0:continue
                at=vm.sum(-1)==n;xx=flat[at][vm[at]].reshape(-1,n,8)
                yy=self.caprmil(xx)-self.caprmil(torch.zeros_like(xx))
                result[at]=torch.zeros_like(result[at]).masked_scatter(vm[at][...,None],torch.tanh(yy).flatten())
            h=result.reshape_as(h)
        q=self.scores(h,b);site_valid=valid.any(-1)
        safe=valid.clone();safe[...,0]=safe[...,0] | ~site_valid
        if self.pooling in ('measure','identity'):
            kernel=k if self.pooling=='measure' else torch.eye(k.shape[-1],device=k.device,dtype=k.dtype).expand_as(k)
            w=measure(kernel,q,safe,self.tau)
        elif self.pooling=='geometry':w=measure(k,torch.zeros_like(q),safe,0.)
        elif self.pooling=='attention':w=torch.softmax((q/self.tau).masked_fill(~safe,-torch.inf),-1)
        elif self.pooling in ('mean','mlp','caprmil'):w=safe.to(q.dtype)/safe.sum(-1,keepdim=True)
        elif self.pooling=='gmp':
            # Murray & Perronnin dual GMP; DGMP author implementation uses the
            # same ridge system and L2-normalizes the pooled descriptor.
            phi=coordinates(h,self.readout)/coordinate_scale
            phi=phi*valid[...,None]
            gram=phi@phi.transpose(-1,-2)
            eye=torch.eye(gram.shape[-1],device=gram.device,dtype=gram.dtype)
            w=torch.linalg.solve(gram+self.tau*eye,safe.to(q.dtype).unsqueeze(-1)).squeeze(-1)*valid
        else:raise ValueError(self.pooling)
        if self.readout=='lse':
            terms=[(w[...,None]*h).sum(-2)]
            for sign in [1.,-1.]:
                for t in [.5,2.,8.]:terms.append(torch.log1p((w[...,None]*torch.expm1(t*(sign*h).clamp_min(0))).sum(-2))/t)
            v=torch.stack(terms,-1).flatten(-2)/coordinate_scale
        else:v=(w[...,None]*coordinates(h,self.readout)).sum(-2)/coordinate_scale
        if self.pooling=='gmp':v=torch.nn.functional.normalize(v,dim=-1)
        v=v*site_valid[...,None]
        safe_site=site_valid.clone();safe_site[:,0]=safe_site[:,0] | ~site_valid.any(-1)
        alpha=torch.softmax((v@self.g).masked_fill(~safe_site,-torch.inf),-1)
        r=(alpha[...,None]*v).sum(-2)
        z=anchor+r@self.beta
        return (z,dict(q=q,w=w,v=v,alpha=alpha)) if return_details else z
