"""Simplex kernel QP with unique atoms and active-set implicit gradients.

No entropy or instance-level diagonal regularizer is added to the objective.
Padding diagonals in the linear system only constrain inactive variables to zero.
"""
import numpy as np
import torch
from scipy.optimize import minimize

def gaussian_kernel(h,sigma):
    if sigma<=0:raise ValueError('Bandwidth must be positive and source-frozen')
    d=h[:,None,:]-h[None,:,:]
    return np.exp(-np.sum(d*d,axis=-1)/(2*sigma*sigma))

def reference_measure(h,q,sigma,tau):
    h=np.asarray(h,dtype=np.float64);q=np.asarray(q,dtype=np.float64)
    atoms,inverse=np.unique(h,axis=0,return_inverse=True)
    scores=np.array([q[inverse==j][0] for j in range(len(atoms))])
    if any(np.ptp(q[inverse==j])>1e-12 for j in range(len(atoms))):
        raise ValueError('Exact copies must have identical conditional scores')
    k=gaussian_kernel(atoms,sigma)
    result=minimize(lambda w:(.5*w@k@w-tau*scores@w,k@w-tau*scores),
        np.ones(len(atoms))/len(atoms),jac=True,bounds=[(0,1)]*len(atoms),
        constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)},
        method='SLSQP',options={'ftol':1e-13,'maxiter':2000})
    if not result.success:raise RuntimeError(result.message)
    w=result.x;active=w>1e-8;gradient=k@w-tau*scores
    nu=-float(gradient[active].mean())
    residual=max(abs(w.sum()-1),float(np.max(abs(gradient[active]+nu))),
                 max(0.,-float((gradient[~active]+nu).min())) if (~active).any() else 0.)
    return atoms,w,dict(kkt_residual=residual,kernel_condition=float(np.linalg.cond(k)),
        objective=float(result.fun),unique_atoms=len(atoms),original_instances=len(h))

def equality_system(k,active,rhs):
    n=k.shape[-1];a=active.to(k.dtype)
    mat=torch.zeros((*k.shape[:-2],n+1,n+1),dtype=k.dtype,device=k.device)
    mat[...,:n,:n]=k*a[...,None]*a[...,None,:]+torch.diag_embed(1-a)
    mat[...,:n,n]=a;mat[...,n,:n]=a
    b=torch.cat([rhs*a,torch.ones_like(a[...,:1])],dim=-1)
    return torch.linalg.solve(mat,b.unsqueeze(-1)).squeeze(-1),mat

def active_set(k,q,valid,tau,tol=1e-9,max_iter=128):
    """Batched feasible active-set method, checked against float64 SLSQP."""
    shape=q.shape;n=shape[-1];k=k.reshape(-1,n,n);q=q.reshape(-1,n);valid=valid.reshape(-1,n)
    if not valid.any(-1).all():raise ValueError('Empty bags must be handled before QP')
    active=valid.clone();w=active.to(k.dtype)/active.sum(-1,keepdim=True)
    done=torch.zeros(len(q),dtype=torch.bool,device=q.device)
    for iteration in range(max_iter):
        sol,_=equality_system(k,active,tau*q);candidate=sol[:,:n]
        negative=(candidate < -tol)&active&~done[:,None]
        blocked=negative.any(-1)
        ratios=torch.where(negative,w/(w-candidate).clamp_min(1e-300),torch.inf)
        step=ratios.min(-1).values.clamp(max=1.)
        w=torch.where((~done)[:,None],w+step[:,None]*(candidate-w),w)
        # Only remove the first boundary reached, preserving primal feasibility.
        hit=blocked[:,None] & active & (w<=tol)
        active=active & ~hit
        w=torch.where(hit,torch.zeros_like(w),w)
        feasible=~blocked&~done
        grad=torch.bmm(k,w[:,:,None]).squeeze(-1)-tau*q+sol[:,n,None]
        eligible=valid&~active
        outside=torch.where(eligible,grad,torch.inf)
        value,index=outside.min(-1)
        add=feasible&(value < -tol)
        active[torch.arange(len(q),device=q.device)[add],index[add]]=True
        done=done | (feasible&~add)
        if bool(done.all()):break
    if not bool(done.all()):raise RuntimeError('QP active set did not converge')
    residual=(w.sum(-1)-1).abs().max()
    gradient=torch.bmm(k,w[:,:,None]).squeeze(-1)-tau*q
    nu=-(gradient*active).sum(-1)/active.sum(-1)
    primal=torch.where(active,(gradient+nu[:,None]).abs(),0.).max()
    dual=torch.where(valid&~active,(-gradient-nu[:,None]).clamp_min(0),0.).max()
    if float(torch.maximum(residual,torch.maximum(primal,dual)))>1e-7:
        raise RuntimeError('QP KKT tolerance failed')
    return w.reshape(shape),active.reshape(shape)

class KernelMeasure(torch.autograd.Function):
    @staticmethod
    def forward(ctx,k,q,valid,tau):
        w,active=active_set(k.double(),q.double(),valid,tau)
        ctx.save_for_backward(k.double(),active);ctx.tau=tau;ctx.q_dtype=q.dtype
        return w.to(q.dtype)
    @staticmethod
    def backward(ctx,gradient):
        k,active=ctx.saved_tensors;n=k.shape[-1]
        _,matrix=equality_system(k,active,torch.zeros_like(gradient,dtype=k.dtype))
        rhs=torch.cat([gradient.double()*active,torch.zeros_like(gradient[...,:1],dtype=k.dtype)],-1)
        solution=torch.linalg.solve(matrix,rhs.unsqueeze(-1)).squeeze(-1)[...,:n]
        return None,(ctx.tau*solution*active).to(ctx.q_dtype),None,None

def measure(k,q,valid,tau):return KernelMeasure.apply(k,q,valid,float(tau))
