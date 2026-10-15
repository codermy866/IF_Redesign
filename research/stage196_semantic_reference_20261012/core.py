"""Capacity-matched evidence encodings and a precisely named CFV-risk constraint."""
import numpy as np
from scipy.special import expit, logsumexp
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, average_precision_score

ENCODINGS=('RAW','MOMENT','FOURIER','FOCAL','LAPLACE','UNSIGNED_LAPLACE','ATTENTION')
LAMBDAS=(.003,.03,.3)
RHO=.25
ARMS={
    'M1_MOMENT':('MOMENT','NONE'),
    'M2_MOMENT_CFV':('MOMENT','CFV_RISK'),
    'M3_LAPLACE':('LAPLACE','NONE'),
    'M4_LAPLACE_CFV':('LAPLACE','CFV_RISK'),
    'RAW':('RAW','NONE'), 'FOURIER':('FOURIER','NONE'),
    'FOCAL':('FOCAL','NONE'), 'UNSIGNED_LAPLACE':('UNSIGNED_LAPLACE','NONE'),
    'ATTENTION_POOLING':('ATTENTION','NONE'),
    'LAPLACE_SIMPLE_CONSISTENCY':('LAPLACE','SIMPLE_CONSISTENCY'),
}


def scales(delta,mask):
    return np.asarray([max(float(np.quantile(np.abs(delta[:,sl][mask[:,sl]]),.9)),.05)
                       if mask[:,sl].any() else .05 for sl in (slice(0,12),slice(12,20))])


def encode(delta,mask,scale,kind):
    """26 zero-anchored coordinates, identical masks/scales for every encoding."""
    if kind not in ENCODINGS:raise ValueError(kind)
    blocks=[]
    for m,sl in enumerate((slice(0,12),slice(12,20))):
        valid=mask[:,sl].astype(bool);count=valid.sum(1);den=np.maximum(count,1)
        u=np.where(valid,np.tanh(delta[:,sl]/scale[m]),0.)
        avg=lambda x:(x*valid).sum(1)/den
        coordinates=[avg(u)]
        if kind=='RAW':
            q=np.linspace(0,1,12)
            values=np.zeros((len(u),12))
            for i in range(len(u)):
                if count[i]:values[i]=np.quantile(u[i,valid[i]],q)
            coordinates.extend(values.T)
        elif kind=='FOURIER':
            coordinates.extend(avg(np.cos(np.pi*k*u)-1) for k in range(1,7))
            coordinates.extend(avg(np.sin(np.pi*k*u)) for k in range(1,7))
        else:
            for sign in (1.,-1.):
                v=np.abs(u) if kind=='UNSIGNED_LAPLACE' else np.maximum(sign*u,0.)
                if kind=='MOMENT':coordinates.extend(avg(v**k) for k in range(1,7))
                elif kind in ('LAPLACE','UNSIGNED_LAPLACE'):
                    coordinates.extend(avg(-np.expm1(-t*v)) for t in np.geomspace(.5,16,6))
                elif kind=='ATTENTION':
                    for t in np.geomspace(.5,16,6):
                        mass=np.exp(t*v)*valid
                        coordinates.append((mass*v).sum(1)/np.maximum(mass.sum(1),1e-300))
                else:
                    for t in np.geomspace(.5,16,6):
                        total=(np.exp(t*v)*valid).sum(1)/den
                        coordinates.append(np.where(count>0,np.log(np.maximum(total,1e-300))/t,0.))
        blocks.append(np.column_stack(coordinates))
    result=np.column_stack(blocks)
    assert result.shape==(len(delta),26) and np.isfinite(result).all()
    return result


def source_state(delta,mask,kind):
    scale=scales(delta,mask);raw=encode(delta,mask,scale,kind)
    return dict(scale=scale,std=np.maximum(raw.std(0),.005),kind=kind)


def transform(delta,mask,state):
    return np.clip(encode(delta,mask,state['scale'],state['kind'])/state['std'],-16,16)


def fixed_deletion_design(delta,mask,x,state,visual_scale):
    """Fixed-contribution approximation ONLY; not full-model counterfactuals."""
    dx=np.zeros((len(delta),20,26))
    for j in range(20):
        altered=mask.copy();altered[:,j]=False
        dx[:,j]=x-transform(delta,altered,state)
    return dict(dx=dx,db=visual_scale*delta,mask=mask)


def objective(w,x,base,y,groups,lam,constraint,deletion=None):
    z=base+RHO*(x@w);ll=np.logaddexp(0,z)-y*z
    baseline=np.logaddexp(0,base)-y*base
    value=.5*lam*(w@w);dz=np.zeros(len(z));extra_gradient=np.zeros(len(w))
    centers=np.unique(groups)
    for c in centers:
        at=groups==c;weight=1/(at.sum()*len(centers))
        value+=weight*ll[at].sum();dz[at]+=weight*(expit(z[at])-y[at])
        if constraint=='CFV_RISK':
            increment=float((ll[at]-baseline[at]).mean()-.001)
            value+=max(increment,0.)/len(centers)
            if increment>0:dz[at]+=weight*(expit(z[at])-y[at])
        elif constraint=='SIMPLE_CONSISTENCY':
            if deletion is None:raise ValueError('Missing fixed-deletion approximation')
            valid=deletion['mask'][at];count=np.maximum(valid.sum(1),1)
            difference=deletion['db'][at]+RHO*np.einsum('nvd,d->nv',deletion['dx'][at],w)
            weights=valid/count[:,None]/(at.sum()*len(centers))
            value+=.1*np.sum(weights*difference**2)
            extra_gradient+=.2*RHO*np.einsum('nv,nvd->d',weights*difference,deletion['dx'][at])
        elif constraint!='NONE':raise ValueError(constraint)
    return float(value),RHO*x.T@dz+lam*w+extra_gradient


def fit(delta,mask,base,y,groups,kind,constraint,lam,visual_scale):
    state=source_state(delta,mask,kind);x=transform(delta,mask,state)
    deletion=fixed_deletion_design(delta,mask,x,state,visual_scale) if constraint=='SIMPLE_CONSISTENCY' else None
    opt=minimize(lambda w:objective(w,x,base,y,groups,lam,constraint,deletion),
                 np.zeros(26),jac=True,method='L-BFGS-B',
                 options=dict(maxiter=1000,maxls=100,ftol=1e-11,gtol=1e-7))
    if not np.isfinite(opt.fun) or (not opt.success and np.max(np.abs(opt.jac))>1e-4):
        raise RuntimeError('Readout failed to converge: '+str(opt.message))
    return dict(state=state,w=opt.x,kind=kind,constraint=constraint,lambda_=lam,
                rho=RHO,parameters=26,gradient_max=float(np.max(np.abs(opt.jac))))


def predict(head,delta,mask,base):
    return base+head['rho']*transform(delta,mask,head['state'])@head['w']


def metrics(y,z):
    y=np.asarray(y);z=np.asarray(z);p=expit(z)
    return dict(N=len(y),CIN2plus=int(y.sum()),
        AUROC=float(roc_auc_score(y,z)) if len(np.unique(y))==2 else None,
        AUPRC=float(average_precision_score(y,z)) if y.sum() else None,
        NLL=float(np.mean(np.logaddexp(0,z)-y*z)),Brier=float(np.mean((p-y)**2)))
