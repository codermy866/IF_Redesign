"""Source-fitted, zero-anchored Qwen semantic/evidence readouts.

Centered matrices learn a basis; inference does NOT subtract a mean. Gray
evidence therefore remains zero. All candidate readouts have 32 coefficients.
"""
import numpy as np
from scipy.special import expit, logsumexp
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, average_precision_score

VARIANTS = ('SEMANTIC32', 'CENTERED32', 'WITHIN32', 'HYBRID32', 'CROSS32', 'WITHINPRODUCT32')
FAMILIES = tuple(v + '__' + loss for v in VARIANTS for loss in ('MEAN', 'ROBUST'))
LAMBDAS = (.01, .1, 1., 10.)
RHO = .25

def subset(data, at):
    return {k: v[at] for k, v in data.items()}

def magnitude(x):
    return max(float(np.quantile(np.abs(x), .9)), .05) if len(x) else .05

def evidence(data, state):
    blocks=[]; raw=[]
    for m, sl in enumerate((slice(0,12), slice(12,20))):
        mask=data['mask'][:,sl].astype(bool)
        u=np.tanh(data['delta'][:,sl]/state['modality_scales'][m])
        u=np.where(mask,u,0.); raw.append(u)
        coords=[u]
        for sign in (1., -1.):
            positive=np.maximum(sign*u,0.)
            coords.extend([-np.expm1(-tau*positive) for tau in (.125,.25,.5,1.,2.,4.)])
        blocks.append(np.column_stack([(v*mask).sum(1)/np.maximum(mask.sum(1),1) for v in coords]))
    any_view=data['mask'].any(1)
    both=data['mask'][:,:12].any(1)&data['mask'][:,12:].any(1)
    a=np.where(any_view,np.tanh(data['A']/state['A_scale']),0.)
    joint=np.where(both,np.tanh(data['I']/state['I_scale']),0.)
    return np.column_stack(raw+blocks+[a,joint])  # 20 + 26 + 2

def basis(matrix, dim=32):
    _, _, vt=np.linalg.svd(matrix,full_matrices=False)
    k=min(dim,len(vt)); result=np.zeros((matrix.shape[1],dim))
    result[:,:k]=vt[:k].T
    return result

def fit_state(data, groups):
    h=data['hidden'].astype(float); centered=h-h.mean(0)
    within=h.copy()
    for c in np.unique(groups):
        at=groups==c
        within[at]=(h[at]-h[at].mean(0))/np.sqrt(at.sum())
    state=dict(modality_scales=[magnitude(data['delta'][:,sl][data['mask'][:,sl]])
              for sl in (slice(0,12),slice(12,20))], A_scale=magnitude(data['A']), I_scale=magnitude(data['I']))
    state.update(S=basis(h), C=basis(centered), W=basis(within), E=basis(evidence(data,state)))
    for kind, matrix in (('S',h),('C',h),('W',h),('E',evidence(data,state))):
        state[kind+'_std']=np.maximum((matrix@state[kind]).std(0),.005)
    for variant in VARIANTS:
        state[variant+'_std']=np.maximum(raw_features(data,state,variant).std(0),.005)
    return state

def raw_features(data,state,variant):
    h=data['hidden']; s=np.clip(h@state['C']/state['C_std'],-16,16)
    e=np.clip(evidence(data,state)@state['E']/state['E_std'],-16,16)
    if variant in ('SEMANTIC32','CENTERED32','WITHIN32'):
        kind=dict(SEMANTIC32='S',CENTERED32='C',WITHIN32='W')[variant]
        result=h@state[kind]/state[kind+'_std']
    elif variant=='HYBRID32':result=np.column_stack((s[:,:16],e[:,:16]))
    elif variant=='CROSS32':
        cross=(s[:,:4,None]*e[:,None,:4]).reshape(len(s),16)
        result=np.column_stack((s[:,:8],e[:,:8],cross))
    elif variant=='WITHINPRODUCT32':
        terms=[]
        for block in (s,e):
            terms.extend([block[:,j]**2 for j in range(4)])
            terms.extend([block[:,j]*block[:,(j+1)%4] for j in range(4)])
        result=np.column_stack((s[:,:8],e[:,:8],*terms))
    else:raise ValueError(variant)
    return np.where(data['mask'].any(1)[:,None],result,0.)

def features(data,state,variant):
    result=np.clip(raw_features(data,state,variant)/state[variant+'_std'],-16,16)
    if result.shape!=(len(data['clinical']),32) or not np.isfinite(result).all():
        raise ValueError('Invalid feature matrix')
    return result

def pairs(y,groups):
    rng=np.random.default_rng(190); design=[]
    for c in np.unique(groups):
        p=np.flatnonzero((groups==c)&(y==1));n=np.flatnonzero((groups==c)&(y==0))
        if not len(p) or not len(n):raise ValueError('Empty source center-outcome stratum')
        if len(p)*len(n)<=2048:p,n=np.repeat(p,len(n)),np.tile(n,len(p))
        else:p,n=rng.choice(p,2048),rng.choice(n,2048)
        design.append((p,n,np.flatnonzero(groups==c)))
    return design

def objective(w,x,clinical,y,design,lam,robust):
    z=clinical+RHO*(x@w);risks=[];grads=[];bce=0.;db=np.zeros(len(z))
    for p,n,at in design:
        margin=z[p]-z[n];risks.append(np.logaddexp(0,-margin).mean())
        dz=np.zeros(len(z)); v=-expit(-margin)/len(p)
        np.add.at(dz,p,v);np.add.at(dz,n,-v);grads.append(dz)
        bce+=(np.logaddexp(0,z[at])-y[at]*z[at]).mean()/len(design)
        db[at]=(expit(z[at])-y[at])/(len(at)*len(design))
    risks=np.asarray(risks)
    if robust:
        weights=np.exp(risks/.1-logsumexp(risks/.1))
        value=.1*(logsumexp(risks/.1)-np.log(len(risks)))
    else:weights=np.full(len(risks),1/len(risks));value=risks.mean()
    gradient=np.asarray(grads).T@weights+.1*db
    return float(value+.1*bce+.5*lam*(w@w)),RHO*x.T@gradient+lam*w

def fit_head(x,clinical,y,groups,lam,robust):
    design=pairs(y,groups)
    opt=minimize(lambda w:objective(w,x,clinical,y,design,lam,robust),np.zeros(32),jac=True,
                 method='L-BFGS-B',options=dict(maxiter=1000,maxls=100,ftol=1e-11,gtol=1e-7))
    if not np.isfinite(opt.fun) or (not opt.success and abs(opt.jac).max()>1e-4):
        raise RuntimeError('Readout convergence failure: '+str(opt.message))
    return dict(w=opt.x.tolist(),lambda_=float(lam),rho=RHO,parameters=32,
                objective=float(opt.fun),gradient_max=float(abs(opt.jac).max()),optimizer_success=bool(opt.success))

def predict(head,data,state):
    return data['clinical']+RHO*(features(data,state,head['family'].split('__')[0])@np.asarray(head['w']))

def center_auc(y,z,groups):
    return float(np.mean([roc_auc_score(y[groups==c],z[groups==c]) for c in np.unique(groups)
                         if len(np.unique(y[groups==c]))==2]))

def metrics(y,z):
    p=expit(z);bins=np.minimum((10*p).astype(int),9);ece=0.
    for i in range(10):
        at=bins==i
        if at.any():ece+=at.mean()*abs(float(p[at].mean()-y[at].mean()))
    return dict(N=int(len(y)),AUROC=float(roc_auc_score(y,p)),AUPRC=float(average_precision_score(y,p)),
                NLL=float(np.mean(np.logaddexp(0,z)-y*z)),Brier=float(np.mean((p-y)**2)),ECE=float(ece))

def train_all(data,y,groups,progress=None):
    """Reuse source-fitted bases across lambdas; refit them inside every CV fold."""
    state=fit_state(data,groups);cv=[]
    for c in np.unique(groups):
        tr=np.flatnonzero(groups!=c);va=np.flatnonzero(groups==c)
        td=subset(data,tr);vd=subset(data,va);cs=fit_state(td,groups[tr])
        cv.append((tr,va,cs,td,vd))
    heads={}
    for variant in VARIANTS:
        x=features(data,state,variant)
        projections=[(tr,va,features(td,cs,variant),features(vd,cs,variant)) for tr,va,cs,td,vd in cv]
        for loss in ('MEAN','ROBUST'):
            menu=[]
            for lam in LAMBDAS:
                scores=[]
                for tr,va,xt,xv in projections:
                    h=fit_head(xt,data['clinical'][tr],y[tr],groups[tr],lam,loss=='ROBUST')
                    z=data['clinical'][va]+RHO*(xv@np.asarray(h['w']))
                    scores.append(float(roc_auc_score(y[va],z)))
                menu.append(dict(lambda_=lam,center_AUROC=float(np.mean(scores)),center_scores=scores))
            selected=max(menu,key=lambda r:(r['center_AUROC'],r['lambda_']))
            h=fit_head(x,data['clinical'],y,groups,selected['lambda_'],loss=='ROBUST')
            name=variant+'__'+loss;h.update(family=name,cv_menu=menu,conditional_CV=True);heads[name]=h
            if progress:progress(name)
    return state,heads
