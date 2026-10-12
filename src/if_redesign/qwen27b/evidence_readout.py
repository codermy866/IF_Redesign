"""Patient evidence distributions and bounded source-only ranking correction."""
import numpy as np
from scipy.special import expit, logsumexp
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, average_precision_score

FACTORIAL = tuple(f'{enc}_{interaction}_{loss}' for enc in ('SPECTRAL', 'MOMENT')
                 for interaction in ('JOINT', 'NOJOINT') for loss in ('ROBUST', 'MEAN'))
FAMILIES = FACTORIAL+tuple('DECISION_'+family for family in FACTORIAL)

def subset(data, ix):
    return {k: v[ix] for k, v in data.items()}

def scale(values):
    return max(float(np.quantile(np.abs(values), .9)), .05) if len(values) else .05

def fit_state(data):
    return dict(modality_scales=[scale(data['delta'][:, sl][data['mask'][:, sl]])
                for sl in (slice(0, 12), slice(12, 20))],
                A_scale=scale(data['A']), I_scale=scale(data['I']), fit_scope='source_train_only')

def features(data, state, family):
    encoding, interaction, _ = family.removeprefix('DECISION_').split('_'); blocks=[]
    for m, sl in enumerate((slice(0, 12), slice(12, 20))):
        mask=data['mask'][:, sl].astype(bool); u=np.tanh(data['delta'][:, sl]/state['modality_scales'][m])
        if encoding == 'SPECTRAL':
            coords=[u, np.abs(u)]
            for frequency in (.5, 1., 2.):
                coords.extend((np.sin(frequency*u), -2*np.sin(frequency*u/2)**2))
        else:
            coords=[u**j for j in range(1, 7)]+[np.abs(u), np.abs(u)**3]
        blocks.append(np.column_stack([(v*mask).sum(1)/np.maximum(mask.sum(1), 1) for v in coords]))
    both=data['mask'][:, :12].any(1)&data['mask'][:, 12:].any(1)
    any_view=data['mask'].any(1)
    A=np.where(any_view, np.tanh(data['A']/state['A_scale']), 0.)
    I=np.where(both, np.tanh(data['I']/state['I_scale']), 0.) if interaction=='JOINT' else np.zeros(len(A))
    result=np.column_stack(blocks+[A, I])
    if not np.isfinite(result).all(): raise ValueError('Nonfinite evidence features')
    return result

def pairs(y, groups, seed=191):
    rng=np.random.default_rng(seed); result=[]
    for center in np.unique(groups):
        p=np.flatnonzero((groups==center)&(y==1)); n=np.flatnonzero((groups==center)&(y==0))
        if not len(p) or not len(n): raise ValueError('Empty center-outcome stratum')
        if len(p)*len(n)<=2048: pos,neg=np.repeat(p,len(n)),np.tile(n,len(p))
        else: pos,neg=rng.choice(p,2048),rng.choice(n,2048)
        result.append((pos,neg,np.flatnonzero(groups==center)))
    return result

def objective(w, x, clinical, y, design, lam, robust, bounded=True):
    t=x@w; increment=np.tanh(t) if bounded else t
    derivative=1-increment**2 if bounded else np.ones(len(t))
    z=clinical+increment; risks=[]; gradients=[]; bce=0.; dz_bce=np.zeros(len(z))
    for p,n,at in design:
        margin=z[p]-z[n]; risks.append(np.logaddexp(0,-margin).mean())
        dz=np.zeros(len(z)); v=-expit(-margin)/len(p)
        np.add.at(dz,p,v); np.add.at(dz,n,-v); gradients.append(dz)
        bce+=np.mean(np.logaddexp(0,z[at])-y[at]*z[at])/len(design)
        dz_bce[at]=(expit(z[at])-y[at])/(len(design)*len(at))
    risks=np.asarray(risks)
    if robust:
        weights=np.exp(risks/.1-logsumexp(risks/.1))
        loss=.1*(logsumexp(risks/.1)-np.log(len(risks)))
    else: weights=np.full(len(risks),1/len(risks)); loss=risks.mean()
    dz=np.asarray(gradients).T@weights+.1*dz_bce
    return float(loss+.1*bce+.5*lam*(w@w)), x.T@(dz*derivative)+lam*w

def fit_head(data, y, groups, family, lam):
    state=fit_state(data); raw=features(data,state,family); std=np.maximum(raw.std(0),.005)
    x=np.clip(raw/std,-16,16); design=pairs(y,groups)
    fun=lambda w: objective(w,x,offset(data,family),y,design,lam,family.endswith('ROBUST'))
    opt=minimize(fun,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',
                 options=dict(maxiter=1000,maxls=100,ftol=1e-11,gtol=1e-7))
    if not np.isfinite(opt.fun) or (not opt.success and np.abs(opt.jac).max()>1e-4):
        raise RuntimeError('Bounded evidence head failed to converge: '+str(opt.message))
    return dict(state=state,std=std.tolist(),w=opt.x.tolist(),family=family,lambda_=float(lam),
                B=1.,parameters=18,active_parameters=17 if '_NOJOINT_' in family else 18,
                optimizer_success=bool(opt.success),gradient_max=float(np.abs(opt.jac).max()),objective=float(opt.fun))

def predict(head, data):
    x=np.clip(features(data,head['state'],head['family'])/np.asarray(head['std']),-16,16)
    return offset(data,head['family'])+np.tanh(x@np.asarray(head['w']))

def offset(data,family):
    if family.startswith('DECISION_'):
        return data['clinical']+np.where(data['mask'].any(1),data['A'],0.)
    return data['clinical']

def center_auc(y,z,groups):
    return float(np.mean([roc_auc_score(y[groups==c],z[groups==c]) for c in np.unique(groups)
                          if len(np.unique(y[groups==c]))==2]))

def metrics(y,z):
    p=expit(z); ece=0.; bins=np.minimum((10*p).astype(int),9)
    for j in range(10):
        at=bins==j
        if at.any():ece+=at.mean()*abs(float(p[at].mean()-y[at].mean()))
    return dict(N=int(len(y)),AUROC=float(roc_auc_score(y,p)),AUPRC=float(average_precision_score(y,p)),
                NLL=float(np.mean(np.logaddexp(0,z)-y*z)),Brier=float(np.mean((p-y)**2)),ECE=float(ece))

def train_select(data,y,groups,family,lambdas=(.01,.1,1.,10.)):
    menu=[]
    for lam in lambdas:
        scores=[]
        for c in np.unique(groups):
            tr=np.flatnonzero(groups!=c); va=np.flatnonzero(groups==c)
            head=fit_head(subset(data,tr),y[tr],groups[tr],family,lam)
            scores.append(float(roc_auc_score(y[va],predict(head,subset(data,va)))))
        menu.append(dict(lambda_=lam,center_AUROC=float(np.mean(scores)),center_scores=scores))
    best=max(menu,key=lambda row:(row['center_AUROC'],row['lambda_']))
    head=fit_head(data,y,groups,family,best['lambda_']);head.update(cv_menu=menu,conditional_source_CV=True)
    return head

def projection_features(data, state, kind):
    if kind=='SCALAR':return data['A'][:,None]/np.asarray(state['std'])
    return np.clip((data['hidden']@np.asarray(state['basis']))/np.asarray(state['std']),-16,16)

def fit_control(data,y,groups,kind,lam):
    bounded=kind!='PCA_LINEAR'
    if kind=='SCALAR':state=dict(std=np.maximum(data['A'].std(),.005).reshape(1).tolist())
    else:
        _,_,v=np.linalg.svd(data['hidden'],full_matrices=False);basis=v[:18].T
        state=dict(basis=basis.tolist(),std=np.maximum((data['hidden']@basis).std(0),.005).tolist())
    x=projection_features(data,state,kind);design=pairs(y,groups)
    opt=minimize(lambda w:objective(w,x,data['clinical'],y,design,lam,False,bounded),
                 np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',
                 options=dict(maxiter=1000,maxls=100,ftol=1e-11,gtol=1e-7))
    if not np.isfinite(opt.fun) or (not opt.success and np.abs(opt.jac).max()>1e-4):raise RuntimeError('Control optimizer failed')
    return dict(state=state,w=opt.x.tolist(),kind=kind,lambda_=float(lam),parameters=len(opt.x),bounded=bounded)

def predict_control(head,data):
    increment=projection_features(data,head['state'],head['kind'])@np.asarray(head['w'])
    return data['clinical']+(np.tanh(increment) if head['bounded'] else increment)

def train_control(data,y,groups,kind,lambdas=(.01,.1,1.,10.)):
    menu=[]
    for lam in lambdas:
        scores=[]
        for c in np.unique(groups):
            tr=np.flatnonzero(groups!=c);va=np.flatnonzero(groups==c)
            head=fit_control(subset(data,tr),y[tr],groups[tr],kind,lam)
            scores.append(float(roc_auc_score(y[va],predict_control(head,subset(data,va)))))
        menu.append(dict(lambda_=lam,center_AUROC=float(np.mean(scores))))
    best=max(menu,key=lambda row:(row['center_AUROC'],row['lambda_']))
    head=fit_control(data,y,groups,kind,best['lambda_']);head['cv_menu']=menu;return head
