"""Repeated-site phenotype geometry and source cross-center risk ordering."""
import numpy as np
from scipy.special import expit,logsumexp
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score,average_precision_score
GEOMETRIES=('MOMENT','FOCAL','RELIABLE_FOCAL')
FAMILIES=tuple(g+'__'+r for g in GEOMETRIES for r in ('WITHIN','ALL_PAIRS'))
LAMBDAS=(.01,.1,1.,10.);RHO=.25

def subset(data,at):return {k:v[at] for k,v in data.items()}
def scale(x):return max(float(np.quantile(abs(x),.9)),.05) if len(x) else .05
def site_geometry(data,state,kind):
    response=data['site'];mask=data['site_mask'].astype(bool)
    mean=response.mean(2);disagreement=abs(response[:,:,0]-response[:,:,1])
    if kind=='RELIABLE_FOCAL':mean=mean/(1+disagreement/state['disagreement_scale'])
    u=np.where(mask,np.clip(mean/state['site_scale'],-8,8),0.);n=mask.sum(1);den=np.maximum(n,1)
    avg=lambda x:(x*mask).sum(1)/den
    mu=avg(u);std=np.sqrt(np.maximum(avg(u*u)-mu*mu,0.))
    pos=np.maximum(u,0.);neg=np.maximum(-u,0.)
    terms=[mu,avg(abs(u)),std,avg(u*u),avg(u**3),avg(abs(u)**3)]
    for x in (pos,neg):
        if kind=='MOMENT':terms.extend([avg(x**j) for j in (1,2,3)])
        else:
            for tau in (.5,1.,2.):
                # Normalized cumulant generating function: zero -> zero;
                # interpolates mean and focal maxima with fixed positive tau.
                total=(np.exp(tau*x)*mask).sum(1)/den
                terms.append(np.where(n>0,np.log(np.maximum(total,1e-12))/tau,0.))
    terms.extend((pos.max(1),neg.max(1)))
    result=np.column_stack(terms)
    assert result.shape==(len(u),14)
    return result

def raw_features(data,state,kind):
    semantic=data['hidden']@state['basis']/state['semantic_std']
    a=data['A']/state['A_scale'];k=data['K']/state['K_scale']
    x=np.column_stack((semantic,site_geometry(data,state,kind),a,k))
    return np.where(data['any_view'][:,None],x,0.)

def fit_state(data):
    h=data['hidden'].astype(float);_,_,vt=np.linalg.svd(h-h.mean(0),full_matrices=False)
    basis=np.zeros((h.shape[1],16));dim=min(16,len(vt));basis[:,:dim]=vt[:dim].T
    mask=data['site_mask'].astype(bool);response=data['site'];state=dict(basis=basis,
        semantic_std=np.maximum((h@basis).std(0),.005),
        site_scale=scale(response.mean(2)[mask]),
        disagreement_scale=scale(abs(response[:,:,0]-response[:,:,1])[mask]),
        A_scale=scale(data['A']),K_scale=scale(data['K']))
    for kind in GEOMETRIES:state[kind+'_std']=np.maximum(raw_features(data,state,kind).std(0),.005)
    return state

def features(data,state,kind):
    x=np.clip(raw_features(data,state,kind)/state[kind+'_std'],-16,16)
    if x.shape!=(len(data['clinical']),32) or not np.isfinite(x).all():raise ValueError('Invalid patient coordinates')
    return x

def pairs(y,groups,all_pairs):
    rng=np.random.default_rng(195);centers=np.unique(groups);design=[]
    for c in centers:
        for d in (centers if all_pairs else [c]):
            p=np.flatnonzero((groups==c)&(y==1));n=np.flatnonzero((groups==d)&(y==0))
            if not len(p) or not len(n):continue
            if len(p)*len(n)<=2048:p,n=np.repeat(p,len(n)),np.tile(n,len(p))
            else:p,n=rng.choice(p,2048),rng.choice(n,2048)
            design.append((p,n))
    if not design:raise ValueError('No labeled source patient pairs')
    return design

def objective(w,x,c,y,groups,design,lam):
    z=c+RHO*(x@w);loss=0.;dz=np.zeros(len(z))
    for p,n in design:
        margin=z[p]-z[n];loss+=np.logaddexp(0,-margin).mean()/len(design)
        v=-expit(-margin)/(len(p)*len(design));np.add.at(dz,p,v);np.add.at(dz,n,-v)
    centers=np.unique(groups)
    for center in centers:
        at=groups==center;weight=.1/(at.sum()*len(centers))
        loss+=weight*np.sum(np.logaddexp(0,z[at])-y[at]*z[at]);dz[at]+=weight*(expit(z[at])-y[at])
    return float(loss+.5*lam*(w@w)),RHO*x.T@dz+lam*w

def fit_head(data,y,groups,state,family,lam):
    kind,mode=family.split('__');x=features(data,state,kind);design=pairs(y,groups,mode=='ALL_PAIRS')
    opt=minimize(lambda w:objective(w,x,data['clinical'],y,groups,design,lam),np.zeros(32),jac=True,
        method='L-BFGS-B',options=dict(maxiter=1000,maxls=100,ftol=1e-11,gtol=1e-7))
    if not np.isfinite(opt.fun) or (not opt.success and abs(opt.jac).max()>1e-4):raise RuntimeError('Patient readout convergence failed')
    return dict(family=family,w=opt.x.tolist(),lambda_=float(lam),rho=RHO,parameters=32,
                gradient_max=float(abs(opt.jac).max()),optimizer_success=bool(opt.success))

def predict(head,data,state):
    return data['clinical']+RHO*(features(data,state,head['family'].split('__')[0])@np.asarray(head['w']))

def metrics(y,z):
    p=expit(z)
    return dict(N=int(len(y)),AUROC=float(roc_auc_score(y,p)),AUPRC=float(average_precision_score(y,p)),
        NLL=float(np.mean(np.logaddexp(0,z)-y*z)),Brier=float(np.mean((p-y)**2)))

def center_auc(y,z,g):
    return float(np.mean([roc_auc_score(y[g==c],z[g==c]) for c in np.unique(g) if len(np.unique(y[g==c]))==2]))

def train_select(data,y,g,selection,sy,sg):
    state=fit_state(data);heads={};scores={}
    for family in FAMILIES:
        menu=[]
        for lam in LAMBDAS:
            h=fit_head(data,y,g,state,family,lam);z=predict(h,selection,state)
            menu.append(dict(head=h,**metrics(sy,z),center_AUROC=center_auc(sy,z,sg)))
        best=max(menu,key=lambda r:(r['AUROC'],-r['NLL'],r['head']['lambda_']))
        heads[family]=best['head'];heads[family]['selection_menu']=[{k:v for k,v in row.items() if k!='head'} for row in menu]
        scores[family]={k:v for k,v in best.items() if k!='head'}
    return state,heads,scores
