"""Source-fitted frame projection and exact-atom preparation for RIME.

The projection consumes all arrays for convenience, but fits only the explicit
source indices. Input hidden contrasts must already subtract the task gray
reference. Scales must remain frozen when processing held-out patients.
"""
import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd
from .readout import coordinates

def fit_projection(x,valid,train):
    values=np.asarray(x[train]);vm=valid[train]
    count=vm.sum(-1);sites=(count>0).sum(-1)
    weight=vm/np.maximum(count,1)[...,None]/np.maximum(sites,1)[:,None,None]/len(train)
    flat=values[vm].astype(np.float32);weights=weight[vm]
    mean=np.sum(flat*weights[:,None],axis=0,dtype=np.float64)
    # Covariance is centred for basis fitting; inference remains zero-anchored.
    centered=(flat-mean).astype(np.float32)*np.sqrt(weights*len(flat))[:,None].astype(np.float32)
    _,_,vt=randomized_svd(centered,n_components=8,n_iter=3,random_state=17,flip_sign=False)
    basis=vt.T.astype(np.float64)
    for k in range(8):
        if basis[np.argmax(abs(basis[:,k])),k]<0:basis[:,k]*=-1
    projected=flat@basis
    mu=np.sum(projected*weights[:,None],axis=0)
    scale=np.maximum(np.sqrt(np.sum((projected-mu)**2*weights[:,None],axis=0)),.005)
    return dict(basis=basis,scale=scale,training_patients=train,fit_centering=mean,
        gray_zero_anchored_mapping=True,balanced_by_patient_and_site=True)

def project(x,valid,state):
    result=np.zeros((*valid.shape,8),dtype=np.float64)
    for i in range(len(x)):
        result[i]=np.clip(np.asarray(x[i])@state['basis']/state['scale'],-3,3)/3
    return np.where(valid[...,None],result,0.)

def source_bandwidth(h,valid,train):
    # Patient-level medians give each patient equal influence, independent of bags.
    medians=[]
    for i in train:
        ds=[]
        for s in range(h.shape[1]):
            a=h[i,s,valid[i,s]]
            if len(a)>1:
                d=np.linalg.norm(a[:,None]-a[None,:],axis=-1)[np.triu_indices(len(a),1)]
                ds.extend(d[d>0].tolist())
        if ds:medians.append(np.median(ds))
    return max(float(np.median(medians)),.001) if medians else .001

def atoms(h,valid,sigma,kind='unique',threshold=.3):
    from scipy.cluster.hierarchy import fcluster,linkage
    out=np.zeros_like(h);mask=np.zeros_like(valid)
    for i in range(len(h)):
        for s in range(h.shape[1]):
            a=h[i,s,valid[i,s]]
            if not len(a):continue
            if kind=='unique':a=np.unique(a,axis=0)
            elif kind=='cluster' and len(a)>1:
                labels=fcluster(linkage(a,method='complete'),threshold*sigma,criterion='distance')
                representatives=[]
                for c in np.unique(labels):
                    v=a[labels==c];representatives.append(v[np.argmin(np.linalg.norm(v-v.mean(0),axis=1))])
                a=np.stack(representatives)
            out[i,s,:len(a)]=a;mask[i,s,:len(a)]=True
    return out,mask

def coordinate_scale(h,valid,source_indices,readout='laplace'):
    """Fit the common readout scale before changing any pooling method."""
    tr=np.asarray(source_indices,dtype=int)
    if not len(tr):raise ValueError('Coordinate scaling needs source patients')
    with torch.no_grad():
        raw=coordinates(torch.as_tensor(h,dtype=torch.float64),readout).numpy()
    count=valid[tr].sum(-1);sites=(count>0).sum(-1)
    weight=valid[tr]/np.maximum(count,1)[...,None]/np.maximum(sites,1)[:,None,None]/len(tr)
    mean=(raw[tr]*weight[...,None]).sum((0,1,2))
    return np.maximum(np.sqrt(((raw[tr]-mean)**2*weight[...,None]).sum((0,1,2))),.005)

def prepare_inputs(h,valid,b,anchor,sigma,*,source_indices=None,
                   fitted_scale=None,indices=None,pooling='measure',
                   readout='laplace',deduplicate=None,device='cpu',
                   semantic_x=None,initial_semantic_weight=None):
    """Prepare caller-owned tensors; the returned dictionary feeds train_head.

    Supply source_indices to fit scales or fitted_scale to reuse a frozen
    source scale. Deduplication happens after fitting the shared scale.
    The lower-level RIMEHead assumes exact duplicates have already been merged
    for kernel-measure pooling; this wrapper performs that operation.
    """
    h=np.asarray(h,dtype=np.float64);valid=np.asarray(valid,dtype=bool)
    b=np.asarray(b,dtype=np.float64);anchor=np.asarray(anchor,dtype=np.float64)
    if h.ndim!=4 or h.shape[-1]!=8 or valid.shape!=h.shape[:-1]:
        raise ValueError('Expected h[N,sites,frames,8] and valid[N,sites,frames]')
    if not h.shape[1] or not h.shape[2]:raise ValueError('Use a masked slot for empty bags')
    if b.ndim!=2 or len(b)!=len(h) or anchor.shape!=(len(h),):
        raise ValueError('Context and anchor must be patient aligned')
    if sigma<=0 or not np.isfinite(sigma):raise ValueError('Bandwidth must be positive and source-frozen')
    if not all(np.isfinite(v).all() for v in (h,b,anchor)):
        raise ValueError('Nonfinite inputs; impute source clinical inputs before calling')
    if fitted_scale is None:
        if source_indices is None:raise ValueError('Provide source_indices or a frozen fitted_scale')
        fitted_scale=coordinate_scale(h,valid,source_indices,readout)
    scale=np.asarray(fitted_scale,dtype=np.float64)
    if scale.shape!=(56,) or not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Expected 56 positive finite coordinate scales')
    if deduplicate is None:deduplicate=pooling in ('measure','geometry')
    if deduplicate:h,valid=atoms(h,valid,sigma,'unique')
    difference=h[:,:,:,None,:]-h[:,:,None,:,:]
    kernel=np.exp(-np.sum(difference*difference,-1)/(2*sigma*sigma))
    at=np.arange(len(h)) if indices is None else np.asarray(indices,dtype=int)
    tensor=lambda v:torch.as_tensor(v,dtype=torch.float64,device=device)
    result=dict(h=tensor(h[at]),k=tensor(kernel[at]),
        valid=torch.as_tensor(valid[at],dtype=torch.bool,device=device),
        b=tensor(b[at]),anchor=tensor(anchor[at]),coordinate_scale=tensor(scale),indices=at)
    if semantic_x is not None:result['semantic_x']=tensor(np.asarray(semantic_x)[at])
    if initial_semantic_weight is not None:result['initial_semantic_weight']=tensor(initial_semantic_weight)
    return result
