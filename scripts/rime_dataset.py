"""Fold-specific, patient/site-balanced frame PCA and unchanged semantic anchors."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4';os.umask(0o077)
import sys,json,hashlib
import numpy as np
import torch
import joblib
from sklearn.utils.extmath import randomized_svd
from rime_common import *
sys.path.insert(0,str(OLD))
import run_factorial as old_run
import core as old_core

def load_patients():
    lock=json.loads((DEST/'INPUT_LOCK.json').read_text());assert sha(lock['cohort_path'])==lock['cohort_sha256']
    return sorted(json.loads(Path(lock['cohort_path']).read_text()),key=lambda r:r['key'])

def pack_frames():
    rows=load_patients();output=DEST/'private/frame_hidden.npy';maskpath=DEST/'private/frame_mask.npy'
    if (DEST/'FEATURE_MANIFEST.json').exists():
        m=json.loads((DEST/'FEATURE_MANIFEST.json').read_text())
        assert sha(output)==m['packed_sha256'];return
    x=np.lib.format.open_memmap(output,mode='w+',dtype='float32',shape=(len(rows),12,10,5120))
    valid=np.zeros((len(rows),12,10),bool);indices=[]
    for i,row in enumerate(rows):
        for s in range(1,13):
            paths=row['oct_sites'].get(str(s),[])
            if len(paths)>10:raise RuntimeError('Unexpected frame count; audit before modifying protocol')
            for t,path in enumerate(paths):
                f,identity=frame_identity(path)
                with np.load(f,allow_pickle=False) as cache:
                    assert json.loads(str(cache['identity']))==identity
                    x[i,s-1,t]=cache['hidden'];valid[i,s-1,t]=True
                indices.append(dict(patient=i,site=s-1,frame=t,cache=f.name,sha256=sha(f)))
        if i%50==0:print('PACK',i,len(rows),flush=True)
    x.flush();np.save(maskpath,valid)
    save(DEST/'private/PACKED_FRAME_INDEX.json',indices)
    save(DEST/'FEATURE_MANIFEST.json',dict(state='COMPLETE',N=len(rows),frames=int(valid.sum()),
        hidden_dimension=5120,packed_sha256=sha(output),mask_sha256=sha(maskpath),
        index_sha256=sha(DEST/'private/PACKED_FRAME_INDEX.json'),
        input_lock_sha256=sha(DEST/'INPUT_LOCK.json'),label_free=True,
        gray_subtracted=True,local_template='20-slot independent reread',dtype='float32'))

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
        for s in range(12):
            a=h[i,s,valid[i,s]]
            if len(a)>1:
                d=np.linalg.norm(a[:,None]-a[None,:],axis=-1)[np.triu_indices(len(a),1)]
                ds.extend(d[d>0].tolist())
        if ds:medians.append(np.median(ds))
    return max(float(np.median(medians)),.001) if medians else .001

def build_anchor(rows,hidden,mask,A,y,g,train,original_outer=None):
    frame=old_run.clinical_frame(rows)
    if original_outer is not None:
        folds=joblib.load(OLD/'five_center/private/SOURCE_FROZEN_MODELS.joblib')
        fold=folds[original_outer];assert np.array_equal(fold['train'],train)
        prior=fold['prior'];alpha=fold['base_alpha'];ascale=fold['A_scale'];sem=fold['semantic']
    else:
        cv=[];alpha_grid=[0.,.25,.5,1.];base_oof={a:np.zeros(len(rows)) for a in alpha_grid}
        for held in sorted(set(g[train])):
            tr=train[g[train]!=held];va=train[g[train]==held]
            prior=old_run.fit_prior(frame,y,tr);ct=prior.decision_function(frame.iloc[tr]);cvv=prior.decision_function(frame.iloc[va])
            scale=old_run.native_scale(A[tr]);state=old_run.semantic_state(hidden[tr],mask[tr])
            xt=old_run.semantic_features(hidden[tr],mask[tr],state);xv=old_run.semantic_features(hidden[va],mask[va],state)
            for alpha in alpha_grid:base_oof[alpha][va]=cvv+alpha*A[va]/scale
            cv.append((tr,va,ct,cvv,scale,state,xt,xv))
        alpha=max(alpha_grid,key=lambda a:(old_core.metrics(y[train],base_oof[a][train])['AUROC'],-old_core.metrics(y[train],base_oof[a][train])['NLL'],-a))
        choices=[]
        for lam in [.003,.03,.3]:
            oof=np.zeros(len(rows))
            for tr,va,ct,cvv,scale,state,xt,xv in cv:
                head=old_run.fit_head_from_matrix(xt,ct+alpha*A[tr]/scale,y[tr],g[tr],lam)
                oof[va]=cvv+alpha*A[va]/scale+.25*xv@head['w']
            choices.append((old_core.metrics(y[train],oof[train])['AUROC'],-old_core.metrics(y[train],oof[train])['NLL'],lam))
        lam=max(choices)[2];prior=old_run.fit_prior(frame,y,train);ascale=old_run.native_scale(A[train])
        st=old_run.semantic_state(hidden[train],mask[train]);xx=old_run.semantic_features(hidden[train],mask[train],st)
        cl=prior.decision_function(frame)
        sem=dict(state=st,head=old_run.fit_head_from_matrix(xx[train],cl[train]+alpha*A[train]/ascale,y[train],g[train],lam))
    clinical=prior.decision_function(frame);xx=old_run.semantic_features(hidden,mask,sem['state'])
    anchor=clinical+alpha*A/ascale+.25*xx@sem['head']['w']
    clinical_x=prior.named_steps['preprocess'].transform(frame)
    if hasattr(clinical_x,'toarray'):clinical_x=clinical_x.toarray()
    b=np.column_stack([xx,clinical_x]);bscale=np.maximum(b[train].std(0),.005);b=np.clip(b/bscale,-16,16)
    return dict(prior=prior,alpha=alpha,A_scale=ascale,semantic=sem,context_scale=bscale,anchor=anchor,b=b,train=train,
        semantic_x=xx,initial_semantic_weight=.25*sem['head']['w'])

def prepare_fold(outer):
    target=DEST/'private/fold_features'/outer
    if (target/'COMPLETE.json').exists():return
    target.mkdir(parents=True,exist_ok=True)
    rows=load_patients();y=np.array([r['y'] for r in rows]);g=np.array([r['center'] for r in rows])
    x=np.load(DEST/'private/frame_hidden.npy',mmap_mode='r');valid=np.load(DEST/'private/frame_mask.npy')
    oldindex=json.loads((OLD/'private/CACHE_INDEX.json').read_text());hidden=[];mask=[];A=[]
    for row in rows:
        assert sha(oldindex[row['key']]['path'])==oldindex[row['key']]['sha256'],'Original patient cache changed'
        with np.load(oldindex[row['key']]['path']) as f:hidden.append(f['hidden'][0]);mask.append(f['mask']);A.append(f['A'][0])
    hidden=np.stack(hidden);mask=np.stack(mask);A=np.array(A)
    source=np.flatnonzero(g!=outer);parts={'outer':source}
    parts.update({c:source[g[source]!=c] for c in sorted(set(g[source]))})
    for name,tr in parts.items():
        path=target/(name+'.joblib')
        if path.exists():continue
        projection=fit_projection(x,valid,tr);h=project(x,valid,projection);sigma=source_bandwidth(h,valid,tr)
        anchor=build_anchor(rows,hidden,mask,A,y,g,tr,outer if name=='outer' else None)
        joblib.dump(dict(h=h,valid=valid,sigma=sigma,projection=projection,anchor=anchor,train=tr),path)
        print('SOURCE_FRAME_PCA',outer,name,'patients',len(tr),flush=True)
    save(target/'COMPLETE.json',dict(state='COMPLETE',outer=outer,source_only=True,parts=list(parts)))

def grouping(h,valid,rows,kind='real'):
    if kind=='random':
        h=h.copy();valid=valid.copy()
        for i,row in enumerate(rows):
            seed=int(hashlib.sha256(('rime_groups_17'+row['key']).encode()).hexdigest()[:8],16)
            perm=np.random.default_rng(seed).permutation(120)
            h[i]=h[i].reshape(120,8)[perm].reshape(12,10,8)
            valid[i]=valid[i].reshape(120)[perm].reshape(12,10)
    return h,valid

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

def tensor_inputs(part,indices,method,tau,device='cpu',rows=None,frame_budget=10):
    from if_redesign.rime.readout import coordinates
    h=part['h'];valid=part['valid'].copy()
    if frame_budget<10:
        chosen=np.array([0,9,3,6,1,8,2,7,4,5][:frame_budget])
        keep=np.zeros(10,bool);keep[chosen]=True;valid &=keep
    if method=='RANDOM_GROUPING_D':h,valid=grouping(h,valid,rows,'random')
    if method in ['C_MEASURE_MOMENT','D_RIME_LAPLACE','GEOMETRY_LAPLACE','IDENTITY_DEDUP_LAPLACE','NO_CLINICAL_CONTEXT','RANDOM_GROUPING_D']:
        h,valid=atoms(h,valid,part['sigma'],'unique')
    elif method=='B3_CLUSTER_LAPLACE':h,valid=atoms(h,valid,part['sigma'],'cluster',tau)
    elif method=='FLAT_GLOBAL_DEDUP':
        flat=h.reshape(len(h),1,120,8);vm=valid.reshape(len(h),1,120)
        h,valid=atoms(flat,vm,part['sigma'],'unique')
    d=h[:,:, :,None,:]-h[:,:,None,:,:]
    k=np.exp(-np.sum(d*d,-1)/(2*part['sigma']**2))
    ht=torch.tensor(h,dtype=torch.float64,device=device)
    readout='moment' if method.endswith('MOMENT') else 'laplace'
    # Same source normalization for a readout irrespective of pooler; equal
    # patient/site weighting follows the frame basis estimation.
    # Fit one common coordinate scale on original frame features, rather than
    # changing feature scaling with each aggregation/unique-atom operation.
    common_valid=part['valid'].copy()
    if frame_budget<10:common_valid &=keep
    with torch.no_grad():
        raw=coordinates(torch.tensor(part['h'],dtype=torch.float64),readout).numpy()
    tr=part['train'];count=common_valid[tr].sum(-1);sites=(count>0).sum(-1)
    weight=common_valid[tr]/np.maximum(count,1)[...,None]/np.maximum(sites,1)[:,None,None]/len(tr)
    mean=(raw[tr]*weight[...,None]).sum((0,1,2))
    scale=np.maximum(np.sqrt(((raw[tr]-mean)**2*weight[...,None]).sum((0,1,2))),.005)
    b=part['anchor']['b'].copy()
    if method=='NO_CLINICAL_CONTEXT':b[:,26:]=0
    return dict(h=ht[indices],k=torch.tensor(k[indices],dtype=torch.float64,device=device),
        valid=torch.tensor(valid[indices],dtype=torch.bool,device=device),b=torch.tensor(b[indices],dtype=torch.float64,device=device),
        anchor=torch.tensor(part['anchor']['anchor'][indices],dtype=torch.float64,device=device),
        coordinate_scale=torch.tensor(scale,dtype=torch.float64,device=device),
        semantic_x=torch.tensor(part['anchor']['semantic_x'][indices],dtype=torch.float64,device=device),
        initial_semantic_weight=torch.tensor(part['anchor']['initial_semantic_weight'],dtype=torch.float64,device=device),indices=indices)
