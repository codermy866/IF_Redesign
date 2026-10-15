"""Recipient-excluded RIME reference with mask-adapted full native contexts.

Every teacher preprocessing fit, anchor and patient loss excludes recipients.
Remaining independent frame encodings are reused; native prefix, conditional
scores, kernel weights and site aggregation are recomputed for every deletion.
"""
import runtime
from runtime import HERE,BASE,config,save,sha
import time,json,hashlib
import numpy as np,torch,joblib
from scipy.special import expit
from rime_dataset import load_patients,fit_projection,project,source_bandwidth,atoms,old_run
from if_redesign.rime.readout import RIMEHead,coordinates
from if_redesign.rime.training import loss
from b5_native import token

def arrays():
    rows=load_patients();index=json.loads((BASE/'196_controlled_cfv_jev_20261012/private/CACHE_INDEX.json').read_text())
    hidden=np.zeros((len(rows),13,5120));native=np.zeros((len(rows),13))
    for i,row in enumerate(rows):
        item=index[row['key']];assert sha(item['path'])==item['sha256']
        with np.load(item['path'],allow_pickle=False) as f:hidden[i,0]=f['hidden'][0];native[i,0]=f['A'][0]
        for s in range(12):
            name,identity=token(row['key'],s)
            with np.load(HERE/'private/b5_native'/(name+'.npz'),allow_pickle=False) as f:
                assert json.loads(str(f['identity']))==json.loads(json.dumps(identity,sort_keys=True))
                hidden[i,s+1]=f['hidden'];native[i,s+1]=f['A']
    return dict(rows=rows,y=np.array([r['y'] for r in rows]),groups=np.array([r['center'] for r in rows]),
        hidden=hidden,native=native,frame_hidden=np.load(HERE/'private/frame_hidden.npy',mmap_mode='r'),valid=np.load(HERE/'private/frame_mask.npy'))

def teacher(a,source,seed=17):
    source=np.asarray(source);cfg=config()
    binding=dict(source=source.tolist(),labels=a['y'][source].tolist(),seed=seed,config_sha256=sha(HERE/'CONFIG.json'),code_sha256=sha(__file__),input_sha256=sha(HERE/'INPUT_LOCK.json'))
    key=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest();path=HERE/'private/b5_teachers'/(key+'.joblib')
    if path.exists():return joblib.load(path)
    start=time.monotonic();torch.manual_seed(seed)
    if len(source)<27:raise RuntimeError('Fewer than 27 source patients for intact semantic PCA26')
    frame=old_run.clinical_frame(a['rows']);prior=old_run.fit_prior(frame,a['y'],source);cx=prior.named_steps['preprocess'].transform(frame)
    if hasattr(cx,'toarray'):cx=cx.toarray()
    state=old_run.semantic_state(a['hidden'][source,0],np.ones((len(source),1),bool))
    semantic=old_run.semantic_features(a['hidden'].reshape(-1,5120),np.ones((len(a['rows'])*13,1),bool),state).reshape(len(a['rows']),13,26)
    clinical=prior.decision_function(frame);ascale=old_run.native_scale(a['native'][source,0]);alpha=.25
    base=clinical[:,None]+alpha*a['native']/ascale
    semantic_head=old_run.fit_head_from_matrix(semantic[source,0],base[source,0],a['y'][source],a['groups'][source],.003)
    anchor=base+.25*semantic@semantic_head['w']
    context=np.concatenate([semantic,np.broadcast_to(cx[:,None,:],(len(cx),13,cx.shape[1]))],-1)
    context_scale=np.maximum(context[source,0].std(0),.005);context=np.clip(context/context_scale,-16,16)
    pca=fit_projection(a['frame_hidden'],a['valid'],source);raw=project(a['frame_hidden'],a['valid'],pca);sigma=source_bandwidth(raw,a['valid'],source)
    fit_valid=a['valid'][source];count=fit_valid.sum(-1);sites=(count>0).sum(-1)
    weights=fit_valid/count.clip(1)[...,None]/sites.clip(1)[:,None,None]/len(source)
    with torch.no_grad():phi=coordinates(torch.tensor(raw[source],dtype=torch.double),'laplace').numpy()
    mean=(phi*weights[...,None]).sum((0,1,2));scale=np.maximum(np.sqrt(((phi-mean)**2*weights[...,None]).sum((0,1,2))),.005)
    h,valid=atoms(raw,a['valid'],sigma,'unique');difference=h[...,None,:]-h[...,None,:,:]
    kernel=np.exp(-np.sum(difference*difference,-1)/(2*sigma*sigma))
    model=RIMEHead(context.shape[-1],cfg['B5_teacher_tau'],'measure','laplace').double()
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    yy=torch.tensor(a['y'][source],dtype=torch.double);groups=a['groups'][source]
    hs=torch.tensor(h[source],dtype=torch.double);ks=torch.tensor(kernel[source],dtype=torch.double);sc=torch.tensor(scale,dtype=torch.double)
    schedule=np.random.default_rng(seed).permutation(13);curve=[]
    for epoch in range(cfg['B5_teacher_epochs']):
        condition=int(schedule[epoch%13]);vm=valid[source].copy()
        if condition:vm[:,condition-1]=False
        inp=dict(h=hs,k=ks,valid=torch.tensor(vm),b=torch.tensor(context[source,condition]),anchor=torch.tensor(anchor[source,condition]),coordinate_scale=sc)
        optimizer.zero_grad();z=model(**inp);objective=loss(z,yy,groups,model,cfg['B5_teacher_lambda'])
        if not torch.isfinite(objective):raise RuntimeError('Nonfinite RIME reference teacher objective')
        objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        curve.append(dict(epoch=epoch+1,condition=condition,train_objective=float(objective.detach())))
    model.eval();prediction=np.zeros((len(a['rows']),13))
    with torch.no_grad():
        for condition in range(13):
            for start_batch in range(0,len(h),256):
                at=np.arange(start_batch,min(start_batch+256,len(h)));vm=valid[at].copy()
                if condition:vm[:,condition-1]=False
                z=model(h=torch.tensor(h[at]),k=torch.tensor(kernel[at]),valid=torch.tensor(vm),b=torch.tensor(context[at,condition]),anchor=torch.tensor(anchor[at,condition]),coordinate_scale=sc)
                prediction[at,condition]=torch.sigmoid(z).numpy()
    result=dict(prediction=prediction,train=source,binding=binding,state={k:v.detach().cpu() for k,v in model.state_dict().items()},
        semantic_state=state,semantic_head=semantic_head,prior=prior,native_scale=ascale,alpha=alpha,context_scale=context_scale,
        local_projection=pca,sigma=sigma,coordinate_scale=scale,seconds=time.monotonic()-start,patients=len(source),
        training_conditions=13*len(source),epochs=cfg['B5_teacher_epochs'],mask_cycle=schedule,curve=curve,
        effective_parameters=sum(p.numel() for p in model.parameters()),scope='fixed-hyperparameter source-only RIME teacher; actual native full/site-removed context, conditional measure and site aggregation rerun; independent frame encoder reuse',
        patient_condition_weight='one condition per patient per update; all13 conditions in fixed source-blind cycle; centre-balanced patient objective, no condition denominator claimed as patients')
    path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(result,path);return result

def advantage(prediction,y):return (y[:,None]-prediction[:,1:])**2-(y[:,None]-prediction[:,:1])**2

def targets(a,source,destination,seed=17):
    source=np.asarray(source);g=a['groups'];values=np.zeros((len(a['rows']),12));audit=[]
    if len(set(g[source]))<2:raise RuntimeError('At least two source centres required for crossfit')
    for held in sorted(set(g[source])):
        fit=source[g[source]!=held];val=source[g[source]==held]
        if set(fit)&set(val):raise RuntimeError('Teacher recipient in fit')
        ref=teacher(a,fit,seed);assert np.array_equal(ref['train'],fit)
        values[val]=advantage(ref['prediction'][val],a['y'][val])
        audit.append(dict(teacher_train=fit,preprocessing_train=fit,recipients=val,intersection=[],held_source_center=held,seed=seed,
            native_context='complete prefix rerun; RIME anchor/context/conditional measure/site pooling recomputed',seconds=ref['seconds'],training_patient_count=len(fit),teacher_epochs=ref['epochs']))
    if np.max(np.abs(values[source]))>1+1e-12:raise RuntimeError('Illegal Brier difference')
    save(destination,dict(legal_source=source,audit=audit,negative_fraction=float((values[source]<0).mean()),mean=float(values[source].mean()),std=float(values[source].std()),
        auxiliary_patient_total_weights_equal=True,training_conditions_are_not_independent_patients=True,independent_confirmation=False))
    return values

def shuffle(values,source,g,seed,labels=None):
    values=values.copy();original=values.copy();rng=np.random.default_rng(seed);records=[]
    strata=[(center,None) for center in sorted(set(g[source]))] if labels is None else [(center,label) for center in sorted(set(g[source])) for label in [0,1]]
    for center,label in strata:
        at=source[g[source]==center] if label is None else source[(g[source]==center)&(labels[source]==label)]
        if len(at)<2:raise RuntimeError('Cannot shuffle across distinct patients in this prespecified source stratum')
        for site in range(12):
            order=rng.permutation(at);donor=np.roll(order,1);values[order,site]=original[donor,site]
            records.append(dict(center=center,label=label,site=site,recipient=order,donor=donor))
    return values,records
