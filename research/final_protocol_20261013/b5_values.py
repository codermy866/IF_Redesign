"""Outcome-dependent targets from recipient-excluded, mask-adapted reference g.

g is a fixed small linear reference, not a new RIME architecture. Its native
patient hidden/context and visible OCT summary are recomputed for site deletion.
Supervision estimates predictive Brier advantage under this reference only.
"""
import runtime
from runtime import HERE,BASE,config,save,sha
import time,json,hashlib
import numpy as np,joblib
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from rime_dataset import load_patients,fit_projection,project,old_run
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
        hidden=hidden,native=native,frame_hidden=np.load(HERE/'private/frame_hidden.npy',mmap_mode='r'),
        valid=np.load(HERE/'private/frame_mask.npy'))

def teacher(a,source):
    source=np.asarray(source);cfg=config()
    binding=dict(source=source.tolist(),labels=a['y'][source].tolist(),config_sha256=sha(HERE/'CONFIG.json'),
        code_sha256=sha(__file__),input_sha256=sha(HERE/'INPUT_LOCK.json'))
    key=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest();path=HERE/'private/b5_teachers'/(key+'.joblib')
    if path.exists():return joblib.load(path)
    start=time.monotonic()
    if len(source)<27:raise RuntimeError('Fewer than 27 teacher-training patients for intact semantic PCA26')
    frame=old_run.clinical_frame(a['rows']);preprocess=old_run.fit_prior(frame,a['y'],source).named_steps['preprocess']
    cx=preprocess.transform(frame)
    if hasattr(cx,'toarray'):cx=cx.toarray()
    state=old_run.semantic_state(a['hidden'][source,0],np.ones((len(source),1),bool))
    semantic=old_run.semantic_features(a['hidden'].reshape(-1,5120),np.ones((len(a['rows'])*13,1),bool),state).reshape(len(a['rows']),13,26)
    pca=fit_projection(a['frame_hidden'],a['valid'],source);h=project(a['frame_hidden'],a['valid'],pca)
    sites=(h*a['valid'][...,None]).sum(-2)/a['valid'].sum(-1).clip(1)[...,None];available=a['valid'].any(-1)
    raw=[]
    for condition in range(13):
        mask=available.copy()
        if condition:mask[:,condition-1]=False
        octmean=(sites*mask[...,None]).sum(1)/mask.sum(1).clip(1)[:,None]
        raw.append(np.column_stack([semantic[:,condition],octmean,cx,mask.sum(1)/12,a['native'][:,condition]]))
    raw=np.stack(raw,1);mean=raw[source].mean((0,1));scale=np.maximum(raw[source].std((0,1)),.005)
    x=np.clip((raw-mean)/scale,-16,16);groups=a['groups'][source];centers=np.unique(groups)
    weights=np.array([len(source)/(len(centers)*np.sum(groups==c)) for c in groups]);weights=np.repeat(weights,13)
    # Each patient's 13 conditions receive equal total weight, not 13 patients.
    model=LogisticRegression(C=1/(cfg['B5_teacher_lambda']*len(source)*13),max_iter=2000,random_state=17)
    model.fit(x[source].reshape(-1,x.shape[-1]),np.repeat(a['y'][source],13),sample_weight=weights)
    if model.n_iter_.max()>=2000:raise RuntimeError('B5 reference teacher did not converge')
    prediction=model.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(len(a['rows']),13)
    result=dict(prediction=prediction,train=source,binding=binding,model=model,semantic_state=state,local_projection=pca,
        clinical_preprocess=preprocess,feature_mean=mean,feature_scale=scale,seconds=time.monotonic()-start,
        patients=len(source),training_conditions=13*len(source),effective_parameters=x.shape[-1]+1,
        scope='source-only linear teacher, native full/site-removed contexts and visible OCT summary; teacher dependent predictive advantage')
    path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(result,path);return result

def advantage(prediction,y):
    return (y[:,None]-prediction[:,1:])**2-(y[:,None]-prediction[:,:1])**2

def targets(a,source,destination):
    source=np.asarray(source);g=a['groups'];values=np.zeros((len(a['rows']),12));audit=[]
    if len(set(g[source]))<2:raise RuntimeError('At least two source centres required for crossfit')
    for held in sorted(set(g[source])):
        fit=source[g[source]!=held];val=source[g[source]==held]
        if set(fit)&set(val):raise RuntimeError('Teacher recipient in fit')
        ref=teacher(a,fit);assert np.array_equal(ref['train'],fit)
        values[val]=advantage(ref['prediction'][val],a['y'][val])
        audit.append(dict(teacher_train=fit,preprocessing_train=fit,recipients=val,intersection=[],held_source_center=held,
            native_context='complete prefix recomputed separately for each site removal',seconds=ref['seconds'],training_patient_count=len(fit)))
    if np.max(np.abs(values[source]))>1+1e-12:raise RuntimeError('Illegal Brier difference')
    save(destination,dict(legal_source=source,audit=audit,negative_fraction=float((values[source]<0).mean()),
        mean=float(values[source].mean()),std=float(values[source].std()),patient_total_weights_equal=True,
        training_conditions_are_not_independent_patients=True,independent_confirmation=False))
    return values

def shuffle(values,source,g,seed):
    values=values.copy();rng=np.random.default_rng(seed);records=[]
    for center in sorted(set(g[source])):
        at=source[g[source]==center]
        if len(at)<2:raise RuntimeError('Cannot shuffle across patients within centre')
        for site in range(12):
            order=rng.permutation(at);donor=np.roll(order,1);values[order,site]=values[donor,site]
            records.append(dict(center=center,site=site,recipient=order,donor=donor))
    return values,records
