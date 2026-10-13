"""Fixed-context feature diagnostics, separate from complete native interventions."""
import runtime
from runtime import HERE,ORIGINAL,save,config,verify
import hashlib,json,time
import numpy as np,torch,pandas as pd,joblib
from scipy.special import expit
from rime_dataset import load_patients,tensor_inputs,atoms
from core_heads import restore,METHODS

def changed(original,part,h,valid,method):
    if method in ['C_MEASURE_MOMENT','D_RIME_LAPLACE']:h,valid=atoms(h,valid,part['sigma'],'unique')
    delta=h[...,None,:]-h[...,None,:,:];kernel=np.exp(-np.sum(delta*delta,-1)/(2*part['sigma']**2))
    return {**original,'h':torch.tensor(h,dtype=torch.double),'k':torch.tensor(kernel,dtype=torch.double),'valid':torch.tensor(valid,dtype=torch.bool)}

def main():
    verify();torch.set_num_threads(2);rows=load_patients();lookup={r['key']:i for i,r in enumerate(rows)}
    subset=json.loads((ORIGINAL/'private/MECHANISM_PATIENTS.json').read_text());records=[];start=time.monotonic()
    for outer in sorted({r['center'] for r in rows}):
        part=joblib.load(HERE/'private/fold_features'/outer/'outer.joblib')
        cases=[r for r in subset if r['center']==outer];indices=np.array([lookup[r['key']] for r in cases]);cache={}
        for method in METHODS:
            for seed in config()['seeds']:
                checkpoint=torch.load(HERE/'private/head_runs'/outer/f'{method}_seed{seed}_OUTER.pt',map_location='cpu',weights_only=False)
                model=restore(checkpoint,part['anchor']['b'].shape[1],'cpu')
                original=tensor_inputs(part,indices,method,checkpoint['metadata']['tau'],'cpu',rows)
                cache[method,seed]=(model,{k:v for k,v in original.items() if k!='indices'})
        for position,case in enumerate(cases):
            i=lookup[case['key']];h=part['h'][i:i+1];vm=part['valid'][i:i+1]
            rng=np.random.default_rng(int(hashlib.sha256(('robust199'+case['key']).encode()).hexdigest()[:8],16))
            site=int(rng.integers(12));frame=int(rng.choice(np.flatnonzero(vm[0,site])))
            conditions=[]
            perm=np.stack([rng.permutation(h.shape[2]) for _ in range(12)])
            hh=np.stack([h[0,s,perm[s]] for s in range(12)])[None];vv=np.stack([vm[0,s,perm[s]] for s in range(12)])[None]
            conditions.append(('within_site_permutation',hh,vv))
            duplicate=np.pad(h,((0,0),(0,0),(0,2),(0,0)));valid=np.pad(vm,((0,0),(0,0),(0,2)))
            duplicate[0,site,-2:]=h[0,site,frame];valid[0,site,-2:]=True
            conditions.append(('exact_copy',duplicate,valid))
            for magnitude in [.01,.05,.1]:
                noisy=duplicate.copy();noisy[0,site,-2:]=np.clip(noisy[0,site,-2:]+rng.normal(0,magnitude*part['sigma'],(2,8)),-1,1)
                conditions.append((f'near_copy_feature_noise_{magnitude}',noisy,valid.copy()))
            noisy=h.copy();noisy[0,site,frame]=np.clip(noisy[0,site,frame]+rng.normal(0,.3*part['sigma'],8),-1,1)
            conditions.append(('single_frame_feature_noise',noisy,vm.copy()))
            missing=vm.copy();missing[:,site]=False;conditions.append(('local_only_missing_site',h.copy(),missing))
            for method in METHODS:
                for seed in config()['seeds']:
                    model,batch=cache[method,seed]
                    original={k:(v if k in ['coordinate_scale','initial_semantic_weight'] else v[position:position+1]) for k,v in batch.items()}
                    with torch.no_grad():full=float(model(**original));beta_norm=float(model.beta.norm())
                    for condition,hh,vv in conditions:
                        record=dict(key=case['key'],center=outer,method=method,seed=seed,condition=condition,
                            beta_norm=beta_norm,fixed_native_context=True,original_anchor_unchanged=True,clinical_native_robustness=False)
                        try:
                            altered=changed(original,part,hh,vv,method)
                            with torch.no_grad():z=float(model(**altered))
                            record.update(state='COMPLETE',probability_drift=abs(float(expit(z)-expit(full))),logit_drift=abs(z-full),error=None)
                        except (RuntimeError,torch.linalg.LinAlgError) as error:record.update(state='NUMERICAL_FAILURE',probability_drift=None,logit_drift=None,error=str(error))
                        records.append(record)
            save(HERE/'ROBUSTNESS_STATE.json',dict(state='RUNNING',last_center=outer,records=len(records),seconds=time.monotonic()-start))
    frame=pd.DataFrame(records);frame.to_parquet(HERE/'private/FEATURE_ROBUSTNESS.parquet',index=False)
    aggregate=[]
    for (method,condition),group in frame.groupby(['method','condition']):
        aggregate.append(dict(method=method,condition=condition,patients=group.key.nunique(),seeds=3,numerical_failures=int((group.state!='COMPLETE').sum()),
            mean_probability_drift=float(group.probability_drift.mean()),maximum_probability_drift=float(group.probability_drift.max()),
            nonzero_residual_patients=int(group[group.beta_norm>1e-10].key.nunique()),scope='fixed-native-context feature diagnostic, not complete native/image perturbation or clinical causal value'))
    pd.DataFrame(aggregate).to_csv(HERE/'TABLE_FEATURE_ROBUSTNESS.csv',index=False)
    save(HERE/'ROBUSTNESS_COMPLETE.json',dict(state='COMPLETE_WITH_RECORDED_FAILURES',patients=len(subset),seconds=time.monotonic()-start,full_native_results_reported_separately=True))

if __name__=='__main__':main()
