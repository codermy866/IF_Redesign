import runtime
from runtime import HERE,BASE,save,sha,verify,config
import argparse,json,time,numpy as np,torch,joblib
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from data import prepare,load_patients
from recent_training import fit,predict,cfg
from recent_models import make
from pathlib import Path
PARTS={}
def bags(part,at,track,device):
    if track=='A':x=part['h'];scale=None
    else:
        x=np.load(BASE/'199_final_core_confirmation_20261013/private/frame_hidden.npy',mmap_mode='r')
        cache=HERE/'private/raw_scales'/(sha_array(part['train'])+'.npy');cache.parent.mkdir(parents=True,exist_ok=True)
        if cache.exists():scale=np.load(cache)
        else:
            tr=part['train'];sums=np.zeros(x.shape[-1]);squares=np.zeros(x.shape[-1]);count=0
            for start in range(0,len(tr),8):
                idx=tr[start:start+8];v=np.asarray(x[idx],dtype=np.float64)[part['valid'][idx]];sums+=v.sum(0);squares+=(v*v).sum(0);count+=len(v)
            scale=np.maximum(np.sqrt(np.maximum(squares/count-(sums/count)**2,0)),.005);np.save(cache,scale)
    out=[]
    for i in at:
        values=np.asarray(x[i],dtype=np.float32)[part['valid'][i]]
        if scale is not None:values=values/scale.astype(np.float32)
        out.append(torch.tensor(values,dtype=torch.float32,device=device))
    anchor=torch.tensor(part['anchor']['anchor'][at],dtype=torch.float32,device=device).detach()
    return out,anchor
def sha_array(a):
    import hashlib
    return hashlib.sha256(np.asarray(a,dtype=np.int64).tobytes()).hexdigest()[:24]
def verify_recent():
    verify();m=json.loads((HERE/'RECENT_MANIFEST.json').read_text())
    for p,digest in m['execution_files'].items():assert sha(p)==digest,'Recent adapter changed: '+Path(p).name
def run(outer,kind,track,device):
    verify_recent();torch.set_num_threads(1);c=cfg();rows=load_patients();y=np.array([r['y'] for r in rows]);g=np.array([r['center'] for r in rows]);name='TRACK_'+track+'_'+kind
    folder=HERE/'private/recent_runs'/outer;folder.mkdir(parents=True,exist_ok=True)
    if (folder/(name+'.npz')).exists():return
    helds=sorted(set(g[g!=outer]));at=np.concatenate([np.flatnonzero(g==h) for h in helds]);choices=[]
    # Caches are private; source fit patients are excluded from both held domains.
    for wd in c['source_weight_decay_candidates']:
        z=np.full((3,len(rows)),np.nan);a=np.full(len(rows),np.nan);epochs=[];seconds=0.;capped=0
        for si,seed in enumerate(c['seeds']):
            for held in helds:
                verify_recent();part=prepare(outer,held);tr=part['train'];va=np.flatnonzero(g==held)
                assert not np.isin(tr,np.flatnonzero((g==held)|(g==outer))).any()
                key=f'{name}_{held}_wd{wd}_seed{seed}';path=HERE/'private/recent_cases'/outer/(key+'.pt');path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():ck=torch.load(path,map_location='cpu',weights_only=False)
                else:
                    train,ta=bags(part,tr,track,device);val,va_a=bags(part,va,track,device)
                    resume=path.with_suffix('.resume.pt')
                    model,md=fit(kind,train,val,ta,va_a,y[tr],g[tr],y[va],wd,seed,resume_path=resume)
                    ck=dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},metadata=md,prediction=predict(model,val,va_a),anchor=va_a.cpu().numpy(),train=tr,validation=va)
                    torch.save(ck,path);resume.unlink(missing_ok=True)
                    del train,val,ta,va_a,model;torch.cuda.empty_cache()
                assert np.array_equal(ck['train'],tr) and np.array_equal(ck['validation'],va)
                z[si,va]=ck['prediction'];a[va]=ck['anchor'];epochs.append(ck['metadata']['best_epoch']);seconds+=ck['metadata']['fit_seconds'];capped+=int(ck['metadata']['cap_reached_still_improving'])
                save(HERE/f'{name}_{outer}_PROGRESS.json',dict(held_source=held,seed=seed,weight_decay=wd,best_epoch=ck['metadata']['best_epoch'],source_only=True,time=time.time()))
                print('SOURCE_CASE',name,outer,held,wd,seed,flush=True)
        pp=expit(z[:,at]).mean(0);zz=np.log(np.clip(pp,1e-12,None)/np.clip(1-pp,1e-12,None));nl=[];br=[]
        for held in helds:
            mask=g[at]==held;nl.append(float(np.mean(np.logaddexp(0,zz[mask])-y[at][mask]*zz[mask])));br.append(float(np.mean((pp[mask]-y[at][mask])**2)))
        choices.append(dict(weight_decay=wd,mean_source_center_NLL=float(np.mean(nl)),mean_source_center_Brier=float(np.mean(br)),source_AUROC=float(roc_auc_score(y[at],pp)),source_indices=at,source_seed_logits=z[:,at],source_anchor_logits=a[at],fixed_refit_epochs=int(np.median(epochs)),fit_seconds=seconds,source_fit_caps_still_improving=capped))
    selected=min(choices,key=lambda x:(x['mean_source_center_NLL'],x['mean_source_center_Brier'],-x['source_AUROC']))
    save(folder/(name+'_SOURCE_CONFIGURATIONS.json'),choices);save(folder/(name+'_SELECTION_LOCK.json'),selected|dict(source_only=True))
    part=prepare(outer,'outer');tr=part['train'];te=np.flatnonzero(g==outer);train,ta=bags(part,tr,track,device);test,te_a=bags(part,te,track,device);logits=[];metadata=[]
    for seed in c['seeds']:
        path=folder/f'{name}_seed{seed}_OUTER.pt'
        if path.exists():
            ck=torch.load(path,map_location='cpu',weights_only=False);model=make(kind,train[0].shape[-1]).to(device);model.load_state_dict(ck['state']);md=ck['metadata']
        else:
            model,md=fit(kind,train,None,ta,None,y[tr],g[tr],None,selected['weight_decay'],seed,fixed_epochs=selected['fixed_refit_epochs'],resume_path=path.with_suffix('.resume.pt'))
            torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},metadata=md,train=tr),path)
        logits.append(predict(model,test,te_a));metadata.append(md)
    np.savez_compressed(folder/(name+'.npz'),test=te,seed_logits=np.stack(logits),anchor_logits=te_a.cpu().numpy(),source_indices=selected['source_indices'],source_seed_logits=selected['source_seed_logits'],source_y=y[selected['source_indices']],source_g=g[selected['source_indices']],source_anchor_logits=selected['source_anchor_logits'])
    save(folder/(name+'_REFIT_DIAGNOSTICS.json'),metadata);print('RECENT_OUTER_COMPLETE',outer,name,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--outer',required=True);p.add_argument('--kind',required=True);p.add_argument('--track',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();run(a.outer,a.kind,a.track,a.device)
