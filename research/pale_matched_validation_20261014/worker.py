import runtime
from runtime import HERE,PREVIOUS,save,config,verify,sha
import argparse,time,json,hashlib,shutil
import numpy as np,torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from data import prepare,inputs,source_scale,load_patients
from training import fit
from heads import make_head
PARTS={}
def part_for(outer,held):
    key=(outer,held)
    if key not in PARTS:PARTS[key]=prepare(outer,held)
    return PARTS[key]
def case(outer,held,method,lam,seed,device,y,g):
    spec=config()['variants'][method];part=part_for(outer,held);tr=part['train'];va=np.flatnonzero(g==held)
    assert not np.isin(tr,np.flatnonzero((g==outer)|(g==held))).any()
    key=hashlib.sha256(json.dumps(dict(method=method,lambda_=lam,seed=seed,held=held),sort_keys=True).encode()).hexdigest()[:24]
    path=HERE/'private/cases'/outer/(method+'_'+key+'.pt')
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert np.array_equal(saved['train'],tr) and np.array_equal(saved['validation'],va)
        return saved
    scale=source_scale(part,spec);ti=inputs(part,tr,device,scale);vi=inputs(part,va,device,scale)
    model,md=fit(ti,vi,y[tr],g[tr],y[va],lam,seed,spec)
    with torch.no_grad():z=model(**vi).cpu().numpy()
    result=dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},metadata=md,
        prediction=z,train=tr,validation=va,scale=scale,anchor=part['anchor']['anchor'][va],
        config_sha256=sha(HERE/'CONFIG.json'),source_only=True)
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(result,path)
    save(HERE/f'{outer}_PROGRESS.json',dict(method=method,held_source=held,lambda_=lam,seed=seed,epoch=md['best_epoch'],pid=__import__('os').getpid(),time=time.time()))
    return result
def select(outer,method,device,y,g):
    helds=sorted(set(g[g!=outer]));at=np.concatenate([np.flatnonzero(g==h) for h in helds]);records=[]
    for lam in config()['lambdas']:
        logits=np.full((3,len(y)),np.nan);anchors=np.full(len(y),np.nan);epochs=[];seconds=0.
        for si,seed in enumerate(config()['seeds']):
            for held in helds:
                c=case(outer,held,method,lam,seed,device,y,g);va=c['validation']
                logits[si,va]=c['prediction'];anchors[va]=c['anchor'];epochs.append(c['metadata']['best_epoch']);seconds+=c['metadata']['fit_seconds']
        prob=expit(logits[:,at]).mean(0);z=np.log(np.clip(prob,1e-12,None)/np.clip(1-prob,1e-12,None))
        nll=[];brier=[]
        for held in helds:
            mask=g[at]==held;nll.append(float(np.mean(np.logaddexp(0,z[mask])-y[at][mask]*z[mask])));brier.append(float(np.mean((prob[mask]-y[at][mask])**2)))
        records.append(dict(lambda_=lam,mean_source_center_NLL=float(np.mean(nll)),mean_source_center_Brier=float(np.mean(brier)),source_AUROC=float(roc_auc_score(y[at],prob)),
            fixed_refit_epochs=int(np.median(epochs)),source_indices=at,source_seed_logits=logits[:,at],source_anchor_logits=anchors[at],fit_seconds=seconds))
    best=min(records,key=lambda r:(r['mean_source_center_NLL'],r['mean_source_center_Brier'],-r['source_AUROC'],r['fixed_refit_epochs']))
    return best,records
def run(outer,phase,device):
    verify();torch.set_num_threads(1);cfg=config();rows=load_patients();y=np.array([r['y'] for r in rows]);g=np.array([r['center'] for r in rows])
    folder=HERE/'private/runs'/outer;folder.mkdir(parents=True,exist_ok=True)
    methods=cfg['core'] if phase=='core' else cfg['extension']
    for method in methods:
        verify()
        if (folder/(method+'.npz')).exists():continue
        if method=='P3_PALE':
            assert json.loads((HERE/'PREFLIGHT_GATE.json').read_text())['state']=='PASS'
            old=PREVIOUS/'private/runs'/outer
            for source,target in [('A0_PALE.npz',method+'.npz'),('A0_PALE_REFIT_DIAGNOSTICS.json',method+'_REFIT_DIAGNOSTICS.json'),('A0_PALE_SELECTION_LOCK.json',method+'_SELECTION_LOCK.json')]:shutil.copy2(old/source,folder/target)
            for seed in cfg['seeds']:shutil.copy2(old/f'A0_PALE_seed{seed}_OUTER.pt',folder/f'{method}_seed{seed}_OUTER.pt')
            save(folder/(method+'_REUSE_RECEIPT.json'),dict(source=str(old),sha256=sha(old/'A0_PALE.npz'),exact_forward_reproduction=True,training_budget_and_selection_identical=True));continue
        selected,allconfigs=select(outer,method,device,y,g)
        # Lock source choice before creating any target prediction.
        save(folder/(method+'_SOURCE_CONFIGURATIONS.json'),allconfigs)
        save(folder/(method+'_SELECTION_LOCK.json'),selected|dict(selection=cfg['selection'],outer_labels_used=False))
        part=part_for(outer,'outer');tr=part['train'];te=np.flatnonzero(g==outer);scale=source_scale(part,cfg['variants'][method]);ti=inputs(part,tr,device,scale);vi=inputs(part,te,device,scale)
        logits=[];metadata=[]
        for seed in cfg['seeds']:
            path=folder/f'{method}_seed{seed}_OUTER.pt'
            if path.exists():
                ck=torch.load(path,map_location='cpu',weights_only=False);assert np.array_equal(ck['train'],tr)
                model=make_head(ti['b'].shape[-1],cfg['variants'][method]).to(ti['h']);model.load_state_dict(ck['state']);model.eval();md=ck['metadata']
            else:
                model,md=fit(ti,None,y[tr],g[tr],None,selected['lambda_'],seed,cfg['variants'][method],fixed_epochs=selected['fixed_refit_epochs'])
                torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},metadata=md,train=tr,scale=scale,source_selection_frozen=True),path)
            with torch.no_grad():logits.append(model(**vi).cpu().numpy())
            metadata.append(md)
        np.savez_compressed(folder/(method+'.npz'),test=te,seed_logits=np.stack(logits),anchor_logits=part['anchor']['anchor'][te],
            source_indices=selected['source_indices'],source_seed_logits=selected['source_seed_logits'],source_y=y[selected['source_indices']],source_g=g[selected['source_indices']],source_anchor_logits=selected['source_anchor_logits'])
        save(folder/(method+'_REFIT_DIAGNOSTICS.json'),metadata)
        print('OUTER_COMPLETE',outer,method,flush=True)
    save(folder/(phase.upper()+'_COMPLETE.json'),dict(state='COMPLETE',methods=methods,outer=outer,seeds=cfg['seeds']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--outer',required=True);p.add_argument('--phase',choices=['core','extension'],required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();run(a.outer,a.phase,a.device)
