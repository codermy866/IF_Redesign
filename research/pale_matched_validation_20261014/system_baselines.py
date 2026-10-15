"""Modality/system references, kept separate from local mechanism ablations."""
import runtime
from runtime import HERE,BASE,config,save,sha
import numpy as np,torch,json,time,copy,joblib
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from data import prepare,load_patients

def logits(part,y,g,lam,method,seed,native_colpo,validation=None,fixed_epochs=None):
    tr=part['train']
    clinical=part['anchor']['prior'].decision_function(__import__('rime_dataset').old_run.clinical_frame(load_patients()))
    if method=='SYS_CLINICAL':
        return clinical,dict(parameters=None,deterministic=True)
    vm=part['valid'];h=part['h']
    oct=(h*vm[...,None]).sum((1,2))/vm.sum((1,2)).clip(1)[:,None]
    col=native_colpo[:,None];col=(col-col[tr].mean(0))/np.maximum(col[tr].std(0),.005)
    weights=np.array([len(tr)/(len(set(g[tr]))*(g[tr]==c).sum()) for c in g[tr]])
    def linear(x):
        model=LogisticRegression(C=1/(lam*len(tr)),max_iter=3000,random_state=seed)
        model.fit(x[tr],y[tr],sample_weight=weights);assert model.n_iter_.max()<3000
        return model.decision_function(x),model
    if method in ['SYS_OCT','SYS_COLPO']:
        z,model=linear(oct if method=='SYS_OCT' else col)
        return z,dict(parameters=model.coef_.size+1,deterministic=True,state=model)
    if method.startswith('SYS_LATE'):
        zo,_=linear(oct);zc,_=linear(col);x=np.column_stack([clinical,zo,zc])
        if method=='SYS_LATE_PROB':x=expit(x)
        z,model=linear(x)
        return z,dict(parameters=4,deterministic=True,state=model,base_fit_scope='source in-sample; held-source excluded from all fits')
    x=np.column_stack([part['anchor']['b'][:,26:],oct,col]);xx=torch.tensor(x,dtype=torch.double);torch.manual_seed(seed)
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],16),torch.nn.Tanh(),torch.nn.Linear(16,1)).double()
    optimizer=torch.optim.Adam(model.parameters(),lr=.003);yy=torch.tensor(y[tr],dtype=torch.double);gg=g[tr]
    best=None;bad=0;curve=[];limit=config()['max_epochs'] if fixed_epochs is None else fixed_epochs
    for epoch in range(limit+1):
        if epoch:
            optimizer.zero_grad();z=model(xx[tr]).squeeze(-1);objective=z.new_zeros(())
            for c in sorted(set(gg)):
                zz=z[gg==c];label=yy[gg==c];objective+=torch.nn.functional.binary_cross_entropy_with_logits(zz,label)/len(set(gg))
                pos=zz[label==1];neg=zz[label==0]
                if len(pos) and len(neg):objective+=.1*torch.nn.functional.softplus(-(pos[:,None]-neg[None,:])).mean()/len(set(gg))
            objective+=lam*sum(p.square().sum() for p in model.parameters())
            objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if validation is not None:
            with torch.no_grad():z=model(xx[validation]).squeeze(-1).numpy()
            nll=float(np.mean(np.logaddexp(0,z)-y[validation]*z));brier=float(np.mean((expit(z)-y[validation])**2));auc=float(roc_auc_score(y[validation],z));score=(-nll,-brier,auc,-epoch)
            curve.append(dict(epoch=epoch,NLL=nll,Brier=brier,AUROC=auc))
            if best is None or score>best[0]:best=(score,copy.deepcopy(model.state_dict()),epoch);bad=0
            else:bad+=1
            if bad>=config()['patience']:break
        elif epoch==limit:best=(None,copy.deepcopy(model.state_dict()),epoch)
    model.load_state_dict(best[1]);model.eval()
    with torch.no_grad():z=model(xx).squeeze(-1).numpy()
    return z,dict(parameters=sum(p.numel() for p in model.parameters()),best_epoch=best[2],curve=curve,deterministic=False,state=model.state_dict())

def main():
    torch.set_num_threads(1);cfg=config();rows=load_patients();y=np.array([r['y'] for r in rows]);g=np.array([r['center'] for r in rows])
    cache=json.loads((BASE/'196_controlled_cfv_jev_20261012/private/CACHE_INDEX.json').read_text());col=[]
    for row in rows:
        item=cache[row['key']];assert sha(item['path'])==item['sha256']
        with np.load(item['path']) as f:col.append(float(f['only_k'][0]))
    col=np.array(col)
    methods=['SYS_CLINICAL','SYS_OCT','SYS_COLPO','SYS_CONCAT_MLP','SYS_LATE_LOGIT','SYS_LATE_PROB']
    save(HERE/'SYSTEM_BASELINE_SPEC.json',dict(frozen_unix=time.time(),methods=methods,Clinical='Existing legal source clinical prior',OCT='FR1 source-fit8D independent-frame patient mean logistic head',Colpo='Released JEV colposcopy-only decision calibrated on source patients',Concat='Source clinical plus8D OCT plus colpo decision; fixed hidden16 MLP',Late='Source clinical/OCT/colpo predictions, logistic fusion; three coupled lambda choices',limitations='Unequal modality representation dimensions; no backbone-superiority claim. Linear seed outputs are deterministic.'))
    for outer in cfg['centers']:
        folder=HERE/'private/system_runs'/outer;folder.mkdir(parents=True,exist_ok=True)
        source=np.flatnonzero(g!=outer);test=np.flatnonzero(g==outer);helds=sorted(set(g[source]))
        for method in methods:
            if (folder/(method+'.npz')).exists():continue
            start=time.monotonic();candidates=[]
            for lam in ([.03] if method=='SYS_CLINICAL' else cfg['lambdas']):
                z=np.full((3,len(rows)),np.nan);epochs=[]
                for si,seed in enumerate(cfg['seeds']):
                    for held in helds:
                        part=prepare(outer,held);va=np.flatnonzero(g==held)
                        zz,md=logits(part,y,g,lam,method,seed,col,validation=va if method=='SYS_CONCAT_MLP' else None)
                        z[si,va]=zz[va];epochs.append(md.get('best_epoch',0))
                pp=expit(z[:,source]).mean(0);nl=[];br=[]
                for c in helds:
                    at=g[source]==c;zz=np.log(np.clip(pp[at],1e-12,None)/np.clip(1-pp[at],1e-12,None))
                    nl.append(float(np.mean(np.logaddexp(0,zz)-y[source][at]*zz)));br.append(float(np.mean((pp[at]-y[source][at])**2)))
                candidates.append(dict(lambda_=lam,source_seed_logits=z[:,source],mean_source_center_NLL=float(np.mean(nl)),mean_source_center_Brier=float(np.mean(br)),source_AUROC=float(roc_auc_score(y[source],pp)),refit_epochs=int(np.median(epochs))))
            chosen=min(candidates,key=lambda r:(r['mean_source_center_NLL'],r['mean_source_center_Brier'],-r['source_AUROC']))
            save(folder/(method+'_SOURCE_CONFIGURATIONS.json'),candidates);save(folder/(method+'_SELECTION_LOCK.json'),chosen|dict(source_only=True))
            part=prepare(outer,'outer');zz=[];meta=[]
            for seed in cfg['seeds']:
                z,md=logits(part,y,g,chosen['lambda_'],method,seed,col,fixed_epochs=chosen['refit_epochs'] if method=='SYS_CONCAT_MLP' else None)
                zz.append(z[test]);meta.append({k:v for k,v in md.items() if k!='state'})
                joblib.dump(dict(model_state=md.get('state'),train=part['train'],source_only=True),folder/f'{method}_seed{seed}_MODEL.joblib')
            np.savez_compressed(folder/(method+'.npz'),test=test,seed_logits=np.stack(zz),source_indices=source,source_seed_logits=chosen['source_seed_logits'],source_y=y[source],source_g=g[source])
            save(folder/(method+'_DIAGNOSTICS.json'),dict(seconds=time.monotonic()-start,seeds=meta))
            print('SYSTEM_OUTER_COMPLETE',outer,method,flush=True)
    save(HERE/'SYSTEM_BASELINES_COMPLETE.json',dict(state='COMPLETE',methods=methods,time=time.time()))
if __name__=='__main__':main()
