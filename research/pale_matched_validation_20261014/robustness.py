"""Fixed-anchor masking, distinct from full native input deletion."""
import runtime
from runtime import HERE,config,save
import json,hashlib,time
import numpy as np,pandas as pd,torch
from scipy.special import expit,logit
from data import load_patients,prepare,inputs
from heads import make_head
from report import metrics,auc

def main():
    torch.set_num_threads(1);cfg=config();rows=load_patients();g=np.array([r['center'] for r in rows]);y=np.array([r['y'] for r in rows]);predictions={};threshold=np.zeros(len(rows));start=time.monotonic()
    plan_path=HERE/'ROBUSTNESS_PLAN.json'
    if not plan_path.exists():
        save(plan_path,dict(frozen_unix=time.time(),frame_budgets=cfg['frame_budgets'],site_budgets=cfg['site_budgets'],repeats=3,
            seed=201,sampling='SHA256(mask201:repeat:patient_key), nested permutations within sites and site set; outcomes never used',
            trained_models_fixed=True,thresholds_fixed_from_full_input_source_OOF=True,anchor_fixed=True,native_deletion_requires_separate_anchor_recomputation=True))
    for c in cfg['centers']:
        te=np.flatnonzero(g==c);part=prepare(c,'outer');inp=inputs(part,te,'cpu');models=[]
        with np.load(HERE/'private/runs'/c/'P3_PALE.npz') as f:
            source=logit(np.clip(expit(f['source_seed_logits']).mean(0),1e-12,1-1e-12));threshold[te]=np.quantile(source[f['source_y']==1],.05,method='lower')
        for seed in cfg['seeds']:
            ck=torch.load(HERE/'private/runs'/c/f'P3_PALE_seed{seed}_OUTER.pt',map_location='cpu',weights_only=False)
            model=make_head(inp['b'].shape[-1],cfg['variants']['P3_PALE']).double();model.load_state_dict(ck['state']);model.eval();models.append(model)
        for repeat in range(3):
            frame_order=[];site_order=[]
            for i in te:
                seed=int(hashlib.sha256(('mask201:'+str(repeat)+':'+rows[i]['key']).encode()).hexdigest()[:8],16)
                rng=np.random.default_rng(seed);frame_order.append(np.stack([rng.permutation(10) for _ in range(12)]));site_order.append(rng.permutation(12))
            frame_order=np.stack(frame_order);site_order=np.stack(site_order)
            for kind,budgets in [('frames',cfg['frame_budgets']),('sites',cfg['site_budgets'])]:
                for budget in budgets:
                    mask=np.zeros_like(part['valid'][te])
                    if kind=='frames':
                        for i in range(len(te)):
                            for s in range(12):mask[i,s,frame_order[i,s,:budget]]=True
                    else:
                        for i in range(len(te)):mask[i,site_order[i,:budget]]=True
                    mask &=part['valid'][te];changed=inp|dict(valid=torch.tensor(mask,dtype=torch.bool))
                    with torch.no_grad():p=np.stack([expit(m(**changed).numpy()) for m in models]).mean(0)
                    if budget==0:np.testing.assert_allclose(p,expit(part['anchor']['anchor'][te]),atol=1e-12)
                    key=(kind,budget,repeat)
                    if key not in predictions:predictions[key]=np.zeros(len(rows))
                    predictions[key][te]=p
    records=[]
    for (kind,budget,repeat),p in predictions.items():
        for c in ['POOLED']+cfg['centers']:
            at=np.ones(len(rows),bool) if c=='POOLED' else g==c
            records.append(dict(budget_type=kind,budget=budget,repeat=repeat,center=c,N=int(at.sum()),**metrics(y[at],p[at],threshold[at]),
                macro_AUROC=float(np.mean([auc(y[g==cc],p[g==cc]) for cc in cfg['centers']])),scope='fixed-anchor local-branch masking, not native full-system deletion'))
    pd.DataFrame(records).to_csv(HERE/'tables/TABLE_FIXED_ANCHOR_ROBUSTNESS.csv',index=False)
    np.savez_compressed(HERE/'private/FIXED_ANCHOR_MASKING.npz',**{f'{k[0]}_{k[1]}_{k[2]}':p for k,p in predictions.items()})
    save(HERE/'FIXED_ANCHOR_ROBUSTNESS_COMPLETE.json',dict(state='COMPLETE',patients=len(rows),conditions=len(predictions),seconds=time.monotonic()-start,
        scope='fixed-anchor only',full_native_deletion_complete=False))
    print('Fixed-anchor masks complete; full native deletion remains a separate task',flush=True)
if __name__=='__main__':main()
