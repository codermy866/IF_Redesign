"""Full native input sensitivity on the declared200-patient subset only."""
import runtime
from runtime import HERE,config,save
import numpy as np,pandas as pd,json,torch
from scipy.special import expit,logit
from data import load_patients,prepare,inputs
from heads import make_head
from report import metrics,auc
from rime_dataset import old_run

def main():
    torch.set_num_threads(1);rows=load_patients();lookup={r['key']:i for i,r in enumerate(rows)}
    jobs=json.loads((HERE/'private/NATIVE_ROBUSTNESS_QUEUE.json').read_text());missing=sum(not (HERE/'private/native_robustness'/(j['token']+'.npz')).exists() for j in jobs)
    if missing:
        save(HERE/'NATIVE_SUMMARY_STATE.json',dict(state='INCOMPLETE',missing_conditions=missing,patient_denominator=200,not_full984_evaluation=True));return
    parts={};models={};records=[]
    for c in config()['centers']:
        part=prepare(c,'outer');parts[c]=part;models[c]=[]
        for seed in config()['seeds']:
            ck=torch.load(HERE/'private/runs'/c/f'P3_PALE_seed{seed}_OUTER.pt',map_location='cpu',weights_only=False)
            model=make_head(part['anchor']['b'].shape[-1],config()['variants']['P3_PALE']).double();model.load_state_dict(ck['state']);model.eval();models[c].append(model)
    for job in jobs:
        i=lookup[job['key']];row=rows[i];part=parts[job['center']];anchor=part['anchor'];keep=np.array(job['keep'],bool)
        with np.load(HERE/'private/native_robustness'/(job['token']+'.npz')) as f:hidden=f['hidden'];A=float(f['A'])
        any_visual=bool(keep.any() or row['colpo_paths'])
        sem=old_run.semantic_features(hidden[None,:],np.array([[any_visual]]),anchor['semantic']['state'])
        clinicalframe=old_run.clinical_frame([row]);clinical=float(anchor['prior'].decision_function(clinicalframe)[0])
        z0=clinical+anchor['alpha']*A/anchor['A_scale']+.25*float((sem@anchor['semantic']['head']['w']).item())
        cx=anchor['prior'].named_steps['preprocess'].transform(clinicalframe)
        if hasattr(cx,'toarray'):cx=cx.toarray()
        updated_context=np.clip(np.column_stack([sem,cx])/anchor['context_scale'],-16,16)
        base=inputs(part,[i],'cpu')
        changed=base|dict(valid=torch.tensor(keep[None]&part['valid'][i:i+1],dtype=torch.bool),
            anchor=torch.tensor([z0],dtype=torch.double),b=torch.tensor(updated_context,dtype=torch.double))
        with torch.no_grad():prob=float(np.mean([expit(m(**changed).item()) for m in models[job['center']]]))
        records.append(dict(key=job['key'],center=job['center'],budget_type=job['budget_type'],budget=job['budget'],repeat=job['repeat'],y=row['y'],probability=prob,updated_anchor_probability=float(expit(z0)),inclusion_probability=job['inclusion_probability'],full_native_context_updated=True))
    df=pd.DataFrame(records);df.to_parquet(HERE/'private/NATIVE_PATIENT_PREDICTIONS.parquet',index=False);table=[]
    for (kind,budget,repeat),group in df.groupby(['budget_type','budget','repeat']):
        for center in ['POOLED']+config()['centers']:
            block=group if center=='POOLED' else group[group.center==center]
            # Patient-level subset endpoints; population-weighted expected loss
            # reported separately from unweighted subset AUROC.
            y=block.y.to_numpy();p=block.probability.to_numpy();z=logit(np.clip(p,1e-12,1-1e-12));w=1/block.inclusion_probability.to_numpy()
            table.append(dict(budget_type=kind,budget=budget,repeat=repeat,center=center,N=len(block),AUROC=auc(y,p),AUPRC=__import__('sklearn.metrics',fromlist=['average_precision_score']).average_precision_score(y,p),NLL=float(np.mean(np.logaddexp(0,z)-y*z)),design_weighted_NLL=float(np.average(np.logaddexp(0,z)-y*z,weights=w)),Brier=float(np.mean((p-y)**2)),scope='Full native context recomputed, predeclared200-patient subset; not full984 or causal effect'))
    pd.DataFrame(table).to_csv(HERE/'tables/TABLE_NATIVE_ROBUSTNESS.csv',index=False)
    save(HERE/'NATIVE_SUMMARY_STATE.json',dict(state='COMPLETE',patients=200,scope='Input sensitivity only; no clinical causality',conditions=len(jobs)))
    print('Full native sensitivity report complete on200 patients',flush=True)
if __name__=='__main__':main()
