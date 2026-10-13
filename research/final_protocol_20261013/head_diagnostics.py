"""Scorer/kernel weight diagnostics; no phenotype localization claims."""
import runtime
from runtime import HERE,config,save
import numpy as np,pandas as pd,joblib,torch,json
from rime_dataset import load_patients
from factorial_inputs import tensor_inputs,shuffled_scoring_context
from scorers import ScoreControl
from core_heads import restore

def main():
    torch.set_num_threads(2);rows=load_patients();g=np.array([r['center'] for r in rows]);records=[]
    for outer in sorted(set(g)):
        part=joblib.load(HERE/'private/fold_features'/outer/'outer.joblib');tr=np.flatnonzero(g!=outer);te=np.flatnonzero(g==outer)
        folder=HERE/'private/head_runs'/outer;methods=['D_RIME_LAPLACE','B2_GEOMETRY','B2_LOCAL','B2_LOCAL_CAPACITY_MATCHED','B2_CONTEXT_SHUFFLED']
        methods+=[n for n in ['B5_VALUE_AUXILIARY','B5_SHUFFLED_AUXILIARY','B5_STRATIFIED_SHUFFLED','B5_PATIENT_AUXILIARY'] if (folder/(n+'.npz')).exists()]
        for seed in config()['seeds']:
            baseline=None
            for method in methods:
                path=folder/f'{method}_seed{seed}_OUTER.pt'
                if not path.exists():continue
                checkpoint=torch.load(path,map_location='cpu',weights_only=False);md=checkpoint['metadata'];mode=md.get('score_mode','conditional')
                head=ScoreControl(part['anchor']['b'].shape[1],md['tau'],'measure','laplace',mode).double();head.load_state_dict(checkpoint['state']);head.eval()
                inp=tensor_inputs(part,te,'D_RIME_LAPLACE',md['tau'],'cpu',rows)
                if mode=='context_shuffled':
                    train=tensor_inputs(part,tr,'D_RIME_LAPLACE',md['tau'],'cpu',rows);_,inp,_=shuffled_scoring_context(train,inp,seed)
                with torch.no_grad():z,details=head(**{k:v for k,v in inp.items() if k!='indices'},return_details=True)
                q,w,alpha=[details[n].numpy() for n in ['q','w','alpha']];vm=inp['valid'].numpy();site_valid=vm.any(-1)
                if method=='D_RIME_LAPLACE':baseline=(q,w,alpha)
                phi=details['v'].numpy()
                for position,index in enumerate(te):
                    for site in np.flatnonzero(site_valid[position]):
                        mask=vm[position,site];weights=w[position,site,mask];scores=q[position,site,mask]
                        record=dict(key=rows[index]['key'],center=outer,method=method,seed=seed,site=int(site),atoms=int(mask.sum()),
                            score_mean=float(scores.mean()),score_std=float(scores.std()),score_min=float(scores.min()),score_max=float(scores.max()),
                            weight_entropy=float(-(weights*np.log(np.clip(weights,1e-300,None))).sum()),effective_atoms=float(1/np.square(weights).sum()),
                            site_attention=float(alpha[position,site]),signed_readout_norm=float(np.linalg.norm(phi[position,site])),
                            beta_norm=float(head.beta.norm()),evidence_scope='latent scores/weights; not clinical pathology truth')
                        if baseline is not None:record.update(frame_weight_L1_vs_no_aux=float(abs(w[position,site]-baseline[1][position,site]).sum()),
                            site_attention_change_vs_no_aux=float(alpha[position,site]-baseline[2][position,site]))
                        records.append(record)
    frame=pd.DataFrame(records);frame.to_parquet(HERE/'private/SCORING_WEIGHT_DIAGNOSTICS.parquet',index=False)
    numeric=['score_mean','score_std','weight_entropy','effective_atoms','site_attention','beta_norm','frame_weight_L1_vs_no_aux','site_attention_change_vs_no_aux']
    summary=frame.groupby(['method','center','seed'])[numeric].mean().reset_index();summary.to_csv(HERE/'TABLE_SCORING_WEIGHT_DIAGNOSTICS.csv',index=False)
    save(HERE/'SCORING_WEIGHT_DIAGNOSTICS_COMPLETE.json',dict(state='COMPLETE',patient_count=len(rows),site_records_are_not_independent_patients=True,source_context_shuffle_does_not_change_anchor=True))

if __name__=='__main__':main()
