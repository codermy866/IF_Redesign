import runtime
from runtime import HERE,OLD,RECOVERY,BASE,save,sha,config
import json,joblib,numpy as np,torch,pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from rime_dataset import load_patients,tensor_inputs
from if_redesign.rime.readout import RIMEHead
from data import prepare

def main():
    torch.set_num_threads(4);rows=load_patients();y=np.array([r['y'] for r in rows]);g=np.array([r['center'] for r in rows])
    assert len(set(r['key'] for r in rows))==len(rows)
    lock=json.loads((HERE/'INPUT_LOCK.json').read_text());assert sha(lock['cohort_path'])==lock['cohort_sha256']
    p=np.full((3,len(rows)),np.nan);audits=[]
    for outer in sorted(set(g)):
        te=np.flatnonzero(g==outer);tr=np.flatnonzero(g!=outer)
        folder=RECOVERY/'private/head_runs'/outer;part=joblib.load(OLD/'private/fold_features'/outer/'outer.joblib')
        inp=tensor_inputs(part,te,'B1_MLP',.3,'cpu',rows)
        with np.load(folder/'B1_MLP.npz') as f:
            assert np.array_equal(f['test'],te) and np.array_equal(f['source_indices'],tr) and np.array_equal(f['source_y'],y[tr])
            for k,seed in enumerate([17,29,43]):
                ckpt=torch.load(folder/f'B1_MLP_seed{seed}_OUTER.pt',map_location='cpu',weights_only=False)
                assert np.array_equal(ckpt['train'],tr) and ckpt['metadata']['seed']==seed
                head=RIMEHead(inp['b'].shape[1],.3,'mlp','laplace').double();head.load_state_dict(ckpt['state']);head.eval()
                with torch.no_grad():z=head(**{k:v for k,v in inp.items() if k!='indices'}).numpy()
                error=float(abs(z-f['seed_logits'][k]).max());assert error<1e-8
                p[k,te]=expit(z);audits.append(dict(outer=outer,seed=seed,max_saved_logit_error=error,patients=len(te),labels_and_splits_match=True))
    pp=p.mean(0);zz=np.log(np.clip(pp,1e-12,None)/np.clip(1-pp,1e-12,None))
    legacy=dict(AUROC=roc_auc_score(y,pp),NLL=float(np.mean(np.logaddexp(0,zz)-y*zz)),Brier=float(np.mean((pp-y)**2)))
    for key,want in lock['legacy_expected_metrics'].items():
        assert abs(legacy[key]-want)<2e-6,(key,legacy)
    pd.DataFrame(audits).to_csv(HERE/'LEGACY_REPRODUCTION.csv',index=False)
    save(HERE/'LEGACY_REPRODUCTION.json',dict(state='PASS',metrics=legacy,forward_model_restored_from_all15checkpoints=True,
        legacy_preserved=True,legacy_not_used_as_repaired_comparator=True,N=len(y),positive=int(y.sum()),CIN3_known=sum(r['cin3'] is not None for r in rows),CIN3_positive=sum(r['cin3']==1 for r in rows),CIN3_unknown=sum(r['cin3'] is None for r in rows)))
    repaired=prepare(config()['pilot_outer'],config()['pilot_held_source']);save(HERE/'REPAIR_INITIAL_SOURCE_GATE.json',dict(state='PASS',**repaired['repair']))
    print('Legacy model forward/patient/label/seed reproduction PASS',legacy,flush=True)
    print('First source-only shared repair PASS',repaired['repair']['source_clip_fraction'],repaired['repair']['nonconstant_dimensions'],flush=True)
if __name__=='__main__':main()
