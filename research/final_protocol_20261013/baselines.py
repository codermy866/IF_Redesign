"""Cohort-matched linear controls using existing label-free cached encoders."""
import runtime
from runtime import HERE,BASE,save,config,sha
import time,json
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from rime_dataset import load_patients,old_run,old_core

def fit_visual_prior(rows,y,g,source,octmean,colpomean,lam):
    frame=old_run.clinical_frame(rows);clinical=old_run.fit_prior(frame,y,source)
    cx=clinical.named_steps['preprocess'].transform(frame)
    if hasattr(cx,'toarray'):cx=cx.toarray()
    state=[];blocks=[cx]
    for values in [octmean,colpomean]:
        pca=PCA(n_components=8,svd_solver='randomized',random_state=17).fit(values[source])
        xx=pca.transform(values);scale=np.maximum(xx[source].std(0),.005);blocks.append(np.clip(xx/scale,-16,16));state.append((pca,scale))
    x=np.column_stack(blocks);centers=np.unique(g[source]);weight=np.array([len(source)/(len(centers)*np.sum(g[source]==c)) for c in g[source]])
    model=LogisticRegression(C=1/(lam*len(source)),max_iter=2000,random_state=17)
    model.fit(x[source],y[source],sample_weight=weight)
    if model.n_iter_.max()>=2000:raise RuntimeError('No-JEV baseline did not converge')
    return model.decision_function(x),dict(model=model,projection=state,clinical_preprocess=clinical.named_steps['preprocess'],train=source,lambda_=lam,effective_parameters=int(x.shape[1]+1))

def main():
    rows=load_patients();y=np.array([r['y'] for r in rows]);g=np.array([r['center'] for r in rows]);start=time.monotonic()
    path=BASE/'results/oct_evidence_repair_20260912/data.npz'
    manifests=[json.loads(line) for line in (BASE/'results/ices_v1/cluster_manifest.jsonl').read_text().splitlines()]
    manifest={str(r['id']):r for r in manifests}
    with np.load(path,allow_pickle=False) as data:
        lookup={str(case):i for i,case in enumerate(data['ids'])};order=np.array([lookup[r['case_id']] for r in rows])
        assert np.array_equal(data['y'][order],y) and np.array_equal(data['centers'][order],g)
        visual=data['visual'][order].astype(np.float64);valid=data['available'][order]
    for row in rows:
        units=manifest[row['case_id']]['evidence_units'];colpo=[u['path'] for u in units if u['kind']=='colposcopy_view']
        assert [str(Path(p).resolve()) for p in colpo[:8]]==[str(Path(p).resolve()) for p in row['colpo_paths'][:8]],'Colposcopy cache order differs'
        sites={str(int(u['scanner_position'][0])):set(str(Path(p).resolve()) for p in u['frame_paths']) for u in units if u['kind']=='oct_position_cluster'}
        assert all(sites[s]==set(str(Path(p).resolve()) for p in paths) for s,paths in row['oct_sites'].items())
    octmean=(visual[:,25:]*valid[:,25:,None]).sum(1)/valid[:,25:].sum(1).clip(1)[:,None]
    colpomean=(visual[:,1:9]*valid[:,1:9,None]).sum(1)/valid[:,1:9].sum(1).clip(1)[:,None]
    cache=json.loads((BASE/'196_controlled_cfv_jev_20261012/private/CACHE_INDEX.json').read_text());hidden=[];mask=[]
    for row in rows:
        item=cache[row['key']];assert sha(item['path'])==item['sha256']
        with np.load(item['path'],allow_pickle=False) as data:hidden.append(data['hidden'][0]);mask.append(data['mask'])
    hidden=np.stack(hidden);mask=np.stack(mask)
    for outer in sorted(set(g)):
        folder=HERE/'private/baselines'/outer;folder.mkdir(parents=True,exist_ok=True)
        tr=np.flatnonzero(g!=outer);te=np.flatnonzero(g==outer)
        for method in ['NO_JEV_THREE_MODAL_LINEAR','FROZEN_JEV_SEMANTIC_LINEAR_PROBE']:
            if (folder/(method+'.npz')).exists() and (folder/(method+'_ALL_SOURCE_CONFIGURATIONS.json')).exists():continue
            candidates=[]
            for lam in config()['lambdas']:
                oof=np.full(len(rows),np.nan)
                for held in sorted(set(g[tr])):
                    fit=tr[g[tr]!=held];val=tr[g[tr]==held]
                    if method=='NO_JEV_THREE_MODAL_LINEAR':z,_=fit_visual_prior(rows,y,g,fit,octmean,colpomean,lam)
                    else:
                        state=old_run.semantic_state(hidden[fit],mask[fit]);xx=old_run.semantic_features(hidden,mask,state)
                        model=old_run.fit_head_from_matrix(xx[fit],np.zeros(len(fit)),y[fit],g[fit],lam)
                        z=.25*xx@model['w']
                    oof[val]=z[val]
                candidates.append(dict(lambda_=lam,source_oof=oof,**old_core.metrics(y[tr],oof[tr])))
            best=max(candidates,key=lambda c:(c['AUROC'],-c['NLL'],c['lambda_']))
            save(folder/(method+'_SOURCE_SELECTION_LOCK.json'),dict(selected_lambda=best['lambda_'],source_only=True,effective_search_configurations=3))
            if method=='NO_JEV_THREE_MODAL_LINEAR':z,model=fit_visual_prior(rows,y,g,tr,octmean,colpomean,best['lambda_'])
            else:
                state=old_run.semantic_state(hidden[tr],mask[tr]);xx=old_run.semantic_features(hidden,mask,state)
                model=dict(state=state,head=old_run.fit_head_from_matrix(xx[tr],np.zeros(len(tr)),y[tr],g[tr],best['lambda_']),train=tr)
                z=.25*xx@model['head']['w']
            joblib.dump(model,folder/(method+'.joblib'))
            output=folder/(method+'.npz')
            if output.exists():
                with np.load(output) as previous:
                    assert np.allclose(previous['seed_logits'],np.tile(z[te],(3,1)),atol=1e-10,rtol=0)
            else:np.savez_compressed(output,test=te,seed_logits=np.tile(z[te],(3,1)),source_indices=tr,
                source_seed_logits=np.tile(best['source_oof'][tr],(3,1)),source_y=y[tr],source_g=g[tr])
            save(folder/(method+'_ALL_SOURCE_CONFIGURATIONS.json'),[
                {**c,'source_indices':tr,'source_oof':c['source_oof'][tr]} for c in candidates])
            print('BASELINE_SOURCE_FROZEN_AND_OUTER_PREDICTED',outer,method,flush=True)
    save(HERE/'BASELINES_COMPLETE.json',dict(state='COMPLETE',N=len(rows),five_folds=True,
        non_JEV_encoder='frozen ImageNet ResNet50, 120 OCT frames mean within site then patient, first <=8 colposcopy views; clinical age/HPV/TCT',
        matched_data_verified=True,head_capacity_matched_to_RIME=False,preprocessing='original cache preprocessing differs from JEV; do not attribute the difference solely to backbone',
        JEV_probe='source PCA26 native patient hidden, linear head; no clinical covariates',seeds=config()['seeds'],deterministic_seeds_identical=True,seconds=time.monotonic()-start))

if __name__=='__main__':main()
