"""Admission experiment; full native targets before source-only auxiliary fitting."""
import runtime
from runtime import HERE,CPU,GPU,save,config,verify
import os,time,json,subprocess,signal,argparse
import numpy as np,torch,joblib,pandas as pd
from scipy.special import expit
from rime_dataset import tensor_inputs
from run_heads import prob_logits
from train_control import train_control
from b5_values import arrays,targets,teacher,advantage,shuffle

def fit_fold(a,outer,source_only=True):
    verify();cfg=config();rows=a['rows'];y=a['y'];g=a['groups'];tr=np.flatnonzero(g!=outer);te=np.flatnonzero(g==outer)
    folder=HERE/'private/head_runs'/outer;parts={p.stem:joblib.load(p) for p in (HERE/'private/fold_features'/outer).glob('*.joblib')}
    core=json.loads((folder/'D_RIME_LAPLACE_SOURCE_SELECTION_LOCK.json').read_text());tau,lam=core['selected_tau'],core['selected_lambda']
    diag=[]
    for method in ['B5_VALUE_AUXILIARY','B5_SHUFFLED_AUXILIARY','B5_STRATIFIED_SHUFFLED','B5_PATIENT_AUXILIARY']:
        if (folder/(method+'.npz')).exists():continue
        candidates=[]
        for coefficient in cfg['B5_aux_weights']:
            oofs=[];epochs=[]
            for seed in cfg['seeds']:
                oof=np.full(len(y),np.nan)
                for held in sorted(set(g[tr])):
                    fit=tr[g[tr]!=held];val=tr[g[tr]==held];part=parts[held]
                    assert np.array_equal(part['train'],fit)
                    targetdir=HERE/'private/b5_targets'/outer/held;targetdir.mkdir(parents=True,exist_ok=True)
                    targetpath=targetdir/f'VALUES_seed{seed}.npy'
                    if targetpath.exists():u=np.load(targetpath)
                    else:u=targets(a,fit,targetdir/f'CROSSFIT_AUDIT_seed{seed}.json',seed=seed);np.save(targetpath,u)
                    scale=max(float(np.var(u[fit])),cfg['B5_target_variance_floor']);supervision=u.copy()
                    if method in ['B5_SHUFFLED_AUXILIARY','B5_STRATIFIED_SHUFFLED']:
                        supervision,records=shuffle(u,fit,g,seed,labels=y if method=='B5_STRATIFIED_SHUFFLED' else None)
                        save(targetdir/f'{method}_SHUFFLE_seed{seed}.json',dict(records=records,source_only=True))
                    path=folder/f'{method}_aux{coefficient}_seed{seed}_inner{held}.pt'
                    ti=tensor_inputs(part,fit,'D_RIME_LAPLACE',tau,'cpu',rows);vi=tensor_inputs(part,val,'D_RIME_LAPLACE',tau,'cpu',rows)
                    if path.exists():saved=torch.load(path,map_location='cpu',weights_only=False)
                    else:
                        head,md=train_control(ti,vi,y,g,tau,lam,seed,max_epochs=cfg['max_epochs'],patience=cfg['patience'],
                            auxiliary=torch.as_tensor(supervision[fit],dtype=torch.double),aux_weight=coefficient if method=='B5_PATIENT_AUXILIARY' else coefficient/scale,
                            auxiliary_mode='patient' if method=='B5_PATIENT_AUXILIARY' else 'value')
                        with torch.no_grad():z=head(**{k:v for k,v in vi.items() if k!='indices'}).cpu().numpy()
                        saved=dict(state={k:v.cpu() for k,v in head.state_dict().items()},metadata=md,prediction=z,train=fit,validation=val)
                        torch.save(saved,path)
                    # Also reconstruct diagnostics on resume, preserving each
                    # completed patient-level run in its own immutable file.
                    diagnostic=targetdir/f'{method}_aux{coefficient}_seed{seed}_DIAGNOSTICS.parquet'
                    if diagnostic.exists():diag.extend(pd.read_parquet(diagnostic).to_dict('records'))
                    else:
                        from scorers import ScoreControl
                        head=ScoreControl(vi['b'].shape[-1],tau,'measure','laplace','conditional').double()
                        head.load_state_dict(saved['state']);head.eval()
                        reference=teacher(a,fit,seed);held_adv=advantage(reference['prediction'][val],y[val])
                        with torch.no_grad():
                            q=head.scores(vi['h'],vi['b']);vp=((q*vi['valid']).sum(-1)/vi['valid'].sum(-1).clamp_min(1)).numpy()
                        records=[]
                        for i,index in enumerate(val):
                            high=np.argsort(vp[i])[-3:];low=np.argsort(vp[i])[:3]
                            records.append(dict(method=method,outer=outer,held_source=held,seed=seed,aux_coefficient=coefficient,key=rows[index]['key'],
                                value_MSE=float(np.mean((vp[i]-held_adv[i])**2)),zero_MSE=float(np.mean(held_adv[i]**2)),
                                constant_MSE=float(np.mean((u[fit].mean()-held_adv[i])**2)),predicted_value_std=float(vp[i].std()),
                                high_minus_low_actual_Brier_advantage=float(held_adv[i,high].mean()-held_adv[i,low].mean())))
                        pd.DataFrame(records).to_parquet(diagnostic,index=False);diag.extend(records)
                    assert np.array_equal(saved['train'],fit) and np.array_equal(saved['validation'],val)
                    oof[val]=saved['prediction'];epochs.append(saved['metadata']['best_epoch'])
                    save(HERE/f'B5_{outer}_STATE.json',dict(state='SOURCE_CV',method=method,seed=seed,aux_coefficient=coefficient,held_source=held))
                oofs.append(oof)
            z=prob_logits(np.stack(oofs));from sklearn.metrics import roc_auc_score
            candidates.append(dict(aux_coefficient=coefficient,source_AUROC=roc_auc_score(y[tr],z[tr]),source_NLL=float(np.mean(np.logaddexp(0,z[tr])-y[tr]*z[tr])),
                source_seed_logits=np.stack(oofs),fixed_refit_epochs=int(np.median(epochs))))
        save(folder/(method+'_ALL_SOURCE_CONFIGURATIONS.json'),[{**c,'source_indices':tr,'source_seed_logits':c['source_seed_logits'][:,tr]} for c in candidates])
        best=max(candidates,key=lambda c:(c['source_AUROC'],-c['source_NLL'],-c['aux_coefficient']))
        save(folder/(method+'_SOURCE_SELECTION_LOCK.json'),{k:v for k,v in best.items() if k!='source_seed_logits'}|dict(selected_tau=tau,selected_lambda=lam,source_only=True,effective_search_configurations=3))
        chosen=pd.DataFrame([r for r in diag if r['method']==method and np.isclose(r['aux_coefficient'],best['aux_coefficient'])])
        seed_gates=[]
        for seed,group in chosen.groupby('seed'):
            seed_gates.append(dict(seed=int(seed),patients=int(group.key.nunique()),value_MSE=float(group.value_MSE.mean()),
                zero_MSE=float(group.zero_MSE.mean()),constant_MSE=float(group.constant_MSE.mean()),
                high_minus_low_actual_Brier_advantage=float(group.high_minus_low_actual_Brier_advantage.mean()),
                supported=bool(group.value_MSE.mean()<min(group.zero_MSE.mean(),group.constant_MSE.mean()) and group.high_minus_low_actual_Brier_advantage.mean()>0)))
        passed=sum(r['supported'] for r in seed_gates)>=2
        save(folder/(method+'_SOURCE_VALUE_GATE.json'),dict(state='PASS' if passed else 'SOURCE_VALUE_UNSUPPORTED',
            seeds=seed_gates,rule='at least 2 of 3 source-selected seed fits beat zero and source constant MSE and have positive high-low actual mean advantage; no outer values inspected',
            source_only=True,teacher_dependent=True))
        if source_only:continue
        targetdir=HERE/'private/b5_targets'/outer/'outer';targetdir.mkdir(parents=True,exist_ok=True)
        ti=tensor_inputs(parts['outer'],tr,'D_RIME_LAPLACE',tau,'cpu',rows);vi=tensor_inputs(parts['outer'],te,'D_RIME_LAPLACE',tau,'cpu',rows);logits=[]
        for seed in cfg['seeds']:
            path=targetdir/f'VALUES_seed{seed}.npy'
            if path.exists():u=np.load(path)
            else:u=targets(a,tr,targetdir/f'CROSSFIT_AUDIT_seed{seed}.json',seed=seed);np.save(path,u)
            scale=max(float(np.var(u[tr])),cfg['B5_target_variance_floor'])
            supervision=u.copy()
            if method in ['B5_SHUFFLED_AUXILIARY','B5_STRATIFIED_SHUFFLED']:
                supervision,records=shuffle(u,tr,g,seed,labels=y if method=='B5_STRATIFIED_SHUFFLED' else None)
                save(targetdir/f'{method}_SHUFFLE_seed{seed}.json',dict(records=records,source_only=True))
            head,md=train_control(ti,None,y,g,tau,lam,seed,fixed_epochs=best['fixed_refit_epochs'],
                auxiliary=torch.as_tensor(supervision[tr],dtype=torch.double),aux_weight=best['aux_coefficient'] if method=='B5_PATIENT_AUXILIARY' else best['aux_coefficient']/scale,
                auxiliary_mode='patient' if method=='B5_PATIENT_AUXILIARY' else 'value')
            torch.save(dict(state={k:v.cpu() for k,v in head.state_dict().items()},metadata=md,train=tr,source_selection_frozen=True),folder/f'{method}_seed{seed}_OUTER.pt')
            with torch.no_grad():logits.append(head(**{k:v for k,v in vi.items() if k!='indices'}).numpy())
        np.savez_compressed(folder/(method+'.npz'),test=te,seed_logits=np.stack(logits),source_indices=tr,source_seed_logits=best['source_seed_logits'][:,tr],source_y=y[tr],source_g=g[tr])
    diagnostic_files=list((HERE/'private/b5_targets'/outer).glob('*/*_DIAGNOSTICS.parquet'))
    if diagnostic_files:
        pd.concat([pd.read_parquet(path) for path in diagnostic_files],ignore_index=True).to_parquet(HERE/f'private/B5_{outer}_SOURCE_VALUE_DIAGNOSTICS.parquet',index=False)
    save(folder/('B5_SOURCE_COMPLETE.json' if source_only else 'B5_COMPLETE.json'),dict(state='SOURCE_ONLY_COMPLETE' if source_only else 'COMPLETE',posthoc_only_predictor_identical_to_no_auxiliary=True))

def main():
    torch.set_num_threads(4);verify();cfg=config();children=[]
    try:
        # Original campaign has finished/released GPUs; never steal its capacity.
        for split in [0,1]:
            while int(subprocess.check_output(['nvidia-smi',f'--id={split}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<27000:
                verify();time.sleep(20)
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(split)
            with (HERE/f'b5_native_{split}.log').open('a') as log:
                child=subprocess.Popen([str(GPU),'-u',str(HERE/'b5_native.py'),'--split',str(split)],env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append(child)
        while any(c.poll() is None for c in children):
            verify()
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('B5 full-native worker failed')
            time.sleep(10)
        if any(c.returncode for c in children):raise RuntimeError('B5 full-native worker failed')
    finally:
        for c in children:
            if c.poll() is None:c.terminate()
    a=arrays()
    for outer in sorted(set(a['groups'])):fit_fold(a,outer,source_only=True)
    gates={outer:json.loads((HERE/'private/head_runs'/outer/'B5_VALUE_AUXILIARY_SOURCE_VALUE_GATE.json').read_text()) for outer in sorted(set(a['groups']))}
    save(HERE/'B5_SOURCE_ADMISSION_GATE.json',dict(state='PASS' if all(gate['state']=='PASS' for gate in gates.values()) else 'UNSUPPORTED_STOP_BEFORE_ANY_B5_OUTER_PREDICTIONS',
        folds=gates,outer_results_used=False,negative_results_preserved=True))
    if all(gate['state']=='PASS' for gate in gates.values()):
        for outer in sorted(set(a['groups'])):fit_fold(a,outer,source_only=False)
        subprocess.run([str(CPU),str(HERE/'summarize.py'),'--phase','B5'],check=True,timeout=max(1,cfg['deadline_unix']-time.time()))
    else:
        save(HERE/'B5_STOP_REASON.json',dict(state='SOURCE_VALUE_GATE_NOT_SUPPORTED',action='retain all source fits/controls/targets; do not tune or open B5 outer outcomes',
            diagnoses=['check numerical implementation versus passed synthetic gates','reference predictor and target variance','optimization convergence and sample size','site-value hypothesis may be unsupported'],
            actual_cause_not_proven=True,scope='teacher-dependent site auxiliary, distinct from previously failed fixed-context CV1 frame values'))

if __name__=='__main__':main()
