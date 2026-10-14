"""Supplemental activity/cost audit; never used for selection or training."""
import runtime
from runtime import HERE,save,sha,config
import json,time,pandas as pd,torch

def main():
    selected={}
    for folder in (HERE/'private/runs').glob('*'):
        for path in folder.glob('*_SELECTION_LOCK.json'):
            selected[folder.name,path.name.removesuffix('_SELECTION_LOCK.json')]=json.loads(path.read_text())
    costs=[];activity=[]
    for path in (HERE/'private/cases').glob('*/*.pt'):
        item=torch.load(path,map_location='cpu',weights_only=False);m=item['metadata'];outer=path.parent.name
        costs.append(dict(outer=outer,method=m['method'],seed=m['seed'],fit_seconds=m['fit_seconds'],epochs_run=m['epochs_run'],best_epoch=m['best_epoch'],parameters=m['parameters'],scope='unique fitted source case, including pilot cases; resumed cases counted once'))
        choice=selected.get((outer,m['method']))
        chosen=choice is not None and m['candidate']==choice['candidate'] and m['lambda_']==choice['lambda_']
        if m['method']=='A5_ARDR':
            curve=[c for c in m['curve'] if c['epoch']<=m['best_epoch'] and c['loss_terms'] is not None]
            values=[c['loss_terms']['extra'] for c in curve]
            activity.append(dict(outer=outer,seed=m['seed'],candidate=json.dumps(m['candidate'],sort_keys=True),lambda_=m['lambda_'],source_selected=chosen,
                best_epoch=m['best_epoch'],penalty_active_updates_before_selected_epoch=sum(v>1e-12 for v in values),max_ARDR_penalty_before_selected_epoch=max(values,default=0.),
                source_constraint_fraction=m['constraint_satisfaction_fraction'],source_mean_abs_delta=m['source_mean_abs_delta'],source_patient_count=len(item['train']),
                held_source_patient_count=len(item['validation']),recipient_excluded_from_head_and_preprocessing=True))
    for path in (HERE/'private/runs').glob('*/*_OUTER.pt'):
        m=torch.load(path,map_location='cpu',weights_only=False)['metadata']
        costs.append(dict(outer=path.parent.name,method=m['method'],seed=m['seed'],fit_seconds=m['fit_seconds'],epochs_run=m['epochs_run'],best_epoch=m['best_epoch'],parameters=m['parameters'],scope='outer-source refit; no labels from held centre'))
    pd.DataFrame(costs).to_csv(HERE/'ARDR_COMPUTE_CASES.csv',index=False)
    pd.DataFrame(costs).groupby(['method','scope']).agg(fit_count=('seed','size'),fit_seconds=('fit_seconds','sum'),updates=('epochs_run','sum')).reset_index().to_csv(HERE/'ARDR_COMPUTE_TOTALS.csv',index=False)
    pd.DataFrame(activity).to_csv(HERE/'ARDR_OPTIMIZATION_ACTIVITY.csv',index=False)
    repairs=[dict(outer=p.parent.name,partition=p.stem.removesuffix('.audit'),**json.loads(p.read_text())) for p in (HERE/'private/fold_features').glob('*/*.audit.json')]
    pd.DataFrame([{k:v for k,v in r.items() if k!='source'} for r in repairs]).to_csv(HERE/'SHARED_PREPROCESSING_REPAIR_AUDIT.csv',index=False)
    launch=json.loads((HERE/'CONTROLLER_LAUNCH.json').read_text())
    save(HERE/'COMPUTE_BUDGET.json',dict(unique_source_fits=sum(r['scope'].startswith('unique') for r in costs),source_refits=sum(r['scope'].startswith('outer') for r in costs),
        summed_head_fit_seconds=sum(r['fit_seconds'] for r in costs),wall_seconds_since_controller_launch=time.time()-launch['started_unix'],
        additional_native_27B_calls=0,original_frame_cache_cost_not_zero=True,original_deadline_unix=config()['deadline_unix'],deadline_extended=False,
        exact_limit='Six configurations per added arm; three shared nuisance base lambdas; three seeds; one fixed pilot source split and five outer folds',
        shared_preprocessing_partitions=len(repairs),preprocessing_source_clipping_max=max((r['source_clip_fraction'] for r in repairs),default=None)))
    save(HERE/'SUPPLEMENTAL_POSTPROCESSING_MANIFEST.json',dict(script=__file__,sha256=sha(__file__),training_or_selection_changed=False,main_experiment_manifest_sha256=sha(HERE/'EXPERIMENT_MANIFEST.json')))
    print('Recorded unique computation, selected-source penalty activity, and common repair evidence.')
if __name__=='__main__':main()
