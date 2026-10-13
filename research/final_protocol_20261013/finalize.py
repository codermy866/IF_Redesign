"""Actual-result-only paper artifacts and full pending/negative ledger."""
import runtime
from runtime import HERE,save,config
import json,time
import numpy as np,pandas as pd,torch

def main():
    tables={}
    for phase in ['CORE','B1','B2','B5']:
        path=HERE/f'TABLE_{phase}_MAIN.csv'
        if path.exists():tables[phase]=pd.read_csv(path)
    negatives=[];evidence=[]
    for phase in tables:
        path=HERE/f'TABLE_{phase}_CONTRASTS.csv'
        if path.exists():
            frame=pd.read_csv(path)
            negatives.extend(frame[~frame.support].assign(phase=phase).to_dict('records'))
            evidence.extend(frame[frame.support].assign(phase=phase).to_dict('records'))
    value_diagnostics=[]
    for path in (HERE/'private').glob('B5_*_SOURCE_VALUE_DIAGNOSTICS.parquet'):
        frame=pd.read_parquet(path)
        for (method,outer,seed),group in frame.groupby(['method','outer','seed']):
            lock=json.loads((HERE/'private/head_runs'/outer/(method+'_SOURCE_SELECTION_LOCK.json')).read_text())
            chosen=group[np.isclose(group.aux_coefficient,lock['aux_coefficient'])]
            if chosen.empty:continue
            value_diagnostics.append(dict(method=method,outer=outer,seed=int(seed),patients=int(chosen.key.nunique()),
                value_MSE=float(chosen.value_MSE.mean()),zero_MSE=float(chosen.zero_MSE.mean()),constant_MSE=float(chosen.constant_MSE.mean()),
                high_minus_low_actual_Brier_advantage=float(chosen.high_minus_low_actual_Brier_advantage.mean()),
                descriptive_value_learning_supported=bool(chosen.value_MSE.mean()<min(chosen.zero_MSE.mean(),chosen.constant_MSE.mean()) and chosen.high_minus_low_actual_Brier_advantage.mean()>0),
                evaluation='source-selection diagnostic; not independent confirmation'))
    pd.DataFrame(value_diagnostics).to_csv(HERE/'VALUE_PREDICTION_RESULTS.csv',index=False)
    costs=[]
    for path in (HERE/'private/head_runs').glob('*/*.pt'):
        item=torch.load(path,map_location='cpu',weights_only=False);md=item['metadata']
        costs.append(dict(file=str(path.relative_to(HERE)),best_epoch=md['best_epoch'],epochs_run=md['epochs_run'],fit_seconds=md.get('fit_seconds'),
            parameters=md['parameters'],allocated_parameters=md['allocated_parameters'],seed=md['seed']))
    pd.DataFrame(costs).to_csv(HERE/'TABLE_COMPUTE.csv',index=False)
    teacher_costs=[]
    import joblib
    for path in (HERE/'private/b5_teachers').glob('*.joblib'):
        teacher=joblib.load(path);teacher_costs.append(dict(file=str(path.relative_to(HERE)),patients=teacher['patients'],
            training_conditions=teacher['training_conditions'],seconds=teacher['seconds'],effective_parameters=teacher['effective_parameters']))
    pd.DataFrame(teacher_costs).to_csv(HERE/'TABLE_TEACHER_COMPUTE.csv',index=False)
    save(HERE/'COMPUTE_BUDGET.json',dict(original_deadline_unix=config()['deadline_unix'],automatic_extension=False,
        head_fit_seconds=sum(r['fit_seconds'] or 0 for r in costs),teacher_fit_seconds=sum(r['seconds'] for r in teacher_costs),
        teacher_cost_additional_to_baseline=True,upstream_cost_shared_not_zero=True,
        native_workers={p.name:json.loads(p.read_text()) for p in HERE.glob('B5_NATIVE_WORKER_*.json')},
        source_baselines=json.loads((HERE/'BASELINES_COMPLETE.json').read_text()) if (HERE/'BASELINES_COMPLETE.json').exists() else None))
    source_recommendations={}
    for outerdir in (HERE/'private/head_runs').glob('*'):
        choices=[]
        for path in outerdir.glob('*_SOURCE_SELECTION_LOCK.json'):
            selected=json.loads(path.read_text())
            if 'source_AUROC' in selected:choices.append(dict(method=path.name.removesuffix('_SOURCE_SELECTION_LOCK.json'),
                source_AUROC=selected['source_AUROC'],source_NLL=selected['source_NLL'],configuration=selected))
        if choices:source_recommendations[outerdir.name]=max(choices,key=lambda r:(r['source_AUROC'],-r['source_NLL']))
    save(HERE/'SOURCE_CANDIDATE_RECOMMENDATIONS.json',dict(fold_source_only_candidates=source_recommendations,
        scope='source-only recommendations, not a new pooled performance comparison or one globally confirmed architecture',
        final_method_frozen=False,independent_confirmation=False))
    (HERE/'NEGATIVE_RESULTS.md').write_text('# Unsupported or negative comparisons\n\n'+pd.DataFrame(negatives).to_csv(index=False)+'\nOriginal CV1 held-source scores were all zero and did not beat the zero-value baseline. This failure remains in Stage198. Source-selected models and all seeds are retained. No independent cohort or clinical safety guarantee.\n')
    claim=dict(supported_conditional_comparisons=evidence,unsupported_or_negative_comparisons=negatives,value_learning_diagnostics=value_diagnostics,
        patient_bootstrap_conditions_on_fitted_models=True,development_history_not_removed=True,independent_confirmation=False,
        architecture_selected_from_source_only=True,final_external_validation_pending=True,
        B1_is_optimizer_not_original_representation=True,B5_is_teacher_dependent_predictive_site_supervision_not_frame_or_causal_truth=True)
    claim['final_method_frozen']=False
    save(HERE/'CLAIM_LEDGER.json',claim)
    if tables:
        text='\\section{Experiments}\nWe evaluated patient-level CIN2+ prediction in a repeatedly used five-centre development cohort (984 patients, 303 CIN2+). All projections, clinical priors, bandwidths, coordinate scales and selections used source patients only. Five outer centres, source-centre inner validation, and seeds 17, 29 and 43 were retained. This is cross-centre internal development, not independent confirmation. Paired bootstrap used 3000 centre-by-label patient resamples and a fixed new-development comparison family of 16.\\par\n'
        for phase,frame in tables.items():
            text+='\\subsection{'+phase+' actual held-centre results}\n'
            for row in frame.to_dict('records'):
                method=str(row['method']).replace('_','\\_')
                text+=f"{method}: AUROC {row['AUROC']:.4f}, AUPRC {row['AUPRC']:.4f}, NLL {row['NLL']:.4f}, Brier {row['Brier']:.4f}.\\\\\n"
        text+='\\par Confidence intervals, centre-specific results, all seeds, source-selected thresholds, CIN3+ denominators and computational costs accompany the machine-readable tables. Historical exploration and unresolved source-label adjudication constrain generalization claims.\n'
        (HERE/'Experiments.tex').write_text(text)
    save(HERE/'DELIVERY_STATUS.json',dict(actual_result_phases=list(tables),pending_phases=[p for p in ['CORE','B1','B2','B5'] if p not in tables],
        independent_confirmation='UNAVAILABLE',submission_not_performed=True,git_push_not_performed=True,time=time.time()))

if __name__=='__main__':main()
