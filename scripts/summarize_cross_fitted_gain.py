"""Complete source-only cross-fitting ledger, all policies and paired intervals."""
import json
import numpy as np
from run_cross_fitted_gain import OUT,MODES
from summarize_csf_screen import macro

FOLDS=('shiyan','enshi','wuhan','jingzhou','xiangyang')


def main():
    reports=[];contrasts=[];audits=[];target_audits=[];unique=set();appearances=0
    for fold in FOLDS:
        root=OUT/fold/'seed_20260905';r=json.loads((root/'complete.json').read_text());reports.append(r)
        d=np.load(root/'predictions.npz',allow_pickle=False);y=d['y'];c=d['centers'];indices=d['indices']
        unique.update(indices.tolist());appearances+=len(indices)
        assert len(indices)==len(set(indices)) and not r['test_labels_evaluated']
        provenance=json.loads((root/'target_provenance.json').read_text())
        teachers={k:json.loads((root/f'teacher_{k}'/'complete.json').read_text())for k in range(5)}
        for p in provenance:
            m=teachers[p['teacher']];i=p['patient_index']
            if p['kind']=='oof':assert i not in m['fit_indices'] and i not in m['cal_indices'] and i in m['gain_indices']
            else:assert i in m['fit_indices']
        targets=np.load(root/'gain_targets.npz');target_audits.append({'fold':fold,'rows':len(targets['rows']),
             'mean_oof_gain':float(targets['oof_gain'].mean()),'mean_seen_gain':float(targets['seen_gain'].mean()),
             'negative_oof_fraction':float(np.mean(targets['oof_gain']<0)), 'negative_seen_fraction':float(np.mean(targets['seen_gain']<0)),
             'gain_difference_median':float(np.median(targets['seen_gain']-targets['oof_gain']))})
        audit=json.loads((root/'attention_audit.json').read_text());parts={}
        for key in ('label','hpv_group','center'):
            parts[key]={}
            for value in sorted({str(a[key])for a in audit}):
                group=[a for a in audit if str(a[key])==value]
                rho=[a['attention_gain_spearman']for a in group if a['attention_gain_spearman'] is not None]
                gap=[a['positive_minus_negative_site_gain']for a in group if 'positive_minus_negative_site_gain'in a]
                parts[key][value]={'cases':len(group),'mean_within_case_attention_gain_rho':float(np.mean(rho)) if rho else None,
                    'mixed_site_cases':len(gap),'mean_positive_minus_negative_site_gain':float(np.mean(gap)) if gap else None}
        audits.append({'fold':fold,'groups':parts})
        strata=[np.where((c==h)&(y==label))[0]for h in np.unique(c)for label in (0,1)];strata=[s for s in strata if len(s)]
        rng=np.random.default_rng(20260905);boot={m:[]for m in MODES};boot['full']=[]
        for _ in range(1000):
            ix=np.concatenate([rng.choice(s,len(s),replace=True)for s in strata])
            for m in MODES:boot[m].append(macro(y[ix],d[m][ix,4],c[ix]))
            boot['full'].append(macro(y[ix],d['full'][ix],c[ix]))
        for other in [m for m in MODES if m!='gain_oof']+['full']:
            ref=d[other][:,4] if other!='full' else d['full']
            delta=macro(y,d['gain_oof'][:,4],c)-macro(y,ref,c)
            sample=np.array(boot['gain_oof'])-np.array(boot[other])
            contrasts.append({'fold':fold,'left':'gain_oof','right':other,'delta_macro_ap_at4':float(delta),'paired_patient_bootstrap_ci95':np.quantile(sample,[.025,.975]).tolist()})
    table={m:{'mean_fold_macro_ap_at4':float(np.mean([r['policies'][m]['budgets'][4]['macro_auprc']for r in reports])),
              'mean_case_site_recall_at4_across_folds':float(np.mean([r['policies'][m]['mean_case_site_recall_at4']for r in reports]))}for m in MODES}
    best=max(table,key=lambda m:table[m]['mean_fold_macro_ap_at4'])
    result={'completed_outer_folds':5,'fresh_teachers':30,'seeds':1,'unique_validation_patients':len(unique),'fold_appearances':appearances,
            'test_labels_evaluated':False,'policies':table,'all_fold_results':reports,'paired_contrasts':contrasts,'attention_audits':audits,
            'target_optimism_audits':target_audits,'best_development_policy_at4':best,'best_policy_is_confirmatory':False,
            'statistics':'1000 within-fold paired patient bootstrap draws stratified center/outcome; descriptive CIs, no multiplicity-adjusted superiority claim; overlapping folds never treated as independent patients'}
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    lines=['# Cross-fitted diagnostic gain: first complete run','',
           f'Five source folds, 30 freshly trained diagnostic models, one seed; {len(unique)} unique validation patients and {appearances} fold-level validation appearances. Held-out external centres are not evaluated here.','',
           '| Policy | Mean fold macro-AP@4 | Positive-site recall@4 |','|---|---:|---:|']
    for m,v in table.items():lines.append(f"| {m} | {v['mean_fold_macro_ap_at4']:.4f} | {v['mean_case_site_recall_at4_across_folds']:.4f} |")
    lines.extend(['',f'The highest fixed-budget-4 mean on the development set is {best}. This is a development-set choice, not independent superiority evidence.','',
      '## Interpretation boundary','',
      'OOF gain is the observed-label loss difference from a model that did not fit that patient. It is not a clinical causal effect. All candidates have already been imaged, so retained-count curves are offline evidence-retention analyses.', '',
      'The full 0-6 budget curves, full-evidence reference, source-centre audits, and paired intervals are in summary.json. Comparisons should use the same contemporaneous frozen diagnostic model rather than mixing best historical values.', '',
      'This report covers one seed only. A publishable superiority claim requires a locked method, repeated seeds, and independent-centre validation. Brightness, noise, and scanner-style perturbations are not validated lesion-preserving interventions here.'])
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'policies':table,'best_development':best},indent=2),flush=True)


if __name__=='__main__':main()
