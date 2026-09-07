"""All-results paired report. Seeds/fold appearances are not extra patients."""
import json
from pathlib import Path
import numpy as np
from run_csf_screen import ROOT, OUT, MODES

FOLDS = ('shiyan', 'enshi', 'wuhan', 'jingzhou', 'xiangyang')
SEEDS = (20260905, 20260906, 20260907)


def ap(y, p):
    order = np.argsort(-p, kind='stable'); yy = y[order]; pp = p[order]
    endpoints = np.r_[np.where(np.diff(pp) != 0)[0], len(pp)-1]
    tp = np.cumsum(yy)[endpoints]
    return float(np.sum(np.diff(np.r_[0, tp])*tp/(endpoints+1))/sum(y))


def macro(y, p, c):
    return float(np.mean([ap(y[c == h], p[c == h]) for h in np.unique(c) if len(np.unique(y[c == h])) == 2]))


def main():
    modes = list(MODES); memory = ROOT/'results/csf_memory_v1'
    include_memory = len(list(memory.glob('*/seed_*/complete.json'))) == 15
    if include_memory: modes += ['gain_mean', 'gain_transport']
    tables = {m: [] for m in modes}; full_rows = []; zero_rows = []; all_indices = []
    contrasts = []; source_counts = {}; budget_rows = []; selection_rows = []; mechanisms = []
    for fold in FOLDS:
        data = [np.load(OUT/fold/f'seed_{s}'/'predictions.npz', allow_pickle=False) for s in SEEDS]
        reports = [json.loads((OUT/fold/f'seed_{s}'/'complete.json').read_text()) for s in SEEDS]
        more = [np.load(memory/fold/f'seed_{s}'/'predictions.npz', allow_pickle=False) for s in SEEDS] if include_memory else None
        more_reports = [json.loads((memory/fold/f'seed_{s}'/'complete.json').read_text()) for s in SEEDS] if include_memory else None
        y, c, indices = data[0]['val_y'], data[0]['val_centers'], data[0]['val_indices']
        all_indices.extend(indices.tolist())
        assert len(set(indices)) == len(indices)
        for d in data+(more or []):
            assert np.array_equal(d['val_y'], y) and np.array_equal(d['val_indices'], indices)
            assert np.isfinite(d['val_full']).all()
        source_counts[fold] = {h: {'n': int(sum(c == h)), 'positive': int(sum(y[c == h])), 'negative': int(sum(1-y[c == h]))} for h in np.unique(c)}
        pp = {m: np.stack([(data if m in MODES else more)[i]['val_'+m][:, 4] for i in range(3)]) for m in modes}
        pp['full'] = np.stack([d['val_full'] for d in data])
        pp['zero'] = np.stack([d['val_centered'][:, 0] for d in data])
        for m, values in pp.items():
            scores = [macro(y, p, c) for p in values]
            if m in modes: tables[m].append(scores)
            elif m == 'full': full_rows.append(scores)
            else: zero_rows.append(scores)
        for m in modes:
            ds = data if m in MODES else more; rs = reports if m in MODES else more_reports
            for i, seed in enumerate(SEEDS):
                rr = rs[i]['reports']['val']['policies'][m]
                for k in range(5):
                    row = dict(rr['budgets'][k]); row.update(rr['frontier'][k]); row.update({'fold': fold, 'seed': seed, 'mode': m})
                    budget_rows.append(row)
                selection_rows.append({'fold': fold, 'seed': seed, 'mode': m, **rr['selection'], **rr['selected']})
        # Paired bootstrap averages seed-specific metric differences; not an
        # ensemble and not 3*n independent cases. Same patient draw for all seeds.
        pairs = [('centered', 'total_kl'), ('stable', 'centered'), ('cross_center', 'centered')]
        pairs += [(m, r) for m in modes if m not in ('random', 'attention') for r in ('random', 'attention', 'full')]
        strata = [np.where((c == h)&(y == label))[0] for h in np.unique(c) for label in (0, 1)]
        strata = [s for s in strata if len(s)]
        rng = np.random.default_rng(20260906)
        boot = {m: [] for m in pp}
        for _ in range(1000):
            ix = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
            for m, values in pp.items(): boot[m].append(np.mean([macro(y[ix], p[ix], c[ix]) for p in values]))
        for left, right in pairs:
            differences = np.array(boot[left])-np.array(boot[right])
            delta = np.mean([macro(y, a, c)-macro(y, b, c) for a, b in zip(pp[left], pp[right])])
            contrasts.append({'fold': fold, 'left': left, 'right': right, 'seed_mean_delta_ap': float(delta),
                              'paired_patient_bootstrap_ci95': np.quantile(differences, [.025, .975]).tolist()})
        for seed in SEEDS:
            sums = []
            with (OUT/fold/f'seed_{seed}'/'val_chains.jsonl').open() as stream:
                for line in stream:
                    record = json.loads(line)
                    if record['mode'] != 'total_kl': continue
                    for a in record['chain']:
                        sums.append([a['centered'], a['coherence'], a['sensitivity_proxy'], a['total_kl']])
            v = np.array(sums)
            mechanisms.append({'fold': fold, 'seed': seed, 'mean_selected_J': float(v[:, 0].mean()),
                               'mean_selected_coherence': float(v[:, 1].mean()), 'mean_selected_sensitivity': float(v[:, 2].mean()),
                               'coherence_share_of_total': float(v[:, 1].sum()/v[:, 3].sum())})
    table = {}; full = np.array(full_rows)
    for m in modes:
        a = np.array(tables[m]); dr = (a-np.array(tables['random'])).mean(1); da = (a-np.array(tables['attention'])).mean(1); df = (a-full).mean(1)
        gate = m not in ('random', 'attention') and (dr > 0).sum() >= 4 and (da > 0).sum() >= 4 and np.median(dr) >= .01 and np.median(da) >= .01 and np.median(df) >= -.02
        table[m] = {'median_15_run_macro_ap': float(np.median(a)), 'mean_15_run_macro_ap': float(np.mean(a)),
                    'positive_folds_vs_random': int(sum(dr > 0)), 'positive_folds_vs_attention': int(sum(da > 0)),
                    'median_paired_delta_random': float(np.median(dr)), 'median_paired_delta_attention': float(np.median(da)),
                    'median_paired_delta_full': float(np.median(df)), 'advance_to_nested_training': bool(gate),
                    'zero_budget_choices': sum(s['budget'] == 0 for s in selection_rows if s['mode'] == m)}
    result = {'complete_screen_jobs': 15, 'complete_memory_jobs': 15 if include_memory else 0,
              'unique_validation_cases': len(set(all_indices)), 'validation_fold_appearances_per_seed': len(all_indices),
              'test_labels_evaluated': False, 'primary_unit': 'paired patient; mean seed-specific metric, not ensemble',
              'full_median_ap': float(np.median(full)), 'zero_median_ap': float(np.median(zero_rows)),
              'modes': table, 'contrasts': contrasts, 'center_class_counts': source_counts,
              'all_budget_rows': budget_rows, 'all_selected_budget_rows': selection_rows, 'mechanism_audit': mechanisms,
              'bootstrap': '1000 within-fold paired patient draws, stratified center/outcome; descriptive, no multiplicity-adjusted superiority claim; overlapping folds not pooled'}
    (OUT/'summary.json').write_text(json.dumps(result, indent=2))
    lines = ['# CSF complete source-development screen', '',
             f"Screen 15/15; supervised memory {'15/15' if include_memory else 'not yet complete'}. 197 unique validation cases, 788 fold appearances per seed. No outer evaluation.", '',
             '| Rule | Median AP (15 runs) | Median paired delta vs random | vs attention | Gate |',
             '|---|---:|---:|---:|---|']
    for m, v in table.items():
        lines.append(f"| {m} | {v['median_15_run_macro_ap']:.4f} | {v['median_paired_delta_random']:+.4f} | {v['median_paired_delta_attention']:+.4f} | {v['advance_to_nested_training']} |")
    lines += ['', f"Full evidence median AP {result['full_median_ap']:.4f}; clinical+colpo zero-OCT median AP {result['zero_median_ap']:.4f}.", '',
              'All acquisition rules use four OCT positions per model for this table. Risk differences are paired before aggregation: do not subtract medians to infer effect sizes.', '',
              '## Claim boundary', '',
              'These are reused source-development folds and frozen predictors, not fresh held-out-hospital validation. First/last-frame substitutions are finite sensitivity probes, not clinical counterfactuals. Calibration cases were seen by the frozen predictor. No patient-specific or cross-center safety certificate is established. A missing gain is not proof that all possible counterfactual approaches fail.', '',
              'Source-training calibration frequently selects zero OCT; that is a warning to examine calibration and Brier/imbalance alignment, not evidence that OCT is clinically unnecessary. Full models still have higher aggregate AP than the zero-OCT baseline.', '',
              'See summary.json for every budget, source-center class count, seed, selected budget and descriptive paired interval. The gate is exploratory resource allocation; no multiplicity-adjusted superiority is asserted.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'modes': table, 'memory_complete': include_memory}, indent=2), flush=True)


if __name__ == '__main__': main()
