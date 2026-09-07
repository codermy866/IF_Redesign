"""All-arm source-development results, nested masks and patient-paired intervals."""
import json
import argparse
from pathlib import Path
import numpy as np
from run_missing_modality_fusion import ROOT, OUT, FOLDS, SEEDS, ARMS, NAMES, PATTERNS, metrics, threshold90


def weighted_ap_draws(y, p, weights):
    """Vectorized AP for bootstrap patient multiplicities, correctly handling ties."""
    order = np.argsort(-p, kind='stable')
    yy, pp, ww = y[order], p[order], weights[:, order]
    ends = np.r_[np.where(np.diff(pp) != 0)[0], len(pp)-1]
    tp = np.cumsum(ww*yy, axis=1)[:, ends]
    total = np.cumsum(ww, axis=1)[:, ends]
    increments = np.diff(np.c_[np.zeros(len(tp)), tp], axis=1)
    return ((tp/np.maximum(total, 1e-12))*increments).sum(1)/np.maximum(tp[:, -1], 1e-12)


def load_runs():
    rows = []
    for fold in FOLDS:
        for seed in SEEDS:
            path = OUT/fold/f'seed_{seed}'
            if not (path/'complete.json').exists():
                continue
            a = dict(np.load(path/'predictions.npz', allow_pickle=False))
            if (path/'legacy_missingness.npz').exists():
                legacy = np.load(path/'legacy_missingness.npz', allow_pickle=False)
                np.testing.assert_array_equal(a['indices'], legacy['indices'])
                np.testing.assert_array_equal(a['cal_indices'], legacy['cal_indices'])
                a['legacy'], a['cal_legacy'] = legacy['legacy'], legacy['cal_legacy']
            rows.append({'fold': fold, 'seed': seed, 'a': a, 'meta': json.loads((path/'complete.json').read_text())})
    return rows


def paired_bootstrap(rows, modes, draws=1000):
    patient_meta = {}
    for row in rows:
        a = row['a']
        for i, y, center in zip(a['indices'], a['y'], a['centers']):
            value = (int(y), str(center))
            assert int(i) not in patient_meta or patient_meta[int(i)] == value
            patient_meta[int(i)] = value
    ids = sorted(patient_meta)
    lookup = {v: k for k, v in enumerate(ids)}
    weights = np.zeros((draws, len(ids)), dtype=int)
    rng = np.random.default_rng(20260906)
    for stratum in sorted(set(patient_meta.values())):
        group = np.array([lookup[i] for i in ids if patient_meta[i] == stratum])
        draws_ix = rng.integers(len(group), size=(draws, len(group)))
        for b in range(draws):
            weights[b, group] = np.bincount(draws_ix[b], minlength=len(group))
    boot = {m: np.zeros((draws, 7)) for m in modes}
    point = {m: np.zeros(7) for m in modes}
    for row in rows:
        a = row['a']
        w = weights[:, [lookup[int(i)] for i in a['indices']]]
        groups = [a['centers'] == h for h in np.unique(a['centers']) if len(np.unique(a['y'][a['centers'] == h])) == 2]
        for mode in modes:
            for k in range(7):
                boot[mode][:, k] += sum(weighted_ap_draws(a['y'][g], a[mode][g, k], w[:, g]) for g in groups)/len(groups)/len(rows)
                point[mode][k] += metrics(a['y'], a[mode][:, k], a['centers'])['macro_ap']/len(rows)
    comparisons = [('frozen_missing', 'frozen_full'), ('tune_missing', 'tune_full'), ('frozen_full', 'tune_full'), ('frozen_missing', 'tune_missing'), ('frozen_missing', 'late'), ('tune_missing', 'late')]
    result = {}
    for left, right in comparisons:
        delta = boot[left]-boot[right]
        dpoint = point[left]-point[right]
        result[left+' minus '+right] = {name: {'delta': float(dp), 'patient_bootstrap_95ci': np.quantile(db, [.025, .975]).tolist()} for name, dp, db in
            [('complete', dpoint[-1], delta[:, -1]), ('seven_pattern_average', dpoint.mean(), delta.mean(1))]}
    interaction = boot['frozen_missing']-boot['frozen_full']-boot['tune_missing']+boot['tune_full']
    dp = point['frozen_missing']-point['frozen_full']-point['tune_missing']+point['tune_full']
    result['interaction'] = {name: {'delta': float(p), 'patient_bootstrap_95ci': np.quantile(b, [.025, .975]).tolist()} for name, p, b in
        [('complete', dp[-1], interaction[:, -1]), ('seven_pattern_average', dp.mean(), interaction.mean(1))]}
    return result, len(ids)


def progressive_curves(rows, modes):
    output = {}
    # Same global patient-level uniforms across folds, seeds and models. Nested rates.
    n = max(int(i) for row in rows for i in row['a']['indices'])+1
    uniforms = np.random.default_rng(271828).random((20, 3, n))
    for modality, name in enumerate('TCO'):
        absent = np.ones(3, dtype=bool)
        absent[modality] = False
        k = int(np.where((PATTERNS == absent).all(1))[0][0])
        output[name] = {}
        for mode in modes:
            curve = []
            for rate in (0., .25, .5, .75, 1.):
                scores, actual = [], []
                for replicate in range(20):
                    run_scores = []
                    for row in rows:
                        a = row['a']
                        missing = uniforms[replicate, modality, a['indices']] < rate
                        p = np.where(missing, a[mode][:, k], a[mode][:, -1])
                        run_scores.append(metrics(a['y'], p, a['centers'])['macro_ap'])
                        actual.append(float(missing.mean()))
                    scores.append(float(np.mean(run_scores)))
                curve.append({'requested_missing_fraction': rate, 'actual_mean_fraction': float(np.mean(actual)), 'macro_ap': float(np.mean(scores)),
                    'mask_replicate_sd_not_patient_ci': float(np.std(scores, ddof=1))})
            output[name][mode] = curve
    return output


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output
    rows = load_runs()
    if len(rows) != 15:
        raise RuntimeError(f'Only {len(rows)}/15 complete: no complete-matrix report')
    modes = ('late',)+ARMS+(('legacy',) if all('legacy' in r['a'] for r in rows) else ())
    result = {'runs_complete': len(rows), 'new_unimodal_models': 45, 'new_fusion_models': 60, 'test_labels_evaluated': False,
        'validation_status': 'exploratory_source_validation', 'human_review': 'pending', 'availability_audit': rows[0]['meta']['audit'],
        'averages': {}, 'by_seed': {}, 'by_fold': {}, 'by_run': []}
    for row in rows:
        a = row['a']
        entry = {'fold': row['fold'], 'seed': row['seed'], 'cal_n': row['meta']['cal_n'], 'cal_positive_n': row['meta']['cal_positive_n'], 'metrics': {}}
        for mode in modes:
            entry['metrics'][mode] = {name: metrics(a['y'], a[mode][:, k], a['centers'], threshold90(a['cal_y'], a['cal_'+mode][:, k])) for k, name in enumerate(NAMES)}
        entry['unimodal'] = {name: metrics(a['y'], a['unimodal'][:, k], a['centers']) for k, name in enumerate('TCO')}
        result['by_run'].append(entry)
    for mode in modes:
        result['averages'][mode] = {name: {metric: float(np.mean([r['metrics'][mode][name][metric] for r in result['by_run']])) for metric in ('macro_ap', 'auroc', 'ce', 'brier', 'sensitivity', 'specificity')} for name in NAMES}
        for key, values, field in [('by_seed', SEEDS, 'seed'), ('by_fold', FOLDS, 'fold')]:
            result[key][mode] = {str(v): {'complete_macro_ap': float(np.mean([r['metrics'][mode]['TCO']['macro_ap'] for r in result['by_run'] if r[field] == v])),
                'seven_pattern_macro_ap': float(np.mean([r['metrics'][mode][p]['macro_ap'] for r in result['by_run'] if r[field] == v for p in NAMES]))} for v in values}
    result['paired_contrasts'], result['unique_validation_patients'] = paired_bootstrap(rows, modes)
    print(json.dumps({'phase': 'paired_bootstrap_complete'}), flush=True)
    result['progressive_missingness'] = progressive_curves(rows, modes)
    result['inference_caveats'] = ['1000 patient-stratified paired resamples; same patient draws propagated across overlapping folds and all seeds',
        'Unadjusted descriptive intervals; no multiple-testing superiority claim', 'Seed averages are not probability ensembles',
        'Equal seven-pattern deployment mixture is synthetic, not observed clinical availability frequency',
        'Missingness robustness is model dependence, not lesion-preserving causal invariance', 'Legacy differs in architecture, loss, early stopping and capacity']
    (OUT/'summary.json').write_text(json.dumps(result, indent=2))
    text = ['# 模态缺失与融合保护：固定2×2实验', '', '15/15运行完成；45个单模态模型、60个融合模型。仅源中心开发验证，未评价留出医院。', '',
            '| 方法 | 完整输入 macro-AP | 七种可用模式平均 macro-AP | 完整输入 AUROC |', '|---|---:|---:|---:|']
    for mode in modes:
        av = result['averages'][mode]
        text.append(f"| {mode} | {av['TCO']['macro_ap']:.4f} | {np.mean([av[p]['macro_ap'] for p in NAMES]):.4f} | {av['TCO']['auroc']:.4f} |")
    text += ['', '## Paired Factor Effects', '', '| Contrast | Complete-input delta AP [95% CI] | Seven-pattern mean delta AP [95% CI] |', '|---|---|---|']
    for name, effects in result['paired_contrasts'].items():
        cols = []
        for key in ('complete', 'seven_pattern_average'):
            e = effects[key]
            lo, hi = e['patient_bootstrap_95ci']
            cols.append(f"{e['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        text.append('| '+name+' | '+' | '.join(cols)+' |')
    text += ['', f"Unique validation patients: {result['unique_validation_patients']}. Repeated folds and random seeds do not increase the case count. Intervals are unadjusted exploratory paired patient-bootstrap intervals.", '',
             'This run does not use site supervision, OOF gain, or LLM verification, so results must not be attributed to those modules. If all cases have all three modalities, missingness results are simulated stress tests rather than a naturally missing cohort validation.', '',
             'Per-centre, per-seed, seven-pattern metrics, nested missingness curves, internal calibration thresholds, and calibration positive counts are in summary.json. Human clinical review remains pending and this is not clinical deployment certification.']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'status': 'summary_complete', 'runs': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
