"""Matched old/restored clinical-input experiment; patient bootstrap and controls."""
import json
import numpy as np
import torch
import summarize_missing_modality_fusion as sm
from run_missing_modality_fusion import ROOT, OUT, ARMS, NAMES, FOLDS, SEEDS, metrics


def main():
    original_root = OUT
    restored_root = ROOT/'results/missing_modality_fusion_restored_v1'
    sm.OUT = original_root
    old = sm.load_runs()
    sm.OUT = restored_root
    new = sm.load_runs()
    assert len(old) == len(new) == 15
    controls = []
    for before, after in zip(old, new):
        assert (before['fold'], before['seed']) == (after['fold'], after['seed'])
        a, b = before['a'], after['a']
        for key in ('indices', 'y', 'centers', 'cal_indices', 'cal_y'):
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(a['unimodal'][:, 1:], b['unimodal'][:, 1:])
        np.testing.assert_array_equal(a['late'][:, [1, 3, 5]], b['late'][:, [1, 3, 5]])
        for modality in (1, 2):
            relative = f"{before['fold']}/seed_{before['seed']}/branch_{modality}/checkpoint.pt"
            sa = torch.load(original_root/relative, map_location='cpu', weights_only=True)
            sb = torch.load(restored_root/relative, map_location='cpu', weights_only=True)
            assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
        controls.append({'fold': before['fold'], 'seed': before['seed'], 'visual_branches_bitwise_identical': True, 'no_T_late_fusion_identical': True})
    patient_meta = {}
    for row in old:
        a = row['a']
        for i, y, center in zip(a['indices'], a['y'], a['centers']):
            patient_meta[int(i)] = (int(y), str(center))
    ids = sorted(patient_meta)
    lookup = {i: k for k, i in enumerate(ids)}
    weights = np.zeros((1000, len(ids)))
    rng = np.random.default_rng(20260906)
    for stratum in sorted(set(patient_meta.values())):
        group = np.array([lookup[i] for i in ids if patient_meta[i] == stratum])
        samples = rng.integers(len(group), size=(1000, len(group)))
        for k in range(1000):
            weights[k, group] = np.bincount(samples[k], minlength=len(group))
    modes = ('late',)+ARMS
    results = {}
    for mode in modes:
        delta_boot = np.zeros((1000, 7))
        run_deltas = []
        for before, after in zip(old, new):
            a, b = before['a'], after['a']
            w = weights[:, [lookup[int(i)] for i in a['indices']]]
            groups = [a['centers'] == h for h in np.unique(a['centers']) if len(np.unique(a['y'][a['centers'] == h])) == 2]
            delta = []
            for k in range(7):
                delta_boot[:, k] += sum(sm.weighted_ap_draws(a['y'][g], b[mode][g, k], w[:, g])-sm.weighted_ap_draws(a['y'][g], a[mode][g, k], w[:, g]) for g in groups)/len(groups)/15
                delta.append(metrics(a['y'], b[mode][:, k], a['centers'])['macro_ap']-metrics(a['y'], a[mode][:, k], a['centers'])['macro_ap'])
            run_deltas.append({'fold': before['fold'], 'seed': before['seed'], 'deltas': delta})
        point = np.mean([r['deltas'] for r in run_deltas], axis=0)
        results[mode] = {'complete': {'delta_macro_ap': float(point[-1]), 'patient_paired_95ci': np.quantile(delta_boot[:, -1], [.025, .975]).tolist()},
            'seven_patterns': {'delta_macro_ap': float(point.mean()), 'patient_paired_95ci': np.quantile(delta_boot.mean(1), [.025, .975]).tolist()},
            'by_seed_complete_delta': {str(s): float(np.mean([r['deltas'][-1] for r in run_deltas if r['seed'] == s])) for s in SEEDS},
            'by_fold_complete_delta': {f: float(np.mean([r['deltas'][-1] for r in run_deltas if r['fold'] == f])) for f in FOLDS},
            'by_pattern_delta': dict(zip(NAMES, point.tolist())), 'all_run_deltas': run_deltas}
    clinical = {}
    for name, rows in [('before', old), ('after', new)]:
        clinical[name] = {m: float(np.mean([metrics(r['a']['y'], r['a']['unimodal'][:, 0], r['a']['centers'])[m] for r in rows])) for m in ('macro_ap', 'auroc', 'ce', 'brier')}
    clinical_boot = np.zeros(1000)
    for before, after in zip(old, new):
        a, b = before['a'], after['a']
        w = weights[:, [lookup[int(i)] for i in a['indices']]]
        groups = [a['centers'] == h for h in np.unique(a['centers']) if len(np.unique(a['y'][a['centers'] == h])) == 2]
        clinical_boot += sum(sm.weighted_ap_draws(a['y'][g], b['unimodal'][g, 0], w[:, g])-sm.weighted_ap_draws(a['y'][g], a['unimodal'][g, 0], w[:, g]) for g in groups)/len(groups)/15
    clinical['paired_delta_macro_ap'] = clinical['after']['macro_ap']-clinical['before']['macro_ap']
    clinical['paired_95ci'] = np.quantile(clinical_boot, [.025, .975]).tolist()
    report = {'paired_runs': 15, 'unique_validation_patients': len(ids), 'repair_contrasts': results, 'clinical_only': clinical,
        'negative_controls': controls, 'test_labels_evaluated': False, 'clinical_verification': 'pending',
        'interpretation': 'Data source fidelity contrast, not innovation-module efficacy. Descriptive unadjusted bootstrap intervals; no clinical causal or external validation claim.'}
    (restored_root/'clinical_recovery_comparison.json').write_text(json.dumps(report, indent=2))
    lines = ['# 原始临床字段恢复：修复前后配对对照', '', '15个相同划分-种子对照；不改影像、标签、超参数、患者集合或划分。', '',
        '| 方法 | 完整输入 Δmacro-AP [95% CI] | 七种模式平均 Δmacro-AP [95% CI] |', '|---|---|---|']
    for mode, entry in results.items():
        cols = []
        for key in ('complete', 'seven_patterns'):
            e = entry[key]
            lo, hi = e['patient_paired_95ci']
            cols.append(f"{e['delta_macro_ap']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        lines.append('| '+mode+' | '+' | '.join(cols)+' |')
    lines += ['', f"临床单模态 macro-AP：{clinical['before']['macro_ap']:.4f} → {clinical['after']['macro_ap']:.4f}。", '',
        '30个视觉分支检查点逐参数完全相同；后期融合中不含临床输入的C/O/CO预测完全相同。该阴性对照支持变化来自临床字段恢复，而不是意外修改图像训练或患者划分。', '',
        '原始临床字段仍需医学语义复核；本结果属于数据修复效果，不应冒充新的算法模块收益。区间是197名独立患者的探索性配对区间，未多重比较校正。']
    (restored_root/'CLINICAL_RECOVERY_COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status': 'clinical_recovery_comparison_complete', 'clinical_only': clinical, 'contrasts': {m: e['complete'] for m, e in results.items()}}), flush=True)


if __name__ == '__main__':
    main()
