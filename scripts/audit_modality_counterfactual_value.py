"""Retrospective functional dependence, not identified clinical causal effects."""
import collections
import json
import numpy as np
import summarize_missing_modality_fusion as sm
from run_missing_modality_fusion import ROOT, ARMS, NAMES, cross_entropy


def main():
    sm.OUT = ROOT/'results/missing_modality_fusion_restored_v1'
    rows = sm.load_runs()
    assert len(rows) == 15
    report = {'input': 'restored clinical source', 'runs': 15,
        'definition': 'necessity_T=R(CO)-R(TCO); positive means deleting T worsens observed-label CE',
        'interaction': 'O_C_given_T=R(TO)+R(TC)-R(TCO)-R(T), model loss complementarity, not biological synergy',
        'limitations': 'Outcome required only for retrospective audit; not a deployable oracle, clinical causal identification or lesion-preserving nuisance intervention',
        'models': {}}
    for mode in ('late',)+ARMS:
        patients = collections.defaultdict(list)
        metadata = {}
        for row in rows:
            a = row['a']
            risk = {p: cross_entropy(a['y'], a[mode][:, k]) for k, p in enumerate(NAMES)}
            values = np.stack((risk['CO']-risk['TCO'], risk['TO']-risk['TCO'], risk['TC']-risk['TCO'],
                risk['TO']+risk['TC']-risk['TCO']-risk['T']), 1)
            for i, y, center, value in zip(a['indices'], a['y'], a['centers'], values):
                patients[int(i)].append(value)
                metadata[int(i)] = (int(y), str(center))
        ids = sorted(patients)
        mean = np.stack([np.mean(patients[i], axis=0) for i in ids])
        y = np.array([metadata[i][0] for i in ids])
        center = np.array([metadata[i][1] for i in ids])
        groups = [np.where((center == h)&(y == label))[0] for h in np.unique(center) for label in (0, 1)]
        groups = [g for g in groups if len(g)]
        rng = np.random.default_rng(20260906)
        boot = np.stack([mean[np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])].mean(0) for _ in range(1000)])
        result = {'unique_patients': len(ids), 'patient_appearance_averaging': 'First average across seeds and overlapping source folds within patient; then count each patient once', 'effects': {}}
        for k, effect in enumerate(('necessity_T', 'necessity_C', 'necessity_O', 'O_C_given_T')):
            result['effects'][effect] = {'mean_ce_effect': float(mean[:, k].mean()), 'positive_fraction': float(np.mean(mean[:, k] > 0)),
                'ci95_patient_stratified': np.quantile(boot[:, k], [.025, .975]).tolist(),
                'mean_CIN2_negative': float(mean[y == 0, k].mean()), 'mean_CIN2_positive': float(mean[y == 1, k].mean())}
        report['models'][mode] = result
    (sm.OUT/'modality_counterfactual_value_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({m: r['effects']['necessity_T'] for m, r in report['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
