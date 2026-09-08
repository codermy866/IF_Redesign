"""Recover original clinical strings by exact OCT acquisition ID, never outcome."""
import collections
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import torch
from audit_oct_site_labels import read_sheets, WORKBOOK
import train_ices_backbone as tr
from cervix_cogalign.cesl import hpv_group, tct_group

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/clinical_source_recovery_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = torch.load(ROOT/'results/ices_v1/features/ices_position_cluster_resnet50.pt', map_location='cpu', weights_only=False)
    cases = [json.loads(s) for s in (ROOT/'results/ices_v1/cluster_manifest.jsonl').read_text().splitlines()]
    assert [r['id'] for r in cases] == list(base['ids'])
    source = collections.defaultdict(set)
    for _, rows in read_sheets(WORKBOOK):
        for row in rows:
            if 'HPV清洗（高亮表示阳性）' not in row:
                continue
            key = str(row.get('OCT图像Id', '')).strip().upper()
            if key:
                source[key].add((row.get('HPV清洗（高亮表示阳性）', '').strip(), row.get('TCT清洗（高亮表示阳性）', '').strip()))
    recovered = {'hpv': [], 'tct': []}
    reasons = collections.Counter()
    transitions = {k: collections.Counter() for k in recovered}
    center_counts = collections.defaultdict(collections.Counter)
    raw_one_replacements = {k: collections.Counter() for k in recovered}
    for i, case in enumerate(cases):
        prefixes = {re.split('_circle_', Path(p).name, flags=re.I)[0].upper() for u in case['evidence_units'] if u['kind'] == 'oct_position_cluster' for p in u['frame_paths']}
        assert len(prefixes) == 1
        values = source.get(next(iter(prefixes)), set())
        reason = 'matched_unique' if len(values) == 1 else ('conflict' if len(values) > 1 else 'unmatched')
        reasons[reason] += 1
        if reason != 'matched_unique':
            raise ValueError('Exact unambiguous clinical-source join required before recovery')
        hpv, tct = next(iter(values))
        for key, value, group in [('hpv', hpv, hpv_group), ('tct', tct, tct_group)]:
            old = str(base[key+'s'][i]).strip()
            recovered[key].append(value)
            transitions[key][group(old)+' -> '+group(value)] += 1
            center_counts[str(case['center'])][key+'_unknown_before'] += group(old) == 'unknown'
            center_counts[str(case['center'])][key+'_unknown_after'] += group(value) == 'unknown'
            if old in ('1', '1.0'):
                raw_one_replacements[key][value] += 1
    np.savez_compressed(OUT/'clinical_sidecar.npz', ids=np.array(base['ids']), hpvs=np.array(recovered['hpv']), tcts=np.array(recovered['tct']))
    report = {'patients': len(cases), 'join_status': dict(reasons), 'join_key': 'Exact shared OCT acquisition prefix, no outcome or fuzzy patient matching',
        'source_workbook_sha256': hashlib.sha256(WORKBOOK.read_bytes()).hexdigest(), 'transitions': {k: dict(v) for k, v in transitions.items()},
        'cache_one_original_values': {k: dict(v) for k, v in raw_one_replacements.items()}, 'center_unknown_counts': {k: dict(v) for k, v in center_counts.items()},
        'fields_changed': ['HPV source string', 'TCT source string'], 'unchanged': ['age', 'patient cohort', 'case outcome', 'image features', 'splits', 'normalization functions'],
        'clinical_semantic_review': 'pending; restore source text, do not infer meaning of legacy 1 or spreadsheet color'}
    (OUT/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
