"""Build a private site-label sidecar; never select evaluation images by labels."""
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
from collections import Counter, defaultdict
import hashlib
import json
import re
import unicodedata
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = Path('/data2/Colposcopy-3000data/3000_nums.xlsx')
OUT = ROOT/'results/oct_site_supervision_v1'
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def read_sheets(path):
    with ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(t.text or '' for t in si.findall('.//s:t', NS)) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si', NS)]
        for name in sorted(z.namelist()):
            if not re.fullmatch(r'xl/worksheets/sheet\d+\.xml', name): continue
            rows = []
            for row in ET.fromstring(z.read(name)).findall('s:sheetData/s:row', NS):
                cells = {}
                for cell in row.findall('s:c', NS):
                    col = re.sub(r'\d', '', cell.attrib['r']); node = cell.find('s:v', NS)
                    value = '' if node is None else node.text or ''
                    if cell.attrib.get('t') == 's' and value: value = strings[int(value)]
                    elif cell.attrib.get('t') == 'inlineStr': value = ''.join(t.text or '' for t in cell.findall('.//s:t', NS))
                    cells[col] = value
                rows.append(cells)
            if not rows: continue
            header = rows[0]
            yield name, [{header[k]: value for k, value in r.items() if k in header} for r in rows[1:]]


def parse_sites(value):
    text = unicodedata.normalize('NFKC', str(value or '')).strip()
    if not text: return None, 'missing'
    # Ranges must be expanded, never interpreted as just their endpoints.
    sites = set()
    for match in re.finditer(r'(\d+)\s*[-~至到—–]\s*(\d+)', text):
        a, b = map(int, match.groups())
        if not 1 <= a <= b <= 12: return None, 'invalid_range'
        sites.update(range(a, b+1))
    numbers = [int(v) for v in re.findall(r'\d+', text)]
    if any(v < 1 or v > 12 for v in numbers): return None, 'invalid_number'
    if re.search(r'\d+\.\d+', text): return None, 'ambiguous_decimal'
    sites.update(numbers)
    if not sites and text not in ('未发现', '阴性', '未见异常', '正常'): return None, 'unresolved_no_site_text'
    return sorted(sites), 'parsed'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    indexed = defaultdict(list); statuses = Counter(); text_types = Counter(); sheet_rows = {}
    for sheet, rows in read_sheets(WORKBOOK):
        if not rows or 'OCT图像Id' not in rows[0] or 'OCT二次判读' not in rows[0]: continue
        sheet_rows[sheet] = len(rows)
        for row in rows:
            key = str(row.get('OCT图像Id', '')).strip().upper()
            if not key: continue
            sites, status = parse_sites(row.get('OCT二次判读', ''))
            statuses[status] += 1; text_types[row.get('OCT二次判读', '')] += 1
            indexed[key].append((sites, status, sheet))
    labels = {}; conflicts = set()
    for key, records in indexed.items():
        parsed = {tuple(r[0]) for r in records if r[0] is not None}
        if len(parsed) > 1: conflicts.add(key)
        elif len(parsed) == 1 and all(r[0] is not None for r in records): labels[key] = list(next(iter(parsed)))
    cohort = [json.loads(line) for line in (ROOT/'results/ices_v1/cluster_manifest.jsonl').read_text().splitlines()]
    targets = []; valid = []; ids = []; counts = Counter(); centers = defaultdict(Counter)
    with (OUT/'private_site_labels.jsonl').open('w') as out:
        for case in cohort:
            units = [u for u in case['evidence_units'] if u['kind'] == 'oct_position_cluster']
            units = sorted(units, key=lambda u: tuple(u['scanner_position']))
            c_values = [u['scanner_position'][0] for u in units]
            mapped = c_values == list(range(1, 13))
            prefixes = {re.split(r'_circle_', Path(p).name, flags=re.I)[0].upper() for u in units for p in u['frame_paths']}
            key = next(iter(prefixes)) if len(prefixes) == 1 else ''
            reason = 'matched' if key in labels and mapped else ('conflicting_labels' if key in conflicts else 'unmatched_or_unparsed')
            if not mapped: reason = 'invalid_C_site_mapping'
            ok = reason == 'matched'; sites = labels.get(key, []) if ok else []
            target = [int(c in sites) for c in c_values] if mapped else [-1]*12
            ids.append(case['id']); targets.append(target if ok else [-1]*12); valid.append(ok)
            counts[reason] += 1
            if ok:
                counts['positive_sites'] += sum(target); counts['negative_sites'] += 12-sum(target)
                counts['cases_with_positive_sites'] += int(bool(sites)); centers[case['center']]['matched_cases'] += 1
            out.write(json.dumps({'id': case['id'], 'valid': ok, 'reason': reason, 'C_sites': c_values, 'site_labels': targets[-1]}, ensure_ascii=False)+'\n')
    np.savez_compressed(OUT/'site_labels.npz', ids=np.array(ids), labels=np.array(targets, dtype=np.int8), valid=np.array(valid))
    result = {'workbook_sha256': hashlib.sha256(WORKBOOK.read_bytes()).hexdigest(), 'sheet_rows': sheet_rows,
              'parse_status_rows_across_sheets': dict(statuses), 'conflicting_unique_ids': len(conflicts),
              'cohort_n': len(cohort), 'cohort_audit': dict(counts), 'center_coverage': dict(centers),
              'user_confirmed_rule': 'listed numbers are positive sites; unlisted sites are negative; grades are not substituted for site labels',
              'mapping': 'C field 1-12, sorted as cached slots 25-36; NOT S field or lexicographic filename order',
              'label_scope': 'OCT reread site-level annotation, not frame/pixel ground truth or CIN2+ pathology',
              'unresolved_annotation_types': {k: v for k,v in text_types.items() if parse_sites(k)[0] is None}}
    (OUT/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'unresolved_annotation_types'}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
