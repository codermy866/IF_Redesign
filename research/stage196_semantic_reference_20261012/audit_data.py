"""Five-center source audit; never export patient identities to public code."""
import os
os.umask(0o077)
from pathlib import Path
import collections
import concurrent.futures
import hashlib
import importlib.util
import json
import re
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
AUTHORITATIVE = Path(os.environ.get('IF_AUTHORITATIVE_LABELS', 'private/AUTHORITATIVE_LABELS.csv'))
WORKBOOK = Path(os.environ.get('IF_WEAK_LABEL_WORKBOOK', 'private/weak_labels.xlsx'))
CENTERS = {'恩施州中心医院':'enshi','荆州市第一人民医院':'jingzhou',
           '十堰市人民医院':'shiyan','襄阳市中心医院':'xiangyang','武大人民医院':'wuhan'}


def save(name, value):
    p=ROOT/name; p.parent.mkdir(parents=True, exist_ok=True)
    temporary=p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(p)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def image_info(path):
    p=Path(path)
    try:
        stat=p.stat()
        with Image.open(p) as image:
            size=image.size;image.verify()
        return dict(path=str(p.resolve()),valid=True,width=size[0],height=size[1],
                    bytes=stat.st_size,inode=f'{stat.st_dev}:{stat.st_ino}',sha256=sha(p))
    except (OSError,ValueError,Image.DecompressionBombError) as error:
        return dict(path=str(p),valid=False,error=type(error).__name__)


def category(value):
    s=str(value).strip()
    return 'missing' if s.lower() in ('','nan','none','null','missing','未知','未查','未检') else s


def main():
    started=time.monotonic()
    manifest=[json.loads(x) for x in (BASE/'results/ices_v1/cluster_manifest.jsonl').read_text().splitlines()]
    assert len({str(x['id']) for x in manifest})==len(manifest)
    assert len({str(x['patient_id']) for x in manifest})==len(manifest)
    labels=pd.read_csv(AUTHORITATIVE,dtype={'case_id':str}).set_index('case_id',verify_integrity=True)
    with np.load(BASE/'results/oct_evidence_repair_20260912/data.npz',allow_pickle=False) as f:
        clinical={str(case):dict(age=float(age) if np.isfinite(float(age)) and 0<float(age)<120 else None,
                                hpv=category(hpv),tct=category(tct))
                  for case,age,hpv,tct in zip(f['ids'],f['ages'],f['hpvs'],f['tcts'])}
        old_site_known={str(case):bool((s>=0).all()) for case,s in zip(f['ids'],f['site'])}
        old_y=dict(zip(f['ids'].astype(str),f['y'].astype(int)))
    with np.load(BASE/'results/oct_site_supervision_v1/site_labels.npz',allow_pickle=False) as f:
        weak={str(case):dict(reread=s.tolist(),reread_valid=bool(v))
              for case,s,v in zip(f['ids'],f['labels'],f['valid'])}
    # Reuse the established strict CIN3 parser and acquisition-prefix workbook join.
    sys.path.insert(0,str(BASE/'scripts'))
    from audit_oct_site_labels import read_sheets
    spec=importlib.util.spec_from_file_location('strict_cin3',BASE/'73_clinical_deescalation_audit_20261003/extract_cin3.py')
    parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    assert parser.tests()==16 and sha(WORKBOOK)==parser.EXPECTED_SHA
    medical=collections.defaultdict(list);headers=set()
    for sheet,rows in read_sheets(WORKBOOK):
        for row in rows:
            headers.update(row)
            prefix=str(row.get('OCT图像Id','')).strip().upper()
            if prefix:medical[prefix].append(row)
    old_inputs=[json.loads(x) for x in (BASE/'12_oct_incremental_gate/patients.jsonl').read_text().splitlines()]
    old_inputs+=[json.loads(x) for x in (BASE/'14_wuhan_external/wuhan_inputs.jsonl').read_text().splitlines()]
    existing={str(x['case_id']):x for x in old_inputs}
    rows=[];all_paths=set();path_owner=collections.defaultdict(set)
    for patient in manifest:
        case=str(patient['id']);key='p_'+hashlib.sha256(str(patient['patient_id']).encode()).hexdigest()[:16]
        y=int(labels.loc[case,'pathology_cin2plus']);assert y in (0,1) and y==old_y[case]
        center=CENTERS[patient['center']]
        units=patient['evidence_units']
        sites={str(int(u['scanner_position'][0])):sorted(u['frame_paths'],key=lambda s:[int(v) if v.isdigit() else v for v in re.split('(\\d+)',Path(s).name)])
               for u in units if u['kind']=='oct_position_cluster'}
        colpo=[u['path'] for u in sorted((u for u in units if u['kind']=='colposcopy_view'),key=lambda u:u['unit_id'])]
        prefixes={re.split('_circle_',Path(p).name,flags=re.I)[0].upper() for ps in sites.values() for p in ps}
        matched=medical.get(next(iter(prefixes)),[]) if len(prefixes)==1 else []
        parsed=[parser.parse(m.get('病理结果','')) for m in matched]
        determinate={v[0] for v in parsed if v[0] is not None}
        cin3=next(iter(determinate)) if len(determinate)==1 and all(v[0] is not None for v in parsed) else None
        status=parsed[0][1] if len(parsed)==1 else 'unresolved_join_or_multiple_reports'
        if cin3==1:assert y==1,'Pathology endpoint conflict'
        row=dict(key=key,case_id=case,center=center,y=y,cin3=cin3,cin3_parse_status=status,
                 **clinical[case],**weak[case],old_site_known=old_site_known[case],
                 oct_sites=sites,colpo_paths=colpo,old_input=existing.get(case))
        if row['old_input']:assert row['old_input']['patient_key']==key
        rows.append(row)
        for p in [p for ps in sites.values() for p in ps]+colpo:
            all_paths.add(p);path_owner[str(Path(p).resolve())].add(key)
    assert len({r['key'] for r in rows})==len(rows)==984
    save('AUDIT_STATE.json',dict(state='CHECKING_IMAGE_BYTES',patients=len(rows),unique_images=len(all_paths)))
    info={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,item in enumerate(pool.map(image_info,sorted(all_paths)),1):
            info[item['path']]=item
            if i%5000==0:save('AUDIT_STATE.json',dict(state='CHECKING_IMAGE_BYTES',completed=i,total=len(all_paths),seconds=time.monotonic()-started))
    hashes=collections.defaultdict(set);inode_owners=collections.defaultdict(set)
    for path,owners in path_owner.items():
        value=info[path]
        if value['valid']:
            hashes[value['sha256']].update(owners);inode_owners[value['inode']].update(owners)
    shared_hashes={h:sorted(owners) for h,owners in hashes.items() if len(owners)>1}
    shared_paths={p:sorted(owners) for p,owners in path_owner.items() if len(owners)>1}
    private_rows=[];summaries=[]
    for row in rows:
        readable={s:[p for p in paths if info[str(Path(p).resolve())]['valid']] for s,paths in row['oct_sites'].items()}
        valid_colpo=[p for p in row['colpo_paths'] if info[str(Path(p).resolve())]['valid']]
        private_rows.append({**row,'oct_sites':readable,'colpo_paths':valid_colpo})
    for c in sorted(CENTERS.values()):
        rr=[r for r in private_rows if r['center']==c];age=np.array([r['age'] for r in rr if r['age'] is not None])
        summaries.append(dict(center=c,N=len(rr),CIN2plus=sum(r['y'] for r in rr),
            positive_rate=sum(r['y'] for r in rr)/len(rr),
            CIN3_known=sum(r['cin3'] is not None for r in rr),CIN3_positive=sum(r['cin3']==1 for r in rr),
            CIN3_unresolved=sum(r['cin3'] is None for r in rr),weak_patients_known=sum(r['reread_valid'] for r in rr),
            OCT_sites_total=sum(sum(bool(ps) for ps in r['oct_sites'].values()) for r in rr),
            OCT_frames_total=sum(sum(len(ps) for ps in r['oct_sites'].values()) for r in rr),
            colposcopy_total=sum(len(r['colpo_paths']) for r in rr),
            patients_without_OCT=sum(not any(r['oct_sites'].values()) for r in rr),
            patients_without_colposcopy=sum(not r['colpo_paths'] for r in rr),
            age_missing=sum(r['age'] is None for r in rr),hpv_missing=sum(r['hpv']=='missing' for r in rr),
            tct_missing=sum(r['tct']=='missing' for r in rr),
            age_median=float(np.median(age)) if len(age) else None,
            age_q25=float(np.quantile(age,.25)) if len(age) else None,
            age_q75=float(np.quantile(age,.75)) if len(age) else None))
    save('private/COHORT.json',private_rows)
    save('private/IMAGE_AUDIT.json',info)
    save('private/CROSS_PATIENT_IMAGE_COLLISIONS.json',dict(shared_content=shared_hashes,shared_paths=shared_paths))
    pd.DataFrame(summaries).to_csv(ROOT/'FIVE_CENTER_DATA_AUDIT.csv',index=False)
    splits={}
    for outer in sorted(CENTERS.values()):
        source=[r['key'] for r in private_rows if r['center']!=outer]
        test=[r['key'] for r in private_rows if r['center']==outer]
        assert not set(source)&set(test)
        inner={c:dict(train=[r['key'] for r in private_rows if r['center'] not in (outer,c)],
                      validation=[r['key'] for r in private_rows if r['center']==c])
               for c in sorted(CENTERS.values()) if c!=outer}
        splits[outer]=dict(source=source,outer_test=test,inner_source_center_CV=inner)
    save('private/FIVE_CENTER_LOCO_SPLITS.json',splits)
    summary=dict(state='AUDITED_WITH_LIMITATIONS' if not shared_hashes else 'IMAGE_COLLISION_REVIEW_REQUIRED',
        N=len(rows),CIN2plus=sum(r['y'] for r in rows),centers=summaries,
        global_registry_patient_ids_unique=True,patient_keys_unique=True,
        actual_same_person_with_different_registry_ids='NOT_VERIFIABLE_FROM_RELEASED_DEIDENTIFIED_DATA',
        cross_patient_shared_paths=len(shared_paths),cross_patient_shared_image_hashes=len(shared_hashes),
        invalid_images=sum(not x['valid'] for x in info.values()),
        previously_excluded_source_records=len([r for r in rows if r['old_input'] is None]),
        previously_excluded_weak_labels_known=[r['reread_valid'] for r in rows if r['old_input'] is None],
        equipment_fields=[s for s in headers if re.search('设备|型号|仪器|成像系统',s)],
        equipment_metadata='UNKNOWN_UNLESS_LISTED',quality_scope='readability, resolution, file integrity; no clinical quality grading',
        CIN3_definition='strict explicit CIN3/cervical invasion; unresolved/AIS separate, no human adjudication',
        Wuhan_sampling=json.loads((BASE/'14_wuhan_external/268_WUHAN_SAMPLING_GATE.json').read_text()),
        validation_scope='five-center selected-cohort cross-center internal development; not independent confirmation',
        historical_test_exposure=True,new_protocol_outcomes_used_for_selection=False,
        source_hashes={str(p):sha(p) for p in [AUTHORITATIVE,WORKBOOK,BASE/'results/ices_v1/cluster_manifest.jsonl',BASE/'results/oct_site_supervision_v1/site_labels.npz']},
        seconds=time.monotonic()-started)
    save('DATA_AUDIT.json',summary);save('AUDIT_STATE.json',dict(state=summary['state'],seconds=summary['seconds']))
    print(json.dumps({k:summary[k] for k in ['state','N','CIN2plus','cross_patient_shared_paths','cross_patient_shared_image_hashes','invalid_images','previously_excluded_source_records','previously_excluded_weak_labels_known','seconds']},indent=2),flush=True)


if __name__=='__main__':main()
