import runtime
from runtime import HERE,BASE,ROOT,PREVIOUS,config,save,sha
import csv,json,inspect,subprocess,collections,time
import numpy as np,joblib
import rime_dataset
from if_redesign.rime import readout
from data import prepare

def main():
    cfg=config();rows=rime_dataset.load_patients();keys=[r['key'] for r in rows]
    assert len(keys)==len(set(keys))==984
    assert len(set((r['center'],str(r['case_id'])) for r in rows))==len(rows)
    y=np.array([r['y'] for r in rows]);g=np.array([r['center'] for r in rows]);assert set(g)==set(cfg['centers']) and set(y)=={0,1}
    table=[];shared_paths=collections.defaultdict(set)
    for row in rows:
        for paths in row['oct_sites'].values():
            for path in paths:shared_paths[str(path)].add(row['key'])
        for path in row['colpo_paths']:shared_paths[str(path)].add(row['key'])
    assert not any(len(owners)>1 for owners in shared_paths.values()),'Imaging path shared across patients'
    manifest=json.loads((PREVIOUS/'EXPERIMENT_MANIFEST.json').read_text())
    for name,digest in manifest['execution_files'].items():assert sha(name)==digest,'Original dependency changed: '+str(name)
    partitions={}
    for outer in cfg['centers']:
        for held in ['outer']+[c for c in cfg['centers'] if c!=outer]:
            part=prepare(outer,held);tr=part['train'];vm=part['valid'];assert vm.shape==(len(rows),12,10)
            assert np.isfinite(part['h']).all() and np.isfinite(part['anchor']['anchor']).all()
            assert (part['coordinate_scale']>0).all()
            path=PREVIOUS/'private/fold_features'/outer/(held+'.joblib');partitions[str(path)]=sha(path)
    for c in cfg['centers']:
        at=g==c;rr=[r for r in rows if r['center']==c]
        table.append(dict(center=c,patients=int(at.sum()),CIN2_positive=int(y[at].sum()),prevalence=float(y[at].mean()),
            CIN3_known=sum(r['cin3'] is not None for r in rr),CIN3_unknown=sum(r['cin3'] is None for r in rr),CIN3_positive=sum(r['cin3']==1 for r in rr),
            valid_OCT_sites_min=12,valid_OCT_frames_min=120,colposcopy_available=sum(bool(r['colpo_paths']) for r in rr),
            age_missing=sum(r['age'] is None for r in rr),HPV_missing=sum(r['hpv'] is None for r in rr),TCT_missing=sum(r['tct'] is None for r in rr)))
    with (HERE/'tables/TABLE_1_COHORT.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    source=json.loads((HERE/'INPUT_LOCK.json').read_text());assert sha(source['cohort_path'])==source['cohort_sha256']
    # No new training until actual import locations and every source split pass.
    save(HERE/'AUDIT_GATE.json',dict(state='PASS',patient_unique_keys=True,centre_case_identity_unique=True,
        imaging_paths_disjoint=True,global_identity_scope='Known centre/case identifiers and imaging paths; universal cross-centre medical identity is unavailable',
        labels=dict(CIN2_positive=int(y.sum()),CIN3_known=sum(r['cin3'] is not None for r in rows),CIN3_unknown=sum(r['cin3'] is None for r in rows)),
        mask_source='Original frozen full120-frame cache mask',source_partitions_checked=25,
        actual_import_files={'runtime':runtime.__file__,'rime_dataset':rime_dataset.__file__,'readout':readout.__file__},
        original_execution_hashes_preserved=True,new_protocol='FR1_NLL',original_protocol='uncentred_AUROC',independent_confirmation=False))
    records=[dict(experiment='Historical PALE',code='Stage197 frozen checkpoint',input='uncentred/AUROC-first',state='COMPLETE_REPRODUCED',result=str(PREVIOUS/'LEGACY_REPRODUCTION.json'),directly_comparable='Historical only'),
        dict(experiment='ARDR six losses',code='Stage200 frozen',input='FR1/NLL-first',state='COMPLETE',result=str(PREVIOUS/'ARDR_MAIN_COMPARISON.csv'),directly_comparable='P0/P3 reusable after exact gate'),
        dict(experiment='PALE core P0-P5',code='Stage201 independent snapshot',input='FR1/NLL-first',state='PREFLIGHT_PASS_NOT_YET_TRAINED',result=str(HERE/'tables'),directly_comparable='Within new protocol'),
        dict(experiment='Recent baselines Track A/B',code='Pinned author repositories',input='Source-only8D or raw frozen features',state='ADAPTERS_PENDING',result=str(HERE/'vendor'),directly_comparable='Track separated'),
        dict(experiment='Native robustness',code='Existing native bundle',input='Must update affected anchor inputs',state='INPUT_COVERAGE_PENDING',result='Not yet evaluated',directly_comparable='Separate from fixed-anchor masking')]
    with (HERE/'EXPERIMENT_AUDIT.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    execution={str(p):sha(p) for p in HERE.glob('*.py')}
    execution.update({str(p):sha(p) for p in (HERE/'code').rglob('*.py')})
    execution[str(HERE/'CONFIG.json')]=sha(HERE/'CONFIG.json')
    execution[str(HERE/'INPUT_LOCK.json')]=sha(HERE/'INPUT_LOCK.json')
    for p in [ROOT/'scripts/rime_dataset.py',ROOT/'scripts/rime_common.py']:execution[str(p)]=sha(p)
    vendor={p.name:subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip() for p in (HERE/'vendor').iterdir() if (p/'.git').exists()}
    save(HERE/'EXPERIMENT_MANIFEST.json',dict(frozen_unix=time.time(),reference_git_commit='3272ab2b16a28738558a9e9a3dbafdb34a1f4783',execution_files=execution,input=source,source_partitions=partitions,vendor_commits=vendor,
        source_fit_budget_per_lightweight_variant=5*4*3*3,original_deadline_retained=True,outer_selection_forbidden=True,main_method_not_changed=True))
    print('Engineering/data/import audit PASS; 25 legal partitions locked; old results preserved',flush=True)
if __name__=='__main__':main()
