"""Three-seed replication summary; never count seeds as extra patients."""
import json
import argparse
from pathlib import Path
import numpy as np
from run_cross_fitted_gain import OUT,MODES
from summarize_csf_screen import macro

FOLDS=('shiyan','enshi','wuhan','jingzhou','xiangyang')
SEEDS=(20260905,20260906,20260907)


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args();OUT=args.output
    table={m:[]for m in MODES};curves={m:[]for m in MODES};recalls={m:[]for m in MODES};full=[];contrasts=[];all_indices=set();case_counts={};seed_results=[]
    audit_count=0;gain_rows=0
    for fold in FOLDS:
        ds=[np.load(OUT/fold/f'seed_{s}'/'predictions.npz',allow_pickle=False)for s in SEEDS]
        rs=[json.loads((OUT/fold/f'seed_{s}'/'complete.json').read_text())for s in SEEDS]
        indices=ds[0]['indices'];y=ds[0]['y'];c=ds[0]['centers'];all_indices.update(indices.tolist())
        case_counts[fold]={str(h):{'n':int(sum(c==h)),'positive':int(sum(y[c==h]))}for h in np.unique(c)}
        for d in ds:
            np.testing.assert_array_equal(d['indices'],indices);np.testing.assert_array_equal(d['y'],y)
            for m in MODES:
                assert np.isfinite(d[m]).all()
                assert all(len(set(row))==6 for row in d[m+'_actions'])
        for seed,r in zip(SEEDS,rs):
            root=OUT/fold/f'seed_{seed}'
            assert not r['test_labels_evaluated'];gain_rows+=r['gain_action_rows']
            provenance=json.loads((root/'target_provenance.json').read_text())
            teachers={i:json.loads((root/f'teacher_{i}'/'complete.json').read_text())for i in range(5)}
            for p in provenance:
                t=teachers[p['teacher']];i=p['patient_index']
                if p['kind']=='oof':assert i in t['gain_indices'] and i not in t['fit_indices'] and i not in t['cal_indices']
                else:assert i in t['fit_indices']
                audit_count+=1
            seed_results.append({'fold':fold,'seed':seed,'policies':{m:r['policies'][m]['budgets'][4]for m in MODES},'full':r['full']})
        for m in MODES:
            table[m].append([r['policies'][m]['budgets'][4]['macro_auprc']for r in rs])
            curves[m].append([[r['policies'][m]['budgets'][k]['macro_auprc']for k in range(7)]for r in rs])
            recalls[m].append([r['policies'][m]['mean_case_site_recall_at4']for r in rs])
        full.append([r['full']['macro_auprc']for r in rs])
        strata=[np.where((c==h)&(y==label))[0]for h in np.unique(c)for label in (0,1)];strata=[s for s in strata if len(s)]
        rng=np.random.default_rng(20260906);boot={m:[]for m in MODES};boot['full']=[]
        for _ in range(1000):
            ix=np.concatenate([rng.choice(s,len(s),replace=True)for s in strata])
            for m in MODES:boot[m].append(float(np.mean([macro(y[ix],d[m][ix,4],c[ix])for d in ds])))
            boot['full'].append(float(np.mean([macro(y[ix],d['full'][ix],c[ix])for d in ds])))
        for other in [m for m in MODES if m!='gain_oof']+['full']:
            delta=np.mean(table['gain_oof'][-1])-np.mean(full[-1] if other=='full' else table[other][-1])
            sample=np.array(boot['gain_oof'])-np.array(boot[other])
            contrasts.append({'fold':fold,'left':'gain_oof','right':other,'seed_mean_delta_macro_ap_at4':float(delta),
                              'paired_patient_bootstrap_ci95':np.quantile(sample,[.025,.975]).tolist()})
    values={m:{'mean_15_run_macro_ap_at4':float(np.mean(table[m])),
               'macro_ap_mean_by_seed':np.mean(table[m],axis=0).tolist(),
               'macro_ap_mean_by_budget_0_to_6':np.mean(curves[m],axis=(0,1)).tolist(),
               'mean_case_site_recall_at4':float(np.mean(recalls[m]))}for m in MODES}
    best=max(values,key=lambda m:values[m]['mean_15_run_macro_ap_at4'])
    result={'runs_complete':15,'fresh_diagnostic_models':90,'seeds':list(SEEDS),'unique_validation_patients':len(all_indices),
            'policies':values,'full_mean_15_run_macro_ap':float(np.mean(full)),'paired_contrasts':contrasts,'seed_results':seed_results,
            'center_class_counts':case_counts,'best_development_at4':best,'independent_superiority_established':False,
            'test_labels_evaluated':False,'verified_target_provenance_rows':audit_count,'training_action_rows':gain_rows,
            'statistics':'seed-averaged metric differences, not probability ensemble;1000 stratified paired patient bootstrap draws within each fold;descriptive unadjusted intervals;overlapping folds not independent'}
    (OUT/'replication_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    lines=['# Cross-fitted gain：三种子完整重复实验','',
           f'15/15划分-种子运行、90个重新训练的诊断模型；独立验证患者{len(all_indices)}名。未评价留出医院。','',
           '| 方法 | 15次运行平均 macro-AP@4 | 点位召回@4 |','|---|---:|---:|']
    for m,v in values.items():lines.append(f"| {m} | {v['mean_15_run_macro_ap_at4']:.4f} | {v['mean_case_site_recall_at4']:.4f} |")
    lines += ['',f'本次开发集固定预算4的均值最高方法：{best}。结果表包含全部基线；该选择并不等于独立验证优越性。','',
              '完整预算曲线、每个种子的表现、各中心类别数和配对区间见replication_summary.json。统计单位是患者，三种子平均指标不是病例数扩充，也不是概率集成。', '',
              '方法只对已提取的所有候选特征进行离线排序；不能把保留4个点位解释为只采集4个点位。OOF收益不是临床因果效应。不得将历史协议的最好值拼成当前模型的结果。']
    (OUT/'REPLICATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'policies':values,'best_development':best},indent=2),flush=True)


if __name__=='__main__':main()
