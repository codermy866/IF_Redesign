"""Wait for finite queue completion and report all factorial outcomes."""
import argparse
import json
import time
import numpy as np
from train_oct_site_pilot import OUT

FOLDS=('shiyan','enshi','wuhan','jingzhou','xiangyang')
ARMS=('A00','A10','A01','A11')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    start=time.time()
    while len(list((OUT/'runs').glob('*/seed_*/*/complete.json'))) < 20:
        if not args.wait:raise RuntimeError('Pilot incomplete; do not report partial results as final')
        progress=json.loads((OUT/'progress.json').read_text())
        if progress['status']=='failed':raise RuntimeError('Queue failed; inspect run logs')
        if time.time()-start > 3600:raise TimeoutError('Finite one-hour monitor expired')
        time.sleep(10)
    rows={};effects=[];references=[]
    for fold in FOLDS:
        rr={arm:json.loads((OUT/'runs'/fold/'seed_20260905'/arm/'complete.json').read_text())['validation']for arm in ARMS}
        rows[fold]=rr
        references.append(json.loads((OUT/'runs'/fold/'seed_20260905'/'A00'/'complete.json').read_text())['reference'])
        for name,a,b in [('M1_without_M2','A10','A00'),('M1_with_M2','A11','A01'),('M2_without_M1','A01','A00'),('M2_with_M1','A11','A10')]:
            effects.append({'fold':fold,'contrast':name,'site_ap_delta':rr[a]['within_case_site_ap']-rr[b]['within_case_site_ap'],
                            'recall4_delta':rr[a]['site_recall_at4']-rr[b]['site_recall_at4'],
                            'case_macro_ap_delta':rr[a]['selected4']['macro_auprc']-rr[b]['selected4']['macro_auprc']})
        effects.append({'fold':fold,'contrast':'interaction_A11-A10-A01+A00',**{key:sum(sign*rr[arm][metric] for sign,arm in [(1,'A11'),(-1,'A10'),(-1,'A01'),(1,'A00')])for key,metric in [('site_ap_delta','within_case_site_ap'),('recall4_delta','site_recall_at4')]},
                        'case_macro_ap_delta':sum(sign*rr[arm]['selected4']['macro_auprc']for sign,arm in [(1,'A11'),(-1,'A10'),(-1,'A01'),(1,'A00')])})
    summary={arm:{'mean_case_site_ap':float(np.mean([rows[f][arm]['within_case_site_ap']for f in FOLDS])),
                  'mean_recall4':float(np.mean([rows[f][arm]['site_recall_at4']for f in FOLDS])),
                  'mean_fold_macro_ap_at4':float(np.mean([rows[f][arm]['selected4']['macro_auprc']for f in FOLDS]))}for arm in ARMS}
    reference={'mean_case_site_ap':float(np.mean([r['within_case_site_ap']for r in references])),
               'mean_recall4':float(np.mean([r['site_recall_at4']for r in references])),
               'mean_fold_macro_ap_at4':float(np.mean([r['selected4']['macro_auprc']for r in references]))}
    result={'completed':20,'seed_count':1,'test_labels_evaluated':False,'summary':summary,'unmodified_checkpoint_reference':reference,'all_folds':rows,'paired_effects':effects,
            'interpretation':'Exploratory source-only offline retention pilot; no independent causal/clinical validation or seed replication'}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# 点位监督 M1/M2 完整首轮消融','', '20/20 完成；五个源中心划分、一个种子。以下为五个划分指标的算术平均，非独立患者合并统计。','',
           '| 组合 | 点位 AP（病例内、混合阳阴点位病例） | 阳性点位 Recall@4 | 病例 macro-AP@4 |','|---|---:|---:|---:|']
    for arm,v in {'未微调检查点':reference,**summary}.items():lines.append(f"| {arm} | {v['mean_case_site_ap']:.4f} | {v['mean_recall4']:.4f} | {v['mean_fold_macro_ap_at4']:.4f} |")
    lines += ['', 'A00：同期基础微调；A10：点位 BCE；A01：方向性替换排序；A11：两者同时。', '',
              '随机保留四个点位的期望阳性召回率为1/3。点位排序改善不等于病例诊断改善；必须分别检查二者。完整条件消融与交互项见 summary.json。', '',
              '重要：四个微调组合的病例诊断平均AP均低于未微调检查点。M1的点位定位收益不能掩盖诊断退化；A11相对A00的改善尚未恢复既有诊断水平。下一步应隔离诊断骨干与点位监督，检验共享重要性头微调及固定训练轮数造成的退化，而不是直接替换正式模型。', '',
              '本实验读取全部点位特征后排序，是离线证据保留。验证标签仅用于评分，不用于预筛图像。还不能据此声称临床因果必要性、采集成本下降或三随机种子稳健性。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
