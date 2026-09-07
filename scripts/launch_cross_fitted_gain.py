"""Finite five-outer-fold / thirty-fresh-teacher queue, no unrelated job control."""
import subprocess
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
from run_cross_fitted_gain import ROOT,OUT


def worker(gpu,folds):
    records=[]
    for fold in folds:
        path=OUT/fold/'seed_20260905';path.mkdir(parents=True,exist_ok=True)
        if (path/'complete.json').exists():continue
        with (path/'run.log').open('a') as log:
            p=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/run_cross_fitted_gain.py'),'--fold',fold],cwd=ROOT,
              env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
        records.append({'fold':fold,'exit_code':p.returncode});(OUT/f'worker_{gpu}.json').write_text(json.dumps(records,indent=2))
        if p.returncode:break
    return records


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'progress.json').write_text(json.dumps({'status':'running','outer_folds_planned':5,'fresh_teachers_planned':30,'seed':20260905,'started_unix':time.time()}))
    folds=['shiyan','enshi','wuhan','jingzhou','xiangyang']
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(worker,g,folds[g::2])for g in (0,1)];records=[r for t in tasks for r in t.result()]
    complete=len(list(OUT.glob('*/seed_*/complete.json')))
    (OUT/'progress.json').write_text(json.dumps({'status':'complete' if complete==5 else 'failed','completed_outer_folds':complete,'planned_outer_folds':5,'jobs':records},indent=2))
    if complete==5:
        result=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/summarize_cross_fitted_gain.py')],cwd=ROOT)
        if result.returncode:raise RuntimeError('Training completed; summary failed')


if __name__=='__main__':main()
