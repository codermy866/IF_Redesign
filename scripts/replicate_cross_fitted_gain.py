"""Unchanged-protocol replication at two additional predeclared seeds."""
import subprocess
import os
import json
from concurrent.futures import ThreadPoolExecutor
from run_cross_fitted_gain import ROOT,OUT


def worker(gpu,jobs):
    rows=[]
    for fold,seed in jobs:
        path=OUT/fold/f'seed_{seed}';path.mkdir(parents=True,exist_ok=True)
        if (path/'complete.json').exists():continue
        with (path/'run.log').open('a') as log:
            p=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/run_cross_fitted_gain.py'),'--fold',fold,'--seed',str(seed)],cwd=ROOT,
              env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
        rows.append({'fold':fold,'seed':seed,'exit_code':p.returncode});(OUT/f'replication_worker_{gpu}.json').write_text(json.dumps(rows,indent=2))
        if p.returncode:break
    return rows


def main():
    jobs=[(f,s)for s in (20260906,20260907)for f in ('shiyan','enshi','wuhan','jingzhou','xiangyang')]
    (OUT/'replication_progress.json').write_text(json.dumps({'status':'running','additional_runs':10,'additional_teachers':60,'protocol_changes':'none; seeds only'}))
    with ThreadPoolExecutor(max_workers=2)as pool:
        tasks=[pool.submit(worker,g,jobs[g::2])for g in (0,1)];rows=[r for t in tasks for r in t.result()]
    n=len(list(OUT.glob('*/seed_*/complete.json')))
    (OUT/'replication_progress.json').write_text(json.dumps({'status':'complete' if n==15 else 'failed','all_fold_seed_runs_completed':n,'all_planned':15,'jobs':rows},indent=2))
    if n==15:
        result=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/summarize_cross_fitted_gain_replication.py')],cwd=ROOT)
        if result.returncode:raise RuntimeError('Replication finished but summary failed')


if __name__=='__main__':main()
