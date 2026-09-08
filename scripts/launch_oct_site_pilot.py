"""Resumable finite 20-run two-GPU factorial pilot."""
import os
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor
from train_oct_site_pilot import ROOT,OUT


def worker(gpu,jobs):
    status=[]
    for fold,arm in jobs:
        path=OUT/'runs'/fold/'seed_20260905'/arm;path.mkdir(parents=True,exist_ok=True)
        if (path/'complete.json').exists():continue
        with (path/'run.log').open('a') as log:
            p=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/train_oct_site_pilot.py'),'--fold',fold,'--arm',arm],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
        status.append({'fold':fold,'arm':arm,'exit_code':p.returncode})
        (OUT/f'worker_{gpu}.json').write_text(json.dumps(status,indent=2))
        if p.returncode:break
    return status


def main():
    jobs=[(fold,arm)for fold in ('shiyan','enshi','wuhan','jingzhou','xiangyang')for arm in ('A00','A10','A01','A11')]
    (OUT/'progress.json').write_text(json.dumps({'planned':20,'status':'running','seed':20260905}))
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(worker,g,jobs[g::2])for g in (0,1)];rows=[r for t in tasks for r in t.result()]
    n=len(list((OUT/'runs').glob('*/seed_*/*/complete.json')))
    (OUT/'progress.json').write_text(json.dumps({'planned':20,'completed':n,'status':'complete' if n==20 else 'failed','jobs':rows},indent=2))


if __name__=='__main__':main()
