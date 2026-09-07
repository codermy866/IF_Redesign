"""Finite source-only supervised value-memory training queue."""
from concurrent.futures import ThreadPoolExecutor
import subprocess
import json
import os
import time
from run_csf_memory import ROOT, OUT


def worker(gpu, jobs):
    rows = []
    for fold, seed in jobs:
        path = OUT/fold/f'seed_{seed}'; path.mkdir(parents=True, exist_ok=True)
        if (path/'complete.json').exists(): continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', OMP_NUM_THREADS='2')
        start = time.time()
        with (path/'run.log').open('a') as log:
            p = subprocess.run([str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/run_csf_memory.py'), '--fold', fold, '--seed', str(seed)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        rows.append({'fold': fold, 'seed': seed, 'exit_code': p.returncode, 'elapsed_s': time.time()-start})
        (OUT/f'worker_{gpu}.json').write_text(json.dumps(rows, indent=2))
        if p.returncode: break
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'progress.json').write_text(json.dumps({'planned': 15, 'status': 'running', 'started_unix': time.time()}))
    jobs = [(f, s) for s in (20260905, 20260906, 20260907) for f in ('shiyan', 'enshi', 'wuhan', 'jingzhou', 'xiangyang')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, g, jobs[g::2]) for g in (0, 1)]
        rows = [r for future in futures for r in future.result()]
    complete = len(list(OUT.glob('*/seed_*/complete.json')))
    (OUT/'progress.json').write_text(json.dumps({'planned': 15, 'completed': complete, 'status': 'complete' if complete == 15 else 'failed', 'jobs': rows}, indent=2))
    print(json.dumps({'complete': complete, 'planned': 15}), flush=True)


if __name__ == '__main__': main()
