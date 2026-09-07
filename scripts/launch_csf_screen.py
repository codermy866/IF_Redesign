"""Finite, resumable two-GPU queue; never interrupts other experiments."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import os
import json
import time
import hashlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/csf_screen_v1'
FOLDS = ('shiyan', 'enshi', 'wuhan', 'jingzhou', 'xiangyang')


def worker(gpu, jobs):
    results = []
    for fold, seed in jobs:
        path = OUT/fold/f'seed_{seed}'; path.mkdir(parents=True, exist_ok=True)
        if (path/'complete.json').exists():
            results.append({'fold': fold, 'seed': seed, 'status': 'already_complete'}); continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        start = time.time()
        with (path/'run.log').open('a') as log:
            process = subprocess.run([str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/run_csf_screen.py'), '--fold', fold, '--seed', str(seed)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = {'fold': fold, 'seed': seed, 'exit_code': process.returncode, 'elapsed_s': time.time()-start}
        results.append(row)
        (OUT/f'worker_{gpu}.json').write_text(json.dumps(results, indent=2))
        if process.returncode: break
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scripts = ('run_csf_screen.py', 'launch_csf_screen.py')
    (OUT/'execution_manifest.json').write_text(json.dumps({'started_unix': time.time(), 'scripts_sha256': {s: hashlib.sha256((ROOT/'scripts'/s).read_bytes()).hexdigest() for s in scripts}, 'planned_jobs': 15, 'status': 'running'}, indent=2))
    jobs = [(fold, seed) for seed in (20260905, 20260906, 20260907) for fold in FOLDS]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, gpu, jobs[gpu::2]) for gpu in range(2)]
        results = [row for f in futures for row in f.result()]
    complete = len(list(OUT.glob('*/seed_*/complete.json')))
    (OUT/'progress.json').write_text(json.dumps({'planned': 15, 'completed': complete, 'status': 'complete' if complete == 15 else 'failed', 'jobs': results}, indent=2))
    print(json.dumps({'completed': complete, 'planned': 15}), flush=True)


if __name__ == '__main__': main()
